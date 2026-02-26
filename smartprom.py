#!/usr/bin/env python3
import json
import os
import subprocess
import time
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Tuple, Dict, Optional

import prometheus_client

import megaraid

LABELS = [
    "drive",
    "type",
    "model_family",
    "model_name",
    "serial_number",
    "user_capacity",
]
DRIVES = {}
METRICS = {}

# https://www.smartmontools.org/wiki/USB
SAT_TYPES = ["sat", "usbjmicron", "usbprolific", "usbsunplus"]
NVME_TYPES = ["nvme", "sntasmedia", "sntjmicron", "sntrealtek"]
SCSI_TYPES = ["scsi"]

# IO tracking state
DISK_IO_STATE: Dict[str, 'DiskIOState'] = {}

# Cron scheduling state
CRON_ITER = None
NEXT_FORCED_SCAN: Optional[datetime] = None


@dataclass
class DiskIOState:
    """Tracks IO activity state for a single disk."""
    diskstats_name: str  # Device name in /proc/diskstats format (e.g., "sda", "nvme0n1")
    last_reads: int = 0
    last_writes: int = 0
    last_io_time: float = 0.0  # Timestamp when IO activity was last detected
    last_scan_time: float = 0.0  # Timestamp of last successful SMART scan


def parse_diskstats() -> Dict[str, Tuple[int, int]]:
    """
    Parse /proc/diskstats and return dict of {device_name: (reads_completed, writes_completed)}.

    This is a passive read operation that does NOT wake sleeping disks.

    Returns:
        dict: Maps device names like "sda", "nvme0n1" to tuples of (reads, writes)
    """
    stats = {}
    try:
        with open('/proc/diskstats', 'r') as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 14:
                    # Fields: major minor name reads_completed reads_merged sectors_read time_reading
                    #         writes_completed writes_merged sectors_written time_writing
                    #         ios_in_progress time_io weighted_time_io
                    device_name = parts[2]
                    reads_completed = int(parts[3])
                    writes_completed = int(parts[7])
                    stats[device_name] = (reads_completed, writes_completed)
    except FileNotFoundError:
        print("WARNING: /proc/diskstats not found. IO tracking disabled.")
    except Exception as e:
        print(f"WARNING: Error reading /proc/diskstats: {e}")
    return stats


def map_drive_to_diskstats_name(drive: str, drive_attrs: dict) -> str:
    """
    Map a SMART device path to its /proc/diskstats device name.

    Args:
        drive: Device path like "/dev/sda" or "megaraid,0"
        drive_attrs: Drive attributes dict from DRIVES

    Returns:
        Device name as it appears in /proc/diskstats (e.g., "sda", "nvme0n1")
        Returns empty string if mapping cannot be determined (e.g., MegaRAID)
    """
    # MegaRAID devices cannot be passively monitored via /proc/diskstats
    if "megaraid_id" in drive_attrs:
        return ""

    # Standard devices: "/dev/sda" -> "sda", "/dev/nvme0n1" -> "nvme0n1"
    if drive.startswith("/dev/"):
        return drive[5:]  # Strip "/dev/"

    return ""


def init_disk_io_state():
    """
    Initialize DISK_IO_STATE for all discovered drives.
    Called after get_drives() populates DRIVES.
    """
    global DRIVES, DISK_IO_STATE

    current_stats = parse_diskstats()

    for drive, drive_attrs in DRIVES.items():
        diskstats_name = map_drive_to_diskstats_name(drive, drive_attrs)

        reads, writes = 0, 0
        if diskstats_name and diskstats_name in current_stats:
            reads, writes = current_stats[diskstats_name]

        DISK_IO_STATE[drive] = DiskIOState(
            diskstats_name=diskstats_name,
            last_reads=reads,
            last_writes=writes,
            last_io_time=0.0,  # No IO detected yet
            last_scan_time=0.0,
        )

        if diskstats_name:
            print(f"IO tracking enabled for {drive} -> {diskstats_name}")
        else:
            print(f"IO tracking disabled for {drive} (MegaRAID/unmappable, will only scan on forced schedule)")


def update_io_activity():
    """
    Sample /proc/diskstats and update DISK_IO_STATE for all tracked drives.

    Compares current read/write counters against previous values.
    If counters increased, marks the disk as having recent IO activity.
    """
    global DISK_IO_STATE

    current_stats = parse_diskstats()
    current_time = time.time()

    for drive, state in DISK_IO_STATE.items():
        if not state.diskstats_name:
            continue  # Skip MegaRAID or unmappable devices

        if state.diskstats_name in current_stats:
            reads, writes = current_stats[state.diskstats_name]

            # Check if IO counters increased since last check
            if reads > state.last_reads or writes > state.last_writes:
                state.last_io_time = current_time
                print(f"IO activity detected on {drive}")

            # Update stored counters
            state.last_reads = reads
            state.last_writes = writes


def should_scan_disk(drive: str, drive_attrs: dict, force_scan: bool, io_activity_threshold: int) -> bool:
    """
    Determine if a disk should be scanned based on IO activity or forced scan.

    Args:
        drive: Device path
        drive_attrs: Drive attributes
        force_scan: True if cron triggered a forced scan
        io_activity_threshold: Seconds after IO to consider disk "active"

    Returns:
        True if the disk should be scanned
    """
    global DISK_IO_STATE

    # Always scan if forced
    if force_scan:
        return True

    state = DISK_IO_STATE.get(drive)
    if state is None:
        return True  # New drive, scan it

    # For MegaRAID devices without diskstats mapping, only scan on forced
    if not state.diskstats_name:
        return False  # Wait for forced scan

    # Always scan nvme disks
    if drive_attrs["type"] in NVME_TYPES:
        return True

    # Check if disk had IO activity within the threshold window
    time_since_io = time.time() - state.last_io_time

    return time_since_io < io_activity_threshold


def init_cron_schedule(cron_expr: str):
    """
    Initialize the cron schedule iterator if SMARTCTL_CRON_SCHEDULE is set.

    Args:
        cron_expr: Cron expression string (e.g., "0 3 * * *")

    Returns:
        Tuple of (croniter instance or None, next fire time as datetime or None)
    """
    if not cron_expr:
        print("No cron schedule configured. Forced scans disabled.")
        return None, None

    try:
        from croniter import croniter
        cron = croniter(cron_expr, datetime.now())
        next_time = cron.get_next(datetime)
        print(f"Cron schedule enabled: '{cron_expr}', next forced scan at {next_time}")
        return cron, next_time
    except ImportError:
        print(f"WARNING: croniter package not installed. Cron schedule '{cron_expr}' ignored.")
        print("Install with: pip install croniter")
        return None, None
    except Exception as e:
        print(f"WARNING: Invalid cron expression '{cron_expr}': {e}")
        return None, None


def check_cron_trigger() -> bool:
    """
    Check if the cron schedule has triggered a forced scan.

    Returns:
        True if a forced scan should occur
    """
    global CRON_ITER, NEXT_FORCED_SCAN

    if CRON_ITER is None or NEXT_FORCED_SCAN is None:
        return False

    if datetime.now() >= NEXT_FORCED_SCAN:
        # Update to next scheduled time
        NEXT_FORCED_SCAN = CRON_ITER.get_next(datetime)
        print(f"Cron triggered forced scan. Next scheduled at {NEXT_FORCED_SCAN}")
        return True

    return False


def run_smartctl_cmd(args: list) -> Tuple[str, int]:
    """
    Runs the smartctl command on the system
    """
    out = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    stdout, stderr = out.communicate()

    # exit code can be != 0 even if the command returned valid data
    # see EXIT STATUS in
    # https://www.smartmontools.org/browser/trunk/smartmontools/smartctl.8.in
    if out.returncode != 0:
        stdout_msg = stdout.decode("utf-8") if stdout is not None else ""
        stderr_msg = stderr.decode("utf-8") if stderr is not None else ""
        print(
            f"WARNING: Command returned exit code {out.returncode}. "
            f"Stdout: '{stdout_msg}' Stderr: '{stderr_msg}'"
        )

    return stdout.decode("utf-8"), out.returncode


def get_drives() -> dict:
    """
    Returns a dictionary of devices and its types
    """
    disks = {}
    result, _ = run_smartctl_cmd(["smartctl", "--scan-open", "--json=c"])
    result_json = json.loads(result)

    if "devices" in result_json:
        devices = result_json["devices"]

        # Ignore devices that fail on open, such as Virtual Drives created by MegaRAID.
        devices = list(
            filter(
                lambda x: (
                    x.get("open_error", "")
                    != "DELL or MegaRaid controller, please try adding '-d megaraid,N'"
                ),
                devices,
            )
        )

        for device in devices:
            dev = device["name"]
            if re.match(megaraid.MEGARAID_TYPE_PATTERN, device["type"]):
                # If drive is connected by MegaRAID, dev has a bus name like "/dev/bus/0".
                # After retrieving the disk information using the bus name,
                # replace dev with a disk ID such as "megaraid,0".
                disk_attrs = megaraid.get_megaraid_device_info(dev, device["type"])
                disk_attrs["type"] = megaraid.get_megaraid_device_type(
                    dev, device["type"]
                )
                disk_attrs["bus_device"] = dev
                disk_attrs["megaraid_id"] = megaraid.get_megaraid_device_id(
                    device["type"]
                )
                dev = disk_attrs["megaraid_id"]
            else:
                disk_attrs = get_device_info(dev)
                disk_attrs["type"] = device["type"]
            disks[dev] = disk_attrs
            print("Discovered device", dev, "with attributes", disk_attrs)
    else:
        print("No devices found. Make sure you have enough privileges.")
    return disks


def get_device_info(dev: str) -> dict:
    """
    Returns a dictionary of device info
    """
    results, _ = run_smartctl_cmd(["smartctl", "-i", "--json=c", dev])
    results = json.loads(results)
    user_capacity = "Unknown"
    if "user_capacity" in results and "bytes" in results["user_capacity"]:
        user_capacity = str(results["user_capacity"]["bytes"])
    return {
        "model_family": results.get("model_family", "Unknown"),
        "model_name": results.get("model_name", "Unknown"),
        "serial_number": results.get("serial_number", "Unknown"),
        "user_capacity": user_capacity,
    }


def get_smart_status(results: dict) -> int:
    """
    Returns a 1, 0 or -1 depending on if result from
    smart status is True, False or unknown.
    """
    status = results.get("smart_status")
    return +(status.get("passed")) if status is not None else -1


def smart_sat(dev: str) -> dict:
    """
    Runs the smartctl command on a internal or external "sat" device
    and processes its attributes
    """
    results, exit_code = run_smartctl_cmd(
        ["smartctl", "-A", "-H", "-d", "sat", "--json=c", dev]
    )
    results = json.loads(results)

    attributes = table_to_attributes_sat(results["ata_smart_attributes"]["table"])
    attributes["smart_passed"] = (0, get_smart_status(results))
    attributes["exit_code"] = (0, exit_code)
    return attributes


def table_to_attributes_sat(data: dict) -> dict:
    """
    Returns a results["ata_smart_attributes"]["table"]
    processed into an attributes dict
    """
    attributes = {}
    for metric in data:
        code = metric["id"]
        name = metric["name"]
        value = metric["value"]

        # metric['raw']['value'] contains values difficult to understand for temperatures and time up
        # that's why we added some logic to parse the string value
        value_raw = metric["raw"]["string"]
        try:
            # example value_raw: "33" or "43 (Min/Max 39/46)"
            value_raw = int(value_raw.split()[0])
        except:
            # example value_raw: "20071h+27m+15.375s"
            if "h+" in value_raw:
                value_raw = int(value_raw.split("h+")[0])
            else:
                print(
                    f"Raw value of sat metric '{name}' can't be parsed. raw_string: {value_raw} "
                    f"raw_int: {metric['raw']['value']}"
                )
                value_raw = None

        attributes[name] = (int(code), value)
        if value_raw is not None:
            attributes[f"{name}_raw"] = (int(code), value_raw)
    return attributes


def smart_nvme(dev: str) -> dict:
    """
    Runs the smartctl command on a internal or external "nvme" device
    and processes its attributes
    """
    results, exit_code = run_smartctl_cmd(
        ["smartctl", "-A", "-H", "-d", "nvme", "--json=c", dev]
    )
    results = json.loads(results)

    attributes = {"smart_passed": get_smart_status(results), "exit_code": exit_code}
    data = results["nvme_smart_health_information_log"]
    for key, value in data.items():
        if key == "temperature_sensors":
            for i, _value in enumerate(value, start=1):
                attributes[f"temperature_sensor{i}"] = _value
        else:
            attributes[key] = value
    return attributes


def smart_scsi(dev: str) -> dict:
    """
    Runs the smartctl command on a "scsi" device
    and processes its attributes
    """
    results, exit_code = run_smartctl_cmd(
        ["smartctl", "-A", "-H", "-d", "scsi", "--json=c", dev]
    )
    results = json.loads(results)

    attributes = results_to_attributes_scsi(results)
    attributes["smart_passed"] = get_smart_status(results)
    attributes["exit_code"] = exit_code
    return attributes


def results_to_attributes_scsi(data: dict) -> dict:
    """
    Returns the result of smartctl -i on the SCSI device
    processed into an attributes dict
    """
    attributes = {}
    for key, value in data.items():
        if type(value) == dict:
            for _label, _value in value.items():
                if type(_value) == int:
                    attributes[f"{key}_{_label}"] = _value
        elif type(value) == int:
            attributes[key] = value
    return attributes


def collect(force_scan: bool = False, io_activity_threshold: int = 60):
    """
    Collect drive metrics and save them as Gauge type.
    Only scans drives that had recent IO activity unless force_scan is True.

    Args:
        force_scan: If True, scan all drives regardless of IO activity
        io_activity_threshold: Seconds after IO to consider disk "active"
    """
    global LABELS, DRIVES, METRICS, SAT_TYPES, NVME_TYPES, SCSI_TYPES, DISK_IO_STATE

    for drive, drive_attrs in DRIVES.items():
        # Check if this drive should be scanned
        if not should_scan_disk(drive, drive_attrs, force_scan, io_activity_threshold):
            print(f"Skipping idle disk {drive}")
            continue  # Skip idle disk

        # Update last scan time
        if drive in DISK_IO_STATE:
            DISK_IO_STATE[drive].last_scan_time = time.time()

        typ = drive_attrs["type"]
        try:
            print(f"Collecting SMART metrics for {drive} (type: {typ})")
            if "megaraid_id" in drive_attrs:
                attrs = megaraid.smart_megaraid(
                    drive_attrs["bus_device"], drive_attrs["megaraid_id"]
                )
            elif typ in SAT_TYPES:
                attrs = smart_sat(drive)
            elif typ in NVME_TYPES:
                attrs = smart_nvme(drive)
            elif typ in SCSI_TYPES:
                attrs = smart_scsi(drive)
            else:
                continue

            for key, values in attrs.items():
                # Metric name in lower case
                metric = (
                    "smartprom_"
                    + key.replace("-", "_")
                    .replace(" ", "_")
                    .replace(".", "")
                    .replace("/", "_")
                    .lower()
                )

                # Create metric if it does not exist
                if metric not in METRICS:
                    desc = key.replace("_", " ")
                    code = hex(values[0]) if typ in SAT_TYPES else hex(values)
                    print(f"Adding new gauge {metric} ({code})")
                    METRICS[metric] = prometheus_client.Gauge(
                        metric, f"({code}) {desc}", LABELS
                    )

                # Update metric
                metric_val = values[1] if typ in SAT_TYPES else values

                METRICS[metric].labels(
                    drive=drive,
                    type=typ,
                    model_family=drive_attrs["model_family"],
                    model_name=drive_attrs["model_name"],
                    serial_number=drive_attrs["serial_number"],
                    user_capacity=drive_attrs["user_capacity"],
                ).set(metric_val)

        except Exception as e:
            print("Exception:", e)
            pass


def main():
    """
    Starts a server and exposes the metrics
    """
    global DRIVES, CRON_ITER, NEXT_FORCED_SCAN

    # Validate configuration
    exporter_address = os.environ.get("SMARTCTL_EXPORTER_ADDRESS", "0.0.0.0")
    exporter_port = int(os.environ.get("SMARTCTL_EXPORTER_PORT", 9902))
    refresh_interval = int(os.environ.get("SMARTCTL_REFRESH_INTERVAL", 60))
    metrics_file_enable = os.environ.get("SMARTCTL_METRICS_FILE_ENABLE", False)
    metrics_file_path = os.environ.get("SMARTCTL_METRICS_FILE_PATH", "/metrics/")

    # IO monitoring configuration
    io_check_interval = int(os.environ.get("SMARTCTL_IO_CHECK_INTERVAL", 30))
    io_activity_threshold = int(os.environ.get("SMARTCTL_IO_ACTIVITY_THRESHOLD", 60))
    cron_schedule = os.environ.get("SMARTCTL_CRON_SCHEDULE", "").strip()

    # Get drives (test smartctl)
    DRIVES = get_drives()

    # Initialize IO tracking state
    init_disk_io_state()

    # Initialize cron schedule
    CRON_ITER, NEXT_FORCED_SCAN = init_cron_schedule(cron_schedule)

    # Start Prometheus server
    prometheus_client.start_http_server(exporter_port, exporter_address)
    print(f"Server listening in http://{exporter_address}:{exporter_port}/metrics")

    # Perform initial scan of all drives for immediate metrics availability
    print("Performing initial scan of all drives...")
    collect(force_scan=True, io_activity_threshold=io_activity_threshold)
    if metrics_file_enable:
        prometheus_client.write_to_textfile(metrics_file_path + "smartctl.prom", prometheus_client.REGISTRY)

    # Scheduling loop
    last_io_check = time.time()
    last_collect = time.time()

    while True:
        current_time = time.time()

        # Check IO activity periodically
        if current_time - last_io_check >= io_check_interval:
            update_io_activity()
            last_io_check = current_time

        # Check if cron triggered a forced scan
        force_scan = check_cron_trigger()

        # Run collection at refresh_interval (or immediately if forced)
        if force_scan or (current_time - last_collect >= refresh_interval):
            collect(force_scan=force_scan, io_activity_threshold=io_activity_threshold)
            last_collect = current_time

            if metrics_file_enable:
                prometheus_client.write_to_textfile(
                    metrics_file_path + "smartctl.prom",
                    prometheus_client.REGISTRY
                )

        # Sleep for a short interval to check conditions
        time.sleep(min(io_check_interval, refresh_interval, 10))


if __name__ == "__main__":
    main()
