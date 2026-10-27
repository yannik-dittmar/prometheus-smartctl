FROM python:3.14-alpine3.24

# Install smartmontools
RUN apk add --no-cache smartmontools

# Install Python dependencies
COPY requirements.txt /
RUN pip install -r /requirements.txt \
    # remove temporary files
    && rm -rf /root/.cache

COPY ./smartprom.py /megaraid.py /

EXPOSE 9902
ENTRYPOINT ["/usr/local/bin/python", "-u", "/smartprom.py"]

# HELP
# docker build -t matusnovak/prometheus-smartctl:test .
