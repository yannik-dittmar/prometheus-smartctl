FROM --platform=$BUILDPLATFORM ghcr.io/astral-sh/uv:0.13.6 AS uv

# uv is not available for every target platform, so the virtualenv is built on the
# build platform and copied into the final image. This only works because all
# dependencies are pure Python and the interpreter path is the same in both images.
FROM --platform=$BUILDPLATFORM python:3.14-alpine3.24 AS builder
COPY --from=uv /uv /bin/uv

ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app
COPY pyproject.toml uv.lock /app/
RUN uv sync --locked --no-dev --no-cache

FROM python:3.14-alpine3.24

# Install smartmontools
RUN apk add --no-cache smartmontools

COPY --from=builder /app/.venv /app/.venv
# Fail the build if the copied virtualenv does not work on this platform
RUN /app/.venv/bin/python -c "import croniter, prometheus_client"
COPY ./smartprom.py /megaraid.py /

EXPOSE 9902
ENTRYPOINT ["/app/.venv/bin/python", "-u", "/smartprom.py"]

# HELP
# docker build -t matusnovak/prometheus-smartctl:test .
