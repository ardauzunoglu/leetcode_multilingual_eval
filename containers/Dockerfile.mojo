FROM ubuntu:24.04

ARG DEBIAN_FRONTEND=noninteractive
ARG MOJO_VERSION=1.0.0b2
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        clang \
        curl \
        libc6-dev \
        python3 \
        python3-pip \
        python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/mojo \
    && /opt/mojo/bin/pip install --no-cache-dir --pre "mojo==${MOJO_VERSION}" \
    && /opt/mojo/bin/mojo --version

ENV PATH="/opt/mojo/bin:${PATH}"
WORKDIR /work

