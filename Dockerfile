# Use Ubuntu 22.04 slim base
FROM ubuntu:22.04
#FROM gcc:13
#RUN apt-get update && apt-get install -y python3.11 python3.11-venv python3-pip libopenblas-dev pkg-config libgfortran-13-dev cmake git
# Install Python 3.11, pip, venv, and build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
      cmake \
      git \
      python3.11 \
      python3.11-venv \
      python3.11-dev \
      python3-pip \
      gfortran \
      build-essential \
      libopenblas-dev \
      pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip, setuptools, wheel
RUN which python3.11 
RUN python3.11 -m pip install --upgrade pip setuptools wheel

RUN export CC=gcc
RUN export FC=gfortran
# Optional: create a virtual environment inside the container
RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install tblite from source so it links to the installed gfortran
RUN pip install --no-binary=tblite --no-cache-dir tblite

# Verify
RUN python3.11 -c "import tblite.interface; print('tblite C extension loaded successfully')"