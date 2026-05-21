# Use an official Python base image
FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install service dependencies
COPY services/vla-inference/pyproject.toml .
RUN pip install --no-cache-dir .

# Copy the core library
COPY core/ /app/core/
RUN pip install --no-cache-dir ./core

# Copy service code
COPY services/vla-inference/ /app/services/vla-inference/

# Set PYTHONPATH to include the core and services
ENV PYTHONPATH="/app:/app/core:/app/services/vla-inference"

ENTRYPOINT ["python", "services/vla-inference/run_on_media.py"]
