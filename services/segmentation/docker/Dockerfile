FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY services/segmentation/pyproject.toml .
RUN pip install --no-cache-dir .

COPY core/ /app/core/
RUN pip install --no-cache-dir ./core

COPY services/segmentation/ /app/services/segmentation/

ENV PYTHONPATH="/app:/app/core:/app/services/segmentation/src"

ENTRYPOINT ["python", "services/segmentation/src/segmentation.py"]
