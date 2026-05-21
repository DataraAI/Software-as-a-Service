FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY services/image-tagging/pyproject.toml .
RUN pip install --no-cache-dir .

COPY core/ /app/core/
RUN pip install --no-cache-dir ./core

COPY services/image-tagging/ /app/services/image-tagging/

ENV PYTHONPATH="/app:/app/core:/app/services/image-tagging/src"

ENTRYPOINT ["python", "services/image-tagging/src/post_annotation.py"]
