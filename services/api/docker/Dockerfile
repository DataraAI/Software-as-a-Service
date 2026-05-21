FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install API dependencies
COPY services/api/pyproject.toml .
RUN pip install --no-cache-dir .

# Copy core library
COPY core/ /app/core/
RUN pip install --no-cache-dir ./core

# Copy only the necessary services for the API gateway
COPY services/api/ /app/services/api/
COPY services/synthetic-data/src/ /app/services/synthetic-data/src/
COPY services/image-tagging/src/ /app/services/image-tagging/src/
COPY services/vla-inference/ /app/services/vla-inference/

# Set PYTHONPATH
ENV PYTHONPATH="/app"

COPY services/api/docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 8765

ENTRYPOINT ["/entrypoint.sh"]
CMD ["uvicorn", "services.api.src.main:app", "--host", "0.0.0.0", "--port", "8765"]
