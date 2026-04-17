# Software-as-a-Service — runnable environment for scripts that call Data-as-a-Service.
# Heavy ML (SAM3, full torch stacks) is optional: mount ``packages/`` and extend this image locally if needed.

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

COPY datara_client.py .
COPY . .

# Default: verify API connectivity; override with your script.
CMD ["python", "-c", "from datara_client import DataraAPIClient; c=DataraAPIClient(); print(c.health())"]
