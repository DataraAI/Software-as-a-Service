"""Environment configuration for the SaaS generation worker."""

from __future__ import annotations

import os
from urllib.parse import urlparse


DAAS_INTERNAL_API_URL = os.getenv("DAAS_INTERNAL_API_URL", "http://localhost:5151").rstrip("/")
ALLOW_INSECURE_DAAS_HTTP = os.getenv("ALLOW_INSECURE_DAAS_HTTP", "").strip().lower() in {
    "1",
    "true",
    "yes",
}
GENERATION_WORKER_TOKEN = os.getenv("GENERATION_WORKER_TOKEN", "")
CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", "amqp://guest:guest@localhost:5672//")
CELERY_QUEUE_NAME = os.getenv("CELERY_QUEUE_NAME", "lambda_jobs")
WORKER_HTTP_TIMEOUT_SECONDS = int(os.getenv("WORKER_HTTP_TIMEOUT_SECONDS", "30"))
WORKER_LONG_REQUEST_TIMEOUT_SECONDS = int(os.getenv("WORKER_LONG_REQUEST_TIMEOUT_SECONDS", "1800"))


def validate_worker_config(
    *,
    base_url: str = DAAS_INTERNAL_API_URL,
    worker_token: str = GENERATION_WORKER_TOKEN,
    broker_url: str = CELERY_BROKER_URL,
) -> None:
    parsed = urlparse(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("DAAS_INTERNAL_API_URL must be an absolute HTTP(S) URL")
    if (
        parsed.scheme != "https"
        and parsed.hostname not in {"localhost", "127.0.0.1", "::1"}
        and not ALLOW_INSECURE_DAAS_HTTP
    ):
        raise ValueError("DAAS_INTERNAL_API_URL must use HTTPS outside localhost")
    if not worker_token or worker_token.startswith("replace-with-"):
        raise ValueError("GENERATION_WORKER_TOKEN must be configured")
    broker = urlparse(broker_url)
    if broker.scheme not in {"amqp", "amqps"} or not broker.hostname or not broker.username:
        raise ValueError("CELERY_BROKER_URL must be a valid RabbitMQ URL")
    if not broker.password:
        raise ValueError("CELERY_BROKER_URL must include a RabbitMQ password")
    if parsed.hostname not in {"localhost", "127.0.0.1", "::1"} and broker.hostname in {
        "localhost",
        "127.0.0.1",
        "::1",
    }:
        raise ValueError("CELERY_BROKER_URL cannot use localhost outside local development")
