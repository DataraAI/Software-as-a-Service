"""Celery application for the SaaS generation worker."""

from celery import Celery

from generation_worker.config import CELERY_BROKER_URL, CELERY_QUEUE_NAME, validate_worker_config


validate_worker_config()
celery_app = Celery("datara_saas", broker=CELERY_BROKER_URL, include=["generation_worker.tasks"])
celery_app.conf.update(
    accept_content=["json"],
    broker_connection_retry_on_startup=True,
    broker_transport_options={"confirm_publish": True},
    enable_utc=True,
    result_backend=None,
    task_acks_late=True,
    task_default_delivery_mode=2,
    task_default_queue=CELERY_QUEUE_NAME,
    task_ignore_result=True,
    task_reject_on_worker_lost=True,
    task_serializer="json",
    timezone="UTC",
    worker_prefetch_multiplier=1,
)
