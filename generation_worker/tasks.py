"""Celery task and testable message processor."""

from __future__ import annotations

import uuid
import threading
import socket
from typing import Any, Callable

from generation_worker.config import (
    DAAS_INTERNAL_API_URL,
    GENERATION_WORKER_TOKEN,
    WORKER_HTTP_TIMEOUT_SECONDS,
    WORKER_LONG_REQUEST_TIMEOUT_SECONDS,
)
from generation_worker.daas_client import DaasClient, WorkerApiUnavailable
from generation_worker.runner import run_generation


def build_worker_id(_task_id: str | None = None) -> str:
    """Return a unique lease owner for this delivery, not the reusable Celery task ID."""
    return f"{socket.gethostname()}:{uuid.uuid4().hex}"


def _lease_transition_succeeded(response: Any) -> bool:
    return isinstance(response, dict) and response.get("ok") is True


def process_generation_message(
    *,
    job_id: str,
    job_type: str,
    schema_version: int,
    client: Any,
    runner: Callable[[str, dict[str, Any]], dict[str, Any]] = run_generation,
    heartbeat_interval_seconds: float = 30,
) -> str:
    claim = client.claim(job_id, job_type, schema_version)
    outcome = str(claim.get("outcome") or "invalid")
    if outcome != "claimed":
        return outcome

    stop_heartbeat = threading.Event()
    lease_lost = threading.Event()

    def heartbeat() -> None:
        while not stop_heartbeat.wait(heartbeat_interval_seconds):
            try:
                if not _lease_transition_succeeded(client.heartbeat(job_id)):
                    lease_lost.set()
                    return
            except Exception:
                # Completion remains lease-checked by DaaS; a transient heartbeat
                # failure must not expose details or terminate local compute.
                pass

    heartbeat_thread = threading.Thread(target=heartbeat, daemon=True)
    heartbeat_thread.start()
    try:
        if not _lease_transition_succeeded(client.stage(job_id, "Processing")):
            return "leased"
        response = runner(job_type, claim.get("execution") or {})
        if lease_lost.is_set():
            return "leased"
        if response.get("status") != "succeeded":
            client.fail(job_id, str(response.get("error") or "Generation request failed"))
            return "failed"
        if not _lease_transition_succeeded(client.stage(job_id, "Finalizing")):
            return "leased"
        manifest = response.get("manifest")
        completed = client.complete(job_id, manifest if isinstance(manifest, dict) else {})
        return str(completed.get("outcome") or "invalid")
    except WorkerApiUnavailable:
        raise
    except Exception as exc:
        try:
            client.fail(job_id, str(exc))
        except WorkerApiUnavailable:
            raise
        return "failed"
    finally:
        stop_heartbeat.set()
        heartbeat_thread.join(timeout=max(1.0, heartbeat_interval_seconds))


def build_client(worker_id: str) -> DaasClient:
    return DaasClient(
        base_url=DAAS_INTERNAL_API_URL,
        worker_token=GENERATION_WORKER_TOKEN,
        worker_id=worker_id,
        timeout_seconds=WORKER_HTTP_TIMEOUT_SECONDS,
        long_timeout_seconds=WORKER_LONG_REQUEST_TIMEOUT_SECONDS,
    )


try:
    from generation_worker.celery_app import celery_app

    @celery_app.task(bind=True, name="datara.execute_lambda_job", max_retries=None)
    def execute_lambda_job(self, *, job_id: str, job_type: str, schema_version: int) -> None:
        worker_id = build_worker_id(str(self.request.id or ""))
        try:
            outcome = process_generation_message(
                job_id=job_id,
                job_type=job_type,
                schema_version=schema_version,
                client=build_client(worker_id),
            )
        except WorkerApiUnavailable as exc:
            raise self.retry(countdown=15, exc=exc)
        if outcome in {"leased", "out_of_order"}:
            raise self.retry(countdown=10 if outcome == "leased" else 2)
except ImportError:
    celery_app = None
