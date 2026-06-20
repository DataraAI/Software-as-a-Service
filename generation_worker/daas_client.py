"""Private DaaS worker API client."""

from __future__ import annotations

from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class WorkerApiUnavailable(RuntimeError):
    """Raised when DaaS cannot safely accept a worker state transition."""


class DaasClient:
    def __init__(
        self,
        *,
        base_url: str,
        worker_token: str,
        worker_id: str,
        timeout_seconds: int = 30,
        long_timeout_seconds: int = 1800,
        session: Any | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.worker_token = worker_token
        self.worker_id = worker_id
        self.timeout_seconds = timeout_seconds
        self.long_timeout_seconds = long_timeout_seconds
        self.session = session or requests.Session()
        if session is None:
            retry = Retry(
                total=3,
                connect=3,
                read=0,
                status=3,
                backoff_factor=0.5,
                status_forcelist=(502, 503, 504),
                allowed_methods=frozenset({"POST"}),
                raise_on_status=False,
            )
            self.session.mount("https://", HTTPAdapter(max_retries=retry))
            self.session.mount("http://", HTTPAdapter(max_retries=retry))

    def _post(
        self,
        job_id: str,
        action: str,
        payload: dict[str, Any],
        *,
        timeout_seconds: int | None = None,
    ) -> dict[str, Any]:
        try:
            response = self.session.post(
                f"{self.base_url}/api/internal/generation-jobs/{job_id}/{action}",
                json={"worker_id": self.worker_id, **payload},
                headers={"Authorization": f"Bearer {self.worker_token}"},
                timeout=timeout_seconds or self.timeout_seconds,
            )
        except requests.RequestException as exc:
            raise WorkerApiUnavailable("DaaS worker API is temporarily unavailable") from exc
        if response.status_code == 409:
            return response.json()
        if response.status_code >= 500:
            raise WorkerApiUnavailable("DaaS worker API is temporarily unavailable")
        response.raise_for_status()
        return response.json()

    def claim(self, job_id: str, job_type: str, schema_version: int) -> dict[str, Any]:
        return self._post(
            job_id,
            "claim",
            {"job_type": job_type, "schema_version": schema_version},
            timeout_seconds=self.long_timeout_seconds,
        )

    def heartbeat(self, job_id: str) -> dict[str, Any]:
        return self._post(job_id, "heartbeat", {})

    def stage(self, job_id: str, stage: str) -> dict[str, Any]:
        return self._post(job_id, "stage", {"stage": stage})

    def complete(self, job_id: str, result: dict[str, Any]) -> dict[str, Any]:
        return self._post(
            job_id,
            "complete",
            {"result": result},
            timeout_seconds=self.long_timeout_seconds,
        )

    def fail(self, job_id: str, error: str) -> dict[str, Any]:
        return self._post(job_id, "fail", {"error": error[:4000]})
