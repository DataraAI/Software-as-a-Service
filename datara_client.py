"""
Optional HTTP client for the Data-as-a-Service Flask API (legacy / tooling).

Base URL is read from DATARA_API_BASE_URL (default http://127.0.0.1:5000).

The GPU worker image runs ``gpu_api`` instead; DaaS calls that service via
``SAAS_GPU_BASE_URL`` (see ``saas_gpu_client`` in Data-as-a-Service).
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import requests


def _default_base_url() -> str:
    return os.environ.get("DATARA_API_BASE_URL", "http://127.0.0.1:5000").rstrip("/")


class DataraAPIClient:
    """Thin wrapper around the DataraAI backend REST API."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout_s: float = 120.0,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.base_url = (base_url or _default_base_url()).rstrip("/")
        self.timeout_s = timeout_s
        self._session = session or requests.Session()

    def health(self) -> Dict[str, Any]:
        r = self._session.get(f"{self.base_url}/health", timeout=self.timeout_s)
        r.raise_for_status()
        return r.json()

    def list_datasets(self, path: str = "") -> Any:
        params = {"path": path} if path else {}
        r = self._session.get(
            f"{self.base_url}/api/datasets",
            params=params,
            timeout=self.timeout_s,
        )
        r.raise_for_status()
        return r.json()

    def dataset_paths(self) -> Any:
        r = self._session.get(
            f"{self.base_url}/api/dataset-paths",
            timeout=self.timeout_s,
        )
        r.raise_for_status()
        return r.json()

    def stats(self) -> Any:
        r = self._session.get(f"{self.base_url}/api/stats", timeout=self.timeout_s)
        r.raise_for_status()
        return r.json()

    def process_video(self, payload: Dict[str, Any]) -> Any:
        r = self._session.post(
            f"{self.base_url}/api/process_video",
            json=payload,
            timeout=self.timeout_s,
        )
        r.raise_for_status()
        return r.json()

    def generate_ego(self, payload: Dict[str, Any]) -> Any:
        r = self._session.post(
            f"{self.base_url}/api/generate_ego",
            json=payload,
            timeout=self.timeout_s,
        )
        r.raise_for_status()
        return r.json()

    def generate_corner_case(self, payload: Dict[str, Any]) -> Any:
        r = self._session.post(
            f"{self.base_url}/api/generate_corner_case",
            json=payload,
            timeout=self.timeout_s,
        )
        r.raise_for_status()
        return r.json()

    def create_vlm_tags(self, payload: Dict[str, Any]) -> Any:
        r = self._session.post(
            f"{self.base_url}/api/create_vlm_tags",
            json=payload,
            timeout=self.timeout_s,
        )
        r.raise_for_status()
        return r.json()

    def delete_dataset(self, path: str) -> Any:
        r = self._session.post(
            f"{self.base_url}/api/delete_dataset",
            json={"path": path},
            timeout=self.timeout_s,
        )
        r.raise_for_status()
        return r.json()
