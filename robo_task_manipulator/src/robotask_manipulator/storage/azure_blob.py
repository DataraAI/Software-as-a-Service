"""Azure Blob source and annotation storage adapters."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

from robotask_manipulator.config import AzureBlobSettings
from robotask_manipulator.utils.io import ensure_directory
from robotask_manipulator.utils.validation import RoboTaskManipulatorError


class AzureBlobStorageError(RoboTaskManipulatorError):
    """Raised when Azure Blob access fails or is misconfigured."""


class SourceBlobNotFoundError(AzureBlobStorageError):
    """Raised when the requested source blob cannot be downloaded."""


@dataclass(frozen=True)
class BlobReference:
    """Parsed Azure Blob reference from an HTTPS blob URL."""

    raw_url: str
    account_url: str
    container_name: str
    blob_name: str

    @property
    def file_name(self) -> str:
        name = Path(self.blob_name).name
        return name or "source_asset"


class AzureBlobSource:
    """Download source media from Azure Blob given a blob URL reference."""

    def __init__(self, settings: AzureBlobSettings, blob_service_client: Any | None = None) -> None:
        self.settings = settings
        self._blob_service_client = blob_service_client

    def parse_blob_reference(self, blob_url: str) -> BlobReference:
        parsed = urlsplit(blob_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise AzureBlobStorageError("source_blob_url must be a valid Azure Blob URL.")

        path_parts = [part for part in parsed.path.split("/") if part]
        if len(path_parts) < 2:
            raise AzureBlobStorageError("source_blob_url must include both a container and blob path.")

        container_name = path_parts[0]
        blob_name = unquote("/".join(path_parts[1:]))
        allowlist = set(self.settings.source_container_allowlist)
        if allowlist and container_name not in allowlist:
            raise AzureBlobStorageError(
                f"Source blob container '{container_name}' is not allowed. Allowed containers: {sorted(allowlist)}."
            )

        return BlobReference(
            raw_url=blob_url,
            account_url=f"{parsed.scheme}://{parsed.netloc}",
            container_name=container_name,
            blob_name=blob_name,
        )

    def download_to_file(self, blob_url: str, target_path: str | Path) -> Path:
        reference = self.parse_blob_reference(blob_url)
        target = Path(target_path)
        target.parent.mkdir(parents=True, exist_ok=True)

        try:
            blob_client = self._build_source_blob_client(reference)
            payload = blob_client.download_blob().readall()
        except Exception as exc:  # noqa: BLE001
            if _is_blob_not_found_error(exc):
                raise SourceBlobNotFoundError(f"Source blob not found: {reference.raw_url}") from exc
            raise AzureBlobStorageError(f"Failed to download source blob: {reference.raw_url}") from exc

        target.write_bytes(payload)
        return target

    def download_to_temp_file(self, blob_url: str, working_dir: str | Path, source_asset_id: str) -> Path:
        reference = self.parse_blob_reference(blob_url)
        directory = ensure_directory(working_dir) / _normalize_path_component(source_asset_id)
        directory.mkdir(parents=True, exist_ok=True)
        return self.download_to_file(reference.raw_url, directory / reference.file_name)

    def _build_source_blob_client(self, reference: BlobReference):
        if self.settings.connection_string or self.settings.account_url:
            service_client = self._get_blob_service_client(reference.account_url)
            return service_client.get_blob_client(
                container=reference.container_name,
                blob=reference.blob_name,
            )

        try:
            from azure.storage.blob import BlobClient
        except ImportError as exc:  # pragma: no cover
            raise AzureBlobStorageError(
                "azure-storage-blob is required to download source blobs."
            ) from exc

        return BlobClient.from_blob_url(reference.raw_url)

    def _get_blob_service_client(self, default_account_url: str | None = None):
        if self._blob_service_client is not None:
            return self._blob_service_client

        try:
            from azure.storage.blob import BlobServiceClient
        except ImportError as exc:  # pragma: no cover
            raise AzureBlobStorageError(
                "azure-storage-blob is required to create Azure Blob service clients."
            ) from exc

        if self.settings.connection_string:
            self._blob_service_client = BlobServiceClient.from_connection_string(self.settings.connection_string)
            return self._blob_service_client

        account_url = self.settings.account_url or default_account_url
        if not account_url:
            raise AzureBlobStorageError(
                "Azure Blob service credentials are not configured. Set RTM_AZURE_STORAGE_CONNECTION_STRING or "
                "RTM_AZURE_STORAGE_ACCOUNT_URL."
            )

        try:
            from azure.identity import DefaultAzureCredential
        except ImportError as exc:  # pragma: no cover
            raise AzureBlobStorageError(
                "azure-identity is required for account URL based Azure authentication."
            ) from exc

        self._blob_service_client = BlobServiceClient(
            account_url=account_url,
            credential=DefaultAzureCredential(exclude_interactive_browser_credential=True),
        )
        return self._blob_service_client


class AzureAnnotationStore:
    """Persist canonical annotation JSON back to Azure Blob."""

    def __init__(self, settings: AzureBlobSettings, blob_service_client: Any | None = None) -> None:
        self.settings = settings
        self._blob_service_client = blob_service_client

    def build_annotation_blob_name(self, source_asset_id: str, annotation_id: str) -> str:
        components = [self.settings.annotation_prefix.strip("/")] if self.settings.annotation_prefix else []
        normalized_asset_id = _normalize_blob_path(source_asset_id)
        components.extend([normalized_asset_id, f"{annotation_id}.json"])
        return "/".join(component for component in components if component)

    def upload_annotation(
        self,
        *,
        source_asset_id: str,
        annotation_id: str,
        annotation_payload: dict[str, Any],
        metadata: dict[str, str],
        tags: dict[str, str],
    ) -> str:
        blob_name = self.build_annotation_blob_name(source_asset_id, annotation_id)
        service_client = self._get_blob_service_client()

        if self.settings.create_containers:
            try:
                service_client.create_container(self.settings.annotation_container)
            except Exception:  # noqa: BLE001
                pass

        blob_client = service_client.get_blob_client(
            container=self.settings.annotation_container,
            blob=blob_name,
        )

        payload_bytes = json.dumps(annotation_payload, indent=2, sort_keys=True).encode("utf-8")
        upload_kwargs: dict[str, Any] = {
            "overwrite": True,
            "metadata": _stringify_mapping(metadata),
            "tags": _stringify_mapping(tags),
        }

        try:
            from azure.storage.blob import ContentSettings

            upload_kwargs["content_settings"] = ContentSettings(content_type="application/json")
        except ImportError:
            pass

        try:
            blob_client.upload_blob(payload_bytes, **upload_kwargs)
        except Exception as exc:  # noqa: BLE001
            raise AzureBlobStorageError(
                f"Failed to upload annotation blob to container '{self.settings.annotation_container}' path '{blob_name}'."
            ) from exc

        return getattr(blob_client, "url", _build_blob_url(service_client, self.settings.annotation_container, blob_name))

    def _get_blob_service_client(self):
        if self._blob_service_client is not None:
            return self._blob_service_client

        try:
            from azure.storage.blob import BlobServiceClient
        except ImportError as exc:  # pragma: no cover
            raise AzureBlobStorageError(
                "azure-storage-blob is required to create Azure Blob service clients."
            ) from exc

        if self.settings.connection_string:
            self._blob_service_client = BlobServiceClient.from_connection_string(self.settings.connection_string)
            return self._blob_service_client

        if not self.settings.account_url:
            raise AzureBlobStorageError(
                "Azure annotation storage requires RTM_AZURE_STORAGE_CONNECTION_STRING or RTM_AZURE_STORAGE_ACCOUNT_URL."
            )

        try:
            from azure.identity import DefaultAzureCredential
        except ImportError as exc:  # pragma: no cover
            raise AzureBlobStorageError(
                "azure-identity is required for account URL based Azure authentication."
            ) from exc

        self._blob_service_client = BlobServiceClient(
            account_url=self.settings.account_url,
            credential=DefaultAzureCredential(exclude_interactive_browser_credential=True),
        )
        return self._blob_service_client


def _stringify_mapping(values: dict[str, Any]) -> dict[str, str]:
    return {str(key): str(value) for key, value in values.items() if value is not None}


def _normalize_blob_path(value: str) -> str:
    stripped = value.strip().strip("/")
    if not stripped:
        return "unknown-source"
    return stripped.replace("\\", "/")


def _normalize_path_component(value: str) -> str:
    cleaned = value.replace("\\", "_").replace("/", "_").strip()
    return cleaned or "source_asset"


def _build_blob_url(service_client: Any, container_name: str, blob_name: str) -> str:
    account_url = getattr(service_client, "url", "") or getattr(service_client, "primary_endpoint", "")
    return f"{str(account_url).rstrip('/')}/{container_name}/{blob_name}".rstrip("/")


def _is_blob_not_found_error(exc: Exception) -> bool:
    if isinstance(exc, (FileNotFoundError, KeyError)):
        return True
    status_code = getattr(exc, "status_code", None)
    if status_code == 404:
        return True
    error_code = str(getattr(exc, "error_code", "")).lower()
    if "blobnotfound" in error_code or "resourcenotfound" in error_code:
        return True
    message = str(exc).lower()
    return "not found" in message or "does not exist" in message
