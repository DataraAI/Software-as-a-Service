"""Application service for DaaS-triggered annotation requests."""

from __future__ import annotations

import logging
import re
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from robotask_manipulator.api.models import (
    AnnotationImageRequest,
    AnnotationResponse,
    AnnotationSummary,
    DEFAULT_ANNOTATION_INSTRUCTION,
)
from robotask_manipulator.config import AppSettings
from robotask_manipulator.export import JsonArtifactExporter
from robotask_manipulator.main import RoboTaskManipulatorApp
from robotask_manipulator.storage import AzureAnnotationStore, AzureBlobSource
from robotask_manipulator.storage.azure_blob import AzureBlobStorageError, SourceBlobNotFoundError
from robotask_manipulator.utils.io import ensure_directory
from robotask_manipulator.utils.validation import RoboTaskManipulatorError

LOGGER = logging.getLogger(__name__)


class AnnotationServiceRequestError(RoboTaskManipulatorError):
    """Raised when an annotation request is malformed or unsupported."""


class AnnotationProcessingError(RoboTaskManipulatorError):
    """Raised when annotation processing fails after request validation succeeds."""


PipelineRunner = Callable[[dict[str, Any], Path], Any]


class AnnotationService:
    """Orchestrate source fetch, pipeline execution, and annotation persistence."""

    def __init__(
        self,
        settings: AppSettings,
        *,
        pipeline_runner: PipelineRunner | None = None,
        blob_source: AzureBlobSource | None = None,
        annotation_store: AzureAnnotationStore | None = None,
        exporter: JsonArtifactExporter | None = None,
    ) -> None:
        self.settings = settings
        self._pipeline_runner = pipeline_runner
        self._app: RoboTaskManipulatorApp | None = None
        self.blob_source = blob_source or AzureBlobSource(settings.azure)
        self.annotation_store = annotation_store or AzureAnnotationStore(settings.azure)
        self.exporter = exporter or JsonArtifactExporter()

    def ensure_ready(self) -> None:
        """Warm the long-running Lambda.ai service when configured."""
        if self._pipeline_runner is not None:
            return
        app = self._get_app()
        app.task_understanding_service.backend.load()
        app.action_backend.load()

    def health_summary(self) -> dict[str, Any]:
        """Return deployment-facing health information."""
        local_model_path = self.settings.semantic.local_model_path
        return {
            "status": "ok",
            "service": "robotask_manipulator",
            "deployment_target": "lambda_ai_vm",
            "semantic_model_source": self.settings.semantic.model_source,
            "semantic_model_path_exists": bool(local_model_path and Path(local_model_path).exists()),
            "warm_load": self.settings.service.warm_load,
            "async_jobs": "reserved",
        }

    def annotate_image(self, request: AnnotationImageRequest) -> AnnotationResponse:
        """Run one synchronous image annotation request end-to-end."""
        if request.media_type != "image":
            raise AnnotationServiceRequestError("The synchronous annotation endpoint only supports media_type='image'.")

        annotation_id = uuid.uuid4().hex
        episode_id = annotation_id
        task_name = self._resolve_task_name(request.source_asset_id, request.task_name)
        instruction = (request.instruction or DEFAULT_ANNOTATION_INSTRUCTION).strip()
        created_at = datetime.now(timezone.utc).isoformat()
        working_root = ensure_directory(self.settings.service.temp_dir)

        try:
            with tempfile.TemporaryDirectory(
                prefix="rtm-annotation-",
                dir=str(working_root),
            ) as temporary_directory:
                work_dir = Path(temporary_directory)
                local_asset_path = self.blob_source.download_to_temp_file(
                    str(request.source_blob_url),
                    working_dir=work_dir,
                    source_asset_id=request.source_asset_id,
                )
                payload = self._build_payload(
                    annotation_id=annotation_id,
                    episode_id=episode_id,
                    task_name=task_name,
                    instruction=instruction,
                    source_asset_id=request.source_asset_id,
                    source_blob_url=str(request.source_blob_url),
                    tags=request.tags,
                    local_asset_path=local_asset_path,
                )
                output = self._run_pipeline(payload, work_dir)
        except SourceBlobNotFoundError:
            raise
        except AnnotationServiceRequestError:
            raise
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception(
                "Annotation request failed source_asset_id=%s source_blob_url=%s",
                request.source_asset_id,
                request.source_blob_url,
            )
            raise AnnotationProcessingError(
                f"Failed to generate annotation for source asset '{request.source_asset_id}'."
            ) from exc

        annotation_json = self.exporter.serialize_episode(output)
        summary = self._build_summary(output)
        metadata = {
            "source_asset_id": request.source_asset_id,
            "media_type": request.media_type,
            "task_name": task_name,
            "model_id": self.settings.semantic.model_source,
            "created_at": created_at,
            "status": "completed",
        }
        tags = metadata.copy()

        try:
            annotation_blob_url = self.annotation_store.upload_annotation(
                source_asset_id=request.source_asset_id,
                annotation_id=annotation_id,
                annotation_payload=annotation_json,
                metadata=metadata,
                tags=tags,
            )
        except AzureBlobStorageError as exc:
            LOGGER.exception(
                "Failed to persist annotation source_asset_id=%s annotation_id=%s",
                request.source_asset_id,
                annotation_id,
            )
            raise AnnotationProcessingError(
                f"Failed to persist annotation for source asset '{request.source_asset_id}'."
            ) from exc

        return AnnotationResponse(
            annotation_id=annotation_id,
            status="completed",
            source_asset_id=request.source_asset_id,
            annotation_blob_url=annotation_blob_url,
            annotation_json=annotation_json,
            summary=summary,
        )

    def _run_pipeline(self, payload: dict[str, Any], base_dir: Path):
        if self._pipeline_runner is not None:
            return self._pipeline_runner(payload, base_dir)
        return self._get_app().run_payload(payload, base_dir)

    def _get_app(self) -> RoboTaskManipulatorApp:
        if self._app is None:
            self._app = RoboTaskManipulatorApp(self.settings)
        return self._app

    def _build_payload(
        self,
        *,
        annotation_id: str,
        episode_id: str,
        task_name: str,
        instruction: str,
        source_asset_id: str,
        source_blob_url: str,
        tags: list[str],
        local_asset_path: Path,
    ) -> dict[str, Any]:
        metadata = {
            "source_asset_id": source_asset_id,
            "source_blob_url": source_blob_url,
            "tags": list(tags),
            "annotation_id": annotation_id,
        }
        return {
            "episode_id": episode_id,
            "task_name": task_name,
            "instruction": instruction,
            "asset_path": str(local_asset_path),
            "metadata": metadata,
        }

    def _build_summary(self, output) -> AnnotationSummary:
        return AnnotationSummary(
            episode_id=output.episode_id,
            task_name=output.task_name,
            segment_count=len(output.segments),
            action_labels=[str(segment.symbolic_action.label) for segment in output.segments],
            segment_descriptions=[segment.semantic.description for segment in output.segments],
        )

    def _resolve_task_name(self, source_asset_id: str, task_name: str | None) -> str:
        if task_name and task_name.strip():
            return task_name.strip()
        cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", source_asset_id.strip()).strip("_").lower()
        return cleaned or "annotation_task"
