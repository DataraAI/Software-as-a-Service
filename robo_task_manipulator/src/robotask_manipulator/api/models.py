"""HTTP request and response models for the annotation service."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, HttpUrl


DEFAULT_ANNOTATION_INSTRUCTION = "Describe only the visible hand-object action conservatively."


class AnnotationImageRequest(BaseModel):
    """DaaS-triggered synchronous image annotation request."""

    source_asset_id: str = Field(min_length=1)
    source_blob_url: HttpUrl
    media_type: Literal["image"] = "image"
    task_name: str | None = None
    instruction: str | None = None
    tags: list[str] = Field(default_factory=list)


class AnnotationSummary(BaseModel):
    """Compact UI-facing summary derived from the canonical episode output."""

    episode_id: str
    task_name: str
    segment_count: int = Field(ge=0)
    action_labels: list[str] = Field(default_factory=list)
    segment_descriptions: list[str] = Field(default_factory=list)


class AnnotationResponse(BaseModel):
    """Successful synchronous annotation response returned to DaaS."""

    annotation_id: str
    status: Literal["completed"] = "completed"
    source_asset_id: str
    annotation_blob_url: str
    annotation_json: dict[str, Any]
    summary: AnnotationSummary


class AnnotationJobAcceptedResponse(BaseModel):
    """Reserved response shape for future async image/video jobs."""

    job_id: str
    status: Literal["queued"] = "queued"
    message: str


class AnnotationErrorDetail(BaseModel):
    """Structured error information returned to DaaS."""

    code: str
    message: str


class AnnotationErrorResponse(BaseModel):
    """Structured error response for validation, source, and inference failures."""

    status: Literal["failed", "not_implemented"] = "failed"
    error: AnnotationErrorDetail
    source_asset_id: str | None = None
