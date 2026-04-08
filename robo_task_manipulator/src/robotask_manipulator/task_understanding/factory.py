"""Factory for task-understanding backends."""

from __future__ import annotations

from robotask_manipulator.config import SemanticSettings
from robotask_manipulator.task_understanding.base import BaseTaskUnderstandingBackend
from robotask_manipulator.task_understanding.transformers_vlm import TransformersTaskUnderstandingBackend
from robotask_manipulator.utils.validation import InvalidInputError


def build_task_understanding_backend(settings: SemanticSettings) -> BaseTaskUnderstandingBackend:
    """Create the configured task-understanding backend."""
    backend = settings.backend.strip().lower()
    if backend in {"multimodal_vlm", "transformers", "vlm", "blip"}:
        return TransformersTaskUnderstandingBackend(settings)
    raise InvalidInputError(
        f"Unsupported semantic backend '{settings.backend}'. Expected one of: multimodal_vlm, transformers, vlm, blip."
    )
