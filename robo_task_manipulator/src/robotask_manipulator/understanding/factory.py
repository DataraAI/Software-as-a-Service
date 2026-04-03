"""Factory for semantic understanding backends."""

from __future__ import annotations

from robotask_manipulator.config import SemanticSettings
from robotask_manipulator.understanding.base import BaseSemanticBackend
from robotask_manipulator.understanding.transformers_vlm import TransformersVLMBackend
from robotask_manipulator.utils.validation import InvalidInputError


def build_semantic_backend(settings: SemanticSettings) -> BaseSemanticBackend:
    """Create the configured semantic understanding backend."""
    backend = settings.backend.strip().lower()
    if backend in {"blip", "transformers", "vlm"}:
        return TransformersVLMBackend(settings)
    raise InvalidInputError(
        f"Unsupported semantic backend '{settings.backend}'. Expected one of: blip, transformers, vlm."
    )
