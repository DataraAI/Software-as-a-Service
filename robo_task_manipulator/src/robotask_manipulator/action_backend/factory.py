"""Factory for optional action backends."""

from __future__ import annotations

from robotask_manipulator.action_backend.base import BaseActionBackend
from robotask_manipulator.action_backend.none_backend import NoActionBackend
from robotask_manipulator.action_backend.openvla_backend import OpenVLAActionBackend
from robotask_manipulator.action_backend.pi0_backend import Pi0ActionBackend
from robotask_manipulator.config import ActionBackendSettings
from robotask_manipulator.utils.validation import InvalidInputError


def create_action_backend(settings: ActionBackendSettings) -> BaseActionBackend:
    """Create the configured optional action backend."""
    backend = settings.backend.strip().lower()
    if backend == "none":
        return NoActionBackend()
    if backend == "pi0":
        return Pi0ActionBackend(settings)
    if backend == "openvla":
        return OpenVLAActionBackend()
    raise InvalidInputError(
        f"Unsupported action backend '{settings.backend}'. Expected one of: pi0, openvla, none."
    )
