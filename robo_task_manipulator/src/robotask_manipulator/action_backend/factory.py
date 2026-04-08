"""Factory for optional action backends."""

from __future__ import annotations

from robotask_manipulator.action_backend.base import BaseActionBackend
from robotask_manipulator.action_backend.none_backend import NoActionBackend
from robotask_manipulator.action_backend.openvla_backend import OpenVLAActionBackend
from robotask_manipulator.config import ActionBackendSettings
from robotask_manipulator.utils.validation import InvalidInputError, ModelLoadError


def create_action_backend(settings: ActionBackendSettings) -> BaseActionBackend:
    """Create the configured optional action backend."""
    backend = settings.backend.strip().lower()
    if backend == "none":
        return NoActionBackend()
    if backend == "pi0":
        try:
            from robotask_manipulator.action_backend.pi0_backend import Pi0ActionBackend
        except ModuleNotFoundError as exc:
            missing = exc.name or "optional pi0 dependency"
            raise ModelLoadError(
                "pi0 backend dependencies are not installed. Install the base stack with "
                "`pip install -r requirements.txt` for semantic testing, or install the optional "
                "pi0 stack separately when you actually need robot-action proposals."
                f" Missing dependency: {missing}."
            ) from exc
        return Pi0ActionBackend(settings)
    if backend == "openvla":
        return OpenVLAActionBackend()
    raise InvalidInputError(
        f"Unsupported action backend '{settings.backend}'. Expected one of: pi0, openvla, none."
    )
