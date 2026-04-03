"""OpenVLA backend placeholder.

This keeps the backend choice explicit and easy to extend without pretending
the runtime exists in this v1 build.
"""

from __future__ import annotations

from robotask_manipulator.action_backend.base import BaseActionBackend
from robotask_manipulator.schemas import ActionProposal, EpisodeInput, SegmentAnnotation
from robotask_manipulator.utils.validation import ModelLoadError


class OpenVLAActionBackend(BaseActionBackend):
    """Reserved backend slot for future OpenVLA integration."""

    backend_name = "openvla"

    def load(self) -> None:
        raise ModelLoadError(
            "OpenVLA backend is not packaged in this v1 build. Use RTM_ACTION_BACKEND=pi0 or RTM_ACTION_BACKEND=none."
        )

    def propose(self, episode: EpisodeInput, segment: SegmentAnnotation) -> ActionProposal | None:
        self.load()
        return None
