"""No-op action backend."""

from __future__ import annotations

from robotask_manipulator.action_backend.base import BaseActionBackend
from robotask_manipulator.schemas import ActionProposal, EpisodeInput, SegmentAnnotation


class NoActionBackend(BaseActionBackend):
    """Disable robot action proposals while keeping the rest of the product working."""

    backend_name = "none"

    def load(self) -> None:
        """Nothing to load."""

    def propose(self, episode: EpisodeInput, segment: SegmentAnnotation) -> ActionProposal | None:
        return None
