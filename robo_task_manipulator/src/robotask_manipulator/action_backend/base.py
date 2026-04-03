"""Optional action backend interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod

from robotask_manipulator.schemas import ActionProposal, EpisodeInput, SegmentAnnotation


class BaseActionBackend(ABC):
    """Optional robot-oriented backend used after semantic understanding."""

    backend_name = "base"

    @abstractmethod
    def load(self) -> None:
        """Load model resources if needed."""

    @abstractmethod
    def propose(self, episode: EpisodeInput, segment: SegmentAnnotation) -> ActionProposal | None:
        """Produce a robot-oriented proposal for one segment."""
