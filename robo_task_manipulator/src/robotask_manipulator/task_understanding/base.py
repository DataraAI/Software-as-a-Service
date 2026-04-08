"""Base interfaces for task-understanding backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SemanticPrediction:
    """Structured semantic prediction from a task-understanding backend."""

    description: str
    task_intent: str | None
    objects_involved: list[str] = field(default_factory=list)
    object_source: str | None = None
    object_target: str | None = None
    confidence: float = 0.0
    caption: str | None = None
    evidence: dict[str, Any] = field(default_factory=dict)


class BaseTaskUnderstandingBackend(ABC):
    """Task-understanding backend for conservative segment semantics."""

    @abstractmethod
    def load(self) -> None:
        """Load resources if needed."""

    @abstractmethod
    def predict(
        self,
        frame_paths: list[str],
        instruction: str,
        step_index: int,
        total_steps: int,
    ) -> SemanticPrediction:
        """Produce a structured semantic step prediction from ordered frames."""
