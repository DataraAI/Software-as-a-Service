"""Benchmark and evaluation schemas."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from robotask_manipulator.schemas.core import ActionLabel


class BenchmarkStep(BaseModel):
    """Expected step for a benchmark episode."""

    step_index: int = Field(ge=0)
    description: str
    action_label: ActionLabel
    success: bool | None = None


class BenchmarkEpisode(BaseModel):
    """Golden benchmark episode."""

    episode_id: str
    expected_steps: list[BenchmarkStep]
    metadata: dict[str, Any] = Field(default_factory=dict)


class BenchmarkSet(BaseModel):
    """Collection of benchmark episodes."""

    episodes: list[BenchmarkEpisode]


class BatchEvaluationReport(BaseModel):
    """Readable batch evaluation output."""

    episodes: list[dict[str, Any]]
    summary: dict[str, Any]
