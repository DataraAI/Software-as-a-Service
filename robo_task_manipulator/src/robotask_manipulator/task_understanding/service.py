"""Task-understanding service."""

from __future__ import annotations

from robotask_manipulator.schemas import EpisodeInput, SegmentAnnotation, SemanticStep
from robotask_manipulator.task_understanding.base import BaseTaskUnderstandingBackend


class TaskUnderstandingService:
    """Apply task-understanding inference to segmented episodes."""

    def __init__(self, backend: BaseTaskUnderstandingBackend) -> None:
        self.backend = backend

    def annotate(self, episode: EpisodeInput, segments: list[SegmentAnnotation]) -> list[SegmentAnnotation]:
        for segment in segments:
            frame_paths = segment.observation_refs or [segment.representative_frame_ref]
            prediction = self.backend.predict(
                frame_paths=frame_paths,
                instruction=episode.instruction,
                step_index=segment.step_index,
                total_steps=len(segments),
            )
            segment.semantic = SemanticStep(
                description=prediction.description,
                task_intent=prediction.task_intent,
                objects_involved=prediction.objects_involved,
                object_source=prediction.object_source,
                object_target=prediction.object_target,
                confidence=prediction.confidence,
                evidence=prediction.evidence,
            )
            segment.raw_outputs["task_understanding"] = {
                "caption": prediction.caption,
                "evidence": prediction.evidence,
            }
        return segments
