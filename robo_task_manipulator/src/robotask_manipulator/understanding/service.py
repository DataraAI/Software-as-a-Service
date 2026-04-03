"""Semantic understanding service."""

from __future__ import annotations

from robotask_manipulator.schemas import EpisodeInput, SegmentAnnotation, SemanticStep
from robotask_manipulator.understanding.base import BaseSemanticBackend


class SemanticUnderstandingService:
    """Apply semantic VLM understanding to segmented episodes."""

    def __init__(self, backend: BaseSemanticBackend) -> None:
        self.backend = backend

    def annotate(self, episode: EpisodeInput, segments: list[SegmentAnnotation]) -> list[SegmentAnnotation]:
        for segment in segments:
            prediction = self.backend.predict(
                image_path=segment.representative_frame_ref,
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
            segment.raw_outputs["semantic"] = {"caption": prediction.caption, "evidence": prediction.evidence}
        return segments
