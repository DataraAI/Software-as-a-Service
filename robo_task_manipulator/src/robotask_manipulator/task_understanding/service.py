"""Task-understanding service."""

from __future__ import annotations

from robotask_manipulator.schemas import (
    ActionLabel,
    EpisodeInput,
    FrameAnnotation,
    SegmentAnnotation,
    SemanticStep,
    SymbolicActionLabel,
)
from robotask_manipulator.task_understanding.base import BaseTaskUnderstandingBackend


class TaskUnderstandingService:
    """Apply task-understanding inference to segmented episodes."""

    def __init__(self, backend: BaseTaskUnderstandingBackend, frame_context_radius: int = 1) -> None:
        self.backend = backend
        self.frame_context_radius = max(0, frame_context_radius)

    def annotate_frames(self, episode: EpisodeInput) -> list[FrameAnnotation]:
        predictions: list[FrameAnnotation] = []
        total_frames = len(episode.frames)

        for frame_index, frame in enumerate(episode.frames):
            context_frames = self._context_window(episode.frames, frame_index)
            context_refs = [item.asset_ref for item in context_frames]
            prediction = self.backend.predict(
                frame_paths=context_refs,
                instruction=episode.instruction,
                step_index=frame_index,
                total_steps=total_frames,
            )
            semantic = SemanticStep(
                description=prediction.description,
                task_intent=prediction.task_intent,
                objects_involved=prediction.objects_involved,
                object_source=prediction.object_source,
                object_target=prediction.object_target,
                confidence=prediction.confidence,
                evidence=prediction.evidence,
            )
            predictions.append(
                FrameAnnotation(
                    frame_id=frame.frame_id,
                    episode_id=episode.episode_id,
                    frame_index=frame.frame_index,
                    asset_ref=frame.asset_ref,
                    timestamp_s=frame.timestamp_s,
                    context_frame_refs=context_refs,
                    semantic=semantic,
                    symbolic_action=SymbolicActionLabel(
                        label=ActionLabel.UNKNOWN,
                        confidence=0.0,
                        source="unlabeled",
                    ),
                    raw_outputs={
                        "task_understanding": {
                            "caption": prediction.caption,
                            "evidence": prediction.evidence,
                        }
                    },
                )
            )
        return predictions

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

    def _context_window(self, frames, index: int):
        start = max(0, index - self.frame_context_radius)
        end = min(len(frames), index + self.frame_context_radius + 1)
        return frames[start:end]
