"""Simple deterministic segmentation for v1."""

from __future__ import annotations

from pathlib import Path
from statistics import mean

import numpy as np
from PIL import Image

from robotask_manipulator.config import SegmentationSettings
from robotask_manipulator.schemas import (
    ActionLabel,
    EpisodeInput,
    FrameAnnotation,
    SegmentAnnotation,
    SemanticStep,
    SymbolicActionLabel,
)


class Segmenter:
    """Segment frame sequences using fixed windows with optional visual-change boundaries."""

    def __init__(self, settings: SegmentationSettings) -> None:
        self.settings = settings

    def segment(self, episode: EpisodeInput) -> list[SegmentAnnotation]:
        frames = episode.frames
        boundaries = self._build_boundaries(frames)
        segments: list[SegmentAnnotation] = []
        for step_index, (start, end) in enumerate(boundaries):
            window = frames[start : end + 1]
            segment_id = f"{episode.episode_id}-segment-{step_index:03d}"
            representative = window[len(window) // 2]
            confidence = self._confidence(window)
            segments.append(
                SegmentAnnotation(
                    segment_id=segment_id,
                    episode_id=episode.episode_id,
                    step_index=step_index,
                    observation_refs=[frame.asset_ref for frame in window],
                    representative_frame_ref=representative.asset_ref,
                    frame_start_index=window[0].frame_index,
                    frame_end_index=window[-1].frame_index,
                    timestamp_start_s=window[0].timestamp_s,
                    timestamp_end_s=window[-1].timestamp_s,
                    segmentation_confidence=confidence,
                    semantic=SemanticStep(description=f"segment {step_index + 1}", confidence=0.0),
                    symbolic_action=SymbolicActionLabel(label=ActionLabel.UNKNOWN, confidence=0.0, source="unlabeled"),
                    success=None,
                )
            )
        for index, segment in enumerate(segments[:-1]):
            segment.next_step_refs = [segments[index + 1].segment_id]
        return segments

    def summarize_frame_predictions(
        self,
        episode: EpisodeInput,
        frame_predictions: list[FrameAnnotation],
    ) -> list[SegmentAnnotation]:
        if not frame_predictions:
            return []

        groups: list[list[FrameAnnotation]] = []
        current_group = [frame_predictions[0]]
        for frame_prediction in frame_predictions[1:]:
            if self._same_summary_group(current_group[-1], frame_prediction):
                current_group.append(frame_prediction)
            else:
                groups.append(current_group)
                current_group = [frame_prediction]
        groups.append(current_group)

        segments: list[SegmentAnnotation] = []
        for step_index, group in enumerate(groups):
            representative = group[len(group) // 2]
            semantic = self._summarize_semantic(group)
            symbolic_action = self._summarize_symbolic_action(group)
            context_tags = self._summarize_context_tags(group)
            segment = SegmentAnnotation(
                segment_id=f"{episode.episode_id}-segment-{step_index:03d}",
                episode_id=episode.episode_id,
                step_index=step_index,
                observation_refs=[frame.asset_ref for frame in group],
                representative_frame_ref=representative.asset_ref,
                frame_start_index=group[0].frame_index,
                frame_end_index=group[-1].frame_index,
                timestamp_start_s=group[0].timestamp_s,
                timestamp_end_s=group[-1].timestamp_s,
                segmentation_confidence=round(min(0.98, mean(frame.semantic.confidence for frame in group) + 0.08), 2),
                semantic=semantic,
                symbolic_action=symbolic_action,
                context_tags=context_tags,
                success=all(frame.success is not False for frame in group),
                raw_outputs={
                    "frame_prediction_ids": [frame.frame_id for frame in group],
                    "merged_from_frames": len(group),
                },
            )
            segments.append(segment)

        for index, segment in enumerate(segments[:-1]):
            segment.next_step_refs = [segments[index + 1].segment_id]
        return segments

    def _build_boundaries(self, frames) -> list[tuple[int, int]]:
        if len(frames) <= self.settings.frames_per_segment:
            return [(0, len(frames) - 1)]
        boundaries: list[tuple[int, int]] = []
        start = 0
        for index in range(1, len(frames)):
            reached_window = (index - start) >= self.settings.frames_per_segment
            visual_break = self.settings.use_visual_change_breaks and self._visual_change(frames[index - 1].asset_ref, frames[index].asset_ref) >= self.settings.visual_change_threshold
            if reached_window or visual_break:
                boundaries.append((start, index - 1))
                start = index
        boundaries.append((start, len(frames) - 1))
        return boundaries

    def _visual_change(self, left_ref: str, right_ref: str) -> float:
        try:
            with Image.open(Path(left_ref)) as left, Image.open(Path(right_ref)) as right:
                left_arr = np.asarray(left.convert("RGB"), dtype=np.float32) / 255.0
                right_arr = np.asarray(right.convert("RGB").resize(left.size), dtype=np.float32) / 255.0
            return float(np.mean(np.abs(left_arr - right_arr)))
        except Exception:  # noqa: BLE001
            return 0.0

    def _confidence(self, window) -> float:
        return round(min(0.95, 0.55 + 0.08 * len(window)), 2)

    def _same_summary_group(self, left: FrameAnnotation, right: FrameAnnotation) -> bool:
        return (
            left.semantic.description == right.semantic.description
            and left.symbolic_action.label == right.symbolic_action.label
        )

    def _summarize_semantic(self, group: list[FrameAnnotation]) -> SemanticStep:
        representative = group[len(group) // 2]
        return SemanticStep(
            description=representative.semantic.description,
            task_intent=representative.semantic.task_intent,
            objects_involved=representative.semantic.objects_involved,
            object_source=representative.semantic.object_source,
            object_target=representative.semantic.object_target,
            confidence=round(mean(frame.semantic.confidence for frame in group), 2),
            evidence={
                "frame_count": len(group),
                "frame_indices": [frame.frame_index for frame in group],
                "grouping": "consecutive_equal_description_and_label",
            },
        )

    def _summarize_symbolic_action(self, group: list[FrameAnnotation]) -> SymbolicActionLabel:
        representative = group[len(group) // 2]
        return SymbolicActionLabel(
            label=representative.symbolic_action.label,
            confidence=round(mean(frame.symbolic_action.confidence for frame in group), 2),
            source=representative.symbolic_action.source,
            evidence={
                "frame_count": len(group),
                "frame_indices": [frame.frame_index for frame in group],
            },
        )

    def _summarize_context_tags(self, group: list[FrameAnnotation]):
        seen = {}
        for frame in group:
            for tag in frame.context_tags:
                seen[str(tag.name)] = tag
        return list(seen.values())
