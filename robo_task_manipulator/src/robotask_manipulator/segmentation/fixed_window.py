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
        groups = self._merge_summary_groups(groups)

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
                    "grouping": self._grouping_reason(group),
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

    def _merge_summary_groups(self, groups: list[list[FrameAnnotation]]) -> list[list[FrameAnnotation]]:
        if len(groups) <= 1:
            return groups

        merged = [list(group) for group in groups]
        changed = True
        while changed:
            changed = False

            index = 1
            while index < len(merged) - 1:
                left = merged[index - 1]
                middle = merged[index]
                right = merged[index + 1]
                if self._should_merge_bridge_group(left, middle, right):
                    merged[index - 1] = left + middle + right
                    del merged[index : index + 2]
                    changed = True
                    continue
                index += 1

            if changed:
                continue

            index = 0
            while index < len(merged) - 1:
                left = merged[index]
                right = merged[index + 1]
                if self._should_merge_adjacent_groups(left, right):
                    merged[index] = left + right
                    del merged[index + 1]
                    changed = True
                    continue
                index += 1
        return merged

    def _should_merge_bridge_group(
        self,
        left: list[FrameAnnotation],
        middle: list[FrameAnnotation],
        right: list[FrameAnnotation],
    ) -> bool:
        if not self._is_generic_group(middle):
            return False

        if self._is_generic_group(left) and self._is_generic_group(right):
            return True

        if self._same_summary_group(left[-1], right[0]):
            return True

        if self._group_frame_count(middle) <= self._short_generic_group_limit():
            if self._is_generic_group(left) != self._is_generic_group(right):
                return True
            if not self._is_generic_group(left) and not self._is_generic_group(right):
                return self._group_label(left) == self._group_label(right)
        return False

    def _should_merge_adjacent_groups(
        self,
        left: list[FrameAnnotation],
        right: list[FrameAnnotation],
    ) -> bool:
        left_generic = self._is_generic_group(left)
        right_generic = self._is_generic_group(right)

        if left_generic and right_generic:
            return True

        if left_generic and self._group_frame_count(left) <= self._short_generic_group_limit():
            return True

        if right_generic and self._group_frame_count(right) <= self._short_generic_group_limit():
            return True

        return False

    def _is_generic_group(self, group: list[FrameAnnotation]) -> bool:
        best_frame = self._best_summary_frame(group)
        return self._is_generic_frame(best_frame)

    def _group_description(self, group: list[FrameAnnotation]) -> str:
        return self._best_summary_frame(group).semantic.description

    def _group_label(self, group: list[FrameAnnotation]) -> ActionLabel:
        return ActionLabel(str(self._best_summary_frame(group).symbolic_action.label))

    def _group_confidence(self, group: list[FrameAnnotation]) -> float:
        return mean(frame.semantic.confidence for frame in group)

    def _group_frame_count(self, group: list[FrameAnnotation]) -> int:
        return len(group)

    def _short_generic_group_limit(self) -> int:
        return max(3, self.settings.frames_per_segment)

    def _grouping_reason(self, group: list[FrameAnnotation]) -> str:
        return "generic_bridge_merge" if self._is_generic_group(group) and len(group) > 1 else "consecutive_equal_description_and_label"

    def _best_summary_frame(self, group: list[FrameAnnotation]) -> FrameAnnotation:
        return max(
            group,
            key=lambda frame: (
                0 if self._is_generic_frame(frame) else 1,
                frame.symbolic_action.confidence,
                frame.semantic.confidence,
                len(frame.semantic.description.split()),
            ),
        )

    def _is_generic_frame(self, frame: FrameAnnotation) -> bool:
        description = frame.semantic.description
        label = ActionLabel(str(frame.symbolic_action.label))
        confidence = frame.semantic.confidence
        if description in {
            "hand manipulates object",
            "begin hand-object manipulation",
            "finish hand-object manipulation",
            "hold object",
            "pause or hold object",
            "unclear action",
        }:
            return True
        if label == ActionLabel.UNKNOWN:
            return True
        if confidence < 0.38 and any(token in description for token in {"object", "manipulates", "unclear"}):
            return True
        return False

    def _summarize_semantic(self, group: list[FrameAnnotation]) -> SemanticStep:
        representative = self._best_summary_frame(group)
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
                "grouping": self._grouping_reason(group),
            },
        )

    def _summarize_symbolic_action(self, group: list[FrameAnnotation]) -> SymbolicActionLabel:
        representative = self._best_summary_frame(group)
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
