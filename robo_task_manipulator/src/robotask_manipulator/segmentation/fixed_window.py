"""Simple deterministic segmentation for v1."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from robotask_manipulator.config import SegmentationSettings
from robotask_manipulator.schemas import ActionLabel, EpisodeInput, SegmentAnnotation, SemanticStep, SymbolicActionLabel


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
