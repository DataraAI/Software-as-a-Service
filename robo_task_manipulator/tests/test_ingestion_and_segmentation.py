from __future__ import annotations

from pathlib import Path

from robotask_manipulator.config import IngestionSettings, SegmentationSettings
from robotask_manipulator.ingestion import MediaIngestor
from robotask_manipulator.segmentation import Segmenter
from robotask_manipulator.utils.io import load_json_file


def test_frame_sequence_ingestion_and_segmentation() -> None:
    root = Path(__file__).resolve().parents[1]
    payload_path = root / "data" / "sample_inputs" / "sample_workflow_episode_001.json"
    payload = load_json_file(payload_path)

    episode = MediaIngestor(IngestionSettings()).from_payload(payload, payload_path.parent)
    assert episode.media_metadata.media_type == "frame_sequence"
    assert len(episode.frames) == 3
    assert episode.frames[0].asset_ref.endswith("sample_frame_001.ppm")

    segments = Segmenter(SegmentationSettings(frames_per_segment=2, use_visual_change_breaks=False)).segment(episode)
    assert len(segments) == 2
    assert segments[0].next_step_refs == [segments[1].segment_id]
    assert segments[0].frame_start_index == 0
    assert segments[0].frame_end_index == 1
