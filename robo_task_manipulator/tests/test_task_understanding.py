from __future__ import annotations

from pathlib import Path

from robotask_manipulator.config import SemanticSettings
from robotask_manipulator.task_understanding.transformers_vlm import (
    TransformersTaskUnderstandingBackend,
    _build_semantic_prompt,
    _collect_context_hints,
    _normalize_step_description,
    _sample_frame_paths,
)


def test_task_understanding_uses_multiple_frames_with_conservative_fallback() -> None:
    root = Path(__file__).resolve().parents[1]
    frame_paths = [
        str(root / "data" / "sample_inputs" / "sample_frame_001.ppm"),
        str(root / "data" / "sample_inputs" / "sample_frame_002.ppm"),
        str(root / "data" / "sample_inputs" / "sample_frame_003.ppm"),
    ]
    backend = TransformersTaskUnderstandingBackend(
        SemanticSettings(
            backend="multimodal_vlm",
            model_id="missing/test-model",
            offline=True,
            strict=False,
        )
    )
    prediction = backend.predict(
        frame_paths=frame_paths,
        instruction="Open the pod and remove the peas.",
        step_index=0,
        total_steps=2,
    )
    assert prediction.description
    assert len(prediction.evidence["sampled_frame_paths"]) >= 2
    assert prediction.evidence["pipeline_mode"] in {
        "direct-image-text-to-text",
        "image-text-to-text",
        "image-to-text",
        "heuristic",
    }


def test_semantic_prompt_uses_optional_metadata_hints() -> None:
    hints = _collect_context_hints(
        "real_video_test",
        {
            "tags": ["ethernet cable", "laptop port"],
            "scene": "desk setup",
        },
    )
    prompt = _build_semantic_prompt(
        "Describe the visible hand-object manipulation step conservatively.",
        0,
        10,
        task_name="real_video_test",
        context_hints=hints,
    )
    assert "ethernet cable" in prompt
    assert "laptop port" in prompt
    assert "real_video_test" not in prompt


def test_normalize_step_description_prefers_more_specific_clause() -> None:
    normalized = _normalize_step_description("Holding cable near laptop port, plugging cable into port.")
    assert normalized == "plug cable into port"


def test_sample_frame_paths_supports_single_representative_frame() -> None:
    paths = [f"frame_{index}.jpg" for index in range(5)]
    assert _sample_frame_paths(paths, max_samples=1) == ["frame_2.jpg"]
