from __future__ import annotations

from pathlib import Path

from robotask_manipulator.config import SemanticSettings
from robotask_manipulator.task_understanding.transformers_vlm import TransformersTaskUnderstandingBackend


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
    assert prediction.evidence["pipeline_mode"] in {"image-text-to-text", "image-to-text", "heuristic"}
