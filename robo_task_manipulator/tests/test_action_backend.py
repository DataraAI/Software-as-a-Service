from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from robotask_manipulator.action_backend import Pi0ActionBackend, create_action_backend
from robotask_manipulator.config import ActionBackendSettings
from robotask_manipulator.schemas import ActionLabel, SegmentAnnotation, SemanticStep, SymbolicActionLabel
from robotask_manipulator.utils.validation import InvalidInputError, ModelLoadError


def test_action_backend_factory_supports_none() -> None:
    backend = create_action_backend(ActionBackendSettings(backend="none"))
    assert backend.backend_name == "none"


def test_pi0_backend_invalid_checkpoint_raises(tmp_path: Path) -> None:
    checkpoint = tmp_path / "missing_checkpoint"
    checkpoint.mkdir()
    backend = Pi0ActionBackend(
        ActionBackendSettings(
            backend="pi0",
            checkpoint_path=str(checkpoint),
            offline=True,
        )
    )
    with pytest.raises(ModelLoadError):
        backend.load()


def test_pi0_backend_requires_image_inputs(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = Pi0ActionBackend(ActionBackendSettings(backend="pi0"))
    backend._loaded = True
    backend._policy = object()
    backend._preprocessor = lambda batch: batch
    backend._config = SimpleNamespace(chunk_size=4, n_action_steps=4)
    backend._image_feature_keys = ["observation.images.front"]
    backend._state_dim = 4

    segment = SegmentAnnotation(
        segment_id="segment-000",
        episode_id="episode",
        step_index=0,
        observation_refs=[],
        representative_frame_ref="frame.jpg",
        frame_start_index=0,
        frame_end_index=0,
        timestamp_start_s=0.0,
        timestamp_end_s=0.0,
        segmentation_confidence=0.8,
        semantic=SemanticStep(description="pick the item", confidence=0.8),
        symbolic_action=SymbolicActionLabel(label=ActionLabel.UNKNOWN, confidence=0.0, source="test"),
    )

    with pytest.raises(InvalidInputError):
        backend.propose(
            episode=SimpleNamespace(episode_id="episode", instruction="pick the item"),
            segment=segment,
        )
