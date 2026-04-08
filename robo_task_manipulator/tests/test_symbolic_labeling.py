from __future__ import annotations

from robotask_manipulator.schemas import (
    ActionLabel,
    ActionProposal,
    SegmentAnnotation,
    SemanticStep,
    SymbolicActionLabel,
)
from robotask_manipulator.task_understanding import SymbolicActionLabeler


def _segment(description: str) -> SegmentAnnotation:
    return SegmentAnnotation(
        segment_id="episode-segment-000",
        episode_id="episode",
        step_index=0,
        observation_refs=["frame.jpg"],
        representative_frame_ref="frame.jpg",
        frame_start_index=0,
        frame_end_index=0,
        timestamp_start_s=0.0,
        timestamp_end_s=0.0,
        segmentation_confidence=0.8,
        semantic=SemanticStep(description=description, confidence=0.8),
        symbolic_action=SymbolicActionLabel(label=ActionLabel.UNKNOWN, confidence=0.0, source="test"),
    )


def test_symbolic_labeling_uses_semantic_description() -> None:
    labeler = SymbolicActionLabeler()
    segment = _segment("pick the battery from the tray")
    result = labeler.label(segment)
    assert result.label == ActionLabel.PICK.value
    assert result.confidence >= 0.5


def test_symbolic_labeling_uses_action_backend_evidence() -> None:
    labeler = SymbolicActionLabeler()
    segment = _segment("move carefully")
    segment.action_proposal = ActionProposal(
        backend="pi0",
        selected_action=[0.2, 0.0, 0.0, 0.3],
        action_chunk=[[0.2, 0.0, 0.0, 0.3], [0.18, 0.0, 0.0, 0.32]],
        confidence=0.8,
        metadata={"chunk_stats": {"mean_abs": 0.12, "variance": 0.02}},
    )
    result = labeler.label(segment)
    assert result.label in {ActionLabel.PICK.value, ActionLabel.PUSH.value, ActionLabel.HOLD.value}


def test_symbolic_labeling_falls_back_to_unknown() -> None:
    labeler = SymbolicActionLabeler()
    segment = _segment("continue task")
    result = labeler.label(segment)
    assert result.label == ActionLabel.UNKNOWN.value
