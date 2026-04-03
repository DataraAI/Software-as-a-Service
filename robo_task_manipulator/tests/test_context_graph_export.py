from __future__ import annotations

from robotask_manipulator.context import ContextTagger
from robotask_manipulator.graph import SequenceGraphBuilder
from robotask_manipulator.schemas import (
    ActionLabel,
    ActionProposal,
    SegmentAnnotation,
    SemanticStep,
    SymbolicActionLabel,
)
from robotask_manipulator.simulation import IsaacSimFrankaExporter


def _segment(step_index: int, description: str, label: ActionLabel) -> SegmentAnnotation:
    return SegmentAnnotation(
        segment_id=f"episode-segment-{step_index:03d}",
        episode_id="episode",
        step_index=step_index,
        observation_refs=[f"frame_{step_index}.jpg"],
        representative_frame_ref=f"frame_{step_index}.jpg",
        frame_start_index=step_index,
        frame_end_index=step_index,
        timestamp_start_s=float(step_index),
        timestamp_end_s=float(step_index),
        segmentation_confidence=0.8,
        semantic=SemanticStep(description=description, confidence=0.8),
        symbolic_action=SymbolicActionLabel(label=label, confidence=0.8, source="test"),
        action_proposal=ActionProposal(
            backend="pi0",
            selected_action=[0.0, 0.0, 0.0, 0.0],
            action_chunk=[[0.0, 0.0, 0.0, 0.0]],
            metadata={"chunk_stats": {"mean_abs": 0.01, "variance": 0.001}},
        ),
    )


def test_context_tagging_graph_and_sim_export() -> None:
    insert_segment = _segment(0, "insert the battery into the slot", ActionLabel.INSERT)
    retry_segment = _segment(1, "retry alignment with the slot", ActionLabel.RETRY)

    tagger = ContextTagger()
    insert_segment.context_tags = tagger.annotate(insert_segment)
    retry_segment.context_tags = tagger.annotate(retry_segment)
    retry_segment.success = True

    assert any(str(tag.name) == "blocked_insertion" for tag in insert_segment.context_tags)
    assert any(str(tag.name) == "retry_required" for tag in retry_segment.context_tags)

    graph = SequenceGraphBuilder().build([insert_segment, retry_segment])
    assert len(graph.nodes) == 2
    assert any(str(edge.edge_type) == "retry" for edge in graph.edges)

    sim_export = IsaacSimFrankaExporter().build("episode", "battery_task", [insert_segment, retry_segment])
    assert sim_export.robot == "franka_panda"
    assert sim_export.steps[0].primitive == "insert"
