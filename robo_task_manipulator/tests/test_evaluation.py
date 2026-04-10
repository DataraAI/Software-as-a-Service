from __future__ import annotations

from robotask_manipulator.evaluation import EvaluationService
from robotask_manipulator.schemas import (
    ActionLabel,
    BenchmarkEpisode,
    BenchmarkStep,
    EpisodeOutput,
    FrameAnnotation,
    IsaacSimExport,
    MediaMetadata,
    MediaType,
    SegmentAnnotation,
    SemanticStep,
    SymbolicActionLabel,
    TaskGraph,
)


def test_episode_evaluation_summary() -> None:
    output = EpisodeOutput(
        episode_id="episode",
        task_name="task",
        instruction="pick then place",
        input_metadata=MediaMetadata(media_type=MediaType.IMAGE, source_ref="image.jpg"),
        frame_predictions=[
            FrameAnnotation(
                frame_id="frame-000",
                episode_id="episode",
                frame_index=0,
                asset_ref="image.jpg",
                semantic=SemanticStep(description="pick the object", confidence=0.8),
                symbolic_action=SymbolicActionLabel(label=ActionLabel.PICK, confidence=0.8, source="test"),
            ),
            FrameAnnotation(
                frame_id="frame-001",
                episode_id="episode",
                frame_index=1,
                asset_ref="image.jpg",
                semantic=SemanticStep(description="place the object", confidence=0.8),
                symbolic_action=SymbolicActionLabel(label=ActionLabel.PLACE, confidence=0.8, source="test"),
            ),
        ],
        segments=[
            SegmentAnnotation(
                segment_id="segment-000",
                episode_id="episode",
                step_index=0,
                observation_refs=["image.jpg"],
                representative_frame_ref="image.jpg",
                frame_start_index=0,
                frame_end_index=0,
                timestamp_start_s=0.0,
                timestamp_end_s=0.0,
                segmentation_confidence=0.8,
                semantic=SemanticStep(description="pick the object", confidence=0.8),
                symbolic_action=SymbolicActionLabel(label=ActionLabel.PICK, confidence=0.8, source="test"),
            ),
            SegmentAnnotation(
                segment_id="segment-001",
                episode_id="episode",
                step_index=1,
                observation_refs=["image.jpg"],
                representative_frame_ref="image.jpg",
                frame_start_index=1,
                frame_end_index=1,
                timestamp_start_s=1.0,
                timestamp_end_s=1.0,
                segmentation_confidence=0.8,
                semantic=SemanticStep(description="place the object", confidence=0.8),
                symbolic_action=SymbolicActionLabel(label=ActionLabel.PLACE, confidence=0.8, source="test"),
            ),
        ],
        task_graph=TaskGraph(),
        simulation_export=IsaacSimExport(episode_id="episode", task_name="task", steps=[]),
    )
    benchmark = BenchmarkEpisode(
        episode_id="episode",
        expected_steps=[
            BenchmarkStep(step_index=0, description="pick", action_label=ActionLabel.PICK),
            BenchmarkStep(step_index=1, description="place", action_label=ActionLabel.PLACE),
        ],
    )

    summary = EvaluationService().evaluate_episode(output, benchmark)
    assert summary.step_count_difference == 0
    assert summary.step_label_agreement == 1.0
