"""Isaac Sim 5.1 / Franka Panda export adapter."""

from __future__ import annotations

from robotask_manipulator.schemas import IsaacSimExport, IsaacSimStep, SegmentAnnotation


class IsaacSimFrankaExporter:
    """Convert segments into a simulation-ready task plan."""

    def build(self, episode_id: str, task_name: str, segments: list[SegmentAnnotation]) -> IsaacSimExport:
        steps = [
            IsaacSimStep(
                step_index=segment.step_index,
                segment_id=segment.segment_id,
                primitive=str(segment.symbolic_action.label),
                description=segment.semantic.description,
                target_object=segment.semantic.object_target,
                source_object=segment.semantic.object_source,
                confidence=min(segment.semantic.confidence, segment.symbolic_action.confidence),
                status="failed" if segment.success is False else "planned",
                action_proposal=segment.action_proposal,
                tags=[str(tag.name) for tag in segment.context_tags],
            )
            for segment in segments
        ]
        return IsaacSimExport(
            episode_id=episode_id,
            task_name=task_name,
            steps=steps,
            metadata={
                "simulator_version": "5.1",
                "robot": "Franka Panda",
                "representation": "symbolic_task_plan",
            },
        )
