"""Deterministic ordered task graph builder."""

from __future__ import annotations

from robotask_manipulator.schemas import ContextTagName, SegmentAnnotation, TaskEdge, TaskEdgeType, TaskGraph, TaskNode


class SequenceGraphBuilder:
    """Convert ordered segments into a simple sequence graph."""

    def build(self, segments: list[SegmentAnnotation]) -> TaskGraph:
        nodes = [
            TaskNode(
                node_id=f"node-{segment.step_index:03d}",
                segment_id=segment.segment_id,
                step_index=segment.step_index,
                terminal=(segment.step_index == len(segments) - 1),
            )
            for segment in segments
        ]

        edges: list[TaskEdge] = []
        terminal_conditions: list[str] = []
        for index, segment in enumerate(segments):
            current_node = nodes[index]
            if index + 1 < len(nodes):
                edges.append(
                    TaskEdge(
                        source_node_id=current_node.node_id,
                        target_node_id=nodes[index + 1].node_id,
                        edge_type=TaskEdgeType.NEXT,
                        evidence={"segment_id": segment.segment_id},
                    )
                )

            tag_names = {str(tag.name) for tag in segment.context_tags}
            if ContextTagName.RETRY_REQUIRED.value in tag_names and index > 0:
                edges.append(
                    TaskEdge(
                        source_node_id=current_node.node_id,
                        target_node_id=nodes[index - 1].node_id,
                        edge_type=TaskEdgeType.RETRY,
                        confidence=0.8,
                        evidence={"segment_id": segment.segment_id},
                    )
                )
            if ContextTagName.UNKNOWN_FAILURE.value in tag_names:
                terminal_conditions.append(f"{segment.segment_id}: unknown_failure")
            if segment.success is False:
                terminal_conditions.append(f"{segment.segment_id}: failed")

        if nodes:
            edges.append(
                TaskEdge(
                    source_node_id=nodes[-1].node_id,
                    target_node_id=nodes[-1].node_id,
                    edge_type=TaskEdgeType.TERMINAL,
                    confidence=1.0,
                    evidence={"reason": "last_segment"},
                )
            )
        return TaskGraph(nodes=nodes, edges=edges, terminal_conditions=terminal_conditions)
