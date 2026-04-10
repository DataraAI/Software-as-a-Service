"""Conservative symbolic action labeling."""

from __future__ import annotations

from robotask_manipulator.schemas import (
    ActionLabel,
    ActionProposal,
    FrameAnnotation,
    SegmentAnnotation,
    SemanticStep,
    SymbolicActionLabel,
)

KEYWORD_SCORES: dict[ActionLabel, tuple[str, ...]] = {
    ActionLabel.PICK: ("pick", "grasp", "lift", "grab", "take out", "remove"),
    ActionLabel.PLACE: ("place", "put", "set", "deposit"),
    ActionLabel.ALIGN: ("align", "position", "line up"),
    ActionLabel.INSERT: ("insert", "slot", "fit", "plug", "connect"),
    ActionLabel.FASTEN: ("fasten", "tighten", "screw", "bolt"),
    ActionLabel.PUSH: ("push", "press"),
    ActionLabel.PULL: ("pull", "draw", "peel", "open"),
    ActionLabel.HOLD: ("hold", "stabilize", "steady", "grip"),
    ActionLabel.INSPECT: ("inspect", "check", "look", "verify", "examine"),
    ActionLabel.REGRASP: ("regrasp", "adjust grip", "reposition grip"),
    ActionLabel.RETRY: ("retry", "try again", "repeat"),
    ActionLabel.RELEASE: ("release", "let go", "open gripper", "drop"),
    ActionLabel.WAIT: ("wait", "pause", "idle", "stop"),
}


class SymbolicActionLabeler:
    """Turn segment semantics plus optional action proposals into conservative symbolic labels."""

    def label(self, segment: SegmentAnnotation) -> SymbolicActionLabel:
        source = "task_understanding_vlm"
        if segment.action_proposal is not None:
            source = "task_understanding_vlm_plus_action_backend"
        return self._label_from_semantic(
            semantic=segment.semantic,
            action_proposal=segment.action_proposal,
            source=source,
        )

    def label_frame(self, frame: FrameAnnotation) -> SymbolicActionLabel:
        return self._label_from_semantic(
            semantic=frame.semantic,
            action_proposal=None,
            source="task_understanding_vlm",
        )

    def _label_from_semantic(
        self,
        *,
        semantic: SemanticStep,
        action_proposal: ActionProposal | None,
        source: str,
    ) -> SymbolicActionLabel:
        description = semantic.description.lower()
        scores = {label: 0.0 for label in ActionLabel}

        for label, keywords in KEYWORD_SCORES.items():
            for keyword in keywords:
                if keyword in description:
                    scores[label] += 0.58

        chunk_scores = self._score_action_proposal(action_proposal)
        for label, value in chunk_scores.items():
            scores[label] += value

        best_label = ActionLabel.UNKNOWN
        best_score = 0.0
        for label, score in scores.items():
            if score > best_score:
                best_label = label
                best_score = score

        confidence = min(0.95, round(best_score, 2))
        if confidence < 0.4:
            best_label = ActionLabel.UNKNOWN
            confidence = round(max(confidence, 0.25), 2)

        return SymbolicActionLabel(
            label=best_label,
            confidence=confidence,
            source=source,
            evidence={
                "description": semantic.description,
                "objects": semantic.objects_involved,
                "action_backend": action_proposal.backend if action_proposal else "none",
                "chunk_scores": {label.value: value for label, value in chunk_scores.items() if value > 0.0},
            },
        )

    def _score_action_proposal(self, proposal: ActionProposal | None) -> dict[ActionLabel, float]:
        scores = {label: 0.0 for label in ActionLabel}
        if proposal is None or not proposal.action_chunk:
            return scores

        stats = proposal.metadata.get("chunk_stats", {})
        mean_abs = float(stats.get("mean_abs", 0.0))
        variance = float(stats.get("variance", 0.0))
        selected = proposal.selected_action or []

        if mean_abs < 0.03:
            scores[ActionLabel.WAIT] += 0.35
            scores[ActionLabel.INSPECT] += 0.15
        if variance < 0.01 and mean_abs < 0.08:
            scores[ActionLabel.ALIGN] += 0.2
            scores[ActionLabel.HOLD] += 0.15
        if variance > 0.08:
            scores[ActionLabel.REGRASP] += 0.2
            scores[ActionLabel.RETRY] += 0.15
        if selected:
            first_dim = selected[0]
            gripper_dim = selected[-1]
            if first_dim > 0.1:
                scores[ActionLabel.PUSH] += 0.35
            if first_dim < -0.1:
                scores[ActionLabel.PULL] += 0.35
            if gripper_dim > 0.15:
                scores[ActionLabel.PICK] += 0.45
                scores[ActionLabel.HOLD] += 0.22
            if gripper_dim < -0.15:
                scores[ActionLabel.RELEASE] += 0.45
                scores[ActionLabel.PLACE] += 0.25
        return scores
