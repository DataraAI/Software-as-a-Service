"""Conservative symbolic action labeling."""

from __future__ import annotations

from robotask_manipulator.schemas import ActionLabel, ActionProposal, SegmentAnnotation, SymbolicActionLabel

KEYWORD_SCORES: dict[ActionLabel, tuple[str, ...]] = {
    ActionLabel.PICK: ("pick", "grasp", "lift", "grab"),
    ActionLabel.PLACE: ("place", "put", "set", "deposit"),
    ActionLabel.ALIGN: ("align", "position", "line up"),
    ActionLabel.INSERT: ("insert", "slot", "fit", "plug"),
    ActionLabel.FASTEN: ("fasten", "tighten", "screw", "bolt"),
    ActionLabel.PUSH: ("push", "press"),
    ActionLabel.PULL: ("pull", "draw"),
    ActionLabel.HOLD: ("hold", "stabilize", "steady"),
    ActionLabel.INSPECT: ("inspect", "check", "look", "verify"),
    ActionLabel.REGRASP: ("regrasp", "adjust grip", "reposition grip"),
    ActionLabel.RETRY: ("retry", "try again", "repeat"),
    ActionLabel.RELEASE: ("release", "let go", "open gripper"),
    ActionLabel.WAIT: ("wait", "pause", "idle"),
}


class SymbolicActionLabeler:
    """Turn semantic understanding plus optional action proposals into conservative symbolic labels."""

    def label(self, segment: SegmentAnnotation) -> SymbolicActionLabel:
        description = segment.semantic.description.lower()
        scores = {label: 0.0 for label in ActionLabel}

        for label, keywords in KEYWORD_SCORES.items():
            for keyword in keywords:
                if keyword in description:
                    scores[label] += 0.55

        chunk_scores = self._score_action_proposal(segment.action_proposal)
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

        source = "semantic_vlm"
        if segment.action_proposal is not None:
            source = "semantic_vlm_plus_action_backend"

        return SymbolicActionLabel(
            label=best_label,
            confidence=confidence,
            source=source,
            evidence={
                "description": segment.semantic.description,
                "objects": segment.semantic.objects_involved,
                "action_backend": segment.action_proposal.backend if segment.action_proposal else "none",
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
