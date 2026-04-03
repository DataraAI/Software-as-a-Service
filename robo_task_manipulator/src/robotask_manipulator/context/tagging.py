"""Lightweight context and failure tagging for v1."""

from __future__ import annotations

from robotask_manipulator.schemas import ActionLabel, ContextTag, ContextTagName, SegmentAnnotation


class ContextTagger:
    """Heuristic context/failure tagging based on semantic and optional action signals."""

    def annotate(self, segment: SegmentAnnotation) -> list[ContextTag]:
        tags: list[ContextTag] = []
        description = segment.semantic.description.lower()
        caption = str(segment.semantic.evidence.get("caption") or "").lower()
        stats = (segment.action_proposal.metadata if segment.action_proposal else {}).get("chunk_stats", {})
        variance = float(stats.get("variance", 0.0))
        mean_abs = float(stats.get("mean_abs", 0.0))
        label = ActionLabel(str(segment.symbolic_action.label))

        if "occlud" in description or caption.find("occluded") >= 0:
            tags.append(self._tag(ContextTagName.OCCLUSION, 0.55, "semantic_vlm", {"description": description}))
        if label in {ActionLabel.ALIGN, ActionLabel.INSERT} and ("align" in description or "misalign" in description):
            tags.append(self._tag(ContextTagName.MISALIGNMENT, 0.58, "semantic_rule", {"description": description}))
        if label == ActionLabel.INSERT and mean_abs < 0.04:
            tags.append(self._tag(ContextTagName.BLOCKED_INSERTION, 0.62, "action_chunk_rule", {"mean_abs": mean_abs}))
        if label in {ActionLabel.PICK, ActionLabel.HOLD} and variance > 0.08:
            tags.append(self._tag(ContextTagName.UNSTABLE_GRASP, 0.6, "action_chunk_rule", {"variance": variance}))
        if label == ActionLabel.RETRY:
            tags.append(self._tag(ContextTagName.RETRY_REQUIRED, 0.72, "symbolic_action", {"label": str(label)}))
        if "drop" in description:
            tags.append(self._tag(ContextTagName.DROPPED_OBJECT, 0.78, "semantic_rule", {"description": description}))
        if "miss" in description or ("target" in description and label == ActionLabel.RETRY):
            tags.append(self._tag(ContextTagName.MISSED_TARGET, 0.61, "semantic_rule", {"description": description}))
        if not tags and label == ActionLabel.UNKNOWN:
            tags.append(self._tag(ContextTagName.UNKNOWN_FAILURE, 0.3, "fallback", {"reason": "low confidence"}))
        return tags

    def _tag(self, name: ContextTagName, confidence: float, source: str, evidence: dict) -> ContextTag:
        return ContextTag(name=name, confidence=confidence, source=source, evidence=evidence)
