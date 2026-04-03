"""Transformers-based VLM backend for conservative semantic understanding."""

from __future__ import annotations

import logging
from pathlib import Path

from robotask_manipulator.config import SemanticSettings
from robotask_manipulator.understanding.base import BaseSemanticBackend, SemanticPrediction

LOGGER = logging.getLogger(__name__)


class TransformersVLMBackend(BaseSemanticBackend):
    """Use a pretrained image-to-text model for practical semantic captions."""

    def __init__(self, settings: SemanticSettings) -> None:
        self.settings = settings
        self._pipeline = None

    def load(self) -> None:
        if self._pipeline is not None:
            return
        try:
            from transformers import pipeline

            device = 0 if self.settings.device.startswith("cuda") else -1
            self._pipeline = pipeline("image-to-text", model=self.settings.model_id, device=device)
            LOGGER.info("Loaded semantic VLM backend model=%s", self.settings.model_id)
        except Exception as exc:  # noqa: BLE001
            if self.settings.strict:
                raise RuntimeError(f"Failed to load semantic VLM backend '{self.settings.model_id}'.") from exc
            LOGGER.warning("Falling back to conservative semantic heuristic because VLM load failed: %s", exc)
            self._pipeline = False

    def predict(self, image_path: str, instruction: str, step_index: int, total_steps: int) -> SemanticPrediction:
        self.load()
        instruction_steps = _split_instruction(instruction)
        suggested = instruction_steps[min(step_index, len(instruction_steps) - 1)] if instruction_steps else instruction

        caption = None
        confidence = 0.42
        if self._pipeline:
            try:
                result = self._pipeline(str(Path(image_path)), max_new_tokens=32)
                caption = result[0].get("generated_text", "").strip() if result else ""
                confidence = 0.72 if caption else 0.38
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("Semantic VLM inference failed for %s: %s", image_path, exc)

        description = suggested if suggested else (caption or f"step {step_index + 1}")
        objects = _extract_objects(description, caption)
        object_source, object_target = _extract_source_target(description)
        return SemanticPrediction(
            description=description,
            task_intent=instruction,
            objects_involved=objects,
            object_source=object_source,
            object_target=object_target,
            confidence=round(min(0.92, confidence), 2),
            caption=caption,
            evidence={"caption": caption, "instruction_step": suggested, "step_index": step_index, "total_steps": total_steps},
        )


def _split_instruction(instruction: str) -> list[str]:
    normalized = (
        instruction.replace(", then ", ". ")
        .replace(" then ", ". ")
        .replace(", and ", ". ")
        .replace(" and ", ". ")
        .replace(", ", ". ")
    )
    return [part.strip(" .") for part in normalized.split(".") if part.strip(" .")]


def _extract_source_target(text: str) -> tuple[str | None, str | None]:
    lowered = text.lower()
    if " from " in lowered:
        before, after = lowered.split(" from ", 1)
        return after.strip(), before.strip()
    if " into " in lowered:
        before, after = lowered.split(" into ", 1)
        return before.strip(), after.strip()
    if " onto " in lowered:
        before, after = lowered.split(" onto ", 1)
        return before.strip(), after.strip()
    return None, None


def _extract_objects(description: str, caption: str | None) -> list[str]:
    candidates = []
    for source in [description, caption or ""]:
        for token in source.replace(",", " ").replace(".", " ").split():
            token = token.strip().lower()
            if len(token) > 2 and token not in {"then", "with", "from", "into", "onto", "carefully"}:
                candidates.append(token)
    seen = []
    for candidate in candidates:
        if candidate not in seen:
            seen.append(candidate)
    return seen[:6]
