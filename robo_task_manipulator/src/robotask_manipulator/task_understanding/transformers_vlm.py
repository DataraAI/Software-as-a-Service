"""Transformers-based task-understanding backend."""

from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
from PIL import Image

from robotask_manipulator.config import SemanticSettings
from robotask_manipulator.task_understanding.base import BaseTaskUnderstandingBackend, SemanticPrediction

LOGGER = logging.getLogger(__name__)


class TransformersTaskUnderstandingBackend(BaseTaskUnderstandingBackend):
    """Use a pretrained VLM to describe the main visible action across ordered frames."""

    def __init__(self, settings: SemanticSettings) -> None:
        self.settings = settings
        self._pipeline = None
        self._pipeline_mode = "unloaded"

    def load(self) -> None:
        if self._pipeline is not None:
            return
        try:
            from transformers import pipeline

            device = 0 if self.settings.device.startswith("cuda") else -1
            try:
                self._pipeline = pipeline(
                    "image-text-to-text",
                    model=self.settings.model_id,
                    device=device,
                    local_files_only=self.settings.offline,
                )
                self._pipeline_mode = "image-text-to-text"
                LOGGER.info(
                    "Loaded multimodal task-understanding backend model=%s mode=%s",
                    self.settings.model_id,
                    self._pipeline_mode,
                )
                return
            except Exception as vlm_exc:  # noqa: BLE001
                LOGGER.warning(
                    "Falling back from image-text-to-text to image-to-text for model=%s: %s",
                    self.settings.model_id,
                    vlm_exc,
                )

            self._pipeline = pipeline(
                "image-to-text",
                model=self.settings.model_id,
                device=device,
                local_files_only=self.settings.offline,
            )
            self._pipeline_mode = "image-to-text"
            LOGGER.info(
                "Loaded caption-based task-understanding backend model=%s mode=%s",
                self.settings.model_id,
                self._pipeline_mode,
            )
        except Exception as exc:  # noqa: BLE001
            if self.settings.strict:
                raise RuntimeError(
                    f"Failed to load task-understanding backend '{self.settings.model_id}'."
                ) from exc
            LOGGER.warning(
                "Falling back to conservative task-understanding heuristic because VLM load failed: %s",
                exc,
            )
            self._pipeline = False
            self._pipeline_mode = "heuristic"

    def predict(
        self,
        frame_paths: list[str],
        instruction: str,
        step_index: int,
        total_steps: int,
    ) -> SemanticPrediction:
        self.load()
        sampled_paths = _sample_frame_paths(frame_paths)
        motion_score = _estimate_motion(sampled_paths)
        frame_captions: list[str] = []
        raw_text: str | None = None
        confidence = 0.3

        if self._pipeline and self._pipeline_mode == "image-text-to-text":
            raw_text = self._predict_multiframe(sampled_paths, instruction, step_index, total_steps)
            confidence = 0.78 if raw_text else 0.35
        elif self._pipeline and self._pipeline_mode == "image-to-text":
            frame_captions = self._caption_frames(sampled_paths)
            raw_text = _summarize_caption_sequence(frame_captions, motion_score)
            confidence = 0.55 if raw_text else 0.32

        description = _normalize_step_description(raw_text)
        if not description:
            description = _heuristic_step_description(motion_score, step_index, total_steps)
            confidence = 0.28

        object_source, object_target = _extract_source_target(description)
        objects = _extract_objects(description, raw_text, frame_captions)
        return SemanticPrediction(
            description=description,
            task_intent=instruction,
            objects_involved=objects,
            object_source=object_source,
            object_target=object_target,
            confidence=round(min(0.92, confidence), 2),
            caption=raw_text,
            evidence={
                "pipeline_mode": self._pipeline_mode,
                "sampled_frame_paths": sampled_paths,
                "frame_captions": frame_captions,
                "motion_score": motion_score,
                "step_index": step_index,
                "total_steps": total_steps,
                "used_instruction_only_as_context": True,
                "raw_semantic_response": raw_text,
            },
        )

    def _predict_multiframe(
        self,
        sampled_paths: list[str],
        instruction: str,
        step_index: int,
        total_steps: int,
    ) -> str | None:
        images = _load_images(sampled_paths)
        if not images:
            return None

        prompt = (
            "These are ordered frames from one short segment of a human hand manipulation task. "
            "Describe only the single main visible action happening across these frames in 2 to 6 words. "
            "Focus on the hands and objects. Do not repeat the full task instruction unless it is clearly visible. "
            "If the action is unclear, answer 'unclear action'. "
            f"Task context: {instruction}. Segment position: {step_index + 1} of {total_steps}."
        )
        messages = [
            {
                "role": "user",
                "content": [{"type": "image"} for _ in images] + [{"type": "text", "text": prompt}],
            }
        ]
        try:
            outputs = self._pipeline(
                text=messages,
                images=images,
                max_new_tokens=64,
                return_full_text=False,
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Multiframe VLM inference failed for frames=%s: %s", sampled_paths, exc)
            return None
        return _coerce_generated_text(outputs)

    def _caption_frames(self, sampled_paths: list[str]) -> list[str]:
        captions: list[str] = []
        for frame_path in sampled_paths:
            try:
                outputs = self._pipeline(str(Path(frame_path)), max_new_tokens=32)
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("Caption inference failed for %s: %s", frame_path, exc)
                continue
            caption = _coerce_generated_text(outputs)
            if caption:
                captions.append(caption)
        return captions


def _sample_frame_paths(frame_paths: list[str]) -> list[str]:
    unique_paths = []
    for path in frame_paths:
        if path not in unique_paths:
            unique_paths.append(path)
    if len(unique_paths) <= 3:
        return unique_paths
    indices = {0, len(unique_paths) // 2, len(unique_paths) - 1}
    return [unique_paths[index] for index in sorted(indices)]


def _load_images(frame_paths: list[str]) -> list[Image.Image]:
    images: list[Image.Image] = []
    for frame_path in frame_paths:
        try:
            with Image.open(Path(frame_path)) as image:
                images.append(image.convert("RGB"))
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Failed to load frame for task understanding: %s (%s)", frame_path, exc)
    return images


def _estimate_motion(frame_paths: list[str]) -> float:
    if len(frame_paths) < 2:
        return 0.0
    try:
        with Image.open(Path(frame_paths[0])) as first_image, Image.open(Path(frame_paths[-1])) as last_image:
            first = np.asarray(first_image.convert("RGB"), dtype=np.float32) / 255.0
            last = np.asarray(last_image.convert("RGB").resize(first_image.size), dtype=np.float32) / 255.0
        return round(float(np.mean(np.abs(first - last))), 4)
    except Exception:  # noqa: BLE001
        return 0.0


def _coerce_generated_text(outputs) -> str | None:
    if not outputs:
        return None
    first = outputs[0] if isinstance(outputs, list) else outputs
    if isinstance(first, dict):
        generated_text = first.get("generated_text")
        if isinstance(generated_text, str):
            return generated_text.strip()
        if isinstance(generated_text, list):
            pieces = []
            for item in generated_text:
                if isinstance(item, dict) and "text" in item:
                    pieces.append(str(item["text"]))
                elif isinstance(item, str):
                    pieces.append(item)
            return " ".join(piece.strip() for piece in pieces if piece.strip()) or None
    if isinstance(first, str):
        return first.strip()
    return str(first).strip() or None


def _normalize_step_description(raw_text: str | None) -> str | None:
    if not raw_text:
        return None
    text = raw_text.strip()
    text = re.sub(r"^step_description\s*:\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^main visible step\s*:\s*", "", text, flags=re.IGNORECASE)
    text = text.split("\n", 1)[0].strip()
    text = text.strip(" .")
    if not text:
        return None
    if text.lower() in {"unclear", "unclear action", "unknown action"}:
        return "unclear action"
    return text.lower()


def _summarize_caption_sequence(frame_captions: list[str], motion_score: float) -> str | None:
    if not frame_captions:
        return None
    merged = " ".join(caption.strip() for caption in frame_captions if caption.strip())
    merged = re.sub(r"\s+", " ", merged).strip()
    if not merged:
        return None
    if motion_score < 0.015:
        return "holding or pausing with object"
    return merged


def _heuristic_step_description(motion_score: float, step_index: int, total_steps: int) -> str:
    if motion_score < 0.01:
        return "pause or hold object"
    if step_index == 0 and total_steps > 1:
        return "begin hand-object manipulation"
    if step_index == total_steps - 1 and total_steps > 1:
        return "finish hand-object manipulation"
    return "hand manipulates object"


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
    if " out of " in lowered:
        before, after = lowered.split(" out of ", 1)
        return after.strip(), before.strip()
    return None, None


def _extract_objects(description: str, raw_text: str | None, frame_captions: list[str]) -> list[str]:
    candidates = []
    for source in [description, raw_text or "", *frame_captions]:
        for token in re.split(r"[^a-zA-Z]+", source):
            token = token.strip().lower()
            if len(token) > 2 and token not in {
                "then",
                "with",
                "from",
                "into",
                "onto",
                "carefully",
                "visible",
                "action",
                "main",
                "step",
                "hands",
                "human",
            }:
                candidates.append(token)
    seen = []
    for candidate in candidates:
        if candidate not in seen:
            seen.append(candidate)
    return seen[:8]
