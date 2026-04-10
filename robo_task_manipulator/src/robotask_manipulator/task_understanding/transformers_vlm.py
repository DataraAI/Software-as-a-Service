"""Transformers-based task-understanding backend."""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any

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
        self._processor = None
        self._model = None
        self._model_device = settings.device
        self._model_dtype = None
        self._pipeline_mode = "unloaded"

    def load(self) -> None:
        if self._pipeline is not None or self._model is not None:
            return
        try:
            model_source = self.settings.model_source
            if self.settings.offline:
                # The generic transformers pipeline does not consistently accept
                # local_files_only, so use the standard offline env flags instead.
                os.environ.setdefault("HF_HUB_OFFLINE", "1")
                os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

            if _prefers_direct_model_path(model_source, self.settings.local_model_path):
                try:
                    self._load_direct_model()
                    return
                except Exception as direct_exc:  # noqa: BLE001
                    LOGGER.warning(
                        "Falling back from direct model path to transformers pipeline for model=%s: %s",
                        model_source,
                        direct_exc,
                    )

            from transformers import pipeline

            device = 0 if self.settings.device.startswith("cuda") else -1
            try:
                self._pipeline = pipeline(
                    "image-text-to-text",
                    model=model_source,
                    device=device,
                )
                self._pipeline_mode = "image-text-to-text"
                LOGGER.info(
                    "Loaded multimodal task-understanding backend model=%s mode=%s",
                    model_source,
                    self._pipeline_mode,
                )
                return
            except Exception as vlm_exc:  # noqa: BLE001
                LOGGER.warning(
                    "Falling back from image-text-to-text to image-to-text for model=%s: %s",
                    model_source,
                    vlm_exc,
                )

            self._pipeline = pipeline(
                "image-to-text",
                model=model_source,
                device=device,
            )
            self._pipeline_mode = "image-to-text"
            LOGGER.info(
                "Loaded caption-based task-understanding backend model=%s mode=%s",
                model_source,
                self._pipeline_mode,
            )
        except Exception as exc:  # noqa: BLE001
            if self.settings.strict:
                raise RuntimeError(
                    f"Failed to load task-understanding backend '{self.settings.model_source}'."
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

        if self._model and self._pipeline_mode == "direct-image-text-to-text":
            raw_text = self._predict_direct_multiframe(sampled_paths, instruction, step_index, total_steps)
            confidence = 0.82 if raw_text else 0.35
        elif self._pipeline and self._pipeline_mode == "image-text-to-text":
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

    def _load_direct_model(self) -> None:
        """Load models that work best through the official AutoProcessor + model path."""
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        model_source = self.settings.model_source
        self._processor = AutoProcessor.from_pretrained(model_source)
        self._model_dtype = _select_model_dtype(self.settings.device)

        model_kwargs: dict[str, Any] = {"dtype": self._model_dtype}
        if self.settings.device.startswith("cuda"):
            # Prefer flash attention when available, but do not require it.
            model_kwargs["_attn_implementation"] = "flash_attention_2"

        try:
            self._model = AutoModelForImageTextToText.from_pretrained(
                model_source,
                **model_kwargs,
            )
        except Exception as exc:  # noqa: BLE001
            if "_attn_implementation" in model_kwargs:
                LOGGER.info(
                    "Retrying semantic VLM load without flash attention for model=%s after: %s",
                    model_source,
                    exc,
                )
                model_kwargs.pop("_attn_implementation", None)
                self._model = AutoModelForImageTextToText.from_pretrained(
                    model_source,
                    **model_kwargs,
                )
            else:
                raise

        self._model_device = "cuda" if self.settings.device.startswith("cuda") else "cpu"
        self._model = self._model.to(self._model_device).eval()
        self._pipeline_mode = "direct-image-text-to-text"
        LOGGER.info(
            "Loaded direct multimodal task-understanding backend model=%s mode=%s dtype=%s",
            model_source,
            self._pipeline_mode,
            self._model_dtype,
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
            "Focus on the hands, the manipulated object, and any clear source or target. "
            "Prefer specific actions such as align connector, insert cable, hold connector, inspect port, remove tray, or tighten fastener when clearly visible. "
            "Do not repeat the full task instruction unless it is visually supported by the frames. "
            "If the action is unclear, answer exactly 'unclear action'. "
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

    def _predict_direct_multiframe(
        self,
        sampled_paths: list[str],
        instruction: str,
        step_index: int,
        total_steps: int,
    ) -> str | None:
        if not self._processor or not self._model:
            return None

        prompt = (
            "These are ordered frames from one short segment of a human hand manipulation task. "
            "Describe only the single main visible action happening across these frames in 2 to 6 words. "
            "Focus on the hands, the manipulated object, and any clear source or target. "
            "Prefer specific actions such as align connector, insert cable, hold connector, inspect port, "
            "remove tray, or tighten fastener when clearly visible. "
            "Do not repeat the full task instruction unless it is visually supported by the frames. "
            "If the action is unclear, answer exactly 'unclear action'. "
            f"Task context: {instruction}. Segment position: {step_index + 1} of {total_steps}."
        )
        messages = [
            {
                "role": "user",
                "content": (
                    [{"type": "image", "path": frame_path} for frame_path in sampled_paths]
                    + [{"type": "text", "text": prompt}]
                ),
            }
        ]

        try:
            inputs = self._processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
            if self._model_dtype is not None:
                inputs = inputs.to(self._model.device, dtype=self._model_dtype)
            else:
                inputs = inputs.to(self._model.device)
            generated_ids = self._model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=48,
            )
            if "input_ids" in inputs:
                prompt_length = inputs["input_ids"].shape[1]
                generated_ids = generated_ids[:, prompt_length:]
            generated_texts = self._processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Direct multiframe VLM inference failed for frames=%s: %s", sampled_paths, exc)
            return None

        if not generated_texts:
            return None
        return _strip_assistant_prefix(generated_texts[0])


def _sample_frame_paths(frame_paths: list[str]) -> list[str]:
    unique_paths = []
    for path in frame_paths:
        if path not in unique_paths:
            unique_paths.append(path)
    if len(unique_paths) <= 6:
        return unique_paths
    max_samples = 6
    indices = {
        round(index * (len(unique_paths) - 1) / (max_samples - 1))
        for index in range(max_samples)
    }
    return [unique_paths[index] for index in sorted(indices)]


def _prefers_direct_model_path(model_source: str, local_model_path: str | None = None) -> bool:
    lowered = model_source.strip().lower()
    return (
        bool(local_model_path)
        or "smolvlm" in lowered
        or "qwen2.5-vl" in lowered
        or "qwen2-vl" in lowered
    )


def _select_model_dtype(device: str):
    try:
        import torch
    except ImportError:  # pragma: no cover
        return None
    return torch.bfloat16 if device.startswith("cuda") else torch.float32


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


def _strip_assistant_prefix(text: str | None) -> str | None:
    if not text:
        return None
    cleaned = text.strip()
    cleaned = re.sub(r"^assistant\s*[:\-]?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^answer\s*[:\-]?\s*", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip() or None


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
