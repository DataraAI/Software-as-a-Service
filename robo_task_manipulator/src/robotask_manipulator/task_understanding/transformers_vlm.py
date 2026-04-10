"""Transformers-based task-understanding backend."""

from __future__ import annotations

import gc
import logging
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
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
        task_name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> SemanticPrediction:
        self.load()
        sampled_paths = _sample_frame_paths(
            frame_paths,
            max_samples=_max_sampled_frames(self._pipeline_mode),
        )
        context_hints = _collect_context_hints(task_name, metadata)
        motion_score = _estimate_motion(sampled_paths)
        frame_captions: list[str] = []
        raw_text: str | None = None
        confidence = 0.3

        if self._model and self._pipeline_mode == "direct-image-text-to-text":
            raw_text = self._predict_direct_multiframe(
                sampled_paths,
                instruction,
                step_index,
                total_steps,
                task_name=task_name,
                context_hints=context_hints,
            )
            confidence = 0.82 if raw_text else 0.35
        elif self._pipeline and self._pipeline_mode == "image-text-to-text":
            raw_text = self._predict_multiframe(
                sampled_paths,
                instruction,
                step_index,
                total_steps,
                task_name=task_name,
                context_hints=context_hints,
            )
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
                "task_name": task_name,
                "context_hints": context_hints,
                "used_instruction_only_as_context": not bool(
                    context_hints or (task_name and not _is_generic_hint(task_name))
                ),
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
        *,
        task_name: str | None = None,
        context_hints: list[str] | None = None,
    ) -> str | None:
        prompt = _build_semantic_prompt(
            instruction,
            step_index,
            total_steps,
            task_name=task_name,
            context_hints=context_hints,
        )
        for candidate_paths in _candidate_frame_path_sets(sampled_paths):
            images = _load_images(candidate_paths)
            if not images:
                continue
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
                    max_new_tokens=24,
                    return_full_text=False,
                )
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("Multiframe VLM inference failed for frames=%s: %s", candidate_paths, exc)
                if _is_cuda_oom(exc):
                    _release_cuda_memory(self.settings.device, aggressive=True)
                    continue
                return None
            return _coerce_generated_text(outputs)
        return None

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
        *,
        task_name: str | None = None,
        context_hints: list[str] | None = None,
    ) -> str | None:
        if not self._processor or not self._model:
            return None

        prompt = _build_semantic_prompt(
            instruction,
            step_index,
            total_steps,
            task_name=task_name,
            context_hints=context_hints,
        )

        for candidate_paths in _candidate_frame_path_sets(sampled_paths):
            messages = [
                {
                    "role": "user",
                    "content": (
                        [{"type": "image", "path": frame_path} for frame_path in candidate_paths]
                        + [{"type": "text", "text": prompt}]
                    ),
                }
            ]
            inputs = None
            generated_ids = None
            generated_texts = None
            decoded_text = None
            try:
                with torch.inference_mode():
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
                        max_new_tokens=24,
                    )
                if "input_ids" in inputs:
                    prompt_length = inputs["input_ids"].shape[1]
                    generated_ids = generated_ids[:, prompt_length:]
                generated_texts = self._processor.batch_decode(
                    generated_ids,
                    skip_special_tokens=True,
                )
                decoded_text = generated_texts[0] if generated_texts else None
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("Direct multiframe VLM inference failed for frames=%s: %s", candidate_paths, exc)
                if _is_cuda_oom(exc):
                    _release_cuda_memory(self._model_device, aggressive=True)
                    continue
                return None
            finally:
                del inputs
                del generated_ids
                del generated_texts
                _release_cuda_memory(self._model_device)

            if not decoded_text:
                continue
            return _strip_assistant_prefix(decoded_text)
        return None


def _sample_frame_paths(frame_paths: list[str], max_samples: int = 6) -> list[str]:
    unique_paths = []
    for path in frame_paths:
        if path not in unique_paths:
            unique_paths.append(path)
    max_samples = max(1, max_samples)
    if len(unique_paths) <= max_samples:
        return unique_paths
    if max_samples == 1:
        return [unique_paths[len(unique_paths) // 2]]
    indices = {
        round(index * (len(unique_paths) - 1) / (max_samples - 1))
        for index in range(max_samples)
    }
    return [unique_paths[index] for index in sorted(indices)]


def _max_sampled_frames(pipeline_mode: str) -> int:
    if pipeline_mode == "direct-image-text-to-text":
        return 3
    if pipeline_mode == "image-text-to-text":
        return 4
    return 6


def _candidate_frame_path_sets(frame_paths: list[str]) -> list[list[str]]:
    candidates = [
        _sample_frame_paths(frame_paths, max_samples=len(frame_paths)),
        _sample_frame_paths(frame_paths, max_samples=3),
        _sample_frame_paths(frame_paths, max_samples=2),
        _sample_frame_paths(frame_paths, max_samples=1),
    ]
    deduped: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()
    for candidate in candidates:
        key = tuple(candidate)
        if candidate and key not in seen:
            seen.add(key)
            deduped.append(candidate)
    return deduped


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
    text = _strip_assistant_prefix(raw_text) or raw_text.strip()
    text = re.sub(r"^step_description\s*:\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^main visible step\s*:\s*", "", text, flags=re.IGNORECASE)
    text = re.split(r"\n+", text, maxsplit=1)[0].strip()
    text = _select_best_action_clause(text)
    text = re.sub(r"\b(the person|person|human hand|hand is|hand appears to be)\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(a|an|the)\b", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip(" .,:;!-")
    text = _normalize_leading_action(text)
    text = _trim_action_phrase(text)
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
        return "hold object"
    if step_index == 0 and total_steps > 1:
        return "begin hand-object manipulation"
    if step_index == total_steps - 1 and total_steps > 1:
        return "finish hand-object manipulation"
    return "hand manipulates object"


def _build_semantic_prompt(
    instruction: str,
    step_index: int,
    total_steps: int,
    *,
    task_name: str | None = None,
    context_hints: list[str] | None = None,
) -> str:
    hints = list(context_hints or [])
    prompt_parts = [
        "You are labeling ordered frames from a real-world manipulation video.",
        "Return exactly one short lower-case action phrase between 2 and 6 words.",
        "Describe only the single visible hand-object action happening across these frames.",
        "Prefer concrete phrasing in the form verb object [preposition target].",
        (
            "Good examples: hold cable near port, align connector with socket, insert cable into port, "
            "tighten bolt, place container on shelf, wipe surface, open drawer, pick up tool."
        ),
        "If the hand is mainly maintaining contact, prefer a hold phrase over a generic manipulate phrase.",
        "Use the task description or tags only as soft hints and only when they are visually supported.",
        "If the action is unclear, answer exactly 'unclear action'.",
        f"Task description: {instruction.strip()}",
    ]
    if task_name and not _is_generic_hint(task_name):
        prompt_parts.append(f"Optional task label: {task_name.strip()}")
    if hints:
        prompt_parts.append(f"Optional hints: {', '.join(hints)}")
    prompt_parts.append(f"Frame window position: {step_index + 1} of {total_steps}.")
    return " ".join(part.strip() for part in prompt_parts if part and part.strip())


def _collect_context_hints(task_name: str | None, metadata: dict[str, Any] | None) -> list[str]:
    hints: list[str] = []
    if not metadata:
        return hints

    for key in ("tags", "labels", "objects", "object_tags", "activities", "activity_tags", "tools", "locations"):
        value = metadata.get(key)
        if isinstance(value, list):
            hints.extend(_clean_hint_text(item) for item in value if str(item).strip())
        elif isinstance(value, str) and value.strip():
            hints.append(_clean_hint_text(value))

    for key in ("description", "summary", "scene", "environment"):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            hints.append(_clean_hint_text(value))

    deduped: list[str] = []
    seen: set[str] = set()
    for hint in hints:
        lowered = hint.lower()
        if hint and lowered not in seen and not _is_generic_hint(hint):
            seen.add(lowered)
            deduped.append(hint)
    return deduped[:8]


def _clean_hint_text(value: Any) -> str:
    text = str(value).replace("_", " ").strip()
    return re.sub(r"\s+", " ", text)


def _is_generic_hint(value: str) -> bool:
    tokens = [token for token in re.split(r"[^a-z0-9]+", value.lower()) if token]
    if not tokens:
        return True
    generic = {"test", "video", "image", "task", "demo", "sample", "real", "clip"}
    return all(token in generic for token in tokens)


def _select_best_action_clause(text: str) -> str:
    clauses = [
        clause.strip()
        for clause in re.split(r"[;\n]|,\s+|\.\s+|\s+then\s+|\s+and\s+", text)
        if clause.strip()
    ]
    if not clauses:
        return text
    return max(clauses, key=_action_clause_score)


def _action_clause_score(text: str) -> tuple[int, int]:
    lowered = text.lower()
    score = 0
    for keyword, weight in {
        "insert": 8,
        "plug": 8,
        "connect": 7,
        "align": 6,
        "position": 5,
        "tighten": 6,
        "fasten": 6,
        "pick": 5,
        "place": 5,
        "hold": 4,
        "inspect": 4,
        "check": 4,
        "open": 4,
        "close": 4,
        "wipe": 4,
        "move": 2,
        "manipulate": 1,
    }.items():
        if keyword in lowered:
            score += weight
    if any(token in lowered for token in {"into", "onto", "with", "near", "from"}):
        score += 2
    if any(token in lowered for token in {"prepare", "about to", "appears to", "trying to"}):
        score -= 3
    return score, len(lowered)


def _normalize_leading_action(text: str) -> str:
    lowered = text.lower()
    substitutions = {
        r"^holding\b": "hold",
        r"^plugging in\b": "plug",
        r"^plugging\b": "plug",
        r"^inserting\b": "insert",
        r"^aligning\b": "align",
        r"^positioning\b": "position",
        r"^placing\b": "place",
        r"^picking up\b": "pick up",
        r"^picking\b": "pick",
        r"^moving\b": "move",
        r"^inspecting\b": "inspect",
        r"^checking\b": "check",
        r"^tightening\b": "tighten",
        r"^fastening\b": "fasten",
        r"^connecting\b": "connect",
    }
    for pattern, replacement in substitutions.items():
        if re.search(pattern, lowered):
            return re.sub(pattern, replacement, lowered, count=1)
    return lowered


def _trim_action_phrase(text: str, max_words: int = 8) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words])


def _is_cuda_oom(exc: Exception) -> bool:
    message = str(exc).lower()
    return "out of memory" in message or "cuda oom" in message


def _release_cuda_memory(device: str | None, aggressive: bool = False) -> None:
    if not device or not str(device).startswith("cuda"):
        return
    if aggressive:
        gc.collect()
    try:
        import torch
    except ImportError:  # pragma: no cover
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


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
