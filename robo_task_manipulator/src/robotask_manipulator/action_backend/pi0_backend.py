"""Real LeRobot pi0 action backend."""

from __future__ import annotations

import logging
import sys
import time
import types
from importlib.machinery import ModuleSpec
from pathlib import Path
from typing import Any

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from PIL import Image

import lerobot

from robotask_manipulator.action_backend.base import BaseActionBackend
from robotask_manipulator.config import ActionBackendSettings
from robotask_manipulator.schemas import ActionProposal, EpisodeInput, SegmentAnnotation
from robotask_manipulator.utils.validation import InvalidInputError, ModelLoadError, ensure_asset_exists

LOGGER = logging.getLogger(__name__)

_LEROBOT_POLICIES_DIR = Path(lerobot.__file__).resolve().parent / "policies"
if "lerobot.policies" not in sys.modules:
    policies_package = types.ModuleType("lerobot.policies")
    policies_package.__path__ = [str(_LEROBOT_POLICIES_DIR)]
    policies_package.__package__ = "lerobot"
    policies_package.__spec__ = ModuleSpec("lerobot.policies", loader=None, is_package=True)
    policies_package.__spec__.submodule_search_locations = [str(_LEROBOT_POLICIES_DIR)]
    sys.modules["lerobot.policies"] = policies_package

from lerobot.policies.pi0.configuration_pi0 import PI0Config
from lerobot.policies.pi0.modeling_pi0 import PI0Policy
from lerobot.policies.pi0.processor_pi0 import make_pi0_pre_post_processors
from lerobot.utils.constants import ACTION, OBS_STATE


class Pi0ActionBackend(BaseActionBackend):
    """Optional robot-oriented backend using the official LeRobot `PI0Policy`."""

    backend_name = "pi0"

    def __init__(self, settings: ActionBackendSettings) -> None:
        self.settings = settings
        self._loaded = False
        self._config: PI0Config | None = None
        self._policy: PI0Policy | None = None
        self._preprocessor: Any | None = None
        self._image_feature_keys: list[str] = []
        self._action_dim = 0
        self._state_dim = 32

    def load(self) -> None:
        if self._loaded:
            return

        source = self.settings.model_source
        LOGGER.info("Loading LeRobot pi0 backend from %s", source)
        self._ensure_model_artifacts_exist(source)

        try:
            config = PI0Config.from_pretrained(
                source,
                local_files_only=self.settings.offline,
                revision=self.settings.revision,
                cache_dir=self.settings.cache_dir,
            )
            config.device = self.settings.device
            config.dtype = self.settings.dtype
            if hasattr(config, "validate_features"):
                config.validate_features()

            policy = PI0Policy.from_pretrained(
                source,
                config=config,
                local_files_only=self.settings.offline,
                revision=self.settings.revision,
                cache_dir=self.settings.cache_dir,
                strict=self.settings.strict,
            )
            policy.to(self.settings.device)
            requested_dtype = getattr(torch, self.settings.dtype, None)
            if requested_dtype is None:
                raise ModelLoadError(
                    f"Unsupported PI0_DTYPE '{self.settings.dtype}'. Expected a torch dtype like float32 or bfloat16."
                )
            policy.to(dtype=requested_dtype)
            policy.eval()

            config = policy.config
            preprocessor, _postprocessor = make_pi0_pre_post_processors(config, dataset_stats=None)
        except Exception as exc:  # noqa: BLE001
            raise ModelLoadError(
                "Failed to load LeRobot pi0 backend. Check PI0_MODEL_ID / PI0_CHECKPOINT_PATH and checkpoint compatibility."
            ) from exc

        self._config = config
        self._policy = policy
        self._preprocessor = preprocessor
        self._image_feature_keys = list(config.image_features)
        self._action_dim = int(config.output_features[ACTION].shape[0])
        self._state_dim = int(config.max_state_dim)
        self._loaded = True
        LOGGER.info(
            "Loaded pi0 backend. device=%s dtype=%s images=%s action_dim=%s chunk_size=%s",
            config.device,
            config.dtype,
            self._image_feature_keys,
            self._action_dim,
            config.chunk_size,
        )

    def propose(self, episode: EpisodeInput, segment: SegmentAnnotation) -> ActionProposal | None:
        self.load()
        assert self._policy is not None
        assert self._preprocessor is not None
        assert self._config is not None

        image_paths = [ensure_asset_exists(ref) for ref in segment.observation_refs[: len(self._image_feature_keys)]]
        if not image_paths:
            raise InvalidInputError(
                f"Segment {segment.segment_id} does not contain image observations required for pi0 inference."
            )

        batch = self._build_batch(episode, segment, image_paths)
        processed_batch = self._preprocessor(batch)

        started = time.perf_counter()
        try:
            self._policy.reset()
            action_chunk = self._policy.predict_action_chunk(processed_batch)
        except Exception as exc:  # noqa: BLE001
            raise ModelLoadError(
                "LeRobot pi0 inference failed for this segment. Check image/state mapping for the selected embodiment."
            ) from exc
        latency_ms = round((time.perf_counter() - started) * 1000.0, 3)

        chunk_tensor = action_chunk.detach().cpu()
        if chunk_tensor.ndim != 3 or chunk_tensor.shape[0] != 1:
            raise ModelLoadError(
                f"Unexpected pi0 action chunk shape {tuple(chunk_tensor.shape)}. Expected [1, chunk, action_dim]."
            )

        chunk_list = chunk_tensor[0].tolist()
        selected_action = chunk_list[0] if chunk_list else None
        stats = self._summarize_action_chunk(chunk_list)

        return ActionProposal(
            backend=self.backend_name,
            selected_action=selected_action,
            action_chunk=chunk_list,
            confidence=min(0.95, max(0.35, 0.55 + 0.1 * len(chunk_list))),
            metadata={
                "model_source": self.settings.model_source,
                "model_id": self.settings.model_id,
                "checkpoint_path": self.settings.checkpoint_path,
                "device": self.settings.device,
                "dtype": self.settings.dtype,
                "offline": self.settings.offline,
                "chunk_size": int(self._config.chunk_size),
                "n_action_steps": int(self._config.n_action_steps),
                "state_dim": self._state_dim,
                "action_dim": self._action_dim,
                "image_feature_keys": self._image_feature_keys,
                "latency_ms": latency_ms,
                "selected_action_strategy": "first_action",
                "chunk_stats": stats,
                "segment_id": segment.segment_id,
            },
            raw_model_output={
                "asset_refs": segment.observation_refs,
                "selected_action": selected_action,
                "action_chunk": chunk_list,
                "batch_keys": sorted(processed_batch.keys()),
                "segment_id": segment.segment_id,
            },
        )

    def _ensure_model_artifacts_exist(self, source: str) -> None:
        model_path = Path(source)
        if model_path.exists():
            if not model_path.is_dir():
                raise ModelLoadError(
                    f"PI0_CHECKPOINT_PATH must point to a directory containing a LeRobot export: {model_path}"
                )
            model_file = model_path / "model.safetensors"
            if not model_file.exists():
                raise ModelLoadError(f"Local pi0 checkpoint is missing model.safetensors: {model_file}")
            return

        try:
            hf_hub_download(
                repo_id=source,
                filename="model.safetensors",
                revision=self.settings.revision,
                cache_dir=self.settings.cache_dir,
                local_files_only=self.settings.offline,
            )
        except Exception as exc:  # noqa: BLE001
            raise ModelLoadError(
                f"Unable to access LeRobot pi0 weights for '{source}'. Set PI0_MODEL_ID or PI0_CHECKPOINT_PATH correctly."
            ) from exc

    def _build_batch(
        self,
        episode: EpisodeInput,
        segment: SegmentAnnotation,
        image_paths: list[Path],
    ) -> dict[str, Any]:
        image_tensors = [self._load_image_tensor(path) for path in image_paths]
        if len(image_tensors) > len(self._image_feature_keys):
            raise InvalidInputError(
                f"Received {len(image_tensors)} images but the pi0 checkpoint exposes only "
                f"{len(self._image_feature_keys)} image slots."
            )

        batch: dict[str, Any] = {"task": episode.instruction}
        for key, image_tensor in zip(self._image_feature_keys, image_tensors, strict=False):
            batch[key] = image_tensor

        state = self._normalize_state(segment)
        batch[OBS_STATE] = state
        batch["metadata"] = {
            "episode_id": episode.episode_id,
            "segment_id": segment.segment_id,
            "frame_start_index": segment.frame_start_index,
            "frame_end_index": segment.frame_end_index,
        }
        return batch

    def _load_image_tensor(self, path: Path) -> torch.Tensor:
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            array = np.asarray(rgb, dtype=np.float32) / 255.0
        return torch.from_numpy(array).permute(2, 0, 1)

    def _normalize_state(self, segment: SegmentAnnotation) -> list[float]:
        state = segment.raw_outputs.get("state")
        if not state:
            return [0.0] * self._state_dim
        values = [float(item) for item in state]
        trimmed = values[: self._state_dim]
        if len(trimmed) < self._state_dim:
            trimmed.extend([0.0] * (self._state_dim - len(trimmed)))
        return trimmed

    def _summarize_action_chunk(self, action_chunk: list[list[float]]) -> dict[str, float]:
        flat = [abs(value) for action in action_chunk for value in action]
        if not flat:
            return {"mean_abs": 0.0, "max_abs": 0.0, "variance": 0.0}
        mean_abs = sum(flat) / len(flat)
        variance = sum((value - mean_abs) ** 2 for value in flat) / len(flat)
        return {
            "mean_abs": round(mean_abs, 4),
            "max_abs": round(max(flat), 4),
            "variance": round(variance, 4),
        }
