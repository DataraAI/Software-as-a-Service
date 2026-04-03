"""
policies/smolvla_adapter.py — Experimental SmolVLA wrapper.

STATUS: EXPERIMENTAL
  SmolVLA is a vision-language-action model from HuggingFace/LeRobot.
  It is designed for robot arm manipulation tasks, NOT desktop UI control.
  The action vectors it produces (7-DoF joint deltas) are mapped heuristically
  to desktop actions in decoder.py — this will require a fine-tuned UI model
  for production use.

Model source: https://huggingface.co/lerobot/smolvla_base
Requires: lerobot library (install separately — see README)

KNOWN ISSUES:
  - lerobot may not be pip-installable on all platforms without extra setup
  - Model checkpoint is large (~1.5GB)
  - GPU strongly recommended for reasonable inference speed
  - Action vector → desktop action mapping is heuristic and approximate

If SmolVLA doesn't load, use --backend mock for demos.
"""

import importlib
import logging
import os
import sys
import types
from PIL import Image
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "lerobot/smolvla_base"

# LeRobot batch keys (same as lerobot.utils.constants)
_OBS_STATE = "observation.state"
_OBS_LANGUAGE_TOKENS = "observation.language.tokens"
_OBS_LANGUAGE_ATTENTION_MASK = "observation.language.attention_mask"


def _import_smolvla_policy():
    """
    Load SmolVLAPolicy without importing lerobot.policies.__init__.

    That __init__ eagerly imports Groot and transitively diffusers/xformers, which can
    fail (e.g. xformers vs triton) even when only SmolVLA is needed.
    """
    import lerobot

    pkg = "lerobot.policies"
    if pkg not in sys.modules:
        stub = types.ModuleType(pkg)
        stub.__path__ = [os.path.join(os.path.dirname(lerobot.__file__), "policies")]
        sys.modules[pkg] = stub
    mod = importlib.import_module("lerobot.policies.smolvla.modeling_smolvla")
    return mod.SmolVLAPolicy


class SmolVLAAdapter:
    """
    Wraps SmolVLA inference for batch frame processing.

    Each call to predict() runs forward pass on one frame.
    The returned raw action array is decoded by decoder.decode_smolvla().
    """

    def __init__(self, model_id: str = DEFAULT_MODEL_ID):
        self.model_id = model_id
        self._load_model()

    def _load_model(self):
        """Load SmolVLA model and processor from HuggingFace or local path."""
        try:
            SmolVLAPolicy = _import_smolvla_policy()
        except ImportError:
            raise ImportError(
                "lerobot is required for SmolVLA. Install with:\n"
                "  pip install lerobot\n"
                "Or see: https://github.com/huggingface/lerobot\n"
                "Use --backend mock if you don't need a real VLA."
            ) from None

        try:
            logger.info(f"Loading SmolVLA from: {self.model_id}")
            self.policy = SmolVLAPolicy.from_pretrained(self.model_id)
            self.policy.eval()
            self._vl_tokenizer = self.policy.model.vlm_with_expert.processor.tokenizer
            logger.info("SmolVLA loaded successfully")
        except ImportError:
            raise
        except Exception as e:
            raise RuntimeError(f"Failed to load SmolVLA model '{self.model_id}': {e}") from e

    def predict(self, image: Image.Image, task: str, frame_index: int) -> Any:
        """
        Run SmolVLA inference on one frame.

        Args:
            image: PIL image (RGB)
            task: natural language instruction string
            frame_index: position in the sequence

        Returns:
            Raw action array (numpy or list) — decoded by decoder.decode_smolvla()

        Note: Observation keys and shapes come from the checkpoint's config (e.g. smolvla_base
        uses observation.images.camera1/2/3). A single input frame is broadcast to every camera
        key so video frames still drive inference. Language is tokenized; task length is capped
        by the checkpoint's tokenizer_max_length (48 for smolvla_base).
        """
        import torch
        import numpy as np
        from torchvision import transforms

        cfg = self.policy.config
        device = next(self.policy.parameters()).device

        # Float in [0, 1] — policy prepare_images maps to [-1, 1] for SigLIP (no ImageNet norm).
        img = transforms.ToTensor()(image.convert("RGB"))
        if img.shape[0] > 3:
            img = img[:3]
        img = img.unsqueeze(0).to(device=device, dtype=torch.float32)

        observation = {}
        for cam_key in cfg.image_features:
            observation[cam_key] = img

        rs = cfg.robot_state_feature
        state_dim = int(rs.shape[0]) if rs is not None else 6
        observation[_OBS_STATE] = torch.zeros(1, state_dim, device=device, dtype=torch.float32)

        enc = self._vl_tokenizer(
            task,
            max_length=cfg.tokenizer_max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        observation[_OBS_LANGUAGE_TOKENS] = enc["input_ids"].to(device)
        observation[_OBS_LANGUAGE_ATTENTION_MASK] = enc["attention_mask"].to(
            device=device, dtype=torch.bool
        )

        with torch.no_grad():
            action = self.policy.select_action(observation)

        # Return as numpy array for decoder
        if hasattr(action, "cpu"):
            return action.cpu().numpy().flatten()
        return np.array(action).flatten()
