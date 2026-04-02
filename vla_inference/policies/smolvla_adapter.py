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

import logging
from PIL import Image
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "lerobot/smolvla_base"


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
            # lerobot provides SmolVLA under lerobot.common.policies.smolvla
            from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy
            from lerobot.common.policies.smolvla.configuration_smolvla import SmolVLAConfig

            logger.info(f"Loading SmolVLA from: {self.model_id}")
            self.policy = SmolVLAPolicy.from_pretrained(self.model_id)
            self.policy.eval()
            logger.info("SmolVLA loaded successfully")

        except ImportError:
            raise ImportError(
                "lerobot is required for SmolVLA. Install with:\n"
                "  pip install lerobot\n"
                "Or see: https://github.com/huggingface/lerobot\n"
                "Use --backend mock if you don't need a real VLA."
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load SmolVLA model '{self.model_id}': {e}")

    def predict(self, image: Image.Image, task: str, frame_index: int) -> Any:
        """
        Run SmolVLA inference on one frame.

        Args:
            image: PIL image (RGB)
            task: natural language instruction string
            frame_index: position in the sequence

        Returns:
            Raw action array (numpy or list) — decoded by decoder.decode_smolvla()

        Note: SmolVLA expects observations in a specific format (batch dict).
        This is a simplified single-frame wrapper. For proper temporal context,
        you would maintain a rolling observation buffer across frames.
        """
        import torch
        import numpy as np
        from torchvision import transforms

        # Build a minimal observation dict matching SmolVLA's expected format
        # SmolVLA typically expects: image tensor, language instruction, robot state
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        img_tensor = transform(image).unsqueeze(0)  # (1, 3, 224, 224)

        # Minimal observation dict — SmolVLA may require additional keys
        # depending on which checkpoint is loaded. Adjust as needed.
        observation = {
            "observation.images.top": img_tensor,
            "observation.state": torch.zeros(1, 7),  # dummy robot state
        }

        # Language instruction is typically passed via the policy config
        # or as a separate input depending on the SmolVLA variant
        with torch.no_grad():
            action = self.policy.select_action(observation)

        # Return as numpy array for decoder
        if hasattr(action, "cpu"):
            return action.cpu().numpy().flatten()
        return np.array(action).flatten()
