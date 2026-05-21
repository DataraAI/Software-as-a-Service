"""
policies/pi0_adapter.py — PI0 (pi-zero) wrapper for lerobot 0.5.1

STATUS: EXPERIMENTAL — real model, requires model download (~1-2GB)

PI0 is a robot arm policy from Physical Intelligence (pi.ai), available
in lerobot 0.5.1. It takes:
  - One or more camera images (we pass one: the current frame)
  - A language instruction (tokenized via PaliGemma tokenizer)
  - A robot state vector (joint positions etc.)

It outputs a chunk of actions — each action is a vector of joint deltas
(dimension depends on the checkpoint, typically 7-DoF for robot arms).

Model: lerobot/pi0  (on HuggingFace)
Requires: lerobot>=0.5.1, torch, transformers

NOTE: On CPU this will be very slow (minutes per frame).
"""

import logging
import torch
from PIL import Image
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "lerobot/pi0"

# These are the lerobot 0.5.1 observation key constants
OBS_IMAGE_KEY = "observation.images.top"  # primary camera key
OBS_LANGUAGE_TOKENS = "observation.language_tokens"
OBS_LANGUAGE_ATTENTION_MASK = "observation.language_attention_mask"
OBS_STATE = "observation.state"

# Dummy robot state dimension — adjust to match your robot's DoF
# pi0 will pad this up to max_state_dim automatically
ROBOT_STATE_DIM = 7


class PI0Adapter:
    """
    Wraps PI0Policy for batch frame inference.

    Each call to predict() runs one forward pass on one frame.
    Returns a raw numpy action vector (joint deltas).
    """

    def __init__(self, model_id: str = DEFAULT_MODEL_ID):
        self.model_id = model_id
        self._load_model()

    def _load_model(self):
        try:
            from lerobot.policies.pi0 import PI0Policy
            from transformers import AutoTokenizer
            import traceback

            logger.info(f"Loading PI0 policy from: {self.model_id}")
            logger.info("This will download ~1-2GB on first run...")

            self.policy = PI0Policy.from_pretrained(self.model_id)
            self.policy.eval()
            logger.info("PI0 weights loaded successfully")

            tokenizer_id = getattr(
                self.policy.config, "tokenizer_name",
                "google/paligemma-3b-pt-224"
            )
            logger.info(f"Loading tokenizer: {tokenizer_id}")
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
            logger.info("PI0 loaded successfully")

        except Exception as e:
            import traceback
            logger.error(f"PI0 load failed: {e}")
            logger.error(traceback.format_exc())
            raise RuntimeError(f"Failed to load PI0 model '{self.model_id}': {e}")

        except Exception as e:
            import traceback
            logger.error(f"PI0 load failed: {e}")
            logger.error(traceback.format_exc())
            raise RuntimeError(f"Failed to load PI0 model '{self.model_id}': {e}")
        def _tokenize_task(self, task: str, device: torch.device):
            """Tokenize the language instruction for PI0."""
            encoded = self.tokenizer(
                task,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=64,
            )
            return (
                encoded["input_ids"].to(device),
                encoded["attention_mask"].to(device),
            )

    def _image_to_tensor(self, image: Image.Image, device: torch.device) -> torch.Tensor:
        """
        Convert PIL image to tensor in format PI0 expects:
        [B, C, H, W], float32, normalized to [0, 1].
        PI0's _preprocess_images handles the rest (resize, pad, renormalize to [-1,1]).
        """
        import torchvision.transforms.functional as TF

        img = image.convert("RGB")
        tensor = TF.to_tensor(img)  # [C, H, W], float32, [0,1]
        return tensor.unsqueeze(0).to(device)  # [1, C, H, W]

    def predict(self, image: Image.Image, task: str, frame_index: int) -> Any:
        """
        Run PI0 inference on one frame.

        Args:
            image: PIL image (RGB) — the current camera frame
            task: natural language instruction string
            frame_index: position in the sequence (unused by model, for logging)

        Returns:
            numpy array of shape (action_dim,) — raw joint delta action vector
        """
        device = next(self.policy.parameters()).device

        img_tensor = self._image_to_tensor(image, device)
        lang_tokens, lang_mask = self._tokenize_task(task, device)

        # Dummy robot state — zeros means "unknown/home position"
        # In a real pipeline this would come from Isaac Sim joint feedback
        state = torch.zeros(1, ROBOT_STATE_DIM, dtype=torch.float32, device=device)

        batch = {
            OBS_IMAGE_KEY: img_tensor,
            OBS_LANGUAGE_TOKENS: lang_tokens,
            OBS_LANGUAGE_ATTENTION_MASK: lang_mask,
            OBS_STATE: state,
        }

        logger.debug(f"Frame {frame_index}: running PI0 forward pass...")

        with torch.no_grad():
            action = self.policy.select_action(batch)

        # action shape: (batch_size, action_dim) — take first (only) batch item
        result = action[0].cpu().numpy()
        logger.debug(f"Frame {frame_index}: action vector shape={result.shape}, values={result}")
        return result
