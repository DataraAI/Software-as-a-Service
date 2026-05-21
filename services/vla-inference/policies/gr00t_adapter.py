"""
policies/gr00t_adapter.py — NVIDIA GR00T N1 wrapper (placeholder).

STATUS: PLACEHOLDER — NOT YET IMPLEMENTED

GR00T N1 is NVIDIA's open foundation model for humanoid robots.
GitHub: https://github.com/NVIDIA/Isaac-GR00T

GR00T uses a hierarchical architecture:
  - High-level: vision-language goal specification
  - Low-level: joint trajectory generation

To implement:
  1. Follow NVIDIA Isaac GR00T setup (requires CUDA, Isaac Sim optional)
  2. Load the GR00T policy via the provided inference scripts
  3. Wrap the predict() call here
  4. Pass raw action vectors to decoder.decode_gr00t()

This is the most complex backend to integrate — requires NVIDIA GPU,
specific CUDA versions, and potentially Isaac Sim for full pipeline.

Use --backend mock for demos.
"""

import logging
from PIL import Image
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "nvidia/GR00T-N1-2B"


class GR00TAdapter:
    """Placeholder — raises NotImplementedError until implemented."""

    def __init__(self, model_id: str = DEFAULT_MODEL_ID):
        self.model_id = model_id
        logger.warning(
            "GR00T adapter is a placeholder. "
            "See policies/gr00t_adapter.py for implementation notes. "
            "Use --backend mock for a working demo."
        )
        raise NotImplementedError(
            "GR00T adapter not yet implemented. "
            "Use --backend mock or --backend smolvla."
        )

    def predict(self, image: Image.Image, task: str, frame_index: int) -> Any:
        raise NotImplementedError
