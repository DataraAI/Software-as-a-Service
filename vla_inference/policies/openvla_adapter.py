"""
policies/openvla_adapter.py — OpenVLA wrapper (placeholder).

STATUS: PLACEHOLDER — NOT YET IMPLEMENTED

OpenVLA is a 7B-parameter vision-language-action model from Stanford.
Model: https://huggingface.co/openvla/openvla-7b

To implement:
  1. Install: pip install transformers torch pillow
  2. Load with AutoModelForVision2Seq + AutoProcessor
  3. Tokenize (image + task instruction) and run generate()
  4. Decode output token sequence to action array
  5. Pass raw array to decoder.decode_openvla()

OpenVLA also produces robot-arm action vectors (not desktop UI actions).
Same heuristic mapping applies as SmolVLA.

Example usage once implemented:
  python run_on_media.py --video_path video.mp4 --task "..." --backend openvla \
    --model_id openvla/openvla-7b

Requires significant GPU memory (~14GB for inference).
Use --backend mock for demos.
"""

import logging
from PIL import Image
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "openvla/openvla-7b"


class OpenVLAAdapter:
    """Placeholder — raises NotImplementedError until implemented."""

    def __init__(self, model_id: str = DEFAULT_MODEL_ID):
        self.model_id = model_id
        logger.warning(
            "OpenVLA adapter is a placeholder. "
            "See policies/openvla_adapter.py for implementation notes. "
            "Use --backend mock for a working demo."
        )
        raise NotImplementedError(
            "OpenVLA adapter not yet implemented. "
            "Use --backend mock or --backend smolvla."
        )

    def predict(self, image: Image.Image, task: str, frame_index: int) -> Any:
        raise NotImplementedError
