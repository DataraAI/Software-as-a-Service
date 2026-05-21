"""
policies/mock_policy.py — Mock VLA backend.

Always works. No model loading. Produces deterministic, structured actions
that exercise every action type in the schema.

Use this to:
  - verify the full pipeline end-to-end without a GPU
  - demo the system to stakeholders
  - run CI checks
  - develop DaaS consumption logic before a real VLA is integrated

The mock cycles through action templates so the output JSON contains
a representative variety of action types.
"""

import logging
from PIL import Image
from typing import Any, Dict

logger = logging.getLogger(__name__)

# Action templates cycled through per frame
_ACTION_CYCLE = [
    {"action_type": "WAIT", "seconds": 1.0},
    {"action_type": "CLICK", "x": 412, "y": 88},
    {"action_type": "TYPE", "text": "hello from mock VLA"},
    {"action_type": "PRESS", "key": "enter"},
    {"action_type": "SCROLL", "amount": -3.0},
    {"action_type": "HOTKEY", "modifier": "ctrl", "key": "s"},
    {"action_type": "DOUBLE_CLICK", "x": 960, "y": 540},
]


class MockPolicy:
    """
    Mock policy that returns structured action dicts without any model.
    """

    def __init__(self, model_id: str = "mock"):
        self.model_id = model_id
        logger.info("MockPolicy initialized (no model loaded)")

    def predict(self, image: Image.Image, task: str, frame_index: int) -> Dict[str, Any]:
        """
        Return the next action in the cycle.

        Args:
            image: PIL image (ignored by mock)
            task: natural language instruction (ignored by mock)
            frame_index: used to cycle through action templates

        Returns:
            Raw action dict — will be decoded by decoder.decode_mock()
        """
        action = _ACTION_CYCLE[frame_index % len(_ACTION_CYCLE)]
        logger.debug(f"Frame {frame_index}: mock returning {action['action_type']}")
        return action
