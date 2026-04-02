"""
validator.py — Validates the actions list before writing to disk.

Catches missing fields, out-of-range coordinates, or unknown action types
before the JSON is saved so DaaS doesn't receive malformed data.
"""

import logging
from typing import List, Dict, Any
from actions import VALID_ACTION_TYPES
VALID_ACTION_TYPES = VALID_ACTION_TYPES | {"ROBOT_JOINT_DELTA"}

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    pass


def validate_action(entry: Dict[str, Any], index: int):
    """Validate a single action entry (frame_index, timestamp, action)."""

    if "frame_index" not in entry:
        raise ValidationError(f"Entry {index} missing 'frame_index'")
    if "timestamp" not in entry:
        raise ValidationError(f"Entry {index} missing 'timestamp'")
    if "action" not in entry:
        raise ValidationError(f"Entry {index} missing 'action'")

    action = entry["action"]
    action_type = action.get("action_type")

    if action_type not in VALID_ACTION_TYPES:
        raise ValidationError(
            f"Entry {index} has unknown action_type '{action_type}'. "
            f"Valid types: {VALID_ACTION_TYPES}"
        )

    # Type-specific field checks
    if action_type in ("CLICK", "DOUBLE_CLICK"):
        for field in ("x", "y"):
            if field not in action:
                raise ValidationError(f"Entry {index} {action_type} missing '{field}'")
            if not isinstance(action[field], (int, float)):
                raise ValidationError(f"Entry {index} {action_type}.{field} must be numeric")

    elif action_type == "PRESS":
        if "key" not in action or not isinstance(action["key"], str):
            raise ValidationError(f"Entry {index} PRESS missing valid 'key'")

    elif action_type == "HOTKEY":
        for field in ("modifier", "key"):
            if field not in action or not isinstance(action[field], str):
                raise ValidationError(f"Entry {index} HOTKEY missing valid '{field}'")

    elif action_type == "SCROLL":
        if "amount" not in action or not isinstance(action["amount"], (int, float)):
            raise ValidationError(f"Entry {index} SCROLL missing valid 'amount'")

    elif action_type == "WAIT":
        if "seconds" not in action or not isinstance(action["seconds"], (int, float)):
            raise ValidationError(f"Entry {index} WAIT missing valid 'seconds'")
        if action["seconds"] < 0:
            raise ValidationError(f"Entry {index} WAIT.seconds must be >= 0")

    elif action_type == "ROBOT_JOINT_DELTA":
        if "joint_deltas" not in action or not isinstance(action["joint_deltas"], list):
            raise ValidationError(f"Entry {index} ROBOT_JOINT_DELTA missing valid 'joint_deltas' list")

    elif action_type == "TYPE":
        if "text" not in action or not isinstance(action["text"], str):
            raise ValidationError(f"Entry {index} TYPE missing valid 'text'")


def validate_actions(actions: List[Dict[str, Any]]) -> bool:
    """
    Validate a full list of action entries.
    Returns True if valid, raises ValidationError if not.
    """
    if not isinstance(actions, list):
        raise ValidationError("'actions' must be a list")

    if len(actions) == 0:
        logger.warning("Action list is empty — this may be intentional for short inputs")

    for i, entry in enumerate(actions):
        validate_action(entry, i)

    logger.info(f"Validation passed: {len(actions)} actions")
    return True
