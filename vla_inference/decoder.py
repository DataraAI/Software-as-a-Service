"""
decoder.py — Converts raw backend output into structured action schema.

Mock backend produces desktop UI actions (CLICK, TYPE, etc.).
Robot arm backends (pi0, smolvla, openvla, gr00t) produce joint delta
vectors which are passed through as ROBOT_JOINT_DELTA actions for
consumption by Isaac Sim / DaaS.
"""

import logging
from typing import Any, Dict
from actions import (
    ClickAction, WaitAction, PressAction, TypeAction,
    ScrollAction, HotkeyAction, DoubleClickAction,
    CLICK, DOUBLE_CLICK, PRESS, HOTKEY, SCROLL, WAIT, TYPE,
)

logger = logging.getLogger(__name__)


# --- Robot arm action type ---

class RobotAction:
    """
    Structured robot arm action — joint delta vector.
    Used by pi0, smolvla, openvla, gr00t backends.
    DaaS / Isaac Sim consumes joint_deltas directly.
    """
    def __init__(self, joint_deltas: list):
        self.joint_deltas = joint_deltas

    def to_dict(self):
        return {
            "action_type": "ROBOT_JOINT_DELTA",
            "joint_deltas": self.joint_deltas,
        }


# --- Decoders ---

def decode_mock(raw: Dict[str, Any]):
    """Mock backend already returns structured action dicts."""
    action_type = raw.get("action_type", WAIT)

    if action_type == CLICK:
        return ClickAction(x=raw["x"], y=raw["y"])
    elif action_type == DOUBLE_CLICK:
        return DoubleClickAction(x=raw["x"], y=raw["y"])
    elif action_type == PRESS:
        return PressAction(key=raw["key"])
    elif action_type == HOTKEY:
        return HotkeyAction(modifier=raw["modifier"], key=raw["key"])
    elif action_type == SCROLL:
        return ScrollAction(amount=raw["amount"])
    elif action_type == TYPE:
        return TypeAction(text=raw["text"])
    else:
        return WaitAction(seconds=raw.get("seconds", 1.0))


def decode_robot_vector(raw: Any):
    """
    Pass through raw robot action vectors from pi0, SmolVLA, OpenVLA, GR00T.

    These are joint delta vectors (e.g. 7-DoF) for robot arm control.
    Wrapped in a RobotAction dict so Isaac Sim / DaaS can consume them directly.
    """
    import numpy as np
    try:
        arr = np.array(raw, dtype=float).flatten()
        return RobotAction(joint_deltas=arr.tolist())
    except Exception as e:
        logger.warning(f"Could not parse robot action vector: {e}")
        return RobotAction(joint_deltas=[])


def decode(backend: str, raw: Any):
    """Route raw backend output to the correct decoder."""
    if backend == "mock":
        return decode_mock(raw)
    elif backend in ("pi0", "smolvla", "openvla", "gr00t"):
        return decode_robot_vector(raw)
    else:
        logger.warning(f"Unknown backend '{backend}', defaulting to WAIT")
        return WaitAction(seconds=1.0)
