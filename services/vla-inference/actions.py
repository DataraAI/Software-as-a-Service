"""
actions.py — Structured action schema for desktop UI actions.

These are the canonical action types that DaaS will consume.
All backends must produce output that maps to one of these types.
"""

from dataclasses import dataclass
from typing import Union


# --- Action type constants ---

CLICK = "CLICK"
DOUBLE_CLICK = "DOUBLE_CLICK"
PRESS = "PRESS"
HOTKEY = "HOTKEY"
SCROLL = "SCROLL"
WAIT = "WAIT"
TYPE = "TYPE"

VALID_ACTION_TYPES = {CLICK, DOUBLE_CLICK, PRESS, HOTKEY, SCROLL, WAIT, TYPE}


# --- Per-action dataclasses ---

@dataclass
class ClickAction:
    x: int
    y: int
    action_type: str = CLICK

    def to_dict(self):
        return {"action_type": self.action_type, "x": self.x, "y": self.y}


@dataclass
class DoubleClickAction:
    x: int
    y: int
    action_type: str = DOUBLE_CLICK

    def to_dict(self):
        return {"action_type": self.action_type, "x": self.x, "y": self.y}


@dataclass
class PressAction:
    key: str
    action_type: str = PRESS

    def to_dict(self):
        return {"action_type": self.action_type, "key": self.key}


@dataclass
class HotkeyAction:
    modifier: str
    key: str
    action_type: str = HOTKEY

    def to_dict(self):
        return {"action_type": self.action_type, "modifier": self.modifier, "key": self.key}


@dataclass
class ScrollAction:
    amount: float
    action_type: str = SCROLL

    def to_dict(self):
        return {"action_type": self.action_type, "amount": self.amount}


@dataclass
class WaitAction:
    seconds: float
    action_type: str = WAIT

    def to_dict(self):
        return {"action_type": self.action_type, "seconds": self.seconds}


@dataclass
class TypeAction:
    text: str
    action_type: str = TYPE

    def to_dict(self):
        return {"action_type": self.action_type, "text": self.text}


# Union type for type hints
Action = Union[
    ClickAction, DoubleClickAction, PressAction,
    HotkeyAction, ScrollAction, WaitAction, TypeAction
]


def action_to_dict(action: Action) -> dict:
    """Convert any action object to a plain dict."""
    return action.to_dict()
