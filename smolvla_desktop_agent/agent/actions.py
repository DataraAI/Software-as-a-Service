from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional


ActionType = Literal[
    "CLICK",
    "DOUBLE_CLICK",
    "PRESS",
    "HOTKEY",
    "SCROLL",
    "WAIT",
    "TYPE",
]


@dataclass
class StructuredAction:
    action_type: ActionType
    x: Optional[int] = None
    y: Optional[int] = None
    key: Optional[str] = None
    modifier: Optional[str] = None
    text: Optional[str] = None
    amount: Optional[int] = None
    seconds: Optional[float] = None

    def to_executor_string(self) -> str:
        if self.action_type == "CLICK":
            return f"CLICK({self.x},{self.y})"
        if self.action_type == "DOUBLE_CLICK":
            return f"DOUBLE_CLICK({self.x},{self.y})"
        if self.action_type == "PRESS":
            return f"PRESS({self.key})"
        if self.action_type == "HOTKEY":
            return f"HOTKEY({self.modifier},{self.key})"
        if self.action_type == "SCROLL":
            return f"SCROLL({self.amount})"
        if self.action_type == "WAIT":
            return f"WAIT({self.seconds})"
        if self.action_type == "TYPE":
            return f"TYPE({self.text})"
        raise ValueError(f"Unsupported action_type: {self.action_type}")