from __future__ import annotations

from agent.actions import StructuredAction


def validate_action(
    action: StructuredAction,
    screen_width: int,
    screen_height: int,
) -> bool:
    if action.action_type in {"CLICK", "DOUBLE_CLICK"}:
        if action.x is None or action.y is None:
            return False
        if not (0 <= action.x < screen_width and 0 <= action.y < screen_height):
            return False

    if action.action_type == "PRESS":
        if not action.key:
            return False

    if action.action_type == "HOTKEY":
        if not action.modifier or not action.key:
            return False

    if action.action_type == "SCROLL":
        if action.amount is None:
            return False

    if action.action_type == "WAIT":
        if action.seconds is None or action.seconds < 0:
            return False

    if action.action_type == "TYPE":
        if action.text is None:
            return False

    return True