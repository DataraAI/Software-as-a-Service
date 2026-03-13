from __future__ import annotations

import time
import pyautogui

from agent.actions import StructuredAction


pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.05


def execute_action(action: StructuredAction, dry_run: bool = False) -> None:
    if dry_run:
        print(f"[DRY RUN] {action.to_executor_string()}")
        return

    if action.action_type == "CLICK":
        pyautogui.click(action.x, action.y)
        return

    if action.action_type == "DOUBLE_CLICK":
        pyautogui.doubleClick(action.x, action.y)
        return

    if action.action_type == "PRESS":
        pyautogui.press(action.key)
        return

    if action.action_type == "HOTKEY":
        pyautogui.hotkey(action.modifier, action.key)
        return

    if action.action_type == "SCROLL":
        pyautogui.scroll(action.amount)
        return

    if action.action_type == "WAIT":
        time.sleep(action.seconds or 0.5)
        return

    if action.action_type == "TYPE":
        pyautogui.write(action.text or "", interval=0.01)
        return

    raise ValueError(f"Unsupported action type: {action.action_type}")