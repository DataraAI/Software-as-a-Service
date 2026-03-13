from __future__ import annotations

from typing import Sequence
import numpy as np

from agent.actions import StructuredAction

# Assumed action layout from your fine-tuned desktop SmolVLA checkpoint:
#
# vector[0:7]   = action logits/scores for:
#                 [CLICK, DOUBLE_CLICK, PRESS, HOTKEY, SCROLL, WAIT, TYPE]
# vector[7]     = x_norm in [0, 1]
# vector[8]     = y_norm in [0, 1]
# vector[9]     = key_index_norm in [0, 1]
# vector[10]    = modifier_index_norm in [0, 1]
# vector[11]    = scroll_amount (continuous, later rounded)
# vector[12]    = wait_seconds (continuous, positive)
# vector[13:45] = optional text character logits / IDs slot for short TYPE
#
# This is NOT a public SmolVLA standard.
# It is a startup-specific decoding contract for your v1.

ACTION_NAMES = [
    "CLICK",
    "DOUBLE_CLICK",
    "PRESS",
    "HOTKEY",
    "SCROLL",
    "WAIT",
    "TYPE",
]

KEY_VOCAB = [
    "enter", "tab", "esc", "backspace", "delete", "space",
    "up", "down", "left", "right",
    "a", "c", "v", "x", "z", "y", "s", "n",
]

MODIFIER_VOCAB = [
    "ctrl", "alt", "shift", "cmd",
]

TEXT_VOCAB = list("abcdefghijklmnopqrstuvwxyz0123456789 .,_-:/@")


def _clip01(v: float) -> float:
    return float(max(0.0, min(1.0, v)))


def _norm_to_index(v: float, vocab_size: int) -> int:
    v = _clip01(v)
    idx = int(round(v * (vocab_size - 1)))
    return max(0, min(vocab_size - 1, idx))


def _decode_short_text(vec: np.ndarray) -> str:
    # Simple v1 decoder:
    # treat vec as a series of normalized char IDs in [0,1], stop on near-zero tail
    chars = []
    for v in vec:
        if abs(float(v)) < 0.05:
            break
        idx = _norm_to_index(float(v), len(TEXT_VOCAB))
        chars.append(TEXT_VOCAB[idx])
    return "".join(chars).strip()


def decode_action_vector(
    action_vector: Sequence[float],
    screen_width: int,
    screen_height: int,
) -> StructuredAction:
    vec = np.asarray(action_vector, dtype=np.float32).flatten()

    if vec.size < 13:
        raise ValueError(
            f"Expected action vector of length >= 13, got {vec.size}. "
            "Your SmolVLA checkpoint and decoder contract do not match."
        )

    action_scores = vec[0:7]
    action_idx = int(np.argmax(action_scores))
    action_name = ACTION_NAMES[action_idx]

    x = int(round(_clip01(float(vec[7])) * (screen_width - 1)))
    y = int(round(_clip01(float(vec[8])) * (screen_height - 1)))

    key = KEY_VOCAB[_norm_to_index(float(vec[9]), len(KEY_VOCAB))]
    modifier = MODIFIER_VOCAB[_norm_to_index(float(vec[10]), len(MODIFIER_VOCAB))]
    scroll_amount = int(round(float(vec[11])))
    wait_seconds = max(0.1, float(vec[12]))

    if action_name == "CLICK":
        return StructuredAction(action_type="CLICK", x=x, y=y)

    if action_name == "DOUBLE_CLICK":
        return StructuredAction(action_type="DOUBLE_CLICK", x=x, y=y)

    if action_name == "PRESS":
        return StructuredAction(action_type="PRESS", key=key)

    if action_name == "HOTKEY":
        return StructuredAction(action_type="HOTKEY", modifier=modifier, key=key)

    if action_name == "SCROLL":
        return StructuredAction(action_type="SCROLL", amount=scroll_amount)

    if action_name == "WAIT":
        return StructuredAction(action_type="WAIT", seconds=round(wait_seconds, 2))

    if action_name == "TYPE":
        text = _decode_short_text(vec[13:45])
        if not text:
            text = " "
        return StructuredAction(action_type="TYPE", text=text)

    raise ValueError(f"Unsupported decoded action: {action_name}")