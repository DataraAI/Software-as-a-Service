from __future__ import annotations

import numpy as np
from PIL import Image
import mss


def capture_screenshot() -> Image.Image:
    """
    Capture the primary monitor as a PIL RGB image.
    """
    with mss.mss() as sct:
        monitor = sct.monitors[1]  # primary monitor
        shot = sct.grab(monitor)
        img = Image.frombytes("RGB", shot.size, shot.rgb)
        return img


def get_screen_size() -> tuple[int, int]:
    with mss.mss() as sct:
        monitor = sct.monitors[1]
        return monitor["width"], monitor["height"]