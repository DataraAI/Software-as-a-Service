"""
io/media_loader.py — Load frames from a video file or a directory of images.

Returns a list of (frame_index, timestamp_seconds, PIL.Image) tuples.
The rest of the pipeline only sees PIL Images — no OpenCV/numpy leakage upstream.

Dependencies:
  - Pillow (always required)
  - opencv-python (required only for video input)
"""

import logging
from pathlib import Path
from typing import List, Tuple

from PIL import Image

logger = logging.getLogger(__name__)

# Supported image extensions when loading from a directory
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".gif"}


def load_from_video(video_path: str) -> List[Tuple[int, float, Image.Image]]:
    """
    Load all frames from a video file using OpenCV.

    Returns: list of (frame_index, timestamp_seconds, PIL.Image)
    """
    try:
        import cv2
    except ImportError:
        raise ImportError(
            "opencv-python is required for video input. "
            "Install it with: pip install opencv-python"
        )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    frame_index = 0

    logger.info(f"Loading video: {video_path} (fps={fps:.1f})")

    while True:
        ret, bgr_frame = cap.read()
        if not ret:
            break

        timestamp = frame_index / fps

        # OpenCV is BGR; convert to RGB before wrapping in PIL
        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)

        frames.append((frame_index, timestamp, pil_image))
        frame_index += 1

    cap.release()
    logger.info(f"Loaded {len(frames)} frames from video")
    return frames


def load_from_image_dir(image_dir: str) -> List[Tuple[int, float, Image.Image]]:
    """
    Load sorted images from a directory.

    Files are sorted alphabetically (works for zero-padded filenames like
    frame_0001.png, frame_0002.png, ...).

    Timestamp is assigned as frame_index * 0.5 (assumes ~2 fps image sequence).
    Adjust ASSUMED_FPS below if your image sequences have a known rate.

    Returns: list of (frame_index, timestamp_seconds, PIL.Image)
    """
    ASSUMED_FPS = 2.0  # default assumption; adjust as needed

    dir_path = Path(image_dir)
    if not dir_path.is_dir():
        raise NotADirectoryError(f"Not a directory: {image_dir}")

    image_files = sorted([
        f for f in dir_path.iterdir()
        if f.suffix.lower() in IMAGE_EXTENSIONS
    ])

    if not image_files:
        raise FileNotFoundError(
            f"No image files found in {image_dir}. "
            f"Supported extensions: {IMAGE_EXTENSIONS}"
        )

    logger.info(f"Loading {len(image_files)} images from {image_dir}")

    frames = []
    for frame_index, filepath in enumerate(image_files):
        timestamp = frame_index / ASSUMED_FPS
        pil_image = Image.open(filepath).convert("RGB")
        frames.append((frame_index, timestamp, pil_image))

    logger.info(f"Loaded {len(frames)} images")
    return frames


def load_media(
    video_path: str = None,
    image_dir: str = None,
) -> List[Tuple[int, float, Image.Image]]:
    """
    Unified entry point. Exactly one of video_path or image_dir must be set.

    Returns: list of (frame_index, timestamp_seconds, PIL.Image)
    """
    if video_path and image_dir:
        raise ValueError("Provide either --video_path or --image_dir, not both")
    if not video_path and not image_dir:
        raise ValueError("Must provide either --video_path or --image_dir")

    if video_path:
        return load_from_video(video_path)
    else:
        return load_from_image_dir(image_dir)
