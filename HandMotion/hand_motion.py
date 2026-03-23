"""Fetch a frame sequence from URLs and save MediaPipe hand outputs to disk."""

from __future__ import annotations

import argparse
import json
import re
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlparse, urlunparse

import cv2
import mediapipe as mp
import numpy as np

# ~/hand_motion_generated/{media_pipes,hand_poses}/
OUTPUT_ROOT = Path.home() / "hand_motion_generated"
MEDIA_PIPES_DIR = OUTPUT_ROOT / "media_pipes"
HAND_POSES_DIR = OUTPUT_ROOT / "hand_poses"

# Match basename ..._<digits>.<ext> (extension may be .png, .jpg, etc.)
_FRAME_IN_PATH = re.compile(r"_(\d+)(\.[^./]+)$")


def url_for_frame(template_url: str, frame: int, zero_pad: int) -> str:
    """Build URL for `frame` using the same path shape as `template_url`."""
    parsed = urlparse(template_url)
    path = parsed.path
    m = _FRAME_IN_PATH.search(path)
    if not m:
        raise ValueError(
            "URL path must end with _<zero-padded-digits>.<ext> "
            f"(any image extension), e.g. .../clip_0001.jpg — got: {template_url!r}"
        )
    new_path = path[: m.start(1)] + f"{frame:0{zero_pad}d}" + path[m.end(1) :]
    return urlunparse(parsed._replace(path=new_path))


def frame_zero_pad_width(template_url: str) -> int:
    parsed = urlparse(template_url)
    m = _FRAME_IN_PATH.search(parsed.path)
    if not m:
        raise ValueError(
            "URL path must contain _<digits>.<ext> before query/fragment: "
            f"{template_url!r}"
        )
    return len(m.group(1))


def imread_url(url: str) -> np.ndarray | None:
    """Download an image from a URL and decode it as BGR (OpenCV format)."""
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (compatible; HandMotion/1.0)"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read()
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as e:
        print(f"Failed to fetch {url!r}: {e}")
        return None
    arr = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        print(f"Could not decode image from {url!r}")
    return image


def world_landmarks_to_joint_dict(landmark_list) -> dict[str, list[dict[str, float]]]:
    """MediaPipe world landmarks: x,y,z in meters (hand-relative)."""
    return {
        "joints": [
            {"x": float(lm.x), "y": float(lm.y), "z": float(lm.z)}
            for lm in landmark_list.landmark
        ]
    }


def output_stem(frame: int, zero_pad: int) -> str:
    return f"{frame:0{zero_pad}d}"


def main(url_template: str, num_frames: int) -> None:
    if num_frames < 1:
        raise SystemExit("num_frames must be at least 1.")

    pad = frame_zero_pad_width(url_template)

    MEDIA_PIPES_DIR.mkdir(parents=True, exist_ok=True)
    HAND_POSES_DIR.mkdir(parents=True, exist_ok=True)

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.7,
    ) as hands:
        for frame in range(num_frames):
            url = url_for_frame(url_template, frame, pad)
            image = imread_url(url)
            if image is None:
                continue

            stem = output_stem(frame, pad)
            media_path = MEDIA_PIPES_DIR / f"{stem}.png"

            # 2D landmarks (mirrored, same as original notebook)
            results_2d = hands.process(
                cv2.flip(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), 1)
            )
            print(f"Frame {frame} — {url}")
            print(f"  Handedness: {results_2d.multi_handedness}")

            annotated = cv2.flip(image.copy(), 1)
            if results_2d.multi_hand_landmarks:
                image_height, image_width, _ = image.shape
                for hand_landmarks in results_2d.multi_hand_landmarks:
                    print(
                        "  Index finger tip (image px): (",
                        f"{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, "
                        f"{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height})",
                    )
                    mp_drawing.draw_landmarks(
                        annotated,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style(),
                    )
            # Same resolution as the source frame (no resize).
            out_2d = cv2.flip(annotated, 1)
            cv2.imwrite(str(media_path), out_2d)
            print(f"  Saved {media_path}")

            # 3D world landmarks (unflipped RGB, same as original notebook)
            results_3d = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            print(f"  World landmarks: {results_3d.multi_hand_world_landmarks is not None}")
            if not results_3d.multi_hand_world_landmarks:
                continue
            for hi, hand_world_landmarks in enumerate(results_3d.multi_hand_world_landmarks):
                pose_path = HAND_POSES_DIR / f"{stem}_hand{hi}.json"
                payload = world_landmarks_to_joint_dict(hand_world_landmarks)
                pose_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
                print(f"  Saved {pose_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Download frames from URLs (path must contain _<digits>.<ext>), "
            "run MediaPipe Hands, and write outputs under ~/hand_motion_generated/."
        )
    )
    parser.add_argument(
        "--url_template",
        help=(
            "Example URL for one frame; basename must look like ..._0000.jpg "
            "(zero-padded index before extension; .png/.jpg/etc. allowed)."
        ),
        type=str,
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        help="Number of frames: indices [0, num_frames).",
    )
    args = parser.parse_args()
    url_template = args.url_template
    num_frames = args.num_frames
    main(url_template, num_frames)
