#!/usr/bin/env python3
"""
Load per-frame PNG masks from ~/masks/<video_base_name>/<object_id>/ and cut out
each object from the MP4 at the corresponding frame. Writes RGBA PNGs to
~/objects/<video_base_name>/<object_id>/frame_<frame_id>.png with zero-padding
width equal to the number of digits in the video's total frame count (from
ffprobe on the file path, or by reading frames if ffprobe is missing).
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

FRAME_RE = re.compile(r"^frame_(\d+)\.png$", re.IGNORECASE)


def _parse_r_frame_rate(value: str | None) -> float:
    if not value or value in ("0/0", "0/1"):
        return 0.0
    num_s, _, den_s = value.partition("/")
    try:
        num = float(num_s)
        den = float(den_s) if den_s else 1.0
    except ValueError:
        return 0.0
    return num / den if den else 0.0


def _count_frames_by_reading(video_path: Path) -> int:
    """Decode sequentially and count frames (no CAP_PROP_FRAME_COUNT)."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    try:
        n = 0
        while cap.read()[0]:
            n += 1
        return max(1, n)
    finally:
        cap.release()


def frame_count_from_video_path(video_path: Path) -> int:
    """
    Resolve total frame count from the file at video_path using ffprobe metadata
    (nb_frames or duration x frame rate). If ffprobe is unavailable or yields no
    usable value, count frames by reading through the file with OpenCV.
    """
    video_path = video_path.resolve()
    try:
        proc = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=nb_frames,r_frame_rate,duration",
                "-show_entries",
                "format=duration",
                "-of",
                "json",
                str(video_path),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
    except FileNotFoundError:
        return _count_frames_by_reading(video_path)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"ffprobe failed for {video_path}: {e.stderr.strip() or e}"
        ) from e

    payload = json.loads(proc.stdout)
    streams = payload.get("streams") or []
    if not streams:
        return _count_frames_by_reading(video_path)

    s0 = streams[0]
    nb = s0.get("nb_frames")
    if nb not in (None, "", "N/A"):
        try:
            n = int(nb)
            if n > 0:
                return n
        except ValueError:
            pass

    dur_raw = s0.get("duration") or (payload.get("format") or {}).get("duration")
    fps = _parse_r_frame_rate(s0.get("r_frame_rate"))
    if dur_raw is not None and fps > 0:
        try:
            n = round(float(dur_raw) * fps)
            if n > 0:
                return n
        except ValueError:
            pass

    return _count_frames_by_reading(video_path)


def video_base_name(video_path: Path) -> str:
    if video_path.suffix.lower() != ".mp4":
        raise ValueError(f"Expected an .mp4 file, got: {video_path}")
    return video_path.stem


def collect_mask_tasks(masks_root: Path) -> dict[int, list[tuple[int, Path]]]:
    """Map frame index -> list of (object_id, mask_png_path)."""
    frame_tasks: dict[int, list[tuple[int, Path]]] = defaultdict(list)
    if not masks_root.is_dir():
        raise FileNotFoundError(f"Mask directory not found: {masks_root}")

    for obj_dir in sorted(masks_root.iterdir()):
        if not obj_dir.is_dir():
            continue
        if not obj_dir.name.isdigit():
            continue
        oid = int(obj_dir.name)
        for png in sorted(obj_dir.glob("*.png")):
            m = FRAME_RE.match(png.name)
            if not m:
                continue
            frame_idx = int(m.group(1))
            frame_tasks[frame_idx].append((oid, png))
    return frame_tasks


def load_mask_array(mask_path: Path, shape_hw: tuple[int, int]) -> np.ndarray:
    """Binary mask uint8 {0,255}, shape (H,W) matching the video frame."""
    mask = np.array(Image.open(mask_path).convert("L"))
    h, w = shape_hw
    if mask.shape[0] != h or mask.shape[1] != w:
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    return mask


def extract_and_save(
    frame_bgr: np.ndarray,
    mask_path: Path,
    out_path: Path,
) -> None:
    h, w = frame_bgr.shape[:2]
    mask = load_mask_array(mask_path, (h, w))
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    rgba = np.dstack([rgb, mask.astype(np.uint8)])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgba, mode="RGBA").save(out_path)


def run(video_path: Path, home: Path | None = None) -> None:
    home = home or Path.home()
    base = video_base_name(video_path)
    masks_root = home / "masks" / base
    objects_root = home / "objects" / base

    frame_tasks = collect_mask_tasks(masks_root)
    if not frame_tasks:
        raise RuntimeError(f"No mask PNGs found under {masks_root}")

    video_path = video_path.resolve()
    total_frames = frame_count_from_video_path(video_path)
    n_digits = len(str(max(1, total_frames)))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    max_needed = max(frame_tasks.keys())
    frame_idx = 0
    try:
        while frame_idx <= max_needed:
            ret, frame = cap.read()
            if not ret:
                missing = [f for f in frame_tasks if f >= frame_idx]
                raise RuntimeError(
                    f"Video ended before frame {max_needed} (stopped at {frame_idx}). "
                    f"Unprocessed frames: {missing[:10]}{'...' if len(missing) > 10 else ''}"
                )
            if frame_idx in frame_tasks:
                for oid, mask_path in frame_tasks[frame_idx]:
                    out_path = (
                        objects_root / str(oid) / f"frame_{frame_idx:0{n_digits}d}.png"
                    )
                    extract_and_save(frame, mask_path, out_path)
            frame_idx += 1
    finally:
        cap.release()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cut out masked objects from an MP4 using ~/masks/<name>/<id>/frame_*.png"
    )
    parser.add_argument(
        "--video_path",
        type=Path,
        help="Path to the input .mp4 file",
    )
    args = parser.parse_args()
    run(args.video_path)


if __name__ == "__main__":
    main()
