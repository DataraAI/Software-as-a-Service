from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

import cv2


DEFAULT_ERASERDIT_ROOT = str(Path.home() / "packages" / "EraserDiT")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run EraserDiT inpainting on a source/mask video pair")
    parser.add_argument("--source_video")
    parser.add_argument("--mask_video")
    parser.add_argument("--output_dir")
    parser.add_argument("--prompt")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    if args.verify_only:
        return args

    missing = [
        flag
        for flag, value in (
            ("--source_video", args.source_video),
            ("--mask_video", args.mask_video),
            ("--output_dir", args.output_dir),
            ("--prompt", args.prompt),
        )
        if not value
    ]
    if missing:
        parser.error("Missing required arguments: " + ", ".join(missing))
    return args


def resolve_runtime_paths() -> tuple[str, str, str]:
    eraserdit_root = os.getenv("ERASERDIT_ROOT", DEFAULT_ERASERDIT_ROOT)
    eraserdit_python = os.getenv("ERASERDIT_PYTHON_BIN", sys.executable)
    inference_path = os.path.join(eraserdit_root, "inference.py")

    if not os.path.isdir(eraserdit_root):
        raise FileNotFoundError(f"EraserDiT root was not found: {eraserdit_root}")
    if not os.path.isfile(inference_path):
        raise FileNotFoundError(f"EraserDiT inference entrypoint was not found: {inference_path}")

    if os.path.dirname(eraserdit_python):
        if not os.path.isfile(eraserdit_python):
            raise FileNotFoundError(f"EraserDiT python interpreter was not found: {eraserdit_python}")
    elif shutil.which(eraserdit_python) is None:
        raise FileNotFoundError(f"EraserDiT python interpreter was not found on PATH: {eraserdit_python}")

    if shutil.which("ffmpeg") is None:
        raise FileNotFoundError("ffmpeg was not found on PATH (required to crop videos to a multiple of 16)")

    return eraserdit_root, eraserdit_python, inference_path


def verify_runtime() -> tuple[str, str, str]:
    eraserdit_root, eraserdit_python, inference_path = resolve_runtime_paths()
    print(f"EraserDiT runtime verified: root={eraserdit_root} python={eraserdit_python}")
    return eraserdit_root, eraserdit_python, inference_path


def video_properties(video_path: str) -> dict:
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    finally:
        capture.release()
    if frame_count <= 0 or width <= 0 or height <= 0:
        raise RuntimeError(f"Could not read video properties for: {video_path}")
    return {"frame_count": frame_count, "width": width, "height": height}


def crop_to_multiple_of_16(input_path: str, output_path: str) -> None:
    """Center-crop a video so width/height are multiples of 16 — EraserDiT requires this."""
    completed = subprocess.run(
        [
            "ffmpeg", "-y", "-i", input_path,
            "-vf", "crop=trunc(iw/16)*16:trunc(ih/16)*16",
            "-c:a", "copy",
            output_path,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"ffmpeg crop failed for {input_path}: {completed.stderr.strip()}")


def find_result_video(results_dir: Path, run_stem: str) -> Path:
    """EraserDiT writes to results/<timestamp>-<vid_path stem>/*.mp4 relative to its own repo root."""
    matching_dirs = [p for p in results_dir.iterdir() if p.is_dir() and p.name.endswith(f"-{run_stem}")]
    if not matching_dirs:
        raise RuntimeError(f"EraserDiT did not create a results directory for '{run_stem}' under {results_dir}")
    newest_dir = max(matching_dirs, key=lambda p: p.stat().st_mtime)

    candidates = sorted(newest_dir.glob("*.mp4"))
    if not candidates:
        raise RuntimeError(f"EraserDiT did not produce an output video under {newest_dir}")
    return candidates[-1]


def main() -> None:
    args = parse_args()

    eraserdit_root, eraserdit_python, inference_path = verify_runtime()
    if args.verify_only:
        return

    source_props = video_properties(args.source_video)
    mask_props = video_properties(args.mask_video)
    if source_props["frame_count"] != mask_props["frame_count"]:
        raise RuntimeError(
            "Source and mask videos must contain the same number of frames "
            f"(source={source_props['frame_count']}, mask={mask_props['frame_count']})"
        )
    if (source_props["width"], source_props["height"]) != (mask_props["width"], mask_props["height"]):
        raise RuntimeError(
            "Source and mask videos must share the same resolution "
            f"(source={source_props['width']}x{source_props['height']}, "
            f"mask={mask_props['width']}x{mask_props['height']})"
        )

    os.makedirs(args.output_dir, exist_ok=True)
    results_dir = Path(eraserdit_root) / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    run_stem = f"src_{uuid.uuid4().hex[:12]}"

    with tempfile.TemporaryDirectory(prefix="eraserdit_cut_") as temp_dir:
        source_cut = os.path.join(temp_dir, f"{run_stem}.mp4")
        mask_cut = os.path.join(temp_dir, f"{run_stem}_mask.mp4")
        crop_to_multiple_of_16(args.source_video, source_cut)
        crop_to_multiple_of_16(args.mask_video, mask_cut)

        env = os.environ.copy()
        env["PYTHONNOUSERSITE"] = "1"
        completed = subprocess.run(
            [
                eraserdit_python, inference_path,
                "--vid_path", source_cut,
                "--mask_path", mask_cut,
                "--prompt", args.prompt,
            ],
            cwd=eraserdit_root,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if completed.returncode != 0:
            message = completed.stderr.strip() or completed.stdout.strip() or "Unknown subprocess failure"
            raise RuntimeError(message)

    result_video = find_result_video(results_dir, run_stem)
    final_output_path = os.path.join(args.output_dir, "eraserdit_removed.mp4")
    shutil.copyfile(result_video, final_output_path)
    print(final_output_path)


if __name__ == "__main__":
    main()
