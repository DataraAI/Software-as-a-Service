from __future__ import annotations

import argparse
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2


SUPPORTED_WINDOW_LENGTHS = (17, 33, 49)
DEFAULT_ROSE_ROOT = str(Path.home() / "packages" / "ROSE")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ROSE inpainting on a source/mask video pair")
    parser.add_argument("--source_video")
    parser.add_argument("--mask_video")
    parser.add_argument("--output_dir")
    parser.add_argument("--prompt")
    parser.add_argument("--video_length", type=int, default=49)
    parser.add_argument("--sample_height", type=int, default=480)
    parser.add_argument("--sample_width", type=int, default=720)
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


def ensure_supported_window_length(length: int) -> int:
    return min(SUPPORTED_WINDOW_LENGTHS, key=lambda candidate: (abs(candidate - length), candidate))


def resolve_runtime_paths() -> tuple[str, str, str]:
    rose_root = os.getenv("ROSE_ROOT", DEFAULT_ROSE_ROOT)
    rose_python = os.getenv("ROSE_PYTHON_BIN", sys.executable)
    inference_path = os.path.join(rose_root, "inference.py")

    if not os.path.isdir(rose_root):
        raise FileNotFoundError(f"ROSE root was not found: {rose_root}")
    if not os.path.isfile(inference_path):
        raise FileNotFoundError(f"ROSE inference entrypoint was not found: {inference_path}")

    if os.path.dirname(rose_python):
        if not os.path.isfile(rose_python):
            raise FileNotFoundError(f"ROSE python interpreter was not found: {rose_python}")
    elif shutil.which(rose_python) is None:
        raise FileNotFoundError(f"ROSE python interpreter was not found on PATH: {rose_python}")

    return rose_root, rose_python, inference_path


def verify_runtime() -> tuple[str, str, str]:
    rose_root, rose_python, inference_path = resolve_runtime_paths()
    print(f"ROSE runtime verified: root={rose_root} python={rose_python}")
    return rose_root, rose_python, inference_path


def read_video_frames(video_path: str) -> tuple[list, float]:
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0) or 30.0
    frames = []
    try:
        while True:
            success, frame = capture.read()
            if not success:
                break
            frames.append(frame)
    finally:
        capture.release()

    if not frames:
        raise RuntimeError(f"No frames were decoded from: {video_path}")

    return frames, fps


def write_video(frames: list, output_path: str, fps: float) -> None:
    height, width = frames[0].shape[:2]
    codec_attempts = ("avc1", "H264", "mp4v")
    writer = None
    for codec in codec_attempts:
        candidate = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*codec), fps, (width, height))
        if candidate.isOpened():
            writer = candidate
            break
        candidate.release()
    if writer is None:
        raise RuntimeError(f"Failed to create video writer: {output_path}")

    try:
        for frame in frames:
            writer.write(frame)
    finally:
        writer.release()


def pad_frames(frames: list, target_length: int) -> list:
    if len(frames) >= target_length:
        return list(frames[:target_length])
    if not frames:
        raise RuntimeError("Cannot pad an empty frame list")
    padded = list(frames)
    while len(padded) < target_length:
        padded.append(frames[-1].copy())
    return padded


def chunk_start_indices(total_frames: int, window_length: int) -> list[int]:
    if total_frames <= window_length:
        return [0]
    stride = max(window_length - 1, 1)
    starts = list(range(0, total_frames, stride))
    if starts[-1] >= total_frames:
        starts[-1] = max(total_frames - 1, 0)
    return starts


def run_subprocess(command: list[str], *, cwd: str) -> None:
    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"
    completed = subprocess.run(
        command,
        cwd=cwd,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    if completed.returncode != 0:
        message = completed.stderr.strip() or completed.stdout.strip() or "Unknown subprocess failure"
        raise RuntimeError(message)


def find_generated_video(output_dir: str) -> str:
    candidates = sorted(
        [
            str(path)
            for path in Path(output_dir).iterdir()
            if path.is_file() and path.suffix.lower() in {".mp4", ".mov", ".m4v", ".webm"}
        ]
    )
    if not candidates:
        raise RuntimeError(f"ROSE did not produce a video in {output_dir}")
    return candidates[-1]


def main() -> None:
    args = parse_args()

    rose_root, rose_python, inference_path = verify_runtime()
    if args.verify_only:
        return

    source_frames, source_fps = read_video_frames(args.source_video)
    mask_frames, _mask_fps = read_video_frames(args.mask_video)
    if len(source_frames) != len(mask_frames):
        raise RuntimeError("Source and mask videos must contain the same number of frames")

    os.makedirs(args.output_dir, exist_ok=True)

    window_length = ensure_supported_window_length(args.video_length)
    starts = chunk_start_indices(len(source_frames), window_length)
    stitched_frames = []

    with tempfile.TemporaryDirectory(prefix="rose_chunks_") as temp_root:
        for chunk_index, start in enumerate(starts):
            end = min(start + window_length, len(source_frames))
            actual_length = end - start
            if actual_length <= 0:
                continue

            chunk_source_frames = pad_frames(source_frames[start:end], window_length)
            chunk_mask_frames = pad_frames(mask_frames[start:end], window_length)

            chunk_dir = os.path.join(temp_root, f"chunk_{chunk_index:03d}")
            chunk_output_dir = os.path.join(chunk_dir, "output")
            os.makedirs(chunk_output_dir, exist_ok=True)

            chunk_source_path = os.path.join(chunk_dir, "source.mp4")
            chunk_mask_path = os.path.join(chunk_dir, "mask.mp4")
            write_video(chunk_source_frames, chunk_source_path, source_fps)
            write_video(chunk_mask_frames, chunk_mask_path, source_fps)

            run_subprocess(
                [
                    rose_python,
                    "-s",
                    inference_path,
                    "--validation_videos",
                    chunk_source_path,
                    "--validation_masks",
                    chunk_mask_path,
                    "--validation_prompts",
                    args.prompt,
                    "--output_dir",
                    chunk_output_dir,
                    "--video_length",
                    str(window_length),
                    "--sample_size",
                    str(args.sample_height),
                    str(args.sample_width),
                ],
                cwd=rose_root,
            )

            chunk_result_path = find_generated_video(chunk_output_dir)
            chunk_result_frames, _chunk_result_fps = read_video_frames(chunk_result_path)

            trim_start = 1 if stitched_frames else 0
            keep_count = max(actual_length - trim_start, 0)
            stitched_frames.extend(chunk_result_frames[trim_start : trim_start + keep_count])

    if not stitched_frames:
        raise RuntimeError("ROSE did not return any output frames")

    final_output_path = os.path.join(args.output_dir, "rose_removed.mp4")
    write_video(stitched_frames, final_output_path, source_fps)
    print(final_output_path)


if __name__ == "__main__":
    main()
