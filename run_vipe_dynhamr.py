#!/usr/bin/env python3
"""
run_vipe_dynhamr.py
SaaS script to run the ViPE → DynHaMR pipeline on the Lambda VM.

USAGE:
    python run_vipe_dynhamr.py \
        --video    <path to .mp4 file> \
        --seq      <base name of video, no extension> \
        --pipeline <vipe pipeline: default | lyra  (default: default)> \
        --static   <store_true flag, omit for False>
        [--fps     <override auto-detected FPS>]

EXAMPLE:
    python run_vipe_dynhamr.py \
        --video /home/ubuntu/packages/Dyn-HaMR/test/videos/dishWasher_loading.mp4 \
        --seq dishWasher_loading \
        --pipeline default

NOTES:
    - FPS is auto-detected from the video file via OpenCV (cv2). Falls back to
      30 if detection fails. Pass --fps to override manually.
    - The video file must live under <data_root>/videos/<seq>.mp4
      (data_root is auto-derived from the --video path as two levels up).
    - Conda environments required: "vipe" (ViPE) and "dynhamr" (DynHaMR).
"""

import argparse
import time
import logging
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
VIPE_CONDA_ENV   = "vipe"
DYNHAMR_CONDA_ENV = "dynhamr"
DYNHAMR_DIR      = "/home/ubuntu/packages/Dyn-HaMR/dyn-hamr"
FPS_FALLBACK     = 30

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    format="[%(asctime)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout,
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def conda_base() -> str:
    """Return the conda installation root."""
    try:
        result = subprocess.run(
            ["conda", "info", "--base"],
            capture_output=True, text=True
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except FileNotFoundError:
        # If the OS can't find the 'conda' command at all, log it and move on
        print("Warning: 'conda' command not found globally. Using default fallback path.")
    return "/home/ubuntu/miniconda3"


def run_in_conda(env: str, command: str) -> None:
    """
    Run a shell command inside the given conda environment.
    Raises subprocess.CalledProcessError on non-zero exit.
    """
    base = conda_base()
    activate = f"source '{base}/etc/profile.d/conda.sh' && conda activate '{env}'"
    full_cmd  = f"{activate} && {command}"
    subprocess.run(full_cmd, shell=True, executable="/bin/bash", check=True)


def detect_fps(video_path: str) -> float:
    """
    Use OpenCV to read the FPS embedded in the video container.

    Rounds to the nearest integer when within 0.1 of a whole number
    (e.g. 29.97 → 30, 23.976 → 24). Keeps one decimal otherwise.
    Falls back to FPS_FALLBACK on any failure.
    """
    try:
        import cv2  # noqa: PLC0415  (imported here; cv2 lives in the active env)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"OpenCV could not open: {video_path}")

        raw_fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        if not raw_fps or raw_fps <= 0:
            raise RuntimeError(f"Invalid FPS value returned by OpenCV: {raw_fps}")

        rounded = round(raw_fps)
        fps = rounded if abs(raw_fps - rounded) < 0.1 else round(raw_fps, 1)
        log.info("  Detected FPS : %s", fps)
        return fps

    except Exception as exc:  # noqa: BLE001
        log.warning("  [WARN] OpenCV FPS detection failed (%s) — falling back to %s fps", exc, FPS_FALLBACK)
        return FPS_FALLBACK


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the ViPE → DynHaMR pipeline on the Lambda VM."
    )
    parser.add_argument(
        "--video", required=True,
        help="Full path to the input .mp4 file."
    )
    parser.add_argument(
        "--seq", required=True,
        help="Base name of the video file (no extension), used as data.seq."
    )
    parser.add_argument(
        "--fps", type=float, default=None,
        help="Override auto-detected FPS (optional)."
    )
    parser.add_argument(
        "--pipeline", choices=["default", "lyra"], default="default",
        help="ViPE pipeline to use (default: 'default'). "
             "Use 'lyra' for the RoboEyeView/Lyra pipeline."
    )
    parser.add_argument(
        "--static", action="store_true", default=False,
        help="Pass this flag if the scene is static (sets is_static=True in DynHaMR)."
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def fmt_duration(seconds: float) -> str:
    """Format a duration in seconds as Xh Ym Zs."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    parts = []
    if h:
        parts.append(f"{h}h")
    if h or m:
        parts.append(f"{m}m")
    parts.append(f"{s:.1f}s")
    return " ".join(parts)


def main() -> None:
    args = parse_args()
    pipeline_start = time.time()

    # --- Validate inputs ---
    video_path = Path(args.video).resolve()
    if not video_path.is_file():
        log.error("Video file not found: %s", video_path)
        sys.exit(1)

    # data_root is two levels above the video:
    #   <data_root>/videos/<seq>.mp4
    data_root   = video_path.parent.parent
    vipe_output = data_root / f"{args.seq}_vipe_results" / "vipe_results"
    is_static   = "True" if args.static else "False"

    # --- FPS ---
    log.info("=== Detecting FPS via OpenCV ===")
    fps = args.fps if args.fps is not None else detect_fps(str(video_path))
    if args.fps is not None:
        log.info("FPS manually set to %s (skipping auto-detection)", fps)

    # --- Step 1: ViPE ---
    log.info("=== Step 1: Running ViPE (pipeline: %s) ===", args.pipeline)
    log.info("  Input  : %s", video_path)
    log.info("  Output : %s", vipe_output)

    vipe_output.mkdir(parents=True, exist_ok=True)

    vipe_start = time.time()
    run_in_conda(
        VIPE_CONDA_ENV,
        f"vipe infer '{video_path}' --output '{vipe_output}' --pipeline {args.pipeline}",
    )
    vipe_elapsed = time.time() - vipe_start
    log.info("ViPE inference complete. (%s)", fmt_duration(vipe_elapsed))

    # --- Step 2: DynHaMR ---
    log.info("=== Step 2: Running DynHaMR ===")
    log.info("  data.root   : %s", data_root)
    log.info("  data.seq    : %s", args.seq)
    log.info("  data.fps    : %s", fps)
    log.info("  vipe_dir    : %s", vipe_output)
    log.info("  is_static   : %s", is_static)

    dynhamr_cmd = (
        f"cd '{DYNHAMR_DIR}' && python run_opt.py"
        f" data=video_vipe"
        f" data.use_vipe=True"
        f" data.root='{data_root}'"
        f" data.seq='{args.seq}'"
        f" data.frame_opts.fps={fps}"
        f" data.vipe_dir='{vipe_output}'"
        f" is_static={is_static}"
        f" run_opt=True"
        f" run_vis=True"
    )
    dynhamr_start = time.time()
    run_in_conda(DYNHAMR_CONDA_ENV, dynhamr_cmd)
    dynhamr_elapsed = time.time() - dynhamr_start
    log.info("DynHaMR complete. (%s)", fmt_duration(dynhamr_elapsed))

    total = time.time() - pipeline_start
    log.info("=== Pipeline finished successfully ===")
    log.info("  ViPE      : %s", fmt_duration(vipe_elapsed))
    log.info("  DynHaMR   : %s", fmt_duration(dynhamr_elapsed))
    log.info("  Total     : %s", fmt_duration(total))


if __name__ == "__main__":
    main()
