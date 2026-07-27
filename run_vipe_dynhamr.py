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
        [--video-url    <URL to download video from>]
        [--vipe-zip-url <URL to blob ViPE output zip; skips ViPE inference>]

EXAMPLE:
    python run_vipe_dynhamr.py \
        --video-url https://example.com/video.mp4 \
        --seq dishWasher_loading \
        --pipeline default

NOTES:
    - FPS is auto-detected from the video file via OpenCV (cv2). Falls back to
      30 if detection fails. Pass --fps to override manually.
    - The video file must live under <data_root>/videos/<seq>.mp4
      (data_root is auto-derived from the --video path as two levels up).
    - Conda environments required: "vipe" (ViPE) and "dynhamr" (DynHaMR).
    - If --vipe-zip-url is provided, ViPE inference is skipped entirely and the
      blob zip is downloaded and extracted instead.
"""

import argparse
import datetime
import time
import logging
import subprocess
import sys
import urllib.request
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
VIPE_CONDA_ENV    = "vipe"
DYNHAMR_CONDA_ENV = "dynhamr"
DYNHAMR_DIR       = "/home/ubuntu/packages/Dyn-HaMR/dyn-hamr"
FPS_FALLBACK      = 30
VIDEO_DOWNLOAD_ROOT = Path("/tmp/vipe_downloads")

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
        import cv2

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

    except Exception as exc:
        log.warning("  [WARN] OpenCV FPS detection failed (%s) — falling back to %s fps", exc, FPS_FALLBACK)
        return FPS_FALLBACK


def download_video(url: str, seq: str) -> Path:
    """
    Download a video from a URL into the required directory structure:
        /tmp/vipe_downloads/<seq>/videos/<seq>.mp4

    This ensures data_root resolves correctly as two levels above the video file.
    Returns the path to the downloaded video file.
    """
    # Preserve the original extension if it's a known video type, else default to .mp4
    url_path = url.split("?")[0]  # strip query params before checking extension
    suffix = Path(url_path).suffix.lower()
    if suffix not in (".mp4", ".mov", ".m4v", ".webm"):
        suffix = ".mp4"

    video_dir = VIDEO_DOWNLOAD_ROOT / seq / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    dest = video_dir / f"{seq}{suffix}"

    log.info("Downloading video from %s → %s", url, dest)
    urllib.request.urlretrieve(url, dest)
    log.info("Download complete: %s (%s bytes)", dest, dest.stat().st_size)

    return dest


def download_and_extract_vipe_zip(url: str, vipe_output: Path) -> None:
    """
    Download a blob ViPE output zip from a URL and extract it into vipe_output.
    """
    import zipfile
    vipe_output.mkdir(parents=True, exist_ok=True)
    zip_path = vipe_output.parent / "vipe_blob_download.zip"
    log.info("Downloading blob ViPE output from %s", url)
    urllib.request.urlretrieve(url, zip_path)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(vipe_output)
    zip_path.unlink(missing_ok=True)
    log.info("Extracted blob ViPE output to %s", vipe_output)


def zip_vipe_output(vipe_output: Path) -> Path:
    """
    Zip the ViPE output directory for caching.
    Returns the path to the created zip file.
    """
    import zipfile
    zip_path = vipe_output.parent / f"{vipe_output.parent.name}_vipe_output.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path in vipe_output.rglob("*"):
            if file_path.is_file():
                zf.write(file_path, file_path.relative_to(vipe_output))
    return zip_path

def resolve_dynhamr_output_dir(dynhamr_output_root: Path, seq: str) -> Path:
    """
    DynHaMR writes to: output/logs/video-custom/{YYYY-MM-DD}/{seq}-*/
    Resolve the exact directory created for this run by looking for a
    newly created subdirectory matching today's date and the sequence name.
    Raises RuntimeError if not found or ambiguous.
    """
    today = datetime.date.today().isoformat()  # YYYY-MM-DD
    date_dir = dynhamr_output_root / today
    if not date_dir.exists():
        raise RuntimeError(f"DynHaMR output date directory not found: {date_dir}")

    matches = sorted(
        [p for p in date_dir.iterdir() if p.is_dir() and p.name.startswith(f"{seq}-")],
        key=lambda p: p.stat().st_mtime,
    )
    if not matches:
        raise RuntimeError(
            f"No DynHaMR output directory found for seq '{seq}' under {date_dir}"
        )
    if len(matches) > 1:
        log.warning(
            "Multiple DynHaMR output dirs found for seq '%s' under %s: %s — "
            "using most recent: %s",
            seq, date_dir, [str(m) for m in matches], matches[-1],
        )
    return matches[-1]
# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the ViPE → DynHaMR pipeline on the Lambda VM."
    )

    video_source = parser.add_mutually_exclusive_group(required=True)
    video_source.add_argument(
        "--video",
        help="Full path to the input video file."
    )
    video_source.add_argument(
        "--video-url",
        help="URL to download the input video from."
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
        "--pipeline", choices=["default", "lyra"], default="lyra",
        help="ViPE pipeline to use (default: 'lyra'). "
             "Use 'lyra' for the RoboEyeView/Lyra pipeline."
    )
    parser.add_argument(
        "--static", action="store_true", default=False,
        help="Pass this flag if the scene is static (sets is_static=True in DynHaMR)."
    )
    parser.add_argument(
        "--vipe-zip-url",
        help="URL to a blob ViPE output zip; if provided, ViPE inference is skipped."
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

    # --- Resolve video path (download if URL was given) ---
    if args.video_url:
        video_path = download_video(args.video_url, args.seq)
    else:
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

    # --- Step 1: ViPE (or restore from blob) ---
    vipe_elapsed = 0.0
    if args.vipe_zip_url:
        log.info("=== Step 1: Restoring ViPE output from blob ===")
        log.info("  Blob URL : %s", args.vipe_zip_url)
        log.info("  Output    : %s", vipe_output)
        download_and_extract_vipe_zip(args.vipe_zip_url, vipe_output)
        log.info("ViPE blob restored — inference skipped.")
    else:
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

    # --- Locate DynHaMR output directory ---
    # Scoped to today's date and this seq to avoid picking up stale runs.
    dynhamr_output_root = (Path(DYNHAMR_DIR) / ".." / "output" / "logs" / "video-custom").resolve()
    log.info("Locating run output directory under: %s", dynhamr_output_root)
 
    run_output_dir = resolve_dynhamr_output_dir(dynhamr_output_root, args.seq)
    log.info("DynHaMR run output directory: %s", run_output_dir)
    
    # Search only within this run's output directory
    src_cam_videos = sorted(run_output_dir.rglob(f"*_src_cam.mp4"))
    obj_files = sorted(run_output_dir.rglob("smooth_fit/*/*.obj"))
    npz_files = sorted(run_output_dir.rglob("*_000000_joints_world.npz"))

    # --- Emit OUTPUT_RUN_DIR first so the caller can always clean up ---
    print(f"OUTPUT_RUN_DIR: {run_output_dir}")

    # --- NPZ + MCAP ---
    if npz_files:
        npz_file = npz_files[0]
        print(f"OUTPUT_NPZ: {npz_file}")
        mcap_out = npz_file.parent / f"{args.seq}.mcap"

        # REPLACE THIS WITH ACTUAL ANNOTATIONS FILE PATH
        dummy_annotations = '/home/ubuntu/packages/Dyn-HaMR/dyn-hamr/annotations.json'

        mcap_cmd = (
            f"cd /home/ubuntu/Software-as-a-Service && python dynhamr_mcap_file_gen.py "
            f"'{npz_file}' "
            f"--video '{video_path}' "
            f"--o '{mcap_out}' "
            f"--fps {fps} "
        )
        if Path(dummy_annotations).is_file():
            mcap_cmd += f"--annotations '{dummy_annotations}' "

        run_in_conda(DYNHAMR_CONDA_ENV, mcap_cmd)
        print(f"OUTPUT_MCAP: {mcap_out}")
        layout_cmd = (
            f"cd /home/ubuntu/Software-as-a-Service && python generate_layout.py "
            f"'{mcap_out}'"
        )
        run_in_conda(DYNHAMR_CONDA_ENV, layout_cmd)
        layout_out = mcap_out.parent / f"{args.seq}.layout.json"
        print(f"OUTPUT_LAYOUT: {layout_out}")

    else:
        log.warning("No joints_world.npz found under %s", run_output_dir)

    # --- Videos ---
    for video_file in src_cam_videos:
        print(f"OUTPUT_VIDEO: {video_file}")

    # --- OBJ directory ---
    if obj_files:
        print(f"OUTPUT_OBJ: {obj_files[0].parent}")

        # --- USD animation ---
        usd_out = run_output_dir / f"{args.seq}_hand_animation.usdz"
        saas_root = Path("/home/ubuntu/Software-as-a-Service")
        run_in_conda(
            VIPE_CONDA_ENV,
            f"cd '{saas_root}' && python run_hand_mesh_to_usd.py "
            f"--obj-dir '{obj_files[0].parent}' "
            f"--output '{usd_out}' "
            f"--fps {fps}",
        )
        print(f"OUTPUT_USD: {usd_out}")

    # --- ViPE zip (only when freshly computed, not when restored from blob) ---
    if not args.vipe_zip_url:
        vipe_zip_path = zip_vipe_output(vipe_output)
        print(f"OUTPUT_VIPE_ZIP: {vipe_zip_path}")
        

if __name__ == "__main__":
    main()
