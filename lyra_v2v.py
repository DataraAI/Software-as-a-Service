"""
lyra_v2v.py — VIPE + Lyra Gen3C video-to-video pipeline

Runs on the Lambda VM. Downloads an input video from a SAS URL, runs it through
VIPE (vipe conda env), then feeds the VIPE output into Lyra Gen3C (lyra-v2v conda
env) to synthesise alternative camera views, then runs Lyra 3DGS reconstruction
to produce .ply Gaussian Splat files.

Usage:
    python3 lyra_v2v.py \
        --video_url <SAS_URL> \
        --output_dir <DIR> \
        [--vipe_zip_url <ZIP_SAS_URL>] \
        [--duration_seconds 5.0] \
        [--fps 24] \
        [--trajectory left]

Prints a single JSON line to stdout on success:
    {
        "gen3c_output_dir": "/abs/path/gen3c_output",
        "lyra_ply_dir":     "/abs/path/lyra_output",
        "vipe_zip":         "/abs/path/vipe_output.zip"
    }

All progress and subprocess output is written to stderr so stdout stays clean for
the JSON result that the caller parses.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import zipfile
from pathlib import Path

import requests


# ---------------------------------------------------------------------------
# Path constants — all overridable via environment variables
# ---------------------------------------------------------------------------
MINICONDA_ROOT = Path(os.getenv("SAAS_MINICONDA_ROOT", Path.home() / "miniconda3"))
CONDA_BIN      = MINICONDA_ROOT / "bin" / "conda"
VIPE_ROOT      = Path(os.getenv("VIPE_ROOT",  Path.home() / "packages" / "vipe"))
LYRA_ROOT      = Path(os.getenv("LYRA_ROOT",  Path.home() / "packages" / "lyra" / "Lyra-1"))
LYRA_CONDA_ENV = os.getenv("LYRA_CONDA_ENV", "lyra-v2v")
VIPE_CONDA_ENV = os.getenv("VIPE_CONDA_ENV", "vipe")
LYRA_BRANCH    = os.getenv("LYRA_BRANCH", "aryav_gen3c_fix")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_lyra_branch(lyra_root: Path, branch: str = LYRA_BRANCH) -> None:
    """Ensure the Lyra repo is on the correct branch before running.
    The branch only needs to exist locally on the Lambda VM — no push required.
    """
    print(f"[lyra_v2v] Checking out Lyra branch: {branch}", file=sys.stderr)
    result = subprocess.run(
        ["git", "checkout", branch],
        cwd=str(lyra_root),
        stdout=sys.stderr,
        stderr=sys.stderr,
    )
    if result.returncode != 0:
        print(f"[lyra_v2v] Failed to checkout branch {branch}", file=sys.stderr)
        sys.exit(result.returncode)
    print(f"[lyra_v2v] Lyra branch confirmed: {branch}", file=sys.stderr)


def _download(url: str, dest: Path, label: str = "file", timeout: int = 30) -> None:
    print(f"[lyra_v2v] Downloading {label}...", file=sys.stderr)
    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        with open(dest, "wb") as fh:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    fh.write(chunk)
    print(f"[lyra_v2v] Saved {label} to {dest}", file=sys.stderr)


def _zip_dir(source_dir: Path, zip_path: Path) -> None:
    print(f"[lyra_v2v] Zipping {source_dir} -> {zip_path}", file=sys.stderr)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file in source_dir.rglob("*"):
            if file.is_file():
                zf.write(file, file.relative_to(source_dir.parent))


def _run(args_list: list, label: str, cwd: Path | None = None, extra_env: dict | None = None) -> None:
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    print(f"[lyra_v2v] Running {label}...", file=sys.stderr)
    result = subprocess.run(
        args_list,
        cwd=str(cwd) if cwd else None,
        env=env,
        stdout=sys.stderr,
        stderr=sys.stderr,
    )
    if result.returncode != 0:
        print(f"[lyra_v2v] {label} failed with exit code {result.returncode}", file=sys.stderr)
        sys.exit(result.returncode)
    print(f"[lyra_v2v] {label} completed successfully", file=sys.stderr)


def _compute_num_video_frames(duration_seconds: float, fps: int) -> int:
    """
    Compute valid num_video_frames for GEN3C.
    Must satisfy (N - 1) % 120 == 0 → 121, 241, 361, 481, 601, 721...
    """
    target = int(duration_seconds * fps)
    N = max(1, round((target - 1) / 120))
    num_frames = 120 * N + 1
    actual_duration = (num_frames - 1) / fps
    print(f"[lyra_v2v] Requested: {duration_seconds}s @ {fps}fps = {target} frames", file=sys.stderr)
    print(f"[lyra_v2v] Using:     num_video_frames={num_frames} ({actual_duration:.2f}s)", file=sys.stderr)
    return num_frames


def _compute_bullet_times(num_video_frames: int) -> list:
    """Every 6th frame from 0 to num_video_frames."""
    bullet_times = list(range(0, num_video_frames, 6))
    print(f"[lyra_v2v] Bullet times: {len(bullet_times)} total", file=sys.stderr)
    return bullet_times


def _patch_lyra_yaml(lyra_root: Path, bullet_times: list) -> None:
    """Patch target_index_manual in lyra_dynamic.yaml with computed bullet times."""
    yaml_path = lyra_root / "configs" / "demo" / "lyra_dynamic.yaml"
    target_str = str(bullet_times).replace(" ", "")

    with open(yaml_path, "r") as f:
        content = f.read()

    content = re.sub(
        r"target_index_manual:.*",
        f"target_index_manual: {target_str}",
        content
    )

    with open(yaml_path, "w") as f:
        f.write(content)

    print(f"[lyra_v2v] Patched lyra_dynamic.yaml: {len(bullet_times)} bullet times", file=sys.stderr)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description="VIPE + Lyra Gen3C video-to-video pipeline")
    parser.add_argument("--video_url",        type=str, required=True)
    parser.add_argument("--output_dir",       type=str, required=True)
    parser.add_argument("--vipe_zip_url",     type=str, default=None)
    parser.add_argument("--duration_seconds", type=float, default=5.0,
                        help="Target output duration in seconds (default: 5.0)")
    parser.add_argument("--fps",              type=int, default=24,
                        help="FPS of the input video (default: 24)")
    # ADDED BACK: trajectory for single-trajectory generation
    parser.add_argument("--trajectory",       type=str, default="left",
                        help="Camera trajectory (left/right/up/down/zoom_in/zoom_out)")
    args = parser.parse_args(argv)

    output_dir       = Path(args.output_dir).resolve()
    input_video      = output_dir / "input.mp4"
    vipe_output_dir  = output_dir / "vipe_output"
    vipe_zip         = output_dir / "vipe_output.zip"
    gen3c_output_dir = output_dir / "gen3c_output"

    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1: Ensure Lyra repo is on the correct branch (local only)
    # ------------------------------------------------------------------
    _ensure_lyra_branch(LYRA_ROOT)

    # ------------------------------------------------------------------
    # Step 2: Compute frame count and bullet times from duration args
    # ------------------------------------------------------------------
    num_video_frames = _compute_num_video_frames(args.duration_seconds, args.fps)
    bullet_times     = _compute_bullet_times(num_video_frames)

    # ------------------------------------------------------------------
    # Step 3: Download input video from Azure SAS URL
    # ------------------------------------------------------------------
    _download(args.video_url, input_video, label="input video")

    # ------------------------------------------------------------------
    # Step 4: Restore VIPE output from cache, or run VIPE fresh
    # ------------------------------------------------------------------
    vipe_from_cache = False

    if args.vipe_zip_url:
        cached_zip = output_dir / "vipe_output_cached.zip"
        try:
            _download(args.vipe_zip_url, cached_zip, label="cached VIPE zip")
            import zipfile as _zf
            with _zf.ZipFile(cached_zip, "r") as z:
                z.extractall(output_dir)
            if vipe_output_dir.is_dir():
                print("[lyra_v2v] VIPE output restored from cache", file=sys.stderr)
                vipe_from_cache = True
            else:
                print("[lyra_v2v] Cache zip extracted but vipe_output/ missing — falling back to fresh VIPE", file=sys.stderr)
        except Exception as exc:
            print(f"[lyra_v2v] VIPE cache restore failed ({exc}) — falling back to fresh VIPE", file=sys.stderr)
        finally:
            if cached_zip.exists():
                cached_zip.unlink()

    if not vipe_from_cache:
        _run(
            [
                str(CONDA_BIN), "run", "-n", VIPE_CONDA_ENV,
                "vipe", "infer", str(input_video),
                "--output", str(vipe_output_dir),
                "--pipeline", "lyra",
            ],
            label="VIPE inference",
            cwd=VIPE_ROOT,
        )

    # ------------------------------------------------------------------
    # Step 5: Locate the VIPE output video expected by Gen3C
    # ------------------------------------------------------------------
    vipe_rgb_mp4 = vipe_output_dir / "rgb" / "input.mp4"
    if not vipe_rgb_mp4.is_file():
        print(f"[lyra_v2v] Expected VIPE output not found: {vipe_rgb_mp4}", file=sys.stderr)
        sys.exit(1)
    print(f"[lyra_v2v] VIPE output: {vipe_rgb_mp4}", file=sys.stderr)

    # ------------------------------------------------------------------
    # Step 6: Patch lyra_dynamic.yaml with computed bullet times
    # ------------------------------------------------------------------
    _patch_lyra_yaml(LYRA_ROOT, bullet_times)

    # ------------------------------------------------------------------
    # Step 7: Run Lyra Gen3C (single trajectory, with num_video_frames)
    # ------------------------------------------------------------------
    lyra_conda_prefix = MINICONDA_ROOT / "envs" / LYRA_CONDA_ENV
    gen3c_bash_cmd = (
        f"TORCHDYNAMO_DISABLE=1 " 
        f"CUDA_HOME={lyra_conda_prefix} "
        f"PYTHONPATH={LYRA_ROOT} "
        f"torchrun --nproc_per_node=1 "
        f"cosmos_predict1/diffusion/inference/gen3c_dynamic_sdg.py "
        f"--checkpoint_dir checkpoints "
        f"--vipe_path {vipe_rgb_mp4} "
        f"--video_save_folder {gen3c_output_dir} "
        f"--disable_prompt_upsampler "
        f"--num_gpus 1 "
        f"--num_video_frames {num_video_frames} "
        f"--fps {args.fps} "
        f"--foreground_masking "
        f"--trajectory {args.trajectory} "
        f"--center_depth_quantile"
    )
    _run(
        [str(CONDA_BIN), "run", "-n", LYRA_CONDA_ENV, "bash", "-c", gen3c_bash_cmd],
        label="Lyra Gen3C",
        cwd=LYRA_ROOT,
    )

    # ------------------------------------------------------------------
    # Step 8: Run Lyra 3DGS reconstruction (sample.py)
    # ------------------------------------------------------------------
    _run(
        [
            str(CONDA_BIN), "run", "-n", LYRA_CONDA_ENV,
            "accelerate", "launch", "sample.py",
            "--config", "configs/demo/lyra_dynamic.yaml",
            "dataset_name=lyra_dynamic_demo_generated",
            "save_gaussians_orig=true",
        ],
        label="Lyra reconstruction",
        cwd=LYRA_ROOT,
        extra_env={
            "LYRA_GEN3C_OUTPUT_DIR": str(gen3c_output_dir),
            "LYRA_SCENE_SCALE":      "0.1",
        },
    )

    # ------------------------------------------------------------------
    # Step 9: Confirm .ply outputs exist
    # ------------------------------------------------------------------
    lyra_ply_dir = (
        LYRA_ROOT
        / "outputs" / "demo" / "lyra_dynamic"
        / "static_view_indices_fixed_5_0_1_2_3_4"
        / "lyra_dynamic_demo_generated"
    )
    if not lyra_ply_dir.is_dir():
        print(f"[lyra_v2v] Expected Lyra .ply output directory not found: {lyra_ply_dir}", file=sys.stderr)
        sys.exit(1)
    print(f"[lyra_v2v] Lyra .ply output: {lyra_ply_dir}", file=sys.stderr)

    # ------------------------------------------------------------------
    # Step 10: Zip VIPE output for caching on future runs
    # ------------------------------------------------------------------
    _zip_dir(vipe_output_dir, vipe_zip)

    # ------------------------------------------------------------------
    # Done — print JSON result to stdout for the caller to parse
    # ------------------------------------------------------------------
    result = {
    "gen3c_video":      str(gen3c_output_dir / "rgb" / "input.mp4"),
    "vipe_zip":         str(vipe_zip),
}
    print(json.dumps(result))


if __name__ == "__main__":
    main()