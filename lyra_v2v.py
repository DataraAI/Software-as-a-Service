"""
lyra_v2v.py — VIPE + Lyra Gen3C video-to-video pipeline

Runs on the Lambda VM. Downloads an input video from a SAS URL, runs it through
VIPE (vipe conda env), then feeds the VIPE output into Lyra Gen3C (lyra-v2v conda
env) to synthesise alternative camera views.

Usage:
    python3 lyra_v2v.py --video_url <SAS_URL> --output_dir <DIR> [--vipe_zip_url <ZIP_SAS_URL>]

Prints a single JSON line to stdout on success:
    {"gen3c_video": "/abs/path/gen3c_output/0/rgb/input.mp4", "vipe_zip": "/abs/path/vipe_output.zip"}

All progress and subprocess output is written to stderr so stdout stays clean for
the JSON result that the caller parses.
"""

import argparse
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path
from urllib.parse import urlparse

import requests


# ---------------------------------------------------------------------------
# Path constants — all overridable via environment variables so the script
# works on any machine layout without edits.
# ---------------------------------------------------------------------------
MINICONDA_ROOT = Path(os.getenv("SAAS_MINICONDA_ROOT", Path.home()/"miniconda3"))
CONDA_BIN      = MINICONDA_ROOT/"bin"/"conda"
VIPE_ROOT      = Path(os.getenv("VIPE_ROOT",  Path.home()/"packages"/"vipe"))
LYRA_ROOT      = Path(os.getenv("LYRA_ROOT",  Path.home()/"packages"/"lyra" / "Lyra-1"))
LYRA_CONDA_ENV = os.getenv("LYRA_CONDA_ENV", "lyra-v2v")
VIPE_CONDA_ENV = os.getenv("VIPE_CONDA_ENV", "vipe")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _download(url: str, dest: Path, label: str = "file", timeout: int = 30) -> None:
    """Stream-download a URL to dest, following the same pattern as post_annotation.py."""
    print(f"[lyra_v2v] Downloading {label}...", file=sys.stderr)
    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        with open(dest, "wb") as fh:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    fh.write(chunk)
    print(f"[lyra_v2v] Saved {label} to {dest}", file=sys.stderr)


def _zip_dir(source_dir: Path, zip_path: Path) -> None:
    """Zip source_dir so that the archive root is source_dir's name (e.g. vipe_output/)."""
    print(f"[lyra_v2v] Zipping {source_dir} -> {zip_path}", file=sys.stderr)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file in source_dir.rglob("*"):
            if file.is_file():
                zf.write(file, file.relative_to(source_dir.parent))


def _run(args_list: list, label: str, cwd: Path | None = None, extra_env: dict | None = None) -> None:
    """Run a subprocess, streaming output to stderr. Raises on non-zero exit."""
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


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description="VIPE + Lyra Gen3C video-to-video pipeline")
    parser.add_argument("--video_url",    type=str, required=True)
    parser.add_argument("--output_dir",   type=str, required=True)
    parser.add_argument("--vipe_zip_url", type=str, default=None)
    parser.add_argument("--trajectory", type=str, default="left", choices=["up", "down", "left", "right", "zoom_in", "zoom_out"])
    args = parser.parse_args(argv)

    output_dir      = Path(args.output_dir).resolve()
    input_video     = output_dir/"input.mp4"
    vipe_output_dir = output_dir/"vipe_output"
    vipe_zip        = output_dir/"vipe_output.zip"
    gen3c_output_dir = output_dir/"gen3c_output"

    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1: Download input video from Azure SAS URL
    # ------------------------------------------------------------------
    _download(args.video_url, input_video, label="input video")

    # ------------------------------------------------------------------
    # Step 2: Restore VIPE output from cache, or run VIPE fresh
    # ------------------------------------------------------------------
    vipe_from_cache = False

    if args.vipe_zip_url:
        cached_zip = output_dir/"vipe_output_cached.zip"
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
                str(CONDA_BIN), "run", "--no-capture-output", "-n", VIPE_CONDA_ENV,
                "vipe", "infer", str(input_video),
                "--output", str(vipe_output_dir),
                "--pipeline", "lyra",
            ],
            label="VIPE inference",
            cwd=VIPE_ROOT,
        )

    # ------------------------------------------------------------------
    # Step 3: Locate the VIPE output video expected by Gen3C
    # ------------------------------------------------------------------
    vipe_rgb_mp4 = vipe_output_dir / "rgb" / "input.mp4"
    if not vipe_rgb_mp4.is_file():
        print(f"[lyra_v2v] Expected VIPE output not found: {vipe_rgb_mp4}", file=sys.stderr)
        sys.exit(1)
    print(f"[lyra_v2v] VIPE output: {vipe_rgb_mp4}", file=sys.stderr)

    # ------------------------------------------------------------------
    # Step 4: Run Lyra Gen3C
    # CUDA_HOME and PYTHONPATH are set inline inside bash -c so they
    # resolve correctly within the lyra-v2v conda environment.
    # ------------------------------------------------------------------
    lyra_conda_prefix = MINICONDA_ROOT / "envs" / LYRA_CONDA_ENV
    gen3c_bash_cmd = (
        f"CUDA_HOME={lyra_conda_prefix} "
        f"PYTHONPATH={LYRA_ROOT} "
        f"torchrun --nproc_per_node=1 "
        f"cosmos_predict1/diffusion/inference/gen3c_dynamic_sdg.py "
        f"--checkpoint_dir checkpoints "
        f"--vipe_path {vipe_rgb_mp4} "
        f"--video_save_folder {gen3c_output_dir} "
        f"--disable_prompt_upsampler "
        f"--num_gpus 1 "
        f"--foreground_masking "
        f"--trajectory {args.trajectory}"
    )
    _run(
        [str(CONDA_BIN), "run", "--no-capture-output", "-n", LYRA_CONDA_ENV, "bash", "-c", gen3c_bash_cmd],
        label="Lyra Gen3C",
        cwd=LYRA_ROOT,
    )

    # ------------------------------------------------------------------
    # Step 5: Locate the Gen3C output video
    # ------------------------------------------------------------------
    gen3c_output_video = gen3c_output_dir / "rgb" / "input.mp4"
    if not gen3c_output_video.is_file():
        print(f"[lyra_v2v] Expected Gen3C output not found: {gen3c_output_video}", file=sys.stderr)
        sys.exit(1)
    print(f"[lyra_v2v] Gen3C output: {gen3c_output_video}", file=sys.stderr)

    # ------------------------------------------------------------------
    # Step 6: Zip the VIPE output directory for caching on future runs
    # ------------------------------------------------------------------
    _zip_dir(vipe_output_dir, vipe_zip)

    # ------------------------------------------------------------------
    # Done — print JSON result to stdout for the caller to parse
    # ------------------------------------------------------------------
    result = {
        "gen3c_video": str(gen3c_output_video),
        "vipe_zip":    str(vipe_zip),
    }
    print(json.dumps(result))


if __name__ == "__main__":
    main()
