"""
lyra_v2v.py — VIPE + Lyra Gen3C + Lyra Reconstruction + Isaac Sim USD pipeline

Runs on the Lambda VM. Downloads an input video from a SAS URL, runs it through
VIPE (vipe conda env), then feeds the VIPE output into Lyra Gen3C (lyra-v2v conda
env) to synthesise alternative camera views, runs Lyra 3DGS reconstruction to
produce .ply Gaussian Splat files, then runs the Isaac Sim USD pipeline.

Usage:
    python3 lyra_v2v.py --video_url <SAS_URL> --output_dir <DIR> [--vipe_zip_url <ZIP_SAS_URL>]

Prints a single JSON line to stdout on success:
    {
        "gen3c_video":  "/abs/path/gen3c_output/rgb/input.mp4",
        "vipe_zip":     "/abs/path/vipe_output.zip",
        "lyra_ply_zip": "/abs/path/lyra_ply_output.zip",
        "output_usd":   "/abs/path/scene.usd"
    }

All progress and subprocess output is written to stderr so stdout stays clean.
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
# Path constants
# ---------------------------------------------------------------------------
MINICONDA_ROOT = Path(os.getenv("SAAS_MINICONDA_ROOT", Path.home() / "miniconda3"))
CONDA_BIN      = MINICONDA_ROOT / "bin" / "conda"
VIPE_ROOT      = Path(os.getenv("VIPE_ROOT",  Path.home() / "packages" / "vipe"))
LYRA_ROOT      = Path(os.getenv("LYRA_ROOT",  Path.home() / "packages" / "lyra" / "Lyra-1"))
LYRA_CONDA_ENV = os.getenv("LYRA_CONDA_ENV", "lyra-v2v")
VIPE_CONDA_ENV = os.getenv("VIPE_CONDA_ENV", "vipe")
LYRA_BRANCH    = os.getenv("LYRA_BRANCH", "full-sim")
SAAS_ROOT      = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_lyra_branch(lyra_root: Path, branch: str = LYRA_BRANCH) -> None:
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


def _run(args_list: list, label: str, cwd: Path | None = None,
         extra_env: dict | None = None) -> None:
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


def _patch_lyra_yaml(lyra_root: Path, lyra_out_dir: Path) -> None:
    yaml_path = lyra_root / "configs" / "demo" / "lyra_dynamic.yaml"
    bullet_times = list(range(0, 121, 6))  # [0, 6, 12, ..., 120] = 21 bullet times
    target_str = str(bullet_times).replace(" ", "")

    with open(yaml_path, "r") as f:
        content = f.read()

    content = re.sub(
        r"out_dir_inference:.*",
        f"out_dir_inference: {lyra_out_dir}",
        content
    )
    content = re.sub(
        r"target_index_manual:.*",
        f"target_index_manual: {target_str}",
        content
    )

    with open(yaml_path, "w") as f:
        f.write(content)

    print(f"[lyra_v2v] Patched lyra_dynamic.yaml: out_dir_inference={lyra_out_dir}, "
          f"{len(bullet_times)} bullet times", file=sys.stderr)

# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="VIPE + Lyra Gen3C + Lyra Reconstruction + Isaac Sim USD pipeline"
    )
    parser.add_argument("--video_url",    type=str, required=True)
    parser.add_argument("--output_dir",   type=str, required=True)
    parser.add_argument("--vipe_zip_url", type=str, default=None)
    parser.add_argument("--trajectory",   type=str, default="left",
                        choices=["up", "down", "left", "right", "zoom_in", "zoom_out"])
    args = parser.parse_args(argv)

    output_dir       = Path(args.output_dir).resolve()
    input_video      = output_dir / "input.mp4"
    vipe_output_dir  = output_dir / "vipe_output"
    vipe_zip         = output_dir / "vipe_output.zip"
    gen3c_output_dir = output_dir / "gen3c_output"
    lyra_out_dir     = output_dir / "lyra_output"
    lyra_ply_zip     = output_dir / "lyra_ply_output.zip"
    output_usd       = output_dir / "scene.usd"

    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1: Ensure Lyra repo is on the correct branch
    # ------------------------------------------------------------------
    _ensure_lyra_branch(LYRA_ROOT)

    # ------------------------------------------------------------------
    # Step 2: Download input video
    # ------------------------------------------------------------------
    _download(args.video_url, input_video, label="input video")

    # ------------------------------------------------------------------
    # Step 3: Restore VIPE from cache or run fresh
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
                print("[lyra_v2v] Cache zip extracted but vipe_output/ missing "
                      "-- falling back to fresh VIPE", file=sys.stderr)
        except Exception as exc:
            print(f"[lyra_v2v] VIPE cache restore failed ({exc}) "
                  "-- falling back to fresh VIPE", file=sys.stderr)
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
    # Step 4: Locate VIPE output video
    # ------------------------------------------------------------------
    vipe_rgb_mp4 = vipe_output_dir / "rgb" / "input.mp4"
    if not vipe_rgb_mp4.is_file():
        print(f"[lyra_v2v] Expected VIPE output not found: {vipe_rgb_mp4}", file=sys.stderr)
        sys.exit(1)
    print(f"[lyra_v2v] VIPE output: {vipe_rgb_mp4}", file=sys.stderr)

    # ------------------------------------------------------------------
    # Step 5: Patch lyra_dynamic.yaml with per-job output dir
    # ------------------------------------------------------------------
    _patch_lyra_yaml(LYRA_ROOT, lyra_out_dir)

    # ------------------------------------------------------------------
    # Step 6: Run Lyra Gen3C (original command, unchanged)
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
        [str(CONDA_BIN), "run", "-n", LYRA_CONDA_ENV, "bash", "-c", gen3c_bash_cmd],
        label="Lyra Gen3C",
        cwd=LYRA_ROOT,
    )

    # ------------------------------------------------------------------
    # Step 7: Locate Gen3C output video
    # ------------------------------------------------------------------
    gen3c_output_video = gen3c_output_dir / "rgb" / "input.mp4"
    if not gen3c_output_video.is_file():
        print(f"[lyra_v2v] Expected Gen3C output not found: {gen3c_output_video}", file=sys.stderr)
        sys.exit(1)
    print(f"[lyra_v2v] Gen3C output: {gen3c_output_video}", file=sys.stderr)

    # ------------------------------------------------------------------
    # Step 8: Run Lyra reconstruction (sample.py) -- NEW
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
    # Step 9: Locate .ply output and zip it -- NEW
    # ------------------------------------------------------------------
    lyra_ply_dir = (
        lyra_out_dir
        / "static_view_indices_fixed_5_0_1_2_3_4"
        / "lyra_dynamic_demo_generated"
    )
    if not lyra_ply_dir.is_dir():
        print(f"[lyra_v2v] Expected Lyra .ply output not found: {lyra_ply_dir}", file=sys.stderr)
        sys.exit(1)
    print(f"[lyra_v2v] Lyra .ply output: {lyra_ply_dir}", file=sys.stderr)
    _zip_dir(lyra_ply_dir, lyra_ply_zip)

    # ------------------------------------------------------------------
    # Step 10: Run Isaac Sim USD pipeline -- NEW
    # ------------------------------------------------------------------
    _run(
        [
            str(CONDA_BIN), "run", "-n", LYRA_CONDA_ENV,
            "python3", str(SAAS_ROOT / "run_lyra_to_isaacsim.py"),
            "--ply_zip",    str(lyra_ply_zip),
            "--output_dir", str(output_dir / "isaacsim"),
            "--output_usd", str(output_usd),
        ],
        label="Isaac Sim USD pipeline",
        cwd=SAAS_ROOT,
    )

    if not output_usd.exists():
        print(f"[lyra_v2v] Expected USD output not found: {output_usd}", file=sys.stderr)
        sys.exit(1)

    # ------------------------------------------------------------------
    # Step 11: Zip VIPE output for caching (same as original)
    # ------------------------------------------------------------------
    _zip_dir(vipe_output_dir, vipe_zip)

    # ------------------------------------------------------------------
    # Done — print JSON result
    # ------------------------------------------------------------------
    result = {
        "gen3c_video":  str(gen3c_output_video),
        "vipe_zip":     str(vipe_zip),
        "lyra_ply_zip": str(lyra_ply_zip),
        "output_usd":   str(output_usd),
    }
    print(json.dumps(result))


if __name__ == "__main__":
    main()