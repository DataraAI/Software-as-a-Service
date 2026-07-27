"""
run_lyra_to_isaacsim.py -- Lyra .ply -> Isaac Sim USD pipeline

Takes a .ply zip from Azure Blob (downloaded locally), runs the full
Isaac Sim USD generation pipeline, and outputs a ready-to-open .usd file.

Usage:
    python3 run_lyra_to_isaacsim.py \
        --ply_zip /path/to/lyra_ply_output.zip \
        --output_dir /path/to/output/ \
        --output_usd /path/to/final_scene.usd

Pipeline steps:
    1. Unzip .ply files
    2. Convert each .ply to a per-frame .usd (py3dgsPlyToUsd)
    3. Generate Poisson collision proxy mesh for each frame (poisson_script)
    4. Build final animated stage with physics (build_animation_with_proxy)

Prints a single JSON line to stdout on success:
    {"output_usd": "/abs/path/to/final_scene.usd"}
"""

import argparse
import json
import os
import sys
import zipfile
from pathlib import Path

from convert_all_frames import convert_all_frames
from generate_all_proxy_meshes import generate_all_proxy_meshes
from build_animation_with_proxy import build_animation_with_proxy


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Lyra .ply zip -> Isaac Sim animated USD pipeline"
    )
    parser.add_argument("--ply_zip", type=str, required=True,
        help="Path to the lyra_ply_output.zip file")
    parser.add_argument("--output_dir", type=str, required=True,
        help="Base output directory for intermediate and final files")
    parser.add_argument("--output_usd", type=str, required=True,
        help="Path for the final animated .usd file")
    parser.add_argument("--fps", type=float, default=6.0,
        help="Animation playback rate (default: 6.0)")
    parser.add_argument("--poisson_depth", type=int, default=9)
    parser.add_argument("--opacity_floor", type=float, default=0.05)
    parser.add_argument("--density_trim_quantile", type=float, default=0.02)
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir).resolve()
    ply_dir    = output_dir / "ply_frames"
    usd_dir    = output_dir / "usd_frames"
    mesh_dir   = output_dir / "proxy_meshes"

    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1: Unzip .ply files
    # ------------------------------------------------------------------
    print(f"[run_lyra_to_isaacsim] Step 1: Unzipping {args.ply_zip}...", file=sys.stderr)
    ply_zip = Path(args.ply_zip).resolve()
    if not ply_zip.exists():
        print(f"[run_lyra_to_isaacsim] ERROR: ply_zip not found: {ply_zip}", file=sys.stderr)
        sys.exit(1)

    with zipfile.ZipFile(ply_zip, "r") as zf:
        zf.extractall(ply_dir)
    print(f"[run_lyra_to_isaacsim] Unzipped to {ply_dir}", file=sys.stderr)

    # Find the lyra_dynamic_demo_generated folder inside the zip
    # It may be nested, so search for it
    ply_base = None
    for root, dirs, files in os.walk(ply_dir):
        if "lyra_dynamic_demo_generated" in root and any(d.isdigit() for d in dirs):
            ply_base = root
            break
    # Fallback: look for any folder containing numbered subdirs with gaussians_orig
    if ply_base is None:
        for root, dirs, files in os.walk(ply_dir):
            for d in dirs:
                candidate = os.path.join(root, d, "gaussians_orig", "gaussians_0.ply")
                if os.path.exists(candidate):
                    ply_base = root
                    break
            if ply_base:
                break

    if ply_base is None:
        print(f"[run_lyra_to_isaacsim] ERROR: Could not find bullet-time .ply folders in zip", file=sys.stderr)
        sys.exit(1)

    print(f"[run_lyra_to_isaacsim] Found ply_base: {ply_base}", file=sys.stderr)

    # ------------------------------------------------------------------
    # Step 2: Convert .ply -> per-frame .usd
    # ------------------------------------------------------------------
    print(f"[run_lyra_to_isaacsim] Step 2: Converting .ply frames to USD...", file=sys.stderr)
    converted = convert_all_frames(ply_base, str(usd_dir))
    if not converted:
        print(f"[run_lyra_to_isaacsim] ERROR: No frames converted", file=sys.stderr)
        sys.exit(1)
    print(f"[run_lyra_to_isaacsim] Converted {len(converted)} frames", file=sys.stderr)

    # ------------------------------------------------------------------
    # Step 3: Generate Poisson proxy meshes
    # ------------------------------------------------------------------
    print(f"[run_lyra_to_isaacsim] Step 3: Generating collision proxy meshes...", file=sys.stderr)
    processed = generate_all_proxy_meshes(
        ply_base, str(mesh_dir),
        opacity_floor=args.opacity_floor,
        poisson_depth=args.poisson_depth,
        density_trim_quantile=args.density_trim_quantile,
    )
    print(f"[run_lyra_to_isaacsim] Generated {len(processed)} proxy meshes", file=sys.stderr)

    # ------------------------------------------------------------------
    # Step 4: Build final animated USD stage
    # ------------------------------------------------------------------
    print(f"[run_lyra_to_isaacsim] Step 4: Building animated USD stage...", file=sys.stderr)
    build_animation_with_proxy(
        usd_dir=str(usd_dir),
        mesh_dir=str(mesh_dir),
        output_usd=args.output_usd,
        frames_per_second=args.fps,
    )

    if not os.path.exists(args.output_usd):
        print(f"[run_lyra_to_isaacsim] ERROR: Output USD not created: {args.output_usd}", file=sys.stderr)
        sys.exit(1)

    # ------------------------------------------------------------------
    # Done -- print JSON result
    # ------------------------------------------------------------------
    result = {"output_usd": args.output_usd}
    print(json.dumps(result))
    print(f"[run_lyra_to_isaacsim] Done: {args.output_usd}", file=sys.stderr)


if __name__ == "__main__":
    main()