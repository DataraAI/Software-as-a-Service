"""Run Dyn-HaMR using an already-generated ViPE output directory.

This script assumes Dyn-HaMR is already installed on the SaaS VM. It does not
clone repositories or install environments at runtime.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from runner_common import (
    collect_new_video_files,
    collect_new_obj_files,
    copy_artifacts,
    copy_meshes,
    default_dynhamr_root,
    locate_dynhamr_work_dir,
    parse_bool,
    resolve_dynhamr_python_command,
    run_command,
    write_manifest,
)

def run_dynhamr_for_vipe(
    *,
    data_root: Path,
    vipe_dir: Path,
    output_dir: Path,
    seq: str,
    fps: float,
    is_static: bool,
) -> dict[str, Any]:
    data_root = data_root.expanduser().resolve()
    vipe_dir = vipe_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    seq = re.sub(r"[^a-zA-Z0-9_.-]+", "_", seq.strip()).strip("_") or "datara_hand_motion"
    if not data_root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {data_root}")
    if not vipe_dir.is_dir():
        raise FileNotFoundError(f"ViPE output folder not found: {vipe_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    dynhamr_root = default_dynhamr_root()
    dynhamr_work_dir = locate_dynhamr_work_dir(dynhamr_root)
    logs_dir = output_dir / "logs"
    run_started_at = time.time()

    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")
    env.setdefault("PYTHONNOUSERSITE", "1")

    dynhamr_python_cmd = resolve_dynhamr_python_command()
    dynhamr_python_path = Path(dynhamr_python_cmd[0]).expanduser().resolve()
    dynhamr_bin_dir = dynhamr_python_path.parent
    existing_path = env.get("PATH", "")
    env["PATH"] = (
        f"{dynhamr_bin_dir}:{existing_path}" if existing_path else str(dynhamr_bin_dir)
    )
    env.setdefault("CONDA_PREFIX", str(dynhamr_bin_dir.parent))
    dynhamr_args = [
        "run_opt.py",
        "data=video_vipe",
        "data.use_vipe=True",
        f"data.root={data_root}",
        f"data.seq={seq}",
        f"data.frame_opts.fps={fps}",
        f"data.vipe_dir={vipe_dir}",
        f"is_static={'True' if is_static else 'False'}",
        "run_opt=True",
        "run_vis=True",
    ]
    run_command(
        dynhamr_python_cmd + dynhamr_args,
        cwd=dynhamr_work_dir,
        log_path=logs_dir / "dynhamr.log",
        env=env,
    )

    search_roots = [
        dynhamr_work_dir / "outputs" / "logs",
        dynhamr_root / "outputs" / "logs",
        dynhamr_work_dir,
    ]
    obj_files: list[Path] = []
    collection_root = dynhamr_work_dir
    for search_root in search_roots:
        obj_files = collect_new_obj_files(search_root, seq, run_started_at)
        if obj_files:
            collection_root = search_root
            break

    if not obj_files:
        raise RuntimeError("Dyn-HaMR completed but no OBJ files were found")

    copied_meshes = copy_meshes(obj_files, output_dir, collection_root)
    video_output_dir = output_dir / "rendered_videos"
    copied_videos: list[dict[str, str]] = []
    for search_root in search_roots:
        video_files = collect_new_video_files(search_root, seq, run_started_at)
        if video_files:
            copied_videos = copy_artifacts(video_files, video_output_dir, search_root)
            break

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dynhamr_root": str(dynhamr_root),
        "dynhamr_work_dir": str(dynhamr_work_dir),
        "data_root": str(data_root),
        "vipe_dir": str(vipe_dir),
        "output_dir": str(output_dir),
        "seq": seq,
        "fps": fps,
        "is_static": is_static,
        "command": dynhamr_python_cmd + ["run_opt.py"],
        "mesh_count": len(copied_meshes),
        "meshes": copied_meshes,
        "video_count": len(copied_videos),
        "videos": copied_videos,
        "logs": {"dynhamr": str(logs_dir / "dynhamr.log")},
    }
    write_manifest(output_dir, "dynhamr_manifest.json", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Dyn-HaMR from a prepared ViPE output directory")
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--vipe_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--seq", type=str, required=True)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--is_static", default="false")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = run_dynhamr_for_vipe(
        data_root=args.data_root,
        vipe_dir=args.vipe_dir,
        output_dir=args.output_dir,
        seq=args.seq,
        fps=float(args.fps or 30.0),
        is_static=parse_bool(args.is_static),
    )
    print(
        f"Generated {manifest['mesh_count']} OBJ meshes and {manifest.get('video_count', 0)} rendered videos",
        file=sys.stderr,
    )
    print(args.output_dir.expanduser().resolve())


if __name__ == "__main__":
    main()
