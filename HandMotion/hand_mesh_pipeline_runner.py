"""Build a video from ego frames, run ViPE, then run Dyn-HaMR.

This is the SaaS-side entrypoint used by DaaS for the hand-mesh button. It
assumes ViPE and Dyn-HaMR are already installed on the VM and only stores final
OBJ outputs plus manifests in the requested output directory.
"""

from __future__ import annotations

import argparse
import re
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from dyn_hamr_runner import run_dynhamr_for_vipe
from runner_common import build_video, collect_images, parse_bool, write_manifest
from vipe_runner import run_vipe


def run_pipeline(args: argparse.Namespace) -> dict[str, object]:
    image_dir = args.image_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    seq = re.sub(r"[^a-zA-Z0-9_.-]+", "_", args.seq.strip()).strip("_") or "datara_hand_motion"
    fps = float(args.fps or 30.0)
    is_static = parse_bool(args.is_static)

    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image folder not found: {image_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = collect_images(image_dir)
    if not image_paths:
        raise ValueError(f"No supported image files found in {image_dir}")

    with tempfile.TemporaryDirectory(prefix="datara_hand_mesh_pipeline_") as tmp:
        temp_root = Path(tmp)
        dataset_root = temp_root / "dataset"
        video_dir = dataset_root / "videos"
        video_path = video_dir / f"{seq}.mp4"
        vipe_dir = temp_root / "vipe_results"

        print("__STAGE__:staging_frames", flush=True)
        video_metadata = build_video(image_paths, video_path, fps)

        print("__STAGE__:running_vipe", flush=True)
        vipe_manifest = run_vipe(
            video_path=video_path,
            output_dir=vipe_dir,
            pipeline="lyra",
        )

        print("__STAGE__:running_dynhamr", flush=True)
        dynhamr_manifest = run_dynhamr_for_vipe(
            data_root=dataset_root,
            vipe_dir=vipe_dir,
            output_dir=output_dir,
            seq=seq,
            fps=fps,
            is_static=is_static,
        )

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_image_dir": str(image_dir),
        "output_dir": str(output_dir),
        "seq": seq,
        "fps": fps,
        "is_static": is_static,
        "source_frame_count": len(image_paths),
        "video_metadata": video_metadata,
        "vipe": vipe_manifest,
        "dynhamr": dynhamr_manifest,
    }
    write_manifest(output_dir, "hand_mesh_pipeline_manifest.json", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Datara hand-mesh pipeline")
    parser.add_argument("--image_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--seq", type=str, required=True)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--is_static", default="false")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = run_pipeline(args)
    print(
        f"Generated {manifest['dynhamr']['mesh_count']} OBJ meshes",
        file=sys.stderr,
    )
    print(args.output_dir.expanduser().resolve())


if __name__ == "__main__":
    main()
