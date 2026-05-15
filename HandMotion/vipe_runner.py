"""Run ViPE on a prepared video for hand-motion workflows.

This script assumes ViPE is already installed on the SaaS VM. It does not
clone repositories or install environments at runtime.
"""

from __future__ import annotations

import argparse
import os
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from runner_common import default_vipe_work_dir, run_command, write_manifest


def run_vipe(
    *,
    video_path: Path,
    output_dir: Path,
    pipeline: str = "lyra",
    work_dir: Path | None = None,
) -> dict[str, Any]:
    video_path = video_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if not video_path.is_file():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = output_dir / "logs"
    resolved_work_dir = (work_dir or default_vipe_work_dir()).expanduser().resolve()
    resolved_work_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")
    vipe_cmd = shlex.split(os.environ.get("VIPE_CMD", "micromamba run -n vipe vipe"))
    run_command(
        vipe_cmd + ["infer", str(video_path), "--output", str(output_dir), "--pipeline", pipeline],
        cwd=resolved_work_dir,
        log_path=logs_dir / "vipe.log",
        env=env,
    )

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "video_path": str(video_path),
        "output_dir": str(output_dir),
        "pipeline": pipeline,
        "work_dir": str(resolved_work_dir),
        "logs": {"vipe": str(logs_dir / "vipe.log")},
    }
    write_manifest(output_dir, "vipe_manifest.json", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ViPE inference on a video")
    parser.add_argument("--video_path", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--pipeline", type=str, default="lyra")
    parser.add_argument("--work_dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = run_vipe(
        video_path=args.video_path,
        output_dir=args.output_dir,
        pipeline=str(args.pipeline or "lyra").strip() or "lyra",
        work_dir=args.work_dir,
    )
    print(
        f"ViPE inference completed with pipeline {manifest['pipeline']}",
        file=sys.stderr,
    )
    print(args.output_dir.expanduser().resolve())


if __name__ == "__main__":
    main()
