"""Download a semantic VLM snapshot to local disk for offline inference."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--output-dir", required=True, help="Local directory for the downloaded model snapshot.")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--local-dir-use-symlinks", action="store_true", help="Allow symlinks in the local cache layout.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    destination = Path(args.output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=args.model_id,
        local_dir=str(destination),
        revision=args.revision,
        local_dir_use_symlinks=args.local_dir_use_symlinks,
    )
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
