"""Run a lightweight Colab-oriented image/video demo with hinted payloads."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scripts.run_single_inference import run as run_single_inference


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-path", default="test_image.jpg", help="Optional uploaded image path.")
    parser.add_argument("--video-path", default="test_video.mp4", help="Optional uploaded video path.")
    parser.add_argument("--skip-image", action="store_true", help="Skip the image demo even if the file exists.")
    parser.add_argument("--skip-video", action="store_true", help="Skip the video demo even if the file exists.")
    parser.add_argument("--input-dir", default=str(ROOT / "data" / "sample_inputs"))
    parser.add_argument("--output-dir", default=str(ROOT / "data" / "outputs"))
    parser.add_argument("--config", default=str(ROOT / "configs" / "colab_refined_video.yaml"))
    parser.add_argument("--instruction", default="Describe only the visible hand-object action conservatively.")
    parser.add_argument("--task-name", default="ethernet_cable_insert")
    parser.add_argument("--tag", action="append", default=[], help="Soft semantic hint. Repeat for multiple tags.")
    parser.add_argument("--image-episode-id", default="real-image-001")
    parser.add_argument("--video-episode-id", default="real-video-001")
    parser.add_argument("--zip-outputs", action="store_true", help="Create a zip archive of the output directory.")
    parser.add_argument("--zip-name", default="robotask_outputs", help="Archive filename without extension.")
    return parser.parse_args()


def run_demo(
    *,
    image_path: str | Path | None,
    video_path: str | Path | None,
    input_dir: str | Path,
    output_dir: str | Path,
    config_path: str | Path | None,
    instruction: str,
    task_name: str,
    tags: list[str] | None = None,
    skip_image: bool = False,
    skip_video: bool = False,
    image_episode_id: str = "real-image-001",
    video_episode_id: str = "real-video-001",
    zip_outputs: bool = False,
    zip_name: str = "robotask_outputs",
) -> dict[str, Any]:
    """Create payloads, run inference, and print concise summaries."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    cleaned_tags = [tag.strip() for tag in (tags or []) if tag and tag.strip()]
    results: dict[str, Any] = {
        "payload_paths": {},
        "output_paths": {},
        "raw_output_paths": {},
        "archive_path": None,
    }

    if not skip_image and image_path:
        image_candidate = Path(image_path)
        if image_candidate.exists():
            payload_path = _write_payload(
                asset_path=image_candidate,
                payload_dir=input_dir,
                episode_id=image_episode_id,
                task_name=task_name,
                instruction=instruction,
                tags=cleaned_tags,
            )
            output_path, raw_output_path, output = run_single_inference(
                input_path=payload_path,
                output_dir=output_dir,
                config_path=config_path,
                action_backend="none",
            )
            results["payload_paths"]["image"] = payload_path
            results["output_paths"]["image"] = output_path
            results["raw_output_paths"]["image"] = raw_output_path
            _print_output_summary("image", output, output_path, raw_output_path)
        else:
            print(f"Skipping image demo because the file was not found: {image_candidate}")

    if not skip_video and video_path:
        video_candidate = Path(video_path)
        if video_candidate.exists():
            payload_path = _write_payload(
                asset_path=video_candidate,
                payload_dir=input_dir,
                episode_id=video_episode_id,
                task_name=task_name,
                instruction=instruction,
                tags=cleaned_tags,
            )
            output_path, raw_output_path, output = run_single_inference(
                input_path=payload_path,
                output_dir=output_dir,
                config_path=config_path,
                action_backend="none",
            )
            results["payload_paths"]["video"] = payload_path
            results["output_paths"]["video"] = output_path
            results["raw_output_paths"]["video"] = raw_output_path
            _print_output_summary("video", output, output_path, raw_output_path)
        else:
            print(f"Skipping video demo because the file was not found: {video_candidate}")

    if not results["output_paths"]:
        raise FileNotFoundError("No valid image or video inputs were found for the Colab demo.")

    if zip_outputs:
        archive_path = shutil.make_archive(
            str(output_dir.parent / zip_name),
            "zip",
            root_dir=output_dir,
        )
        results["archive_path"] = Path(archive_path)
        print(f"\nCreated output archive: {archive_path}")

    return results


def _write_payload(
    *,
    asset_path: Path,
    payload_dir: Path,
    episode_id: str,
    task_name: str,
    instruction: str,
    tags: list[str],
) -> Path:
    payload = {
        "episode_id": episode_id,
        "task_name": task_name,
        "instruction": instruction,
        "asset_path": _payload_asset_ref(asset_path, payload_dir),
    }
    if tags:
        payload["metadata"] = {"tags": tags}

    payload_path = payload_dir / f"{episode_id.replace('-', '_')}.json"
    payload_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload_path


def _payload_asset_ref(asset_path: Path, payload_dir: Path) -> str:
    resolved_asset = asset_path.resolve()
    try:
        return str(resolved_asset.relative_to(payload_dir.resolve()))
    except ValueError:
        return str(resolved_asset)


def _print_output_summary(modality: str, output: Any, output_path: Path, raw_output_path: Path | None) -> None:
    print(f"\n{modality.upper()} RESULT")
    print(f"episode_id: {output.episode_id}")
    print(f"frame_predictions: {len(output.frame_predictions)}")
    print(f"grouped_segments: {len(output.segments)}")
    print(f"output_json: {output_path}")
    if raw_output_path is not None:
        print(f"raw_debug_json: {raw_output_path}")
    print("segments:")
    for segment in output.segments[:20]:
        print(
            f"  {segment.step_index}: {segment.semantic.description} | "
            f"{segment.symbolic_action.label} | "
            f"frames {segment.frame_start_index}-{segment.frame_end_index}"
        )
    if len(output.segments) > 20:
        print(f"  ... {len(output.segments) - 20} more segments omitted")


def main() -> int:
    """CLI entry point."""
    args = parse_args()
    run_demo(
        image_path=args.image_path,
        video_path=args.video_path,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        config_path=args.config,
        instruction=args.instruction,
        task_name=args.task_name,
        tags=args.tag,
        skip_image=args.skip_image,
        skip_video=args.skip_video,
        image_episode_id=args.image_episode_id,
        video_episode_id=args.video_episode_id,
        zip_outputs=args.zip_outputs,
        zip_name=args.zip_name,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
