#!/usr/bin/env python3
"""
run_on_media.py — Batch VLA inference on video or image directory.

Takes media input, runs a VLA backend per frame, writes structured
actions JSON to a unique output file path, and prints that path.

This is SaaS output — the JSON is consumed downstream by DaaS.
There is NO live execution, no screenshot capture, no pyautogui here.

Usage examples:
  # Demo with mock backend (no GPU needed):
  python run_on_media.py --image_dir ./sample_images --task "open the file menu" --backend mock

  # Video input:
  python run_on_media.py --video_path ./recording.mp4 --task "click on Settings" --backend mock

  # Experimental SmolVLA (requires lerobot + GPU):
  python run_on_media.py --image_dir ./frames --task "drag file to folder" \\
    --backend smolvla --model_id lerobot/smolvla_base
"""

import argparse
import logging
import sys
from typing import Optional

from media_io.media_loader import load_media
from media_io.action_writer import generate_run_id, build_output_payload, save_actions
from decoder import decode
from validator import validate_actions

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_policy(backend: str, model_id: str):
    """
    Load the appropriate policy backend.
    Raises on unsupported or unimplemented backends.
    """
    if backend == "mock":
        from policies.mock_policy import MockPolicy
        return MockPolicy(model_id=model_id or "mock")

    elif backend == "smolvla":
        from policies.smolvla_adapter import SmolVLAAdapter
        return SmolVLAAdapter(model_id=model_id or "lerobot/smolvla_base")

    elif backend == "pi0":
        from policies.pi0_adapter import PI0Adapter
        return PI0Adapter(model_id=model_id or "lerobot/pi0")

    elif backend == "openvla":
        from policies.openvla_adapter import OpenVLAAdapter
        return OpenVLAAdapter(model_id=model_id or "openvla/openvla-7b")

    elif backend == "gr00t":
        from policies.gr00t_adapter import GR00TAdapter
        return GR00TAdapter(model_id=model_id or "nvidia/GR00T-N1-2B")

    else:
        raise ValueError(
            f"Unknown backend: '{backend}'. "
            f"Choose from: mock, smolvla, openvla, gr00t"
        )


def run_inference(
    video_path: Optional[str],
    image_dir: Optional[str],
    task: str,
    output_dir: str,
    backend: str,
    model_id: str,
) -> str:
    """
    Core inference loop.

    1. Load frames from media
    2. Load policy backend
    3. Run inference per frame
    4. Decode raw output → structured action
    5. Validate
    6. Save to unique JSON file
    7. Return output path
    """

    # --- Load media ---
    logger.info(f"Loading media (video={video_path}, image_dir={image_dir})")
    frames = load_media(video_path=video_path, image_dir=image_dir)
    logger.info(f"Loaded {len(frames)} frames")

    # --- Load policy ---
    logger.info(f"Loading backend: {backend}")
    policy = load_policy(backend, model_id)

    # --- Run inference per frame ---
    action_entries = []

    for frame_index, timestamp, pil_image in frames:
        logger.debug(f"Processing frame {frame_index} (t={timestamp:.3f}s)")

        try:
            raw_output = policy.predict(pil_image, task, frame_index)
            action = decode(backend, raw_output)
            action_dict = action.to_dict()
        except Exception as e:
            # Don't let one bad frame crash the whole run — log and use WAIT
            logger.warning(f"Frame {frame_index} inference failed: {e} — inserting WAIT")
            action_dict = {"action_type": "WAIT", "seconds": 1.0}

        action_entries.append({
            "frame_index": frame_index,
            "timestamp": round(timestamp, 4),
            "action": action_dict,
        })

    # --- Validate ---
    validate_actions(action_entries)

    # --- Save ---
    run_id = generate_run_id()
    resolved_model_id = model_id or policy.model_id

    payload = build_output_payload(
        run_id=run_id,
        video_path=video_path,
        image_dir=image_dir,
        task=task,
        backend=backend,
        model_id=resolved_model_id,
        actions=action_entries,
    )

    output_path = save_actions(
        payload=payload,
        output_dir=output_dir,
        run_id=run_id,
    )

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Batch VLA inference: media in → actions JSON out",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Mock backend (always works, no GPU needed):
  python run_on_media.py --image_dir ./sample_images --task "open file menu" --backend mock

  # Video input:
  python run_on_media.py --video_path ./screen_recording.mp4 --task "click settings" --backend mock

  # SmolVLA (experimental, requires lerobot):
  python run_on_media.py --image_dir ./frames --task "drag file" --backend smolvla
        """
    )

    # Input — exactly one required
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--video_path",
        type=str,
        help="Path to input video file (.mp4, .avi, etc.)"
    )
    input_group.add_argument(
        "--image_dir",
        type=str,
        help="Path to directory of sorted images (.jpg, .png, etc.)"
    )

    # Task instruction
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help='Natural language instruction, e.g. "open the file menu"'
    )

    # Output
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Directory to write the actions JSON file (default: outputs/)"
    )

    # Backend
    parser.add_argument(
        "--backend",
        type=str,
        choices=["mock", "smolvla", "pi0", "openvla", "gr00t"],
        default="mock",
        help="VLA backend to use (default: mock)"
    )

    # Model ID
    parser.add_argument(
        "--model_id",
        type=str,
        default="",
        help="HuggingFace model ID or local path (optional, uses backend default if omitted)"
    )

    # Verbosity
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug-level logging"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        output_path = run_inference(
            video_path=args.video_path,
            image_dir=args.image_dir,
            task=args.task,
            output_dir=args.output_dir,
            backend=args.backend,
            model_id=args.model_id,
        )
        # This is the primary output — print the path so DaaS can consume it
        print(output_path)
        sys.exit(0)

    except Exception as e:
        logger.error(f"Inference failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
