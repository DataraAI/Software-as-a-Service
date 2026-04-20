#!/usr/bin/env python3
"""
Run LingBot-VLA policy on an input video file.

Usage:
    export QWEN25_PATH=/path/to/Qwen2.5-VL-3B-Instruct
    python -m deploy.run_on_frames \
        --model_path /path/to/lingbot-vla-checkpoint \
        --video_path /path/to/video.mp4 \
        --task "open the microwave" \
        --output actions.npy
"""

import argparse
import os
import sys
from pathlib import Path

# Local vendored deps: ~/packages (lerobot, lingbot_vla), ~/packages/lingbot_vla for lingbotvla.
_packages = Path.home() / "packages"
_lerobot_root = _packages / "lerobot"
_lingbot_vla_root = _packages / "lingbot_vla"
if _packages.is_dir() and str(_packages) not in sys.path:
    sys.path.insert(0, str(_packages))
if _lerobot_root.is_dir() and str(_lerobot_root) not in sys.path:
    sys.path.insert(0, str(_lerobot_root))
if _lingbot_vla_root.is_dir() and str(_lingbot_vla_root) not in sys.path:
    sys.path.insert(0, str(_lingbot_vla_root))

import numpy as np
from tqdm import tqdm

try:
    import imageio
except ImportError:
    raise ImportError("Reading video requires imageio. Install with: pip install imageio[ffmpeg]")


def load_frames_from_video(video_path: str) -> list[np.ndarray]:
    """Load all frames from a video file as RGB numpy arrays."""
    path = Path(video_path)
    if not path.is_file():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    reader = imageio.get_reader(video_path)
    frames = []
    for frame in reader:
        # imageio typically returns RGB; ensure uint8 (H, W, 3)
        if frame.ndim == 2:
            frame = np.stack([frame] * 3, axis=-1)
        elif frame.shape[-1] == 4:
            frame = frame[..., :3]
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        frames.append(frame)
    reader.close()

    if not frames:
        raise ValueError(f"No frames read from video: {video_path}")
    return frames


def build_observation(
    image: np.ndarray,
    task: str,
    state_dim: int = 8,
) -> dict:
    """Build observation dict for the policy. Uses the same image for all three camera views."""
    # Policy expects (H, W, 3) uint8 for each view
    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            image = (np.clip(image, 0, 1) * 255).astype(np.uint8)
        else:
            image = np.clip(image, 0, 255).astype(np.uint8)

    obs = {
        "observation.images.cam_high": image.copy(),
        "observation.images.cam_left_wrist": image.copy(),
        "observation.images.cam_right_wrist": image.copy(),
        "observation.state": np.zeros(state_dim, dtype=np.float32),
        "task": task,
    }
    return obs


def main():
    parser = argparse.ArgumentParser(
        description="Run LingBot-VLA on an input video file."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to LingBot-VLA checkpoint (directory containing .safetensors and config).",
    )
    parser.add_argument(
        "--video_path",
        type=str,
        required=True,
        help="Path to input video file (e.g. .mp4, .avi, .mov).",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="do the task",
        help="Task instruction for the policy (e.g. 'open the microwave').",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for predicted actions (.npy). If not set, actions are printed to stdout.",
    )
    parser.add_argument(
        "--use_length",
        type=int,
        default=50,
        help="Action chunk length (same as deploy.lingbot_robotwin_policy).",
    )
    parser.add_argument(
        "--state_dim",
        type=int,
        default=8,
        help="Robot state dimension (will be padded to max_state_dim by the model).",
    )
    args = parser.parse_args()

    if os.environ.get("QWEN25_PATH") is None:
        print(
            "Warning: QWEN25_PATH is not set. Set it to the Qwen2.5-VL-3B-Instruct path."
        )

    frames = load_frames_from_video(args.video_path)
    print(f"Loaded {len(frames)} frames from {args.video_path}")

    from packages.lingbot_vla.deploy.lingbot_robotwin_policy import QwenPiServer

    server = QwenPiServer(
        path_to_pi_model=args.model_path,
        use_length=args.use_length,
        chunk_ret=True,
    )

    actions_list = []
    for image in tqdm(frames, desc="Inference"):
        obs = build_observation(image, args.task, state_dim=args.state_dim)
        out = server.infer(obs)
        action = out["action"]  # shape (use_length, action_dim) when chunk_ret=True
        if action is not None:
            # One action per frame: take first in the chunk
            actions_list.append(action[0])

    actions = np.stack(actions_list, axis=0)

    if args.output:
        np.save(args.output, actions)
        print(f"Saved actions with shape {actions.shape} to {args.output}")
    else:
        print("Predicted actions (shape %s):" % (actions.shape,))
        print(actions)


if __name__ == "__main__":
    main()
