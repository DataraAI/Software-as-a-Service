#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time

from agent.capture import capture_screenshot, get_screen_size
from agent.decoder import decode_action_vector
from agent.executor import execute_action
from agent.policy import PolicyConfig, SmolVLADesktopPolicy
from agent.validator import validate_action


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a local SmolVLA-based desktop VLA loop."
    )
    parser.add_argument(
        "--model_id",
        type=str,
        required=True,
        help="Hugging Face model id or local path to your fine-tuned SmolVLA checkpoint.",
    )
    parser.add_argument(
        "--instruction",
        type=str,
        required=True,
        help="Natural language instruction for the agent.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Inference device.",
    )
    parser.add_argument(
        "--state_dim",
        type=int,
        default=8,
        help="Optional state vector dimension.",
    )
    parser.add_argument(
        "--image_key",
        type=str,
        default="observation.images.main",
        help="Image feature key expected by your fine-tuned checkpoint.",
    )
    parser.add_argument(
        "--state_key",
        type=str,
        default="observation.state",
        help="State feature key expected by your fine-tuned checkpoint.",
    )
    parser.add_argument(
        "--task_key",
        type=str,
        default="task",
        help="Language/task feature key expected by your fine-tuned checkpoint.",
    )
    parser.add_argument(
        "--loop_delay",
        type=float,
        default=0.25,
        help="Delay between loop iterations in seconds.",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=20,
        help="Maximum number of control loop iterations.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Do not execute actions locally; only print them.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = PolicyConfig(
        model_id=args.model_id,
        device=args.device,
        state_dim=args.state_dim,
        image_key=args.image_key,
        state_key=args.state_key,
        task_key=args.task_key,
    )

    policy = SmolVLADesktopPolicy(cfg)
    screen_width, screen_height = get_screen_size()

    print(f"Screen size: {screen_width}x{screen_height}")
    print(f"Running SmolVLA desktop loop with model: {args.model_id}")

    state = [0.0] * args.state_dim

    for step in range(args.max_steps):
        screenshot = capture_screenshot()

        action_vector = policy.predict_action_vector(
            image=screenshot,
            instruction=args.instruction,
            state=state,
        )

        action = decode_action_vector(
            action_vector=action_vector,
            screen_width=screen_width,
            screen_height=screen_height,
        )

        is_valid = validate_action(
            action=action,
            screen_width=screen_width,
            screen_height=screen_height,
        )

        if not is_valid:
            print(f"[step {step}] invalid action skipped: {action}")
            time.sleep(args.loop_delay)
            continue

        print(f"[step {step}] {action.to_executor_string()}")
        execute_action(action, dry_run=args.dry_run)

        time.sleep(args.loop_delay)


if __name__ == "__main__":
    main()