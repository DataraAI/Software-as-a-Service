"""Run one image, video, or frame-sequence payload through RoboTaskManipulator."""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from robotask_manipulator.config import load_settings
from robotask_manipulator.main import RoboTaskManipulatorApp
from robotask_manipulator.utils.io import load_json_file
from robotask_manipulator.utils.logging import get_logger


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(ROOT / "data" / "sample_inputs" / "sample_workflow_episode_001.json"))
    parser.add_argument("--output-dir", default=str(ROOT / "data" / "outputs"))
    parser.add_argument("--config", default=None, help="Optional YAML settings file.")
    parser.add_argument("--semantic-model", default=None, help="Override the semantic VLM model id.")
    parser.add_argument("--semantic-model-path", default=None, help="Optional local semantic VLM directory.")
    parser.add_argument("--semantic-backend", default=None, help="Override the semantic backend name.")
    parser.add_argument("--semantic-offline", action="store_true", help="Use local files only for the semantic VLM.")
    parser.add_argument("--action-backend", default=None, help="Choose pi0, openvla, or none.")
    parser.add_argument("--model-id", default=None, help="Override the pi0 model id when pi0 is enabled.")
    parser.add_argument("--checkpoint", default=None, help="Optional local pi0 checkpoint directory.")
    parser.add_argument("--device", default=None, help="Override the pi0 device.")
    parser.add_argument("--dtype", default=None, help="Override the pi0 dtype.")
    parser.add_argument("--offline", action="store_true", help="Use local files only for pi0.")
    return parser.parse_args()


def run(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    config_path: str | Path | None = None,
    semantic_model: str | None = None,
    semantic_model_path: str | None = None,
    semantic_backend: str | None = None,
    semantic_offline: bool | None = None,
    action_backend: str | None = None,
    model_id: str | None = None,
    checkpoint: str | None = None,
    device: str | None = None,
    dtype: str | None = None,
    offline: bool | None = None,
) -> tuple[Path, Path | None, object]:
    """Run the full product pipeline for one payload."""
    logger = get_logger("robotask_manipulator.run_single_inference")
    settings = load_settings(config_path)
    settings = replace(
        settings,
        semantic=replace(
            settings.semantic,
            model_id=semantic_model or settings.semantic.model_id,
            local_model_path=semantic_model_path or settings.semantic.local_model_path,
            backend=semantic_backend or settings.semantic.backend,
            offline=semantic_offline if semantic_offline is not None else settings.semantic.offline,
        ),
        action_backend=replace(
            settings.action_backend,
            backend=action_backend or settings.action_backend.backend,
            model_id=model_id or settings.action_backend.model_id,
            checkpoint_path=checkpoint or settings.action_backend.checkpoint_path,
            device=device or settings.action_backend.device,
            dtype=dtype or settings.action_backend.dtype,
            offline=offline if offline is not None else settings.action_backend.offline,
        ),
    )

    app = RoboTaskManipulatorApp(settings)
    input_path = Path(input_path)
    payload = load_json_file(input_path)
    output = app.run_payload(payload, input_path.parent)
    output_path, raw_output_path = app.export_episode(output, output_dir)

    logger.info(
        "Processed episode=%s frame_predictions=%s segments=%s labels=%s output=%s",
        output.episode_id,
        len(output.frame_predictions),
        len(output.segments),
        [str(segment.symbolic_action.label) for segment in output.segments],
        output_path,
    )
    return output_path, raw_output_path, output


def main() -> int:
    """CLI entry point."""
    args = parse_args()
    run(
        args.input,
        args.output_dir,
        config_path=args.config,
        semantic_model=args.semantic_model,
        semantic_model_path=args.semantic_model_path,
        semantic_backend=args.semantic_backend,
        semantic_offline=args.semantic_offline,
        action_backend=args.action_backend,
        model_id=args.model_id,
        checkpoint=args.checkpoint,
        device=args.device,
        dtype=args.dtype,
        offline=args.offline,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
