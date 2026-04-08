"""Batch-process payloads and emit a dataset manifest plus optional evaluation report."""

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
from robotask_manipulator.evaluation import EvaluationService
from robotask_manipulator.main import RoboTaskManipulatorApp
from robotask_manipulator.schemas import BenchmarkSet
from robotask_manipulator.utils.io import list_json_files, load_json_file
from robotask_manipulator.utils.logging import get_logger


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default=str(ROOT / "data" / "sample_inputs"))
    parser.add_argument("--output-dir", default=str(ROOT / "data" / "outputs"))
    parser.add_argument("--config", default=None, help="Optional YAML settings file.")
    parser.add_argument("--semantic-model", default=None, help="Override the semantic VLM model id.")
    parser.add_argument("--semantic-backend", default=None, help="Override the semantic backend name.")
    parser.add_argument("--semantic-offline", action="store_true", help="Use local files only for the semantic VLM.")
    parser.add_argument("--action-backend", default=None, help="Choose pi0, openvla, or none.")
    parser.add_argument("--model-id", default=None, help="Override the pi0 model id when pi0 is enabled.")
    parser.add_argument("--checkpoint", default=None, help="Optional local pi0 checkpoint directory.")
    parser.add_argument("--device", default=None, help="Override the pi0 device.")
    parser.add_argument("--dtype", default=None, help="Override the pi0 dtype.")
    parser.add_argument("--offline", action="store_true", help="Use local files only for pi0.")
    parser.add_argument("--benchmark", default=None, help="Optional benchmark set JSON path.")
    return parser.parse_args()


def run_batch(args: argparse.Namespace) -> tuple[int, object]:
    """Process all payloads and return status plus manifest."""
    logger = get_logger("robotask_manipulator.batch_annotate")
    settings = load_settings(args.config)
    settings = replace(
        settings,
        semantic=replace(
            settings.semantic,
            model_id=args.semantic_model or settings.semantic.model_id,
            backend=args.semantic_backend or settings.semantic.backend,
            offline=args.semantic_offline if args.semantic_offline is not None else settings.semantic.offline,
        ),
        action_backend=replace(
            settings.action_backend,
            backend=args.action_backend or settings.action_backend.backend,
            model_id=args.model_id or settings.action_backend.model_id,
            checkpoint_path=args.checkpoint or settings.action_backend.checkpoint_path,
            device=args.device or settings.action_backend.device,
            dtype=args.dtype or settings.action_backend.dtype,
            offline=args.offline if args.offline is not None else settings.action_backend.offline,
        ),
    )
    app = RoboTaskManipulatorApp(settings)

    input_files = [
        path for path in list_json_files(args.input_dir)
        if not path.name.endswith(".raw.json") and path.name != settings.export.manifest_name
    ]
    outputs = []
    failures = 0

    for input_file in input_files:
        try:
            payload = load_json_file(input_file)
            output = app.run_payload(payload, input_file.parent)
            app.export_episode(output, args.output_dir)
            outputs.append(output)
            logger.info(
                "Processed %s -> segments=%s labels=%s",
                input_file.name,
                len(output.segments),
                [str(segment.symbolic_action.label) for segment in output.segments],
            )
        except Exception as exc:  # noqa: BLE001
            failures += 1
            logger.exception("Failed to process %s: %s", input_file, exc)

    manifest = app.export_manifest(outputs, args.output_dir)

    if args.benchmark and outputs:
        benchmark = BenchmarkSet.model_validate(load_json_file(args.benchmark))
        evaluator = EvaluationService()
        report = evaluator.evaluate_batch(outputs, benchmark)
        evaluator.write_report_files(
            report,
            Path(args.output_dir) / settings.export.evaluation_json_name,
            Path(args.output_dir) / settings.export.evaluation_csv_name,
        )

    return (1 if failures else 0), manifest


def main() -> int:
    """CLI entry point."""
    args = parse_args()
    status, _manifest = run_batch(args)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
