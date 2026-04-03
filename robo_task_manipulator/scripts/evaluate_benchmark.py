"""Evaluate exported episodes against a benchmark set."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from robotask_manipulator.config import load_settings
from robotask_manipulator.evaluation import EvaluationService
from robotask_manipulator.schemas import BenchmarkSet, EpisodeOutput
from robotask_manipulator.utils.io import list_json_files, load_json_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-dir", required=True)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--config", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    settings = load_settings(args.config)
    outputs = []
    for path in list_json_files(args.predictions_dir):
        if path.name.endswith(".raw.json") or path.name == settings.export.manifest_name:
            continue
        outputs.append(EpisodeOutput.model_validate(load_json_file(path)))
    benchmark = BenchmarkSet.model_validate(load_json_file(args.benchmark))
    evaluator = EvaluationService()
    report = evaluator.evaluate_batch(outputs, benchmark)
    evaluator.write_report_files(
        report,
        Path(args.predictions_dir) / settings.export.evaluation_json_name,
        Path(args.predictions_dir) / settings.export.evaluation_csv_name,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
