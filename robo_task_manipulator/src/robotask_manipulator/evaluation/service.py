"""Benchmark evaluation service."""

from __future__ import annotations

import csv
from pathlib import Path

from robotask_manipulator.schemas import BatchEvaluationReport, BenchmarkEpisode, BenchmarkSet, DatasetManifest, EvaluationSummary, EpisodeOutput
from robotask_manipulator.utils.io import write_json_file


class EvaluationService:
    """Compute simple, useful v1 metrics against benchmark truth."""

    def evaluate_episode(self, output: EpisodeOutput, benchmark: BenchmarkEpisode) -> EvaluationSummary:
        predicted_labels = [segment.symbolic_action.label for segment in output.segments]
        expected_labels = [step.action_label for step in benchmark.expected_steps]
        matched = sum(
            1 for predicted, expected in zip(predicted_labels, expected_labels, strict=False) if predicted == expected
        )
        label_denominator = max(len(expected_labels), len(predicted_labels), 1)
        label_agreement = matched / label_denominator

        ordering_matches = sum(
            1 for index, segment in enumerate(output.segments[: len(benchmark.expected_steps)]) if segment.step_index == index
        )
        ordering_agreement = ordering_matches / max(len(benchmark.expected_steps), 1)
        step_count_difference = len(output.segments) - len(benchmark.expected_steps)

        success_summary = "pass" if label_agreement >= 0.7 and step_count_difference == 0 else "needs_review"
        return EvaluationSummary(
            step_count_difference=step_count_difference,
            step_label_agreement=round(label_agreement, 3),
            ordering_agreement=round(ordering_agreement, 3),
            success_summary=success_summary,
            details={
                "expected_labels": [str(label) for label in expected_labels],
                "predicted_labels": [str(label) for label in predicted_labels],
            },
        )

    def evaluate_batch(self, outputs: list[EpisodeOutput], benchmark_set: BenchmarkSet) -> BatchEvaluationReport:
        benchmark_index = {episode.episode_id: episode for episode in benchmark_set.episodes}
        rows = []
        for output in outputs:
            benchmark = benchmark_index.get(output.episode_id)
            if benchmark is None:
                continue
            summary = self.evaluate_episode(output, benchmark)
            rows.append(
                {
                    "episode_id": output.episode_id,
                    "step_count_difference": summary.step_count_difference,
                    "step_label_agreement": summary.step_label_agreement,
                    "ordering_agreement": summary.ordering_agreement,
                    "success_summary": summary.success_summary,
                }
            )

        episode_count = len(rows)
        avg_label = sum(row["step_label_agreement"] for row in rows) / episode_count if rows else 0.0
        avg_order = sum(row["ordering_agreement"] for row in rows) / episode_count if rows else 0.0
        return BatchEvaluationReport(
            episodes=rows,
            summary={
                "episodes_evaluated": episode_count,
                "average_step_label_agreement": round(avg_label, 3),
                "average_ordering_agreement": round(avg_order, 3),
            },
        )

    def write_report_files(
        self,
        report: BatchEvaluationReport,
        json_path: str | Path,
        csv_path: str | Path,
    ) -> tuple[Path, Path]:
        json_output = write_json_file(json_path, report.model_dump(mode="json"))
        csv_output = Path(csv_path)
        csv_output.parent.mkdir(parents=True, exist_ok=True)
        with csv_output.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "episode_id",
                    "step_count_difference",
                    "step_label_agreement",
                    "ordering_agreement",
                    "success_summary",
                ],
            )
            writer.writeheader()
            writer.writerows(report.episodes)
        return json_output, csv_output


def manifest_to_outputs(manifest: DatasetManifest) -> list[str]:
    """Compatibility helper for small downstream scripts."""
    return [record.episode_output_path for record in manifest.records]
