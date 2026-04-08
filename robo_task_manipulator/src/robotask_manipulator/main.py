"""Practical end-to-end RoboTaskManipulator v1 pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from robotask_manipulator.action_backend import create_action_backend
from robotask_manipulator.config import AppSettings, load_settings
from robotask_manipulator.context import ContextTagger
from robotask_manipulator.evaluation import EvaluationService
from robotask_manipulator.export import JsonArtifactExporter
from robotask_manipulator.graph import SequenceGraphBuilder
from robotask_manipulator.ingestion import MediaIngestor
from robotask_manipulator.schemas import BenchmarkEpisode, DatasetManifest, EpisodeInput, EpisodeOutput
from robotask_manipulator.segmentation import Segmenter
from robotask_manipulator.simulation import IsaacSimFrankaExporter
from robotask_manipulator.task_understanding import (
    SymbolicActionLabeler,
    TaskUnderstandingService,
    build_task_understanding_backend,
)
from robotask_manipulator.utils.io import ensure_directory, load_json_file
from robotask_manipulator.utils.validation import (
    validate_benchmark_episode,
    validate_dataset_manifest,
    validate_episode_input,
    validate_episode_output,
)


class RoboTaskManipulatorApp:
    """Coordinates the end-to-end v1 product pipeline."""

    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings
        self.ingestor = MediaIngestor(settings.ingestion)
        self.segmenter = Segmenter(settings.segmentation)
        self.task_understanding_service = TaskUnderstandingService(
            build_task_understanding_backend(settings.semantic)
        )
        self.labeler = SymbolicActionLabeler()
        self.action_backend = create_action_backend(settings.action_backend)
        self.context_tagger = ContextTagger()
        self.graph_builder = SequenceGraphBuilder()
        self.sim_exporter = IsaacSimFrankaExporter()
        self.artifact_exporter = JsonArtifactExporter()
        self.evaluator = EvaluationService()

    def run_payload(self, payload: dict[str, Any], base_dir: str | Path) -> EpisodeOutput:
        episode = validate_episode_input(self.ingestor.from_payload(payload, base_dir))
        segments = self.segmenter.segment(episode)
        segments = self.task_understanding_service.annotate(episode, segments)

        self.action_backend.load()
        for segment in segments:
            segment.raw_outputs["state"] = self._segment_state(episode, segment)
            segment.action_proposal = self.action_backend.propose(episode, segment)
            segment.symbolic_action = self.labeler.label(segment)
            segment.context_tags = self.context_tagger.annotate(segment)
            segment.success = self._infer_success(segment)

        task_graph = self.graph_builder.build(segments)
        sim_payload = self.sim_exporter.build(episode.episode_id, episode.task_name, segments)
        evaluation = self._evaluate_episode_if_available(episode, segments, task_graph, sim_payload)

        output = EpisodeOutput(
            episode_id=episode.episode_id,
            task_name=episode.task_name,
            instruction=episode.instruction,
            input_metadata=episode.media_metadata,
            segments=segments,
            task_graph=task_graph,
            simulation_export=sim_payload,
            evaluation=evaluation,
            batch_metadata={"frame_count": len(episode.frames)},
        )
        return validate_episode_output(output)

    def export_episode(self, output: EpisodeOutput, output_dir: str | Path) -> tuple[Path, Path | None]:
        directory = ensure_directory(output_dir)
        episode_path = directory / f"{output.episode_id}.json"
        raw_path = directory / f"{output.episode_id}.raw.json" if self.settings.export.include_raw_debug else None
        self.artifact_exporter.write_episode(output, episode_path)
        if raw_path is not None:
            self.artifact_exporter.write_raw_debug(output, raw_path)
        return episode_path, raw_path

    def export_manifest(self, outputs: list[EpisodeOutput], output_dir: str | Path) -> Path:
        directory = ensure_directory(output_dir)
        records = []
        for output in outputs:
            episode_path = directory / f"{output.episode_id}.json"
            raw_path = directory / f"{output.episode_id}.raw.json" if self.settings.export.include_raw_debug else None
            records.append(
                self.artifact_exporter.build_manifest_record(
                    output,
                    episode_path,
                    raw_path,
                    split=self._suggest_split(output.episode_id),
                )
            )
        manifest = validate_dataset_manifest(
            DatasetManifest(
                records=records,
                summary={
                    "episodes": len(records),
                    "splits": self._split_summary(records),
                },
            )
        )
        path = directory / self.settings.export.manifest_name
        self.artifact_exporter.write_manifest(manifest, path)
        return path

    def _segment_state(self, episode: EpisodeInput, segment) -> list[float] | None:
        frame_index = segment.frame_end_index
        frame_map = {frame.frame_index: frame for frame in episode.frames}
        frame = frame_map.get(frame_index)
        return frame.state if frame is not None else None

    def _infer_success(self, segment) -> bool | None:
        failure_names = {str(tag.name) for tag in segment.context_tags}
        if "unknown_failure" in failure_names:
            return None
        if any(name in failure_names for name in {"blocked_insertion", "dropped_object", "missed_target"}):
            return False
        return True

    def _evaluate_episode_if_available(self, episode: EpisodeInput, segments, task_graph, sim_payload):
        benchmark = self._benchmark_from_episode(episode)
        if benchmark is None:
            return None
        draft_output = EpisodeOutput(
            episode_id=episode.episode_id,
            task_name=episode.task_name,
            instruction=episode.instruction,
            input_metadata=episode.media_metadata,
            segments=segments,
            task_graph=task_graph,
            simulation_export=sim_payload,
        )
        return self.evaluator.evaluate_episode(draft_output, benchmark)

    def _benchmark_from_episode(self, episode: EpisodeInput) -> BenchmarkEpisode | None:
        if not episode.benchmark:
            return None
        payload = dict(episode.benchmark)
        payload.setdefault("episode_id", episode.episode_id)
        return validate_benchmark_episode(payload)

    def _suggest_split(self, episode_id: str) -> str:
        bucket = sum(ord(char) for char in episode_id) % 10
        if bucket == 0:
            return "test"
        if bucket == 1:
            return "val"
        return "train"

    def _split_summary(self, records) -> dict[str, int]:
        summary: dict[str, int] = {}
        for record in records:
            split = str(record.split)
            summary[split] = summary.get(split, 0) + 1
        return summary


def build_app(config_path: str | Path | None = None) -> RoboTaskManipulatorApp:
    """Build the app from settings."""
    return RoboTaskManipulatorApp(load_settings(config_path))


def run_payload_file(input_path: str | Path, config_path: str | Path | None = None) -> EpisodeOutput:
    """Small helper used by scripts and tests."""
    path = Path(input_path)
    payload = load_json_file(path)
    app = build_app(config_path)
    return app.run_payload(payload, path.parent)
