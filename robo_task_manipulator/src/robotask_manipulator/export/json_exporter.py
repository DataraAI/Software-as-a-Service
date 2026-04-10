"""Episode and manifest export."""

from __future__ import annotations

from pathlib import Path

from robotask_manipulator.schemas import DatasetManifest, DatasetManifestRecord, EpisodeOutput
from robotask_manipulator.utils.io import write_json_file


class JsonArtifactExporter:
    """Write product outputs in a debug-friendly, evaluation-friendly shape."""

    def write_episode(self, output: EpisodeOutput, path: str | Path) -> Path:
        return write_json_file(path, output.model_dump(mode="json"))

    def write_raw_debug(self, output: EpisodeOutput, path: str | Path) -> Path:
        debug_payload = {
            "episode_id": output.episode_id,
            "frame_predictions": [
                {
                    "frame_id": frame.frame_id,
                    "frame_index": frame.frame_index,
                    "raw_outputs": frame.raw_outputs,
                }
                for frame in output.frame_predictions
            ],
            "segments": [
                {
                    "segment_id": segment.segment_id,
                    "raw_outputs": segment.raw_outputs,
                    "action_proposal": segment.action_proposal.model_dump(mode="json") if segment.action_proposal else None,
                }
                for segment in output.segments
            ],
        }
        return write_json_file(path, debug_payload)

    def write_manifest(self, manifest: DatasetManifest, path: str | Path) -> Path:
        return write_json_file(path, manifest.model_dump(mode="json"))

    def build_manifest_record(
        self,
        output: EpisodeOutput,
        episode_output_path: str | Path,
        raw_output_path: str | Path | None = None,
        split: str = "unspecified",
    ) -> DatasetManifestRecord:
        return DatasetManifestRecord(
            episode_id=output.episode_id,
            task_name=output.task_name,
            split=split,
            episode_output_path=str(Path(episode_output_path)),
            raw_output_path=str(Path(raw_output_path)) if raw_output_path else None,
            num_frame_predictions=len(output.frame_predictions),
            num_segments=len(output.segments),
            action_labels=[segment.symbolic_action.label for segment in output.segments],
            success=all(segment.success is not False for segment in output.segments),
        )
