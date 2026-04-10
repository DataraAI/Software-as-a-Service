"""Central product settings for RoboTaskManipulator v1."""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml


def _parse_bool(value: str | bool | None, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class IngestionSettings:
    """Settings for photo/video ingestion and frame extraction."""

    video_frame_stride: int = 1
    max_frames: int = 0
    prefer_visual_change: bool = True
    visual_change_threshold: float = 0.18


@dataclass(frozen=True)
class SegmentationSettings:
    """Settings for deterministic segmentation."""

    frames_per_segment: int = 4
    use_visual_change_breaks: bool = True
    visual_change_threshold: float = 0.18


@dataclass(frozen=True)
class SemanticSettings:
    """Settings for semantic step understanding."""

    backend: str = "multimodal_vlm"
    model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    local_model_path: str | None = None
    device: str = "cpu"
    offline: bool = False
    strict: bool = False
    frame_context_radius: int = 1

    @property
    def model_source(self) -> str:
        return self.local_model_path or self.model_id


@dataclass(frozen=True)
class ActionBackendSettings:
    """Settings for optional robot-oriented action proposals."""

    backend: str = "none"
    model_id: str = "lerobot/pi0_base"
    checkpoint_path: str | None = None
    device: str = "cpu"
    dtype: str = "float32"
    offline: bool = False
    revision: str | None = None
    cache_dir: str | None = None
    strict: bool = True

    @property
    def model_source(self) -> str:
        return self.checkpoint_path or self.model_id


@dataclass(frozen=True)
class ExportSettings:
    """Settings for JSON/export output."""

    include_raw_debug: bool = True
    manifest_name: str = "dataset_manifest.json"
    evaluation_json_name: str = "evaluation_report.json"
    evaluation_csv_name: str = "evaluation_report.csv"


@dataclass(frozen=True)
class AppSettings:
    """Top-level product configuration."""

    ingestion: IngestionSettings = IngestionSettings()
    segmentation: SegmentationSettings = SegmentationSettings()
    semantic: SemanticSettings = SemanticSettings()
    action_backend: ActionBackendSettings = ActionBackendSettings()
    export: ExportSettings = ExportSettings()

    def with_overrides(self, **overrides: Any) -> "AppSettings":
        cleaned = {key: value for key, value in overrides.items() if value is not None}
        return replace(self, **cleaned)


def _load_yaml(config_path: str | Path | None) -> dict[str, Any]:
    if config_path is None:
        return {}
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_settings(config_path: str | Path | None = None) -> AppSettings:
    """Load app settings from env and optionally YAML."""
    payload = _load_yaml(config_path)

    ingestion_cfg = {**payload.get("ingestion", {})}
    segmentation_cfg = {**payload.get("segmentation", {})}
    semantic_cfg = {**payload.get("semantic", {})}
    action_cfg = {**payload.get("action_backend", payload.get("pi0", {}))}
    export_cfg = {**payload.get("export", {})}

    semantic_model_id = os.getenv("RTM_SEMANTIC_MODEL_ID", semantic_cfg.get("model_id", SemanticSettings.model_id))
    semantic_model_path = os.getenv("RTM_SEMANTIC_MODEL_PATH", semantic_cfg.get("local_model_path"))
    semantic_backend = os.getenv("RTM_SEMANTIC_BACKEND", semantic_cfg.get("backend", SemanticSettings.backend))
    action_backend = os.getenv("RTM_ACTION_BACKEND", action_cfg.get("backend", ActionBackendSettings.backend))

    return AppSettings(
        ingestion=IngestionSettings(
            video_frame_stride=int(ingestion_cfg.get("video_frame_stride", IngestionSettings.video_frame_stride)),
            max_frames=int(ingestion_cfg.get("max_frames", IngestionSettings.max_frames)),
            prefer_visual_change=_parse_bool(
                ingestion_cfg.get("prefer_visual_change"),
                default=IngestionSettings.prefer_visual_change,
            ),
            visual_change_threshold=float(
                ingestion_cfg.get("visual_change_threshold", IngestionSettings.visual_change_threshold)
            ),
        ),
        segmentation=SegmentationSettings(
            frames_per_segment=int(segmentation_cfg.get("frames_per_segment", SegmentationSettings.frames_per_segment)),
            use_visual_change_breaks=_parse_bool(
                segmentation_cfg.get("use_visual_change_breaks"),
                default=SegmentationSettings.use_visual_change_breaks,
            ),
            visual_change_threshold=float(
                segmentation_cfg.get("visual_change_threshold", SegmentationSettings.visual_change_threshold)
            ),
        ),
        semantic=SemanticSettings(
            backend=semantic_backend,
            model_id=semantic_model_id,
            local_model_path=semantic_model_path,
            device=os.getenv("RTM_SEMANTIC_DEVICE", semantic_cfg.get("device", SemanticSettings.device)),
            offline=_parse_bool(
                os.getenv("RTM_SEMANTIC_OFFLINE", semantic_cfg.get("offline")),
                default=SemanticSettings.offline,
            ),
            strict=_parse_bool(os.getenv("RTM_SEMANTIC_STRICT", semantic_cfg.get("strict")), default=SemanticSettings.strict),
            frame_context_radius=int(
                os.getenv(
                    "RTM_FRAME_CONTEXT_RADIUS",
                    semantic_cfg.get("frame_context_radius", SemanticSettings.frame_context_radius),
                )
            ),
        ),
        action_backend=ActionBackendSettings(
            backend=action_backend,
            model_id=os.getenv("PI0_MODEL_ID", action_cfg.get("model_id", ActionBackendSettings.model_id)),
            checkpoint_path=os.getenv("PI0_CHECKPOINT_PATH", action_cfg.get("checkpoint_path")),
            device=os.getenv("PI0_DEVICE", action_cfg.get("device", ActionBackendSettings.device)),
            dtype=os.getenv("PI0_DTYPE", action_cfg.get("dtype", ActionBackendSettings.dtype)),
            offline=_parse_bool(os.getenv("PI0_OFFLINE", action_cfg.get("offline")), default=ActionBackendSettings.offline),
            revision=os.getenv("PI0_REVISION", action_cfg.get("revision")),
            cache_dir=os.getenv("PI0_CACHE_DIR", action_cfg.get("cache_dir")),
            strict=_parse_bool(os.getenv("PI0_STRICT", action_cfg.get("strict")), default=ActionBackendSettings.strict),
        ),
        export=ExportSettings(
            include_raw_debug=_parse_bool(export_cfg.get("include_raw_debug"), default=ExportSettings.include_raw_debug),
            manifest_name=export_cfg.get("manifest_name", ExportSettings.manifest_name),
            evaluation_json_name=export_cfg.get("evaluation_json_name", ExportSettings.evaluation_json_name),
            evaluation_csv_name=export_cfg.get("evaluation_csv_name", ExportSettings.evaluation_csv_name),
        ),
    )
