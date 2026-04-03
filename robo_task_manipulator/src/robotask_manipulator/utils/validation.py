"""Validation helpers for RoboTaskManipulator v1."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import ValidationError

from robotask_manipulator.schemas import (
    ActionLabel,
    BenchmarkEpisode,
    DatasetManifest,
    EpisodeInput,
    EpisodeOutput,
)


class RoboTaskManipulatorError(Exception):
    """Base project exception."""


class InputAssetMissingError(RoboTaskManipulatorError):
    """Raised when an asset path does not exist."""


class UnsupportedActionLabelError(RoboTaskManipulatorError):
    """Raised when a predicted action label is outside the controlled vocabulary."""


class AnnotationSchemaError(RoboTaskManipulatorError):
    """Raised when schema validation fails."""


class ModelLoadError(RoboTaskManipulatorError):
    """Raised when a configured model cannot be loaded."""


class InvalidInputError(RoboTaskManipulatorError):
    """Raised when the caller provides an unsupported or incomplete input."""


def ensure_asset_exists(path: str | Path) -> Path:
    """Ensure an input asset exists."""
    asset_path = Path(path)
    if not asset_path.is_absolute():
        asset_path = Path.cwd() / asset_path
    asset_path = asset_path.resolve()
    if not asset_path.exists():
        raise InputAssetMissingError(f"Missing input asset: {asset_path}")
    return asset_path


def validate_supported_action_label(label: str | ActionLabel) -> ActionLabel:
    """Validate a candidate action label against the controlled vocabulary."""
    try:
        return label if isinstance(label, ActionLabel) else ActionLabel(label)
    except ValueError as exc:
        raise UnsupportedActionLabelError(f"Unsupported action label: {label}") from exc


def validate_episode_input(episode: EpisodeInput) -> EpisodeInput:
    """Re-validate ingested episode input."""
    try:
        return EpisodeInput.model_validate(episode.model_dump())
    except ValidationError as exc:
        raise AnnotationSchemaError(str(exc)) from exc


def validate_episode_output(output: EpisodeOutput) -> EpisodeOutput:
    """Re-validate final episode output."""
    try:
        return EpisodeOutput.model_validate(output.model_dump())
    except ValidationError as exc:
        raise AnnotationSchemaError(str(exc)) from exc


def validate_dataset_manifest(manifest: DatasetManifest) -> DatasetManifest:
    """Validate a dataset manifest object."""
    try:
        return DatasetManifest.model_validate(manifest.model_dump())
    except ValidationError as exc:
        raise AnnotationSchemaError(str(exc)) from exc


def validate_benchmark_episode(payload: dict[str, Any]) -> BenchmarkEpisode:
    """Validate one benchmark episode payload."""
    try:
        return BenchmarkEpisode.model_validate(payload)
    except ValidationError as exc:
        raise AnnotationSchemaError(str(exc)) from exc
