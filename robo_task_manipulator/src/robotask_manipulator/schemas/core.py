"""Core product schemas."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class MediaType(str, Enum):
    IMAGE = "image"
    VIDEO = "video"
    FRAME_SEQUENCE = "frame_sequence"


class ActionLabel(str, Enum):
    PICK = "pick"
    PLACE = "place"
    ALIGN = "align"
    INSERT = "insert"
    FASTEN = "fasten"
    PUSH = "push"
    PULL = "pull"
    HOLD = "hold"
    INSPECT = "inspect"
    REGRASP = "regrasp"
    RETRY = "retry"
    RELEASE = "release"
    WAIT = "wait"
    UNKNOWN = "unknown"


class ContextTagName(str, Enum):
    UNSTABLE_GRASP = "unstable_grasp"
    MISALIGNMENT = "misalignment"
    BLOCKED_INSERTION = "blocked_insertion"
    RETRY_REQUIRED = "retry_required"
    DROPPED_OBJECT = "dropped_object"
    MISSED_TARGET = "missed_target"
    OCCLUSION = "occlusion"
    UNKNOWN_FAILURE = "unknown_failure"


class TaskEdgeType(str, Enum):
    NEXT = "next"
    RETRY = "retry"
    TERMINAL = "terminal"
    ABORT = "abort"


class DatasetSplit(str, Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"
    UNSPECIFIED = "unspecified"


class MediaMetadata(BaseModel):
    """Describes the input media source."""

    model_config = ConfigDict(use_enum_values=True)

    media_type: MediaType
    source_ref: str
    width: int | None = Field(default=None, ge=1)
    height: int | None = Field(default=None, ge=1)
    fps: float | None = Field(default=None, ge=0.0)
    duration_s: float | None = Field(default=None, ge=0.0)
    frame_count: int | None = Field(default=None, ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class FrameObservation(BaseModel):
    """One ordered observation frame."""

    frame_id: str
    asset_ref: str
    frame_index: int = Field(ge=0)
    timestamp_s: float | None = Field(default=None, ge=0.0)
    state: list[float] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class EpisodeInput(BaseModel):
    """Canonical ingested episode input."""

    episode_id: str
    task_name: str
    instruction: str
    media_metadata: MediaMetadata
    frames: list[FrameObservation]
    metadata: dict[str, Any] = Field(default_factory=dict)
    benchmark: dict[str, Any] | None = None

    @field_validator("frames")
    @classmethod
    def validate_non_empty_frames(cls, value: list[FrameObservation]) -> list[FrameObservation]:
        if not value:
            raise ValueError("frames must not be empty")
        return sorted(value, key=lambda item: (item.frame_index, item.timestamp_s or 0.0))


class SemanticStep(BaseModel):
    """Structured semantic interpretation for a segment."""

    description: str
    task_intent: str | None = None
    objects_involved: list[str] = Field(default_factory=list)
    object_source: str | None = None
    object_target: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: dict[str, Any] = Field(default_factory=dict)


class ActionProposal(BaseModel):
    """Optional robot-oriented action backend proposal."""

    backend: str
    selected_action: list[float] | None = None
    action_chunk: list[list[float]] = Field(default_factory=list)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)
    raw_model_output: dict[str, Any] = Field(default_factory=dict)


class SymbolicActionLabel(BaseModel):
    """Conservative symbolic action output."""

    model_config = ConfigDict(use_enum_values=True)

    label: ActionLabel
    confidence: float = Field(ge=0.0, le=1.0)
    source: str
    evidence: dict[str, Any] = Field(default_factory=dict)


class ContextTag(BaseModel):
    """Lightweight context/failure tag."""

    model_config = ConfigDict(use_enum_values=True)

    name: ContextTagName
    confidence: float = Field(ge=0.0, le=1.0)
    source: str
    evidence: dict[str, Any] = Field(default_factory=dict)


class SegmentAnnotation(BaseModel):
    """One ordered task segment."""

    segment_id: str
    episode_id: str
    step_index: int = Field(ge=0)
    observation_refs: list[str] = Field(default_factory=list)
    representative_frame_ref: str
    frame_start_index: int = Field(ge=0)
    frame_end_index: int = Field(ge=0)
    timestamp_start_s: float | None = Field(default=None, ge=0.0)
    timestamp_end_s: float | None = Field(default=None, ge=0.0)
    segmentation_confidence: float = Field(ge=0.0, le=1.0)
    semantic: SemanticStep
    symbolic_action: SymbolicActionLabel
    context_tags: list[ContextTag] = Field(default_factory=list)
    success: bool | None = None
    next_step_refs: list[str] = Field(default_factory=list)
    action_proposal: ActionProposal | None = None
    raw_outputs: dict[str, Any] = Field(default_factory=dict)


class TaskNode(BaseModel):
    node_id: str
    segment_id: str
    step_index: int = Field(ge=0)
    terminal: bool = False


class TaskEdge(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    source_node_id: str
    target_node_id: str
    edge_type: TaskEdgeType
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    evidence: dict[str, Any] = Field(default_factory=dict)


class TaskGraph(BaseModel):
    nodes: list[TaskNode] = Field(default_factory=list)
    edges: list[TaskEdge] = Field(default_factory=list)
    terminal_conditions: list[str] = Field(default_factory=list)


class IsaacSimStep(BaseModel):
    """Simulation-ready Franka Panda step."""

    step_index: int = Field(ge=0)
    segment_id: str
    primitive: str
    description: str
    target_object: str | None = None
    source_object: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    status: str = "planned"
    action_proposal: ActionProposal | None = None
    tags: list[str] = Field(default_factory=list)


class IsaacSimExport(BaseModel):
    """Isaac Sim 5.1 / Franka Panda task plan."""

    simulator: str = "isaac_sim_5.1"
    robot: str = "franka_panda"
    episode_id: str
    task_name: str
    steps: list[IsaacSimStep]
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvaluationSummary(BaseModel):
    """Optional per-episode evaluation output."""

    step_count_difference: int
    step_label_agreement: float = Field(ge=0.0, le=1.0)
    ordering_agreement: float = Field(ge=0.0, le=1.0)
    success_summary: str
    details: dict[str, Any] = Field(default_factory=dict)


class EpisodeOutput(BaseModel):
    """Final per-episode product output."""

    episode_id: str
    task_name: str
    instruction: str
    input_metadata: MediaMetadata
    segments: list[SegmentAnnotation]
    task_graph: TaskGraph
    simulation_export: IsaacSimExport
    evaluation: EvaluationSummary | None = None
    batch_metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("segments")
    @classmethod
    def validate_segments(cls, value: list[SegmentAnnotation]) -> list[SegmentAnnotation]:
        if not value:
            raise ValueError("segments must not be empty")
        return value

    @model_validator(mode="after")
    def validate_links(self) -> "EpisodeOutput":
        segment_ids = {segment.segment_id for segment in self.segments}
        for segment in self.segments:
            for next_ref in segment.next_step_refs:
                if next_ref not in segment_ids:
                    raise ValueError(f"Unknown next_step_ref: {next_ref}")
        return self


class DatasetManifestRecord(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    episode_id: str
    task_name: str
    split: DatasetSplit = DatasetSplit.UNSPECIFIED
    episode_output_path: str
    raw_output_path: str | None = None
    num_segments: int = Field(ge=0)
    action_labels: list[ActionLabel] = Field(default_factory=list)
    success: bool | None = None


class DatasetManifest(BaseModel):
    manifest_version: str = "1.0"
    records: list[DatasetManifestRecord]
    summary: dict[str, Any] = Field(default_factory=dict)
