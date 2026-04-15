from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from robotask_manipulator.api.app import create_api_app
from robotask_manipulator.api.models import AnnotationImageRequest
from robotask_manipulator.config import (
    AppSettings,
    AzureBlobSettings,
    SemanticSettings,
    ServiceSettings,
)
from robotask_manipulator.schemas import (
    ActionLabel,
    ContextTag,
    EpisodeOutput,
    FrameAnnotation,
    IsaacSimExport,
    IsaacSimStep,
    MediaMetadata,
    MediaType,
    SegmentAnnotation,
    SemanticStep,
    SymbolicActionLabel,
    TaskEdge,
    TaskEdgeType,
    TaskGraph,
    TaskNode,
)
from robotask_manipulator.service import AnnotationService
from robotask_manipulator.storage import AzureAnnotationStore, AzureBlobSource, AzureBlobStorageError


class FakeDownloadStream:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload

    def readall(self) -> bytes:
        return self.payload


class FakeBlobClient:
    def __init__(self, service_client: "FakeBlobServiceClient", container: str, blob: str) -> None:
        self.service_client = service_client
        self.container = container
        self.blob = blob
        self.url = f"{self.service_client.url}/{self.container}/{self.blob}"

    def download_blob(self) -> FakeDownloadStream:
        key = (self.container, self.blob)
        if key not in self.service_client.download_map:
            raise FileNotFoundError(self.blob)
        return FakeDownloadStream(self.service_client.download_map[key])

    def upload_blob(self, data: bytes, **kwargs) -> None:
        self.service_client.uploads.append(
            {
                "container": self.container,
                "blob": self.blob,
                "data": data,
                "kwargs": kwargs,
            }
        )


class FakeBlobServiceClient:
    def __init__(self, download_map: dict[tuple[str, str], bytes] | None = None) -> None:
        self.download_map = download_map or {}
        self.uploads: list[dict[str, object]] = []
        self.created_containers: list[str] = []
        self.url = "https://example.blob.core.windows.net"

    def get_blob_client(self, *, container: str, blob: str) -> FakeBlobClient:
        return FakeBlobClient(self, container, blob)

    def create_container(self, container: str) -> None:
        self.created_containers.append(container)


def _build_episode_output(task_name: str, instruction: str, asset_ref: str) -> EpisodeOutput:
    semantic = SemanticStep(
        description="hold cable near port",
        task_intent=instruction,
        objects_involved=["cable", "port"],
        object_source="cable",
        object_target="port",
        confidence=0.91,
    )
    symbolic = SymbolicActionLabel(
        label=ActionLabel.HOLD,
        confidence=0.88,
        source="semantic_labeler",
    )
    frame_prediction = FrameAnnotation(
        frame_id="frame-000",
        episode_id="episode-001",
        frame_index=0,
        asset_ref=asset_ref,
        timestamp_s=0.0,
        context_frame_refs=[asset_ref],
        semantic=semantic,
        symbolic_action=symbolic,
        context_tags=[ContextTag(name="occlusion", confidence=0.12, source="heuristic")],
        success=True,
        raw_outputs={},
    )
    segment = SegmentAnnotation(
        segment_id="segment-000",
        episode_id="episode-001",
        step_index=0,
        observation_refs=[asset_ref],
        representative_frame_ref=asset_ref,
        frame_start_index=0,
        frame_end_index=0,
        timestamp_start_s=0.0,
        timestamp_end_s=0.0,
        segmentation_confidence=0.95,
        semantic=semantic,
        symbolic_action=symbolic,
        context_tags=[],
        success=True,
        next_step_refs=[],
        raw_outputs={},
    )
    return EpisodeOutput(
        episode_id="episode-001",
        task_name=task_name,
        instruction=instruction,
        input_metadata=MediaMetadata(
            media_type=MediaType.IMAGE,
            source_ref=asset_ref,
            width=640,
            height=480,
            frame_count=1,
            metadata={},
        ),
        frame_predictions=[frame_prediction],
        segments=[segment],
        task_graph=TaskGraph(
            nodes=[TaskNode(node_id="node-0", segment_id="segment-000", step_index=0, terminal=True)],
            edges=[TaskEdge(source_node_id="node-0", target_node_id="node-0", edge_type=TaskEdgeType.TERMINAL)],
            terminal_conditions=["done"],
        ),
        simulation_export=IsaacSimExport(
            episode_id="episode-001",
            task_name=task_name,
            steps=[
                IsaacSimStep(
                    step_index=0,
                    segment_id="segment-000",
                    primitive="hold",
                    description="hold cable near port",
                    target_object="port",
                    source_object="cable",
                    confidence=0.88,
                )
            ],
        ),
        batch_metadata={"frame_count": 1, "frame_prediction_count": 1, "segment_count": 1},
    )


def _build_service(
    tmp_path: Path,
    *,
    fake_blob_client: FakeBlobServiceClient | None = None,
    pipeline_runner=None,
    auth_token: str | None = None,
) -> AnnotationService:
    settings = AppSettings(
        semantic=SemanticSettings(local_model_path=str(tmp_path / "local-model")),
        service=ServiceSettings(temp_dir=str(tmp_path / "service_tmp"), auth_token=auth_token),
        azure=AzureBlobSettings(
            account_url="https://example.blob.core.windows.net",
            source_container_allowlist=("media",),
            annotation_container="datablob-annotations",
            annotation_prefix="annotations",
        ),
    )
    blob_client = fake_blob_client or FakeBlobServiceClient()
    blob_source = AzureBlobSource(settings.azure, blob_service_client=blob_client)
    annotation_store = AzureAnnotationStore(settings.azure, blob_service_client=blob_client)
    return AnnotationService(
        settings,
        pipeline_runner=pipeline_runner,
        blob_source=blob_source,
        annotation_store=annotation_store,
    )


def test_annotation_image_request_rejects_non_image_media_type() -> None:
    with pytest.raises(ValidationError):
        AnnotationImageRequest.model_validate(
            {
                "source_asset_id": "asset-123",
                "source_blob_url": "https://example.blob.core.windows.net/media/example.jpg",
                "media_type": "video",
            }
        )


def test_blob_reference_parsing_and_allowlist_validation() -> None:
    source = AzureBlobSource(AzureBlobSettings(source_container_allowlist=("media",)))
    reference = source.parse_blob_reference(
        "https://example.blob.core.windows.net/media/folder/example.jpg?sig=abc"
    )

    assert reference.container_name == "media"
    assert reference.blob_name == "folder/example.jpg"

    with pytest.raises(AzureBlobStorageError):
        source.parse_blob_reference("https://example.blob.core.windows.net/private/example.jpg")


def test_annotation_path_generation_uses_source_asset_id() -> None:
    store = AzureAnnotationStore(
        AzureBlobSettings(annotation_container="datablob-annotations", annotation_prefix="annotations")
    )

    assert store.build_annotation_blob_name("asset-123", "ann-456") == "annotations/asset-123/ann-456.json"


def test_service_response_serialization_matches_uploaded_annotation_json(tmp_path: Path) -> None:
    fake_blob_client = FakeBlobServiceClient({("media", "example.jpg"): b"image-bytes"})

    def fake_runner(payload: dict[str, object], _: Path) -> EpisodeOutput:
        return _build_episode_output(str(payload["task_name"]), str(payload["instruction"]), str(payload["asset_path"]))

    service = _build_service(tmp_path, fake_blob_client=fake_blob_client, pipeline_runner=fake_runner)
    response = service.annotate_image(
        AnnotationImageRequest(
            source_asset_id="asset-123",
            source_blob_url="https://example.blob.core.windows.net/media/example.jpg",
            media_type="image",
            tags=["ethernet cable", "laptop port"],
        )
    )

    assert response.annotation_json["task_name"] == "asset_123"
    assert response.summary.segment_descriptions == ["hold cable near port"]
    assert len(fake_blob_client.uploads) == 1
    uploaded_json = json.loads(fake_blob_client.uploads[0]["data"].decode("utf-8"))
    assert uploaded_json == response.annotation_json


def test_healthz_returns_service_status(tmp_path: Path) -> None:
    model_path = tmp_path / "local-model"
    model_path.mkdir()
    settings = AppSettings(
        semantic=SemanticSettings(local_model_path=str(model_path)),
        service=ServiceSettings(temp_dir=str(tmp_path / "service_tmp")),
    )
    app = create_api_app(settings=settings, annotation_service=AnnotationService(settings, pipeline_runner=lambda *_: None))
    client = TestClient(app)

    response = client.get("/healthz")

    assert response.status_code == 200
    payload = response.json()
    assert payload["deployment_target"] == "lambda_ai_vm"
    assert payload["semantic_model_path_exists"] is True


def test_annotation_image_endpoint_success(tmp_path: Path) -> None:
    fake_blob_client = FakeBlobServiceClient({("media", "example.jpg"): b"image-bytes"})

    def fake_runner(payload: dict[str, object], _: Path) -> EpisodeOutput:
        return _build_episode_output(str(payload["task_name"]), str(payload["instruction"]), str(payload["asset_path"]))

    service = _build_service(
        tmp_path,
        fake_blob_client=fake_blob_client,
        pipeline_runner=fake_runner,
        auth_token="secret-token",
    )
    app = create_api_app(settings=service.settings, annotation_service=service)
    client = TestClient(app)

    response = client.post(
        "/v1/annotations/image",
        headers={"Authorization": "Bearer secret-token"},
        json={
            "source_asset_id": "asset-123",
            "source_blob_url": "https://example.blob.core.windows.net/media/example.jpg",
            "media_type": "image",
            "tags": ["ethernet cable", "laptop port"],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "completed"
    assert payload["source_asset_id"] == "asset-123"
    assert payload["summary"]["action_labels"] == ["hold"]
    assert len(fake_blob_client.uploads) == 1


def test_annotation_image_endpoint_returns_404_for_missing_source_blob(tmp_path: Path) -> None:
    service = _build_service(
        tmp_path,
        fake_blob_client=FakeBlobServiceClient(),
        pipeline_runner=lambda *_: pytest.fail("pipeline should not run when source blob is missing"),
    )
    app = create_api_app(settings=service.settings, annotation_service=service)
    client = TestClient(app)

    response = client.post(
        "/v1/annotations/image",
        json={
            "source_asset_id": "asset-123",
            "source_blob_url": "https://example.blob.core.windows.net/media/example.jpg",
            "media_type": "image",
        },
    )

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "source_not_found"


def test_annotation_image_endpoint_returns_500_on_pipeline_failure(tmp_path: Path) -> None:
    fake_blob_client = FakeBlobServiceClient({("media", "example.jpg"): b"image-bytes"})

    def failing_runner(_: dict[str, object], __: Path) -> EpisodeOutput:
        raise RuntimeError("boom")

    service = _build_service(tmp_path, fake_blob_client=fake_blob_client, pipeline_runner=failing_runner)
    app = create_api_app(settings=service.settings, annotation_service=service)
    client = TestClient(app)

    response = client.post(
        "/v1/annotations/image",
        json={
            "source_asset_id": "asset-123",
            "source_blob_url": "https://example.blob.core.windows.net/media/example.jpg",
            "media_type": "image",
        },
    )

    assert response.status_code == 500
    assert response.json()["error"]["code"] == "annotation_failed"
    assert fake_blob_client.uploads == []
