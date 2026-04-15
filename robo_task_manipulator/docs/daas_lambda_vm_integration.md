# Lambda.ai + DaaS Integration

This document defines the Milestone 1 service contract for running RoboTaskManipulator on a persistent Lambda.ai VM and calling it from the DaaS backend.

## Deployment Shape

- Host the FastAPI app from `scripts/run_annotation_service.py` as a long-running process on the Lambda.ai VM.
- Keep the semantic model local on the VM via `RTM_SEMANTIC_MODEL_PATH` or `semantic.local_model_path`.
- Use Azure Blob credentials from the VM environment or YAML config.
- Use an internal bearer token for DaaS backend -> Lambda.ai service requests.

## Milestone 1 Request

`POST /v1/annotations/image`

```json
{
  "source_asset_id": "asset-123",
  "source_blob_url": "https://account.blob.core.windows.net/media/example.jpg",
  "media_type": "image",
  "task_name": "ethernet_cable_insert",
  "instruction": "Describe only the visible hand-object action conservatively.",
  "tags": ["ethernet cable", "laptop port", "network connector"]
}
```

Notes:
- `media_type` must be `image` for Milestone 1.
- `task_name` is optional; the service derives a stable fallback from `source_asset_id` when omitted.
- `instruction` is optional; the service defaults to a conservative visible-action prompt.
- `tags` are soft hints only and should reflect known asset metadata.

## Milestone 1 Response

On success the service returns:
- `annotation_id`
- `status=completed`
- `source_asset_id`
- `annotation_blob_url`
- `annotation_json`
- `summary`

The exact `annotation_json` payload is also persisted to Azure Blob under:

`annotations/{source_asset_id}/{annotation_id}.json`

The upload includes metadata and index tags for:
- `source_asset_id`
- `media_type`
- `task_name`
- `model_id`
- `created_at`
- `status`

## DaaS Backend Contract

DaaS should call the Lambda.ai service from the backend, not directly from the browser.

Expected DaaS flow:
- enable `Generate Annotation JSON` only for image assets backed by Azure Blob
- call the Lambda.ai service with a blocking spinner for sync image requests
- use a request timeout of `90s`
- render `annotation_json` directly in the asset detail view on success
- persist `annotation_id`, `annotation_blob_url`, `created_at`, and `status`
- expose a `View latest annotation JSON` action on the asset
- show the structured error response and do not persist a success record when the service fails

## Milestone 2 Reserved Routes

The API reserves these routes for future async image/video jobs:
- `POST /v1/annotations/jobs`
- `GET /v1/annotations/jobs/{job_id}`

Milestone 2 should reuse the same canonical annotation blob format and add an external job status store rather than relying on local VM memory.
