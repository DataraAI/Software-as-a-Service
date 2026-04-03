# RoboTaskManipulator

RoboTaskManipulator is a practical v1 backend for turning photos, videos, or ordered extracted frames into conservative task steps, symbolic robot-action labels, Isaac Sim task plans, and lightweight evaluation reports.

v1 is built around pretrained inference. It does not require training a new model.

## What It Does

Input:
- single image payloads
- video payloads with automatic frame extraction
- explicit ordered frame-sequence payloads

Output:
- ordered task segments
- semantic step descriptions from a VLM layer
- conservative symbolic action labels
- lightweight context/failure tags
- ordered task graph links
- Isaac Sim 5.1 / Franka Panda export
- per-episode JSON, raw debug JSON, batch manifest, and optional evaluation reports

## Product Pipeline

```mermaid
flowchart TD
    A["Input media<br/>image, video, or extracted frames"] --> B["ingestion/<br/>detect media and extract frames"]
    B --> C["segmentation/<br/>fixed-window step candidates"]
    C --> D["understanding/<br/>VLM semantic step understanding"]
    D --> E["understanding/labeling.py<br/>conservative symbolic labels"]
    E --> F["action_backend/<br/>optional pi0/OpenVLA/none proposals"]
    F --> G["context/<br/>heuristic context and failure tags"]
    G --> H["graph/<br/>ordered task graph"]
    H --> I["simulation/<br/>Isaac Sim Franka export"]
    I --> J["export/<br/>episode JSON + raw debug + manifest"]
    J --> K["evaluation/<br/>benchmark JSON + CSV report"]
```

## Model Roles

Semantic understanding:
- primary engine for v1
- implemented through `understanding/`
- default backend is a pretrained Transformers image-to-text model (`Salesforce/blip-image-captioning-base`)
- if the VLM is unavailable, the code falls back to conservative instruction-guided heuristics instead of fabricating certainty

Action backend:
- optional
- implemented through `action_backend/`
- default is `none`
- supported choices are `none`, `pi0`, and a reserved `openvla` slot
- `pi0` is used for robot-oriented action proposals and action chunks, not as the main semantic engine

## Folder Layout

- `configs/`: example runtime configuration
- `data/`: sample inputs, benchmark examples, and generated outputs
- `docs/`: architecture notes and project-specific guidance
- `scripts/`: runnable entry points for single inference, batch annotation, and evaluation
- `src/robotask_manipulator/ingestion/`: media detection and frame extraction
- `src/robotask_manipulator/segmentation/`: deterministic temporal segmentation
- `src/robotask_manipulator/understanding/`: semantic VLM backend plus symbolic labeling
- `src/robotask_manipulator/action_backend/`: optional robot-oriented backends such as pi0
- `src/robotask_manipulator/context/`: context and failure tagging
- `src/robotask_manipulator/graph/`: ordered task graph construction
- `src/robotask_manipulator/simulation/`: Isaac Sim / Franka Panda export
- `src/robotask_manipulator/evaluation/`: benchmark scoring and report generation
- `src/robotask_manipulator/export/`: final JSON artifact writing
- `src/robotask_manipulator/schemas/`: typed data contracts
- `src/robotask_manipulator/utils/`: focused shared utilities
- `src/robotask_manipulator/main.py`: the end-to-end product orchestrator
- `tests/`: product-focused tests

## Input Payloads

Single image:

```json
{
  "episode_id": "demo-001",
  "task_name": "pick_and_place",
  "instruction": "Pick the block and place it on the pad.",
  "asset_path": "sample_frame_001.ppm",
  "state": [0.0, 0.0, 0.0, 0.0]
}
```

Frame sequence:

```json
{
  "episode_id": "workflow-001",
  "task_name": "battery_insert_sequence",
  "instruction": "Pick the battery, align it with the slot, insert it, then inspect the fit.",
  "frames": [
    {"frame_id": "frame-000", "asset_ref": "sample_frame_001.ppm", "frame_index": 0, "timestamp_s": 0.0},
    {"frame_id": "frame-001", "asset_ref": "sample_frame_002.ppm", "frame_index": 1, "timestamp_s": 0.5}
  ]
}
```

Video:
- provide `asset_path` or `media_path` pointing to a video file
- v1 extracts frames with a simple configurable stride
- raw video decoding is supported through OpenCV-based extraction, but explicit extracted frames are still the easiest path for clean evaluation

## Running It

Install:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Single inference:

```bash
py -3 scripts\run_single_inference.py --input data\sample_inputs\sample_workflow_episode_001.json
```

Use real pi0 proposals:

```bash
py -3 scripts\run_single_inference.py ^
  --input data\sample_inputs\sample_workflow_episode_001.json ^
  --action-backend pi0 ^
  --model-id lerobot/pi0_base
```

Batch mode:

```bash
py -3 scripts\batch_annotate.py --input-dir data\sample_inputs
```

Batch mode with evaluation:

```bash
py -3 scripts\batch_annotate.py ^
  --input-dir data\sample_inputs ^
  --benchmark data\benchmarks\sample_benchmark.json
```

Standalone evaluation:

```bash
py -3 scripts\evaluate_benchmark.py ^
  --predictions-dir data\outputs ^
  --benchmark data\benchmarks\sample_benchmark.json
```

## Configuration

Config can come from:
- `configs/settings.example.yaml`
- environment variables
- CLI overrides

Useful env vars:
- `RTM_SEMANTIC_BACKEND`
- `RTM_SEMANTIC_MODEL_ID`
- `RTM_SEMANTIC_DEVICE`
- `RTM_ACTION_BACKEND`
- `PI0_MODEL_ID`
- `PI0_CHECKPOINT_PATH`
- `PI0_DEVICE`
- `PI0_DTYPE`
- `PI0_OFFLINE`

## Isaac Sim Export

The simulation layer generates a simulation-ready task plan for:
- simulator: Isaac Sim 5.1
- robot: Franka Panda

Each exported step includes:
- ordered primitive label
- semantic description
- source/target object fields when available
- confidence
- status
- optional action proposal
- context/failure tags

The exporter is intentionally focused on generating clean structured plans, not directly launching Isaac Sim.

## Evaluation Flow

Benchmark episodes define:
- expected ordered steps
- expected symbolic labels
- optional success outcomes

The evaluation layer currently computes:
- step count difference
- step label agreement
- ordering agreement
- per-episode pass / needs-review summary

Reports are written as:
- JSON
- CSV

## Known Limitations

- Semantic understanding is VLM-driven but still conservative and not guaranteed to hit benchmark-quality labels on every frame.
- The default BLIP-style semantic backend is practical for v1, not robotics-specialized.
- `pi0` proposals are optional and embodiment-sensitive.
- Video segmentation is heuristic and window-based rather than learned.
- Context/failure tags are lightweight heuristics and do not imply true physics certainty.
- Isaac Sim export is plan-oriented; direct sim execution is intentionally out of scope for v1.
- `OpenVLA` is represented as a clean backend slot but is not wired up in this build.

## Current Default Behavior

- semantic understanding: enabled
- symbolic labeling: enabled
- action backend: `none`
- Isaac Sim export: enabled
- evaluation: optional when benchmark truth is provided

That default keeps the product easy to run end to end on photos/videos while preserving a path to richer robot-oriented proposals with `pi0`.
