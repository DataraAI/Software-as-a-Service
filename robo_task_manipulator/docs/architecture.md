# Architecture

RoboTaskManipulator is organized by product responsibility, not by abstract pipeline layers. The goal is that changing one part of the system should point you to one obvious folder.

## Responsibility Map

### `ingestion/`
- owns media type detection
- owns video frame extraction
- owns canonical `EpisodeInput` creation
- defaults to full-frame video extraction unless explicitly downsampled

### `segmentation/`
- owns grouped step summaries
- merges consecutive frame predictions into readable summary segments
- keeps the rest of the pipeline compatible with Isaac Sim export and evaluation

### `task_understanding/`
- owns frame-level task understanding
- owns the semantic backend abstraction
- owns rolling-context VLM inference and conservative symbolic labeling

### `action_backend/`
- owns optional robot-oriented backends
- keeps LeRobot `pi0` integration isolated
- keeps the rest of the product working when no action backend is enabled

### `context/`
- owns lightweight context and failure tags
- uses semantic and optional action evidence

### `graph/`
- owns ordered task graph construction
- outputs deterministic next/retry/terminal links

### `simulation/`
- owns Isaac Sim 5.1 / Franka Panda export
- converts segments into a simulation-ready task plan

### `evaluation/`
- owns benchmark schemas, scoring, and report writing
- computes simple v1 metrics instead of heavy research metrics

### `export/`
- owns final episode JSON, raw debug JSON, and manifest artifacts

### `schemas/`
- owns typed contracts shared across the product

### `utils/`
- owns small shared helpers only

### `main.py`
- owns the end-to-end orchestration path

## Product Flow

1. `ingestion/` converts media into a canonical ordered episode and extracts all frames by default.
2. `task_understanding/` produces one semantic prediction per frame using a small rolling context window.
3. `task_understanding/labeling.py` converts frame semantics into conservative symbolic actions.
4. `context/` adds lightweight per-frame tags.
5. `segmentation/` groups consecutive frame predictions into summary segments.
6. `action_backend/` optionally adds robot-oriented action proposals to the summary segments.
7. `graph/` links the summary sequence.
8. `simulation/` builds the Isaac Sim task plan.
9. `export/` writes episode artifacts.
10. `evaluation/` scores grouped outputs when benchmark truth exists.

## Practical Design Choices

- Frame-level task understanding is the primary source of meaning in the current build.
- Grouped steps are a derived summary view built after frame predictions.
- `pi0` is optional and used for action proposals, not as the main semantic engine.
- The default semantic target is a local Qwen2.5-VL 7B model path or Hub id.
- Grouped step formation is intentionally simple and deterministic.
- Context/failure tagging is heuristic and explicit about confidence.
- Isaac Sim integration is export-oriented rather than execution-oriented.
- Evaluation prioritizes easy-to-read metrics over research complexity.

## Implemented Now

- images, videos, and extracted frame sequences as inputs
- automatic full-frame extraction for videos
- one semantic prediction per frame
- grouped summary segments derived from frame predictions
- conservative symbolic labeling with `unknown` fallback
- optional `pi0` action proposals
- context/failure tagging
- ordered task graphs
- Isaac Sim / Franka Panda export
- batch manifests
- benchmark evaluation reports

## Future Work

- stronger robotics-specific semantic models
- richer embodiment-specific action interpretation
- better segmentation heuristics for long videos
- more detailed Isaac Sim object mapping
- true OpenVLA runtime integration
- stronger benchmark coverage and error analysis tools
