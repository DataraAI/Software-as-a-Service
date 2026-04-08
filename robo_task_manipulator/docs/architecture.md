# Architecture

RoboTaskManipulator is organized by product responsibility, not by abstract pipeline layers. The goal is that changing one part of the system should point you to one obvious folder.

## Responsibility Map

### `ingestion/`
- owns media type detection
- owns video frame extraction
- owns canonical `EpisodeInput` creation

### `segmentation/`
- owns deterministic temporal segmentation
- owns fixed-window and visual-change heuristics
- outputs ordered segment skeletons

### `task_understanding/`
- owns segment-level task understanding
- owns the semantic backend abstraction
- owns conservative symbolic labeling

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

1. `ingestion/` converts media into a canonical ordered episode.
2. `segmentation/` splits that episode into candidate steps.
3. `task_understanding/` produces semantic step descriptions from ordered frames within a segment.
4. `task_understanding/labeling.py` converts semantics into conservative symbolic actions.
5. `action_backend/` optionally adds robot-oriented action proposals.
6. `context/` adds lightweight tags.
7. `graph/` links the sequence.
8. `simulation/` builds the Isaac Sim task plan.
9. `export/` writes episode artifacts.
10. `evaluation/` scores outputs when benchmark truth exists.

## Practical Design Choices

- Segment-level task understanding is the primary source of step meaning in v1.
- `pi0` is optional and used for action proposals, not as the main semantic engine.
- Segmentation is intentionally simple and deterministic.
- Context/failure tagging is heuristic and explicit about confidence.
- Isaac Sim integration is export-oriented rather than execution-oriented.
- Evaluation prioritizes easy-to-read metrics over research complexity.

## Implemented Now

- images, videos, and extracted frame sequences as inputs
- automatic frame extraction for videos
- deterministic segmentation
- VLM-driven semantic step understanding
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
