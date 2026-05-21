# VLA Inference Pipeline — SaaS Batch Interface

> **SaaS component only.** This system produces robot arm action files from video or image input. It does not execute actions. A separate DaaS component consumes the output.

---

## Overview

This project is a batch inference wrapper around existing Vision-Language-Action (VLA) models. It takes a video recording or a directory of images, runs a VLA model on each frame, and saves the resulting robot arm joint delta actions to a uniquely-named JSON file on disk.

**Problem it solves:** The team needed a clean interface between media capture and robot execution. Instead of a live agent loop (the prior incorrect implementation), this system decouples inference from execution — making it testable, hardware-independent, and easy to swap models.

**How it works:**
1. Load frames from a video file or image directory
2. Run a VLA backend per frame (mock, pi0, smolvla, etc.)
3. Decode raw model output into a structured action schema
4. Validate the action list
5. Write the results to a unique JSON file
6. Print the output file path to stdout for DaaS to consume

---

## Key Features

- **Two input modes** — video file (`.mp4`, `.avi`, etc.) or sorted image directory (`.png`, `.jpg`, etc.)
- **Swappable backends** — plug in any VLA model by implementing a single `predict()` interface
- **Structured action schema** — supports both UI actions (CLICK, TYPE, PRESS, etc.) and robot arm actions (ROBOT_JOINT_DELTA)
- **Unique output per run** — each run writes to `outputs/actions_<timestamp>_<uuid>.json`
- **Mock backend** — always works, no GPU needed, cycles through all action types for demos and CI
- **Validation** — every action entry is validated before write; malformed actions raise errors with clear messages
- **Verbose mode** — `--verbose` flag enables debug-level logging for tracing issues

---

## Architecture / Workflow

```
Input
  --video_path  OR  --image_dir
  --task "natural language instruction"
  --backend mock|pi0|smolvla|openvla|gr00t
         │
         ▼
  media_io/media_loader.py
  ┌─────────────────────────────┐
  │  video → OpenCV frame extract│
  │  images → Pillow sorted glob │
  │  returns (index, timestamp,  │
  │           PIL.Image) list    │
  └─────────────────────────────┘
         │
         ▼  per frame
  policies/<backend>.py
  ┌─────────────────────────────┐
  │  predict(image, task, idx)  │
  │  → raw output               │
  │  mock: structured dict      │
  │  pi0/smolvla: numpy array   │
  │  (joint delta vector)       │
  └─────────────────────────────┘
         │
         ▼
  decoder.py
  ┌─────────────────────────────┐
  │  routes by backend type     │
  │  mock → UI Action objects   │
  │  robot → RobotAction        │
  │          (joint_deltas)     │
  └─────────────────────────────┘
         │
         ▼
  validator.py
  ┌─────────────────────────────┐
  │  checks required fields,    │
  │  types, value ranges        │
  │  raises ValidationError     │
  │  on any malformed entry     │
  └─────────────────────────────┘
         │
         ▼
  media_io/action_writer.py
  ┌─────────────────────────────┐
  │  generates run_id           │
  │  builds JSON payload        │
  │  writes unique output file  │
  └─────────────────────────────┘
         │
         ▼
Output
  outputs/actions_2026-04-02T05-18-28Z_ab12cd34.json
  (path printed to stdout for DaaS)
```

### VLA pipeline and action generation

Real VLA backends (pi0, smolvla, openvla, gr00t) are robot arm models. They output **7-DoF joint delta vectors** — not desktop UI coordinates. These are passed through as `ROBOT_JOINT_DELTA` actions, which Isaac Sim / DaaS consumes directly.

The mock backend outputs structured UI-style actions (CLICK, TYPE, etc.) and is the correct choice for pipeline testing, demos, and CI.

---

## Tech Stack

### Core
| Technology | Version | Purpose |
|---|---|---|
| Python | 3.12 | Primary language |
| Pillow | ≥10.0 | Image loading and conversion |
| opencv-python | ≥4.8 | Video frame extraction |

### ML / Model Loading
| Technology | Version | Purpose |
|---|---|---|
| lerobot | 0.5.1 | VLA policy loading (pi0, pi05, smolvla_base) |
| PyTorch | ≥2.0 (CPU build) | Tensor ops and model inference |
| torchvision | ≥0.15 | Image tensor preprocessing |
| transformers | ≥5.x | Tokenizer loading (PaliGemma for pi0) |
| huggingface_hub | ≥0.22 | Model weight download and caching |

### Dev Environment
| Technology | Notes |
|---|---|
| Windows 11 / PowerShell | Development machine |
| VS Code | Primary IDE |
| Python venv | Dependency isolation |

---

## Project Structure

```
vla_inference/
├── run_on_media.py           # CLI entrypoint — start here
├── actions.py                # Action schema definitions (all action types)
├── decoder.py                # Routes raw backend output → typed Action objects
├── validator.py              # Validates every action entry before write
│
├── media_io/                 # NOTE: named media_io (not io) — avoids Python stdlib collision
│   ├── __init__.py
│   ├── media_loader.py       # Loads video frames (OpenCV) or image dir (Pillow)
│   └── action_writer.py      # Generates run_id, builds payload, writes JSON
│
├── policies/
│   ├── __init__.py
│   ├── mock_policy.py        # ✅ Always works — use for demos and testing
│   ├── pi0_adapter.py        # ⚠️ Experimental — lerobot 0.5.1, needs GPU/RAM
│   ├── smolvla_adapter.py    # ⚠️ Experimental — needs update for lerobot 0.5.1
│   ├── openvla_adapter.py    # 🔲 Placeholder — not yet implemented
│   └── gr00t_adapter.py      # 🔲 Placeholder — not yet implemented
│
├── outputs/                  # Auto-created — action JSON files written here
├── requirements.txt
└── README.md
```

**Important naming note:** The media I/O folder is named `media_io/`, not `io/`. Python's standard library has a built-in module named `io` — using that name as a package causes a `ModuleNotFoundError` at import time.

---

## Setup Instructions

### Prerequisites

- Python 3.12+ (use `py` launcher on Windows if `python` is not on PATH)
- Git

### 1. Clone and navigate

```bash
git clone https://github.com/DataraAI/Software-as-a-Service/tree/issue-16-vla
cd vla_inference
```

### 2. Create and activate virtual environment

**Windows (PowerShell):**
```powershell
# If you get "running scripts is disabled", run this first (one-time):
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

py -m venv .venv
.venv\Scripts\activate
```

**Mac/Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

**Minimal (mock backend only — no GPU needed):**
```bash
pip install Pillow opencv-python
```

**Full install (includes pi0 / smolvla support):**
```bash
pip install Pillow opencv-python
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu  # CPU only
pip install git+https://github.com/huggingface/lerobot.git                       # lerobot 0.5.1
pip install transformers huggingface_hub
```

> **GPU install:** Replace the PyTorch CPU line with the appropriate CUDA build from [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally). A GPU is strongly recommended for real model inference — see [Known Issues](#known-issues).

---

## How to Run

### Mock backend (recommended starting point — no GPU needed)

```bash
# From an image directory
python run_on_media.py \
  --image_dir ./sample_images \
  --task "pick up the red cube" \
  --backend mock \
  --output_dir ./outputs

# From a video file
python run_on_media.py \
  --video_path ./recording.mp4 \
  --task "pick up the red cube" \
  --backend mock \
  --output_dir ./outputs
```

### pi0 backend (experimental — requires lerobot + GPU/sufficient RAM)

```bash
python run_on_media.py \
  --image_dir ./sample_images \
  --task "pick up the red cube" \
  --backend pi0 \
  --model_id lerobot/pi0_base \
  --output_dir ./outputs
```

> **First run:** pi0_base weights (~14GB) will download automatically to your HuggingFace cache. This takes several minutes.

### Debug mode

Add `--verbose` to any command for full debug logging:

```bash
python run_on_media.py --image_dir ./sample_images --task "..." --backend mock --verbose
```

### Create sample test images (for quick validation)

```python
# Run once to generate 5 test frames in sample_images/
python -c "
from PIL import Image
import os
os.makedirs('sample_images', exist_ok=True)
for i in range(5):
    Image.new('RGB', (640, 480), color=(i*40, 100, 200)).save(f'sample_images/frame_{i:04d}.png')
"
```

### CLI reference

```
python run_on_media.py [--video_path PATH | --image_dir PATH]
                       --task TEXT
                       [--output_dir PATH]      default: outputs/
                       [--backend BACKEND]       default: mock
                       [--model_id HF_ID]        optional, uses backend default if omitted
                       [--verbose]
```

---

## Example Output

**Console output:**
```
outputs/actions_2026-04-02T05-18-28Z_ab12cd34.json
```

**JSON file (mock backend):**
```json
{
  "run_id": "2026-04-02T05-18-28Z_ab12cd34",
  "input": {
    "video_path": null,
    "image_dir": "./sample_images",
    "task": "pick up the red cube"
  },
  "backend": "mock",
  "model_id": "mock",
  "actions": [
    {
      "frame_index": 0,
      "timestamp": 0.0,
      "action": {
        "action_type": "WAIT",
        "seconds": 1.0
      }
    },
    {
      "frame_index": 1,
      "timestamp": 0.5,
      "action": {
        "action_type": "CLICK",
        "x": 412,
        "y": 88
      }
    }
  ]
}
```

**JSON file (pi0 / real robot backend):**
```json
{
  "run_id": "2026-04-02T06-01-14Z_ff3c9a21",
  "input": {
    "video_path": null,
    "image_dir": "./isaac_frames",
    "task": "pick up the red cube"
  },
  "backend": "pi0",
  "model_id": "lerobot/pi0_base",
  "actions": [
    {
      "frame_index": 0,
      "timestamp": 0.0,
      "action": {
        "action_type": "ROBOT_JOINT_DELTA",
        "joint_deltas": [0.012, -0.034, 0.008, 0.001, -0.021, 0.003, 0.95]
      }
    }
  ]
}
```

### Action types (UI schema — mock backend)

| Type | Fields |
|---|---|
| `CLICK` | `x`, `y` |
| `DOUBLE_CLICK` | `x`, `y` |
| `PRESS` | `key` |
| `HOTKEY` | `modifier`, `key` |
| `SCROLL` | `amount` |
| `WAIT` | `seconds` |
| `TYPE` | `text` |

### Robot arm schema (real VLA backends)

| Type | Fields |
|---|---|
| `ROBOT_JOINT_DELTA` | `joint_deltas` (list of floats, typically 7-DoF) |

---

## Current Status

| Component | Status | Notes |
|---|---|---|
| Mock backend end-to-end | ✅ Working | All action types, validation, unique output path |
| Video input (OpenCV) | ✅ Working | BGR→RGB conversion, fps-based timestamps |
| Image directory input | ✅ Working | Sorted glob, 2fps assumed |
| Action schema + validation | ✅ Working | All UI types + ROBOT_JOINT_DELTA |
| Unique JSON output | ✅ Working | Timestamp + UUID8 filename |
| pi0_adapter (lerobot 0.5.1) | ⚠️ Implemented, blocked | Config loads; inference OOM on dev machine |
| pi0_base weights | ✅ Downloaded | 14GB cached in HuggingFace local cache |
| smolvla_adapter | ⚠️ Experimental | Written for older lerobot API; needs 0.5.1 update |
| openvla_adapter | 🔲 Placeholder | Not yet implemented |
| gr00t_adapter | 🔲 Placeholder | Not yet implemented |

---

## Known Issues

### pi0_base OOM on CPU-only machines
- **Issue:** `lerobot/pi0_base` is 14GB. Windows silently kills the process when RAM is exhausted. No error, no traceback — the process just exits.
- **Root cause:** Hardware constraint. Dev machine has Intel UHD 620 (no NVIDIA GPU) and 8.4GB free RAM.
- **Workaround A:** Test `lerobot/smolvla_base` — likely much smaller.
- **Workaround B:** Run on Google Colab with a free T4 GPU (`Runtime → Change runtime type → T4 GPU`). The pipeline code is unchanged.
- **Long-term fix:** NVIDIA GPU workstation (RTX 3080+ recommended, 16GB+ VRAM).

### SmolVLA removed from lerobot 0.5.1
- **Issue:** `lerobot.common.policies.smolvla` no longer exists. The existing `smolvla_adapter.py` uses the old module path.
- **Workaround:** Use `lerobot/smolvla_base` checkpoint via the pi0 or a new adapter — the checkpoint exists on HuggingFace, but the module path needs updating for 0.5.1.

### Robot state is a dummy zero vector
- **Issue:** `pi0_adapter.py` passes `torch.zeros(1, 7)` as the robot state observation. This means the model has no information about the current joint positions.
- **Impact:** Output actions may be less accurate than with real joint feedback.
- **Fix:** Replace with actual joint state from Isaac Sim at inference time.

### PowerShell multi-line commands
- **Issue:** Writing multi-line Python scripts via PowerShell `echo` or here-strings produces UTF-16 BOM files that Python rejects.
- **Workaround:** Always create `.py` files in VS Code editor (Ctrl+N), save as UTF-8, then run with `python filename.py`.

---

## Adding a New Backend

Adding a new VLA model requires three changes:

**1. Create `policies/your_backend.py`:**
```python
class YourBackend:
    def __init__(self, model_id: str):
        self.model_id = model_id
        # load your model here

    def predict(self, image: PIL.Image, task: str, frame_index: int) -> Any:
        # run inference, return raw output
        pass
```

**2. Add a case to `load_policy()` in `run_on_media.py`:**
```python
elif backend == "your_backend":
    from policies.your_backend import YourBackend
    return YourBackend(model_id=model_id or "default/model-id")
```

**3. Add a decode function in `decoder.py`:**
```python
def decode(backend: str, raw: Any):
    ...
    elif backend == "your_backend":
        return decode_robot_vector(raw)  # or write a custom decoder
```

---

## Future Improvements

### Immediate (unblock real model inference)
- Test `lerobot/smolvla_base` size and update adapter for lerobot 0.5.1 API
- Run pi0_base on Google Colab to validate real joint delta output
- Update `smolvla_adapter.py` to use `lerobot.policies.smolvla` path in 0.5.1

### Near-term (Isaac Sim integration)
- Replace dummy zero robot state in `pi0_adapter.py` with real joint feedback from Isaac Sim
- Align `ROBOT_JOINT_DELTA` schema fields with DaaS consumption format (confirm joint_names, coordinate frame, units)
- Test pipeline with actual Isaac Sim rendered frames instead of solid-color test images
- Share sample output JSON with DaaS team and confirm end-to-end consumption

### Medium-term (production readiness)
- GPU workstation setup (NVIDIA RTX 3080+ or cloud GPU) for practical inference speed
- Implement `openvla_adapter.py` (transformers-based, simpler install than lerobot)
- Add frame subsampling (`--every_n_frames`) for long videos
- Add progress bar for large video inputs
- Write unit tests for decoder, validator, and mock backend

### Long-term
- Fine-tune a VLA model on domain-specific Isaac Sim data for higher-quality robot actions
- Implement GR00T adapter (requires NVIDIA Isaac Sim environment)
- Add batch size > 1 for faster GPU inference

---

## Quick Start (TL;DR)

```bash
# 1. Activate your virtual environment
.venv\Scripts\activate          # Windows
source .venv/bin/activate        # Mac/Linux

# 2. Install minimal dependencies
pip install Pillow opencv-python

# 3. Create 5 test images
python -c "
from PIL import Image; import os
os.makedirs('sample_images', exist_ok=True)
[Image.new('RGB',(640,480),(i*40,100,200)).save(f'sample_images/frame_{i:04d}.png') for i in range(5)]
"

# 4. Run the pipeline
python run_on_media.py \
  --image_dir ./sample_images \
  --task "pick up the red cube" \
  --backend mock \
  --output_dir ./outputs

# 5. Check output — path is printed to terminal
#    Open the JSON file in outputs/ to see the result
```

**Expected output:**
```
outputs/actions_2026-04-02T05-18-28Z_ab12cd34.json
```

---

## Notes for Onboarding Engineers

- Start with `--backend mock`. It always works, needs no GPU, and exercises the full pipeline.
- `run_on_media.py` is the single entry point. Read it top to bottom — it's ~220 lines and well-commented.
- The folder is named `media_io/` not `io/` — this is intentional. Do not rename it.
- All VLA backends (pi0, smolvla, openvla, gr00t) produce robot arm joint vectors, not desktop UI actions. The `ROBOT_JOINT_DELTA` output type is what Isaac Sim / DaaS expects.
- If you're testing on a machine without an NVIDIA GPU, use mock for development and Colab for real model validation.
