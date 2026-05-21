# Datara SaaS Platform

A sophisticated platform for Robotics and Vision, combining synthetic data generation, video segmentation, 3D hand tracking, and Vision-Language-Action (VLA) inference.

## Project Structure

This repository follows a modular micro-service architecture to ensure dependency isolation and efficient CI/CD.

```text
Datara-SaaS/
├── core/                         # Shared Internal Library (Low-dependency)
│   ├── datara_core/
│   │   ├── io/                   # Media loading, image/video utils, MCAP writers
│   │   ├── schemas/              # Unified Action & Annotation schemas (Pydantic)
│   │   └── config.py             # Global settings (env vars, device maps)
│
├── services/                     # Independent ML Microservices (Heavy-dependency)
│   ├── vla-inference/            # Action generation (lerobot, torch-cuda)
│   ├── image-tagging/            # Qwen-based annotation (transformers, torch)
│   ├── segmentation/             # SAM3 Mask generation (specialized SAM3 env)
│   └── synthetic-data/           # Corner case generation (diffusers, controlnet)
│
├── pipelines/                    # Multi-service Orchestration
│   ├── annotation_pipeline.py    # Calls segmentation -> inpainting -> tagging
│   └── hand_motion_capture.py    # Calls mediapipe -> mcap
│
└── .github/workflows/            # Granular CI/CD
```

## Services Overview

### VLA Inference
Produces robot arm action files from video or image input.
- **Location:** `services/vla-inference/`
- **Key Models:** Pi0, SmolVLA, OpenVLA.

### Image Tagging (VLM)
Generates semantic tags for images using Qwen 2.5 VL.
- **Location:** `services/image-tagging/`

### Segmentation (SAM3)
Generates object masks (e.g., humans) from video sequences.
- **Location:** `services/segmentation/`

### Synthetic Data (Corner Cases)
Generates synthetic "corner case" images using Stable Diffusion + ControlNet.
- **Location:** `services/synthetic-data/`

## Core Library
The `core/` directory contains shared logic that is common across multiple services, such as:
- MCAP file generation for Foxglove Studio.
- Common media I/O utilities.
- Pydantic schemas for data exchange.

## CI/CD Strategy
Each service in `services/` is designed to be built and deployed independently.
- **Dependency Isolation:** Services can use different versions of Torch/CUDA without conflicts.
- **Granular Triggers:** CI workflows only run on changes to their respective service directories.

## Setup

This platform is optimized for Ubuntu 24.03 LTS with GH200 (ARM64) or NVIDIA RTX 3080+ GPUs.

### Development
1. Install [Ruff](https://github.com/astral-sh/ruff) for linting.
2. Each service has its own `requirements.txt` or `pyproject.toml`.
