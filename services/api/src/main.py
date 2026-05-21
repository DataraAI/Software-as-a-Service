"""
FastAPI service exposing GPU scripts for remote Data-as-a-Service.
Updated for modular repository structure.

Run with: uvicorn services.api.src.main:app --host 0.0.0.0 --port 8765
"""

from __future__ import annotations

import base64
import json
import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Annotated, Any, Optional

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

# App root is the repository root
APP_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DEFAULT_REQUEST_TIMEOUT_S = int(os.environ.get("GPU_JOB_TIMEOUT_S", "7200"))


def _require_auth(authorization: Annotated[Optional[str], Header()] = None) -> None:
    expected = os.environ.get("GPU_API_KEY", "").strip()
    if not expected:
        return
    if not authorization or authorization != f"Bearer {expected}":
        raise HTTPException(status_code=401, detail="Unauthorized")


AuthDep = Depends(_require_auth)


def _run_script(
    argv: list[str],
    *,
    timeout_s: int = DEFAULT_REQUEST_TIMEOUT_S,
) -> str:
    # Set PYTHONPATH so modules can find each other
    env = {**os.environ}
    ppath = env.get("PYTHONPATH", "")
    new_ppath = str(APP_ROOT)
    if ppath:
        new_ppath = f"{new_ppath}:{ppath}"
    env["PYTHONPATH"] = new_ppath

    proc = subprocess.run(
        argv,
        cwd=str(APP_ROOT),
        capture_output=True,
        text=True,
        timeout=timeout_s,
        env=env,
    )
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip()
        raise HTTPException(
            status_code=502,
            detail={"message": "GPU script failed", "stderr": tail[-8000:]},
        )
    lines = [ln.strip() for ln in (proc.stdout or "").splitlines() if ln.strip()]
    if not lines:
        raise HTTPException(status_code=502, detail="GPU script produced no stdout")
    return lines[-1]


def _read_and_b64(path: str) -> tuple[str, str]:
    p = Path(path)
    if not p.is_file():
        raise HTTPException(status_code=502, detail=f"Expected output file missing: {path}")
    raw = p.read_bytes()
    try:
        p.unlink(missing_ok=True)
    except OSError:
        pass
    return p.name, base64.standard_b64encode(raw).decode("ascii")


class EgoBody(BaseModel):
    prompt: str
    imageURL: str
    container_name: str


class CornerBody(BaseModel):
    prompt: str
    imageURL: str
    container_name: str
    seed: int = 1
    mask_preset: Optional[str] = None


class VlmBody(BaseModel):
    prompt: str
    imageURL: str


app = FastAPI(title="Datara GPU API", version="1.0.0")


@app.get("/health")
def health() -> dict[str, Any]:
    try:
        import torch
        cuda = bool(torch.cuda.is_available())
        dev = torch.cuda.get_device_name(0) if cuda else None
    except Exception:
        cuda = False
        dev = None
    return {"status": "ok", "cuda_available": cuda, "cuda_device": dev}


@app.post("/v1/ego", dependencies=[AuthDep])
def run_ego(body: EgoBody) -> dict[str, Any]:
    argv = [
        sys.executable,
        str(APP_ROOT / "services" / "synthetic-data" / "src" / "image_prompt_tool.py"),
        "--ego_prompt", # Updated flag name based on script analysis
        body.prompt,
        "--imageURL",
        body.imageURL,
        "--container_name",
        body.container_name,
    ]
    out_path = _run_script(argv)
    if not Path(out_path).is_file():
        raise HTTPException(status_code=502, detail=f"Unexpected ego output path: {out_path}")
    filename, b64 = _read_and_b64(out_path)
    return {"filename": filename, "image_base64": b64}


@app.post("/v1/corner-case", dependencies=[AuthDep])
def run_corner(body: CornerBody) -> dict[str, Any]:
    out_root = f"/tmp/datara_corner_{uuid.uuid4().hex}"
    argv = [
        sys.executable,
        str(APP_ROOT / "services" / "synthetic-data" / "src" / "corner_case_tool.py"),
        "--prompt",
        body.prompt,
        "--imageURL",
        body.imageURL,
        "--container_name",
        body.container_name,
        "--seed",
        str(body.seed),
        "--out_root",
        out_root,
    ]
    if body.mask_preset:
        argv.extend(["--mask_preset", body.mask_preset])
    out_path = _run_script(argv)
    if not Path(out_path).is_file() or not str(Path(out_path).resolve()).startswith(
        str(Path(out_root).resolve())
    ):
        shutil.rmtree(out_root, ignore_errors=True)
        raise HTTPException(status_code=502, detail=f"Unexpected corner output path: {out_path}")
    filename, b64 = _read_and_b64(out_path)
    shutil.rmtree(out_root, ignore_errors=True)
    return {"filename": filename, "image_base64": b64}


@app.post("/v1/vlm-tags", dependencies=[AuthDep])
def run_vlm(body: VlmBody) -> dict[str, Any]:
    # Pointing to image-tagging service
    argv = [
        sys.executable,
        str(APP_ROOT / "services" / "image-tagging" / "src" / "qwen_vlm_image.py"),
        "--prompt",
        body.prompt,
        "--egoURL",
        body.imageURL,
    ]
    # This script prints the JSON to stdout or a file. 
    # Based on the previous version it might need --output_json if it supports it.
    # Let's assume it prints to stdout or handles its own path.
    printed = _run_script(argv)
    # The image tagging post_annotation logic usually returns JSON directly or prints it.
    try:
        data = json.loads(printed)
    except Exception:
        if Path(printed).is_file():
            data = json.loads(Path(printed).read_text())
        else:
            raise HTTPException(status_code=502, detail=f"VLM output invalid: {printed}")
    
    return data
