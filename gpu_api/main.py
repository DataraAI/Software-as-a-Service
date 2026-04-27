"""
FastAPI service exposing GPU scripts for remote Data-as-a-Service.

Run with: uvicorn gpu_api.main:app --host 0.0.0.0 --port 8765
Auth: optional Bearer token when GPU_API_KEY is set.
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

APP_ROOT = Path(__file__).resolve().parent.parent
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
    proc = subprocess.run(
        argv,
        cwd=str(APP_ROOT),
        capture_output=True,
        text=True,
        timeout=timeout_s,
        env={**os.environ, "PYTHONPATH": _pythonpath()},
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


def _pythonpath() -> str:
    extra = str(APP_ROOT)
    cur = os.environ.get("PYTHONPATH", "")
    if cur:
        return f"{extra}:{cur}" if extra not in cur else cur
    return extra


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
        str(APP_ROOT / "image_prompt_tool.py"),
        "--prompt",
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
        str(APP_ROOT / "Corner_case_tool.py"),
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
    out_json = f"/tmp/vlm_tags_{uuid.uuid4().hex}.json"
    argv = [
        sys.executable,
        str(APP_ROOT / "Post Annotation" / "qwen_vlm_image.py"),
        "--prompt",
        body.prompt,
        "--egoURL",
        body.imageURL,
        "--output_json",
        out_json,
    ]
    printed = _run_script(argv)
    path = printed if Path(printed).is_file() else out_json
    if not Path(path).is_file():
        raise HTTPException(status_code=502, detail="VLM did not write JSON output")
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    finally:
        Path(path).unlink(missing_ok=True)
    if not isinstance(data, dict) or "VLM_tags" not in data:
        raise HTTPException(status_code=502, detail="VLM output missing VLM_tags")
    return data
