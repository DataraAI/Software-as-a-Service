"""Shared helpers for ViPE and Dyn-HaMR SaaS runners.

These helpers assume the required tools are already installed on the VM.
They do not clone repositories or create environments at runtime.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

VALID_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def image_sort_key(path: Path) -> tuple[int, int, str]:
    stem = path.stem
    match = re.search(r"_(\d+)(?:_|$)", stem)
    if match:
        return (0, int(match.group(1)), path.name.lower())
    if stem.isdigit():
        return (0, int(stem), path.name.lower())
    return (1, 0, path.name.lower())


def collect_images(image_dir: Path) -> list[Path]:
    images = [
        path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in VALID_IMAGE_EXTENSIONS
    ]
    return sorted(images, key=image_sort_key)


def normalize_frame(frame: Any) -> Any:
    import cv2

    if frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    return frame


def build_video(image_paths: list[Path], output_path: Path, fps: float) -> dict[str, int | float]:
    import cv2

    if not image_paths:
        raise ValueError("No source images were provided")

    first = cv2.imread(str(image_paths[0]), cv2.IMREAD_UNCHANGED)
    if first is None:
        raise ValueError(f"Could not read {image_paths[0]}")
    first = normalize_frame(first)
    height, width = first.shape[:2]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {output_path}")

    try:
        for image_path in image_paths:
            frame = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
            if frame is None:
                raise ValueError(f"Could not read {image_path}")
            frame = normalize_frame(frame)
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            writer.write(frame)
    finally:
        writer.release()

    return {
        "frame_count": len(image_paths),
        "fps": float(fps),
        "width": width,
        "height": height,
    }


def default_dynhamr_root() -> Path:
    candidates = [
        Path(os.environ.get("DYNHAMR_ROOT", "")).expanduser() if os.environ.get("DYNHAMR_ROOT") else None,
        Path("~/packages/Dyn-Hamr").expanduser(),
        Path("~/packages/Dyn-HaMR").expanduser(),
        Path("~/packages/Dyn-HAMR").expanduser(),
    ]
    for candidate in candidates:
        if candidate and candidate.exists():
            return candidate.resolve()
    return Path("~/packages/Dyn-Hamr").expanduser().resolve()


def locate_dynhamr_work_dir(dynhamr_root: Path) -> Path:
    nested = dynhamr_root / "dyn-hamr"
    if (nested / "run_opt.py").is_file():
        return nested
    if (dynhamr_root / "run_opt.py").is_file():
        return dynhamr_root
    raise FileNotFoundError(f"Could not find run_opt.py under {dynhamr_root}")


def default_vipe_work_dir() -> Path:
    configured = os.environ.get("VIPE_WORK_DIR")
    if configured:
        candidate = Path(configured).expanduser()
        if candidate.exists():
            return candidate.resolve()
    return Path.cwd().resolve()


def run_command(command: list[str], *, cwd: Path, log_path: Path, env: dict[str, str]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    with log_path.open("a", encoding="utf-8") as log_handle:
        log_handle.write(f"\n[{started}] $ {' '.join(shlex.quote(part) for part in command)}\n")
        log_handle.flush()
        completed = subprocess.run(
            command,
            cwd=str(cwd),
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        log_handle.write(f"\n[exit {completed.returncode}]\n")

    if completed.returncode != 0:
        tail = ""
        try:
            tail = "\n".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-80:])
        except OSError:
            tail = ""
        raise RuntimeError(f"Command failed with exit {completed.returncode}: {' '.join(command)}\n{tail}")


def collect_new_obj_files(search_root: Path, seq: str, started_at: float) -> list[Path]:
    if not search_root.exists():
        return []
    obj_files = sorted(path for path in search_root.rglob("*.obj") if path.is_file())
    if not obj_files:
        return []

    recent = [path for path in obj_files if path.stat().st_mtime >= started_at - 60]
    seq_lower = seq.lower()
    seq_matches = [path for path in recent if seq_lower in path.as_posix().lower()]
    if seq_matches:
        return sorted(seq_matches)
    if recent:
        return sorted(recent)
    return sorted(obj_files, key=lambda path: path.stat().st_mtime, reverse=True)


def safe_output_name(path: Path, root: Path, index: int, used: set[str]) -> str:
    try:
        relative = path.relative_to(root)
    except ValueError:
        relative = Path(path.name)

    if len(relative.parts) == 1:
        candidate = relative.name
    else:
        stem = "__".join(re.sub(r"[^a-zA-Z0-9_.-]+", "_", part) for part in relative.with_suffix("").parts)
        candidate = f"{stem}.obj"

    candidate = re.sub(r"[^a-zA-Z0-9_.-]+", "_", candidate).strip("._-") or f"mesh_{index:04d}.obj"
    if not candidate.lower().endswith(".obj"):
        candidate = f"{candidate}.obj"
    if candidate in used:
        candidate = f"{candidate[:-4]}_{index:04d}.obj"
    used.add(candidate)
    return candidate


def copy_meshes(obj_files: list[Path], output_dir: Path, collection_root: Path) -> list[dict[str, str]]:
    copied: list[dict[str, str]] = []
    used: set[str] = set()
    output_dir.mkdir(parents=True, exist_ok=True)

    for index, obj_path in enumerate(obj_files):
        output_name = safe_output_name(obj_path, collection_root, index, used)
        target = output_dir / output_name
        shutil.copy2(obj_path, target)
        copied.append(
            {
                "source": str(obj_path),
                "output": str(target),
                "name": output_name,
            }
        )
    return copied


def write_manifest(output_dir: Path, file_name: str, payload: dict[str, Any]) -> Path:
    manifest_path = output_dir / file_name
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return manifest_path
