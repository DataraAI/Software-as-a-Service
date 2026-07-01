from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent
HOME = Path.home()

SAAS_EGO_PYTHON_BIN = os.getenv("SAAS_EGO_PYTHON_BIN", str(HOME / "miniconda3" / "envs" / "qwen-angles-2509" / "bin" / "python"))
SAAS_VLM_PYTHON_BIN = os.getenv("SAAS_VLM_PYTHON_BIN", str(HOME / "miniconda3" / "envs" / "qwen-vlm" / "bin" / "python"))
SAAS_CORNER_PYTHON_BIN = os.getenv("SAAS_CORNER_PYTHON_BIN", str(HOME / "miniconda3" / "envs" / "addit-sam2" / "bin" / "python"))
SAAS_SEGMENTATION_PYTHON_BIN = os.getenv("SAAS_PYTHON_BIN", str(HOME / "miniconda3" / "envs" / "sam3" / "bin" / "python"))
SAAS_ROSE_PYTHON_BIN = os.getenv("SAAS_ROSE_PYTHON_BIN", str(HOME / "miniconda3" / "envs" / "rose_runtime" / "bin" / "python"))
SAAS_VIPE_PYTHON_BIN = os.getenv("SAAS_VIPE_PYTHON_BIN", str(HOME / "miniconda3" / "envs" / "vipe" / "bin" / "python"))
SAAS_TASK_INTELLIGENCE_PYTHON_BIN = os.getenv("SAAS_TASK_INTELLIGENCE_PYTHON_BIN", sys.executable)
SAAS_RUNNER_PYTHON_BIN = os.getenv("SAAS_RUNNER_PYTHON_BIN", sys.executable)


def _stderr(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Required file was not found: {path}")
    return path


def _last_non_empty_line(text: str) -> str:
    for line in reversed((text or "").splitlines()):
        candidate = line.strip()
        if candidate:
            return candidate
    return ""


def _redact_command_arg(value: str) -> str:
    text = str(value)
    if text.startswith(("http://", "https://")) and "?" in text:
        return f"{text.split('?', 1)[0]}?[redacted]"
    return text


def _run_command(command: list[str], *, cwd: Path | None = None, extra_env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONPATH"] = f"{HOME}:{REPO_ROOT}:{env.get('PYTHONPATH', '')}".rstrip(":")
    if extra_env:
        env.update(extra_env)

    _stderr(f"[saas_runner] Running: {' '.join(_redact_command_arg(arg) for arg in command)}")
    return subprocess.run(
        command,
        cwd=str(cwd or REPO_ROOT),
        env=env,
        text=True,
        capture_output=True,
    )


def _result(status: str, *, artifacts: dict[str, Any] | None = None, error: str | None = None, logs: dict[str, str] | None = None) -> dict[str, Any]:
    return {
        "status": status,
        "artifacts": artifacts or {},
        "error": error,
        "logs": logs or {},
    }


def _run_and_parse_path(command: list[str], *, cwd: Path | None = None, extra_env: dict[str, str] | None = None) -> dict[str, Any]:
    completed = _run_command(command, cwd=cwd, extra_env=extra_env)
    stdout = completed.stdout or ""
    stderr = completed.stderr or ""
    final_path = _last_non_empty_line(stdout)
    if completed.returncode != 0:
        return _result("failed", error=stderr or stdout or "Remote script failed", logs={"stdout": stdout, "stderr": stderr})
    if not final_path:
        return _result("failed", error="Remote script did not return an output path", logs={"stdout": stdout, "stderr": stderr})
    return _result("succeeded", artifacts={"output_path": final_path}, logs={"stdout": stdout, "stderr": stderr})


def handle_generate_ego(request: dict[str, Any]) -> dict[str, Any]:
    script = _require_file(REPO_ROOT / "image_prompt_tool.py")
    return _run_and_parse_path(
        [
            SAAS_EGO_PYTHON_BIN,
            str(script),
            "--prompt",
            str(request["prompt"]),
            "--imageURL",
            str(request["image_url"]),
            "--container_name",
            str(request["container_name"]),
        ]
    )


def handle_generate_corner_case(request: dict[str, Any]) -> dict[str, Any]:
    script = _require_file(REPO_ROOT / "Corner_case_tool.py")
    return _run_and_parse_path(
        [
            SAAS_CORNER_PYTHON_BIN,
            str(script),
            "--prompt",
            str(request["prompt"]),
            "--imageURL",
            str(request["image_url"]),
            "--container_name",
            str(request["container_name"]),
            "--localization_model",
            str(request.get("localization_model") or "attention_points_sam"),
            "--out_root",
            str(request.get("out_root") or "corner_images_controlnet"),
        ]
    )


def handle_create_vlm_tags(request: dict[str, Any]) -> dict[str, Any]:
    script = _require_file(REPO_ROOT / "Post Annotation" / "qwen_vlm_image.py")
    return _run_and_parse_path(
        [
            SAAS_VLM_PYTHON_BIN,
            str(script),
            "--prompt",
            str(request["prompt"]),
            "--egoURL",
            str(request["image_url"]),
        ]
    )


def handle_generate_masks(request: dict[str, Any]) -> dict[str, Any]:
    script = _require_file(REPO_ROOT / "DataraAI_segmentation.py")
    completed = _run_command(
        [
            SAAS_SEGMENTATION_PYTHON_BIN,
            str(script),
            "--input_mode",
            "folder",
            "--image_dir",
            str(request["image_dir"]),
            "--segment",
            str(request["prompt"]),
            "--output_dir",
            str(request["output_dir"]),
        ]
    )
    stdout = completed.stdout or ""
    stderr = completed.stderr or ""
    output_dir = _last_non_empty_line(stdout)
    if completed.returncode != 0:
        return _result("failed", error=stderr or stdout or "Mask generation failed", logs={"stdout": stdout, "stderr": stderr})
    if not output_dir:
        output_dir = str(request["output_dir"])
    if not Path(output_dir).is_dir():
        return _result("failed", error="Mask generation completed without an output directory", logs={"stdout": stdout, "stderr": stderr})
    return _result("succeeded", artifacts={"output_dir": output_dir}, logs={"stdout": stdout, "stderr": stderr})


def handle_remove_occlusion(request: dict[str, Any]) -> dict[str, Any]:
    script = _require_file(REPO_ROOT / "DataraAI_rose_occlusion.py")
    completed = _run_command(
        [
            SAAS_ROSE_PYTHON_BIN,
            str(script),
            "--source_video",
            str(request["source_video"]),
            "--mask_video",
            str(request["mask_video"]),
            "--output_dir",
            str(request["output_dir"]),
            "--prompt",
            str(request["prompt"]),
            "--video_length",
            str(request.get("video_length") or 49),
            "--sample_height",
            str(request["sample_height"]),
            "--sample_width",
            str(request["sample_width"]),
        ]
    )
    stdout = completed.stdout or ""
    stderr = completed.stderr or ""
    output_path = _last_non_empty_line(stdout)
    if completed.returncode != 0:
        return _result("failed", error=stderr or stdout or "Occlusion removal failed", logs={"stdout": stdout, "stderr": stderr})
    if not output_path or not Path(output_path).is_file():
        return _result("failed", error="Occlusion removal completed without an output video", logs={"stdout": stdout, "stderr": stderr})
    return _result("succeeded", artifacts={"output_path": output_path}, logs={"stdout": stdout, "stderr": stderr})


def handle_generate_hand_mesh(request: dict[str, Any]) -> dict[str, Any]:
    script = _require_file(REPO_ROOT / "run_vipe_dynhamr.py")
    completed = _run_command(
        [
            SAAS_VIPE_PYTHON_BIN,
            str(script),
            "--video-url",
            str(request["video_url"]),
            "--seq",
            str(request["seq"]),
            "--pipeline",
            str(request.get("pipeline") or "default"),
        ]
    )
    stdout = completed.stdout or ""
    stderr = completed.stderr or ""
    if completed.returncode != 0:
        return _result("failed", error=stderr or stdout or "Hand mesh generation failed", logs={"stdout": stdout, "stderr": stderr})

    video_paths: list[str] = []
    artifact_paths: list[str] = []
    mcap_paths: list[str] = []
    npz_paths: list[str] = []
    for line in stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith("OUTPUT_VIDEO: "):
            video_paths.append(stripped.split("OUTPUT_VIDEO: ", 1)[1].strip())
        elif stripped.startswith("OUTPUT_OBJ: "):
            artifact_paths.append(stripped.split("OUTPUT_OBJ: ", 1)[1].strip())
        elif stripped.startswith("OUTPUT_MCAP: "):
            mcap_paths.append(stripped.split("OUTPUT_MCAP: ", 1)[1].strip())
        elif stripped.startswith("OUTPUT_NPZ: "):
            npz_paths.append(stripped.split("OUTPUT_NPZ: ", 1)[1].strip())
    if not video_paths and not artifact_paths and not mcap_paths and not npz_paths:
        return _result("failed", error="Hand mesh runner completed without outputs", logs={"stdout": stdout, "stderr": stderr})
    return _result(
        "succeeded",
        artifacts={
            "video_paths": video_paths,
            "artifact_paths": artifact_paths,
            "mcap_paths": mcap_paths,
            "npz_paths": npz_paths,
        },
        logs={"stdout": stdout, "stderr": stderr},
    )


def handle_generate_task_intelligence(request: dict[str, Any]) -> dict[str, Any]:
    script = _require_file(REPO_ROOT / "Post Annotation" / "vila_subtask_annotator.py")
    return _run_and_parse_path(
        [
            SAAS_TASK_INTELLIGENCE_PYTHON_BIN,
            str(script),
            "--asset_path",
            str(request["video_url"]),
        ]
    )


def handle_generate_video_to_video_views(request: dict[str, Any]) -> dict[str, Any]:
    script = _require_file(REPO_ROOT / "lyra_v2v.py")
    command = [
        SAAS_RUNNER_PYTHON_BIN,
        str(script),
        "--video_url",
        str(request["video_url"]),
        "--output_dir",
        str(request["output_dir"]),
        "--trajectory",
        str(request.get("trajectory") or "left"),
    ]
    vipe_zip_url = str(request.get("vipe_zip_url") or "").strip()
    if vipe_zip_url:
        command.extend(["--vipe_zip_url", vipe_zip_url])

    completed = _run_command(command)
    stdout = completed.stdout or ""
    stderr = completed.stderr or ""
    if completed.returncode != 0:
        return _result("failed", error=stderr or stdout or "Video-to-video generation failed", logs={"stdout": stdout, "stderr": stderr})

    last_line = _last_non_empty_line(stdout)
    try:
        payload = json.loads(last_line)
    except json.JSONDecodeError:
        return _result("failed", error="Video-to-video runner returned invalid JSON", logs={"stdout": stdout, "stderr": stderr})
    return _result(
        "succeeded",
        artifacts={
            "video_path": payload.get("gen3c_video"),
            "vipe_zip_path": payload.get("vipe_zip"),
        },
        logs={"stdout": stdout, "stderr": stderr},
    )


HANDLERS = {
    "generate_ego": handle_generate_ego,
    "generate_corner_case": handle_generate_corner_case,
    "create_vlm_tags": handle_create_vlm_tags,
    "generate_masks": handle_generate_masks,
    "remove_occlusion": handle_remove_occlusion,
    "generate_hand_mesh": handle_generate_hand_mesh,
    "generate_task_intelligence": handle_generate_task_intelligence,
    "generate_video_to_video_views": handle_generate_video_to_video_views,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize Datara SaaS job execution.")
    parser.add_argument("--job-type", required=True)
    parser.add_argument("--request", required=True)
    parser.add_argument("--response", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    handler = HANDLERS.get(args.job_type)
    response_path = Path(args.response).expanduser().resolve()
    response_path.parent.mkdir(parents=True, exist_ok=True)

    if handler is None:
        response = _result("failed", error=f"Unsupported job type: {args.job_type}")
        response_path.write_text(json.dumps(response, indent=2), encoding="utf-8")
        return 1

    request_path = Path(args.request).expanduser().resolve()
    request_payload = json.loads(request_path.read_text(encoding="utf-8"))

    try:
        response = handler(request_payload)
    except Exception as exc:
        response = _result("failed", error=str(exc))

    response_path.write_text(json.dumps(response, indent=2), encoding="utf-8")
    return 0 if response["status"] == "succeeded" else 1


if __name__ == "__main__":
    raise SystemExit(main())
