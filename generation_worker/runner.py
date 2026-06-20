"""Adapter around the normalized SaaS runner registry."""

from __future__ import annotations

import mimetypes
import os
import shutil
import tempfile
from typing import Any
from urllib.parse import quote, urlsplit, urlunsplit

import requests

from saas_runner import HANDLERS


def _download_file(url: str, destination: str) -> None:
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    with requests.get(url, stream=True, timeout=(30, 3600)) as response:
        response.raise_for_status()
        with open(destination, "wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)


def _upload_file(upload_url: str, local_path: str, blob_name: str, content_type: str) -> None:
    parsed = urlsplit(upload_url)
    path = f"{parsed.path.rstrip('/')}/{quote(blob_name, safe='/')}"
    blob_url = urlunsplit((parsed.scheme, parsed.netloc, path, parsed.query, ""))
    with open(local_path, "rb") as handle:
        response = requests.put(
            blob_url,
            data=handle,
            headers={"x-ms-blob-type": "BlockBlob", "Content-Type": content_type},
            timeout=(30, 3600),
        )
    response.raise_for_status()


def _materialize_request(job_type: str, execution: dict[str, Any], workspace: str) -> dict[str, Any]:
    request = {
        key: value
        for key, value in execution.items()
        if key not in {"contract_version", "output_upload_url"}
    }
    output_dir = os.path.join(workspace, "outputs")
    if job_type == "generate_masks":
        input_dir = os.path.join(workspace, "inputs")
        for item in request.pop("input_files", []):
            if not isinstance(item, dict):
                raise ValueError("Invalid mask input")
            name = os.path.basename(str(item.get("name") or ""))
            if not name:
                raise ValueError("Invalid mask input name")
            _download_file(str(item.get("url") or ""), os.path.join(input_dir, name))
        request["image_dir"] = input_dir
        request["output_dir"] = output_dir
    elif job_type == "remove_occlusion":
        input_dir = os.path.join(workspace, "inputs")
        source_video = os.path.join(input_dir, "source.mp4")
        mask_video = os.path.join(input_dir, "mask.mp4")
        _download_file(str(request.pop("source_video_url", "")), source_video)
        _download_file(str(request.pop("mask_video_url", "")), mask_video)
        request["source_video"] = source_video
        request["mask_video"] = mask_video
        request["output_dir"] = output_dir
    elif job_type == "generate_video_to_video_views":
        request["output_dir"] = output_dir
    return request


def _artifact_files(artifacts: dict[str, Any]) -> list[tuple[str, str]]:
    collected: list[tuple[str, str]] = []

    def add_file(role: str, path: str) -> None:
        if os.path.isfile(path):
            collected.append((f"outputs/{role}/{os.path.basename(path)}", path))

    def add_directory(role: str, path: str) -> None:
        if not os.path.isdir(path):
            return
        for root, _dirs, files in os.walk(path):
            for filename in sorted(files):
                local_path = os.path.join(root, filename)
                relative = os.path.relpath(local_path, path).replace("\\", "/")
                collected.append((f"outputs/{role}/{relative}", local_path))

    if artifacts.get("output_path"):
        add_file("result", str(artifacts["output_path"]))
    if artifacts.get("output_dir"):
        add_directory("result", str(artifacts["output_dir"]))
    for path in artifacts.get("video_paths", []) if isinstance(artifacts.get("video_paths"), list) else []:
        add_file("videos", str(path))
    for path in artifacts.get("artifact_paths", []) if isinstance(artifacts.get("artifact_paths"), list) else []:
        if os.path.isdir(str(path)):
            add_directory("artifacts", str(path))
        else:
            add_file("artifacts", str(path))
    for path in artifacts.get("mcap_paths", []) if isinstance(artifacts.get("mcap_paths"), list) else []:
        add_file("mcaps", str(path))
    for path in artifacts.get("npz_paths", []) if isinstance(artifacts.get("npz_paths"), list) else []:
        add_file("npz", str(path))
    if artifacts.get("video_path"):
        add_file("video", str(artifacts["video_path"]))
    if artifacts.get("vipe_zip_path"):
        add_file("vipe", str(artifacts["vipe_zip_path"]))
    return collected


def run_generation(
    job_type: str,
    execution: dict[str, Any],
    *,
    upload_file=_upload_file,
) -> dict[str, Any]:
    if int(execution.get("contract_version") or 0) != 1:
        return {"status": "failed", "error": "Unsupported generation contract"}
    upload_url = str(execution.get("output_upload_url") or "")
    if not upload_url.startswith("https://"):
        return {"status": "failed", "error": "Generation output transfer is unavailable"}
    handler = HANDLERS.get(job_type)
    if handler is None:
        return {"status": "failed", "error": "Unsupported generation request"}

    workspace = tempfile.mkdtemp(prefix="datara_generation_")
    try:
        request = _materialize_request(job_type, execution, workspace)
        response = handler(request)
        if response.get("status") != "succeeded":
            return {"status": "failed", "error": str(response.get("error") or "Generation request failed")}
        artifacts = response.get("artifacts")
        if not isinstance(artifacts, dict):
            return {"status": "failed", "error": "Generation request returned invalid artifacts"}
        files = _artifact_files(artifacts)
        if not files:
            return {"status": "failed", "error": "Generation request returned no artifacts"}
        manifest = []
        for blob_name, local_path in files:
            content_type = mimetypes.guess_type(local_path)[0] or "application/octet-stream"
            upload_file(upload_url, local_path, blob_name, content_type)
            manifest.append(
                {
                    "name": blob_name,
                    "size": os.path.getsize(local_path),
                    "content_type": content_type,
                }
            )
        return {
            "status": "succeeded",
            "manifest": {"contract_version": 1, "artifacts": manifest},
        }
    except Exception as exc:
        return {"status": "failed", "error": str(exc)}
    finally:
        shutil.rmtree(workspace, ignore_errors=True)
