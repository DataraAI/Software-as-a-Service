"""
io/action_writer.py — Saves the inference results to a unique JSON file.

Output filename format: actions_<timestamp>_<uuid_short>.json
Example: actions_2026-04-01T21-42-10Z_ab12cd34.json

The run_id embedded in the JSON matches the filename suffix so DaaS
can correlate filenames to run metadata without parsing the full file.
"""

import json
import os
import uuid
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


def generate_run_id() -> str:
    """
    Generate a unique run ID using UTC timestamp + short UUID.
    Format: 2026-04-01T21-42-10Z_ab12cd34
    Colons replaced with dashes so the ID is safe as a filename component.
    """
    now = datetime.now(timezone.utc)
    timestamp = now.strftime("%Y-%m-%dT%H-%M-%SZ")
    uid = uuid.uuid4().hex[:8]
    return f"{timestamp}_{uid}"


def build_output_payload(
    run_id: str,
    video_path: Optional[str],
    image_dir: Optional[str],
    task: str,
    backend: str,
    model_id: str,
    actions: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Assemble the full output JSON structure.
    """
    return {
        "run_id": run_id,
        "input": {
            "video_path": video_path,
            "image_dir": image_dir,
            "task": task,
        },
        "backend": backend,
        "model_id": model_id,
        "actions": actions,
    }


def save_actions(
    payload: Dict[str, Any],
    output_dir: str,
    run_id: str,
) -> str:
    """
    Write the payload to a unique JSON file in output_dir.

    Returns the full path to the written file.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    filename = f"actions_{run_id}.json"
    output_path = os.path.join(output_dir, filename)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved actions to: {output_path}")
    return output_path
