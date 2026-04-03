"""IO utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def ensure_directory(path: str | Path) -> Path:
    """Create a directory if needed and return its path."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def load_json_file(path: str | Path) -> dict[str, Any]:
    """Load a JSON file into a dictionary."""
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json_file(path: str | Path, payload: dict[str, Any]) -> Path:
    """Write a JSON payload to disk."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return output_path


def read_text_file(path: str | Path) -> str:
    """Read a UTF-8 text file."""
    return Path(path).read_text(encoding="utf-8")


def list_json_files(path: str | Path) -> list[Path]:
    """List JSON files in a directory, excluding generated outputs."""
    directory = Path(path)
    return sorted(
        file_path
        for file_path in directory.glob("*.json")
        if file_path.is_file()
    )
