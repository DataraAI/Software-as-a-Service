import argparse
import json
import os
import re
import shlex
import subprocess
from pathlib import Path


DEFAULT_PROMPT = """
Describe the steps the individual takes in the video in chronological order.

Return ONLY valid JSON exactly in this schema:
[
  {"sub_task":"short_label","start_frame":0,"end_frame":0}
]

Rules:
- List the task-level steps in chronological order.
- Use short conservative sub_task labels.
- Use lowercase labels.
- Use integer frame numbers when they are clear.
- If exact frame numbers are uncertain, use 0 for start_frame and end_frame.
- Do not include explanations, markdown, or extra keys.
- If the video is ambiguous, return [{"sub_task":"unknown","start_frame":0,"end_frame":0}].
""".strip()

MODEL_ID = os.getenv("VILA_MODEL_PATH") or "Efficient-Large-Model/VILA1.5-3b"
DEFAULT_OUTPUT = os.path.join(os.path.expanduser("~"), "sub_task_annotations.json")
DEFAULT_CONDA_ENV = os.getenv("VILA_CONDA_ENV") or "vila"
DEFAULT_CONV_MODE = os.getenv("VILA_CONV_MODE") or "vicuna_v1"
DEFAULT_VILA_INFER_BIN = os.getenv("VILA_INFER_BIN") or "vila-infer"
JSON_VALUE_RE = re.compile(r"(\{.*\}|\[.*\])", re.DOTALL)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompt",
        type=str,
        help="input prompt",
        default=DEFAULT_PROMPT,
    )
    parser.add_argument(
        "--asset_path",
        type=str,
        required=True,
        help="video URL or local video path",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        help="output JSON path",
        default=DEFAULT_OUTPUT,
    )
    parser.add_argument(
        "--model_path",
        type=str,
        help="VILA model path",
        default=MODEL_ID,
    )
    parser.add_argument(
        "--conv_mode",
        type=str,
        help="VILA conversation mode",
        default=DEFAULT_CONV_MODE,
    )
    parser.add_argument(
        "--conda_env",
        type=str,
        help="conda environment containing vila-infer",
        default=DEFAULT_CONDA_ENV,
    )
    parser.add_argument(
        "--vila_infer_bin",
        type=str,
        help="vila-infer executable name or path",
        default=DEFAULT_VILA_INFER_BIN,
    )
    return parser.parse_args()


def normalize_output_path(output_json):
    return os.path.abspath(os.path.expanduser(output_json or DEFAULT_OUTPUT))


def build_prompt(prompt):
    return prompt.strip() or DEFAULT_PROMPT


def build_conda_script(args, prompt):
    vila_command = " ".join(
        [
            shlex.quote(args.vila_infer_bin),
            "--model-path",
            shlex.quote(args.model_path),
            "--conv-mode",
            shlex.quote(args.conv_mode),
            "--text",
            shlex.quote(prompt),
            "--media",
            shlex.quote(args.asset_path),
        ]
    )
    return f"""
set -euo pipefail
for conda_sh in "$HOME/miniconda3/etc/profile.d/conda.sh" "$HOME/anaconda3/etc/profile.d/conda.sh" "/opt/conda/etc/profile.d/conda.sh"; do
    if [ -f "$conda_sh" ]; then
        source "$conda_sh"
        break
    fi
done
conda activate {shlex.quote(args.conda_env)}
trap 'conda deactivate >/dev/null 2>&1 || true' EXIT
{vila_command}
""".strip()


def run_vila_infer(args, prompt):
    script = build_conda_script(args, prompt)
    completed = subprocess.run(
        ["bash", "-lc", script],
        check=False,
        text=True,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise SystemExit(
            "vila-infer failed with exit code "
            f"{completed.returncode}: {completed.stderr.strip() or completed.stdout.strip()}"
        )
    return completed.stdout.strip()


def parse_first_json_value(output_text):
    if not output_text:
        return [{"sub_task": "unknown", "start_frame": 0, "end_frame": 0}]

    decoder = json.JSONDecoder()
    parsed_values = []
    for index, character in enumerate(output_text):
        if character not in "[{":
            continue
        try:
            payload, end_index = decoder.raw_decode(output_text[index:])
            parsed_values.append((index + end_index, payload))
        except json.JSONDecodeError:
            continue

    if parsed_values:
        return max(parsed_values, key=lambda item: item[0])[1]

    match = JSON_VALUE_RE.search(output_text)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    return [{"sub_task": "unknown", "start_frame": 0, "end_frame": 0}]


def normalize_sub_task(value):
    if not isinstance(value, str):
        return "unknown"
    value = re.sub(r"\s+", " ", value.strip().lower())
    return value or "unknown"


def normalize_frame(value):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def first_present(mapping, keys):
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def normalize_segments(payload):
    if isinstance(payload, dict):
        if isinstance(payload.get("subtasks"), list):
            payload = payload["subtasks"]
        elif isinstance(payload.get("steps"), list):
            payload = payload["steps"]
        else:
            payload = [payload]

    if not isinstance(payload, list):
        payload = [{"sub_task": "unknown", "start_frame": 0, "end_frame": 0}]

    segments = []
    for item in payload:
        if isinstance(item, str):
            segment = {
                "sub_task": normalize_sub_task(item),
                "start_frame": 0,
                "end_frame": 0,
            }
        elif isinstance(item, dict):
            label = (
                item.get("sub_task")
                or item.get("subtask")
                or item.get("subtask_name")
                or item.get("step")
                or item.get("action")
                or item.get("description")
                or "unknown"
            )
            start_frame = normalize_frame(
                first_present(item, ("start_frame", "frame_start", "start", "startFrame"))
            )
            end_frame = normalize_frame(
                first_present(item, ("end_frame", "frame_end", "end", "endFrame"))
            )
            segment = {
                "sub_task": normalize_sub_task(label),
                "start_frame": start_frame,
                "end_frame": max(start_frame, end_frame),
            }
        else:
            continue
        segments.append(segment)

    return segments or [{"sub_task": "unknown", "start_frame": 0, "end_frame": 0}]


def write_output(segments, output_json):
    output_path = normalize_output_path(output_json)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(segments, handle, indent=2)
    return output_path


def main():
    args = parse_args()
    prompt = build_prompt(args.prompt)
    raw_output = run_vila_infer(args, prompt)
    payload = parse_first_json_value(raw_output)
    segments = normalize_segments(payload)
    output_path = write_output(segments, args.output_json)
    print(output_path)


if __name__ == "__main__":
    main()
