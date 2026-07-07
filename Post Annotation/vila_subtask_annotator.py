import argparse
import json
import os
import re
import shlex
import subprocess
from pathlib import Path


DEFAULT_PROMPT = (
    "Describe the overall task and the chronological steps (subTasks) the individual takes to complete it in chronological order."
)

MODEL_ID = os.getenv("VILA_MODEL_PATH") or "Efficient-Large-Model/NVILA-8B"
DEFAULT_OUTPUT = "sub_task_annotations.json"
DEFAULT_CONDA_ENV = os.getenv("VILA_CONDA_ENV") or "vila"
DEFAULT_CONV_MODE = os.getenv("VILA_CONV_MODE") or "auto"
DEFAULT_VILA_INFER_BIN = os.getenv("VILA_INFER_BIN") or "vila-infer"
JSON_VALUE_RE = re.compile(r"(\{.*\}|\[.*\])", re.DOTALL)
FRAME_RANGE_RE = re.compile(
    r"(?:frames?|frame\s*range)\s*[:#]?\s*(\d+)\s*(?:-|to|through|\u2013|\u2014)\s*(\d+)",
    re.IGNORECASE,
)
START_END_FRAME_RE = re.compile(
    r"(?:start\s*frame|startFrame|start)\D{0,20}(\d+).*?"
    r"(?:end\s*frame|endFrame|end)\D{0,20}(\d+)",
    re.IGNORECASE,
)
BRACKET_FRAME_RANGE_RE = re.compile(
    r"\[\s*(\d+)\s*(?:-|to|through|\u2013|\u2014)\s*(\d+)\s*\]",
    re.IGNORECASE,
)
STEP_LINE_RE = re.compile(
    r"^\s*(?:[-*\u2022]\s+|\d+[.)]\s+|(?:step|subtask|task)\s*\d+\s*[:.)-]\s*)(.+)$",
    re.IGNORECASE,
)
MODEL_PREFIX_RE = re.compile(r"^\s*(?:assistant|vila|output|answer)\s*:\s*", re.IGNORECASE)


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
        "--raw_output",
        type=str,
        help="optional path for raw vila-infer stdout",
        default="",
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
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
exit_code=0
{vila_command} || exit_code=$?
conda deactivate
exit $exit_code
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
        return None

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

    return None


def normalize_description(value, default="unknown"):
    if not isinstance(value, str):
        return default
    value = re.sub(r"\s+", " ", value.strip())
    return value or default


def normalize_frame(value):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def first_present(mapping, keys):
    normalized = {str(key).lower(): value for key, value in mapping.items()}
    for key in keys:
        if key in mapping:
            return mapping[key]
        normalized_key = str(key).lower()
        if normalized_key in normalized:
            return normalized[normalized_key]
    return None


def clean_output_text(output_text):
    text = output_text or ""
    text = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE)
    text = text.replace("```", "")
    lines = []
    for line in text.splitlines():
        stripped = MODEL_PREFIX_RE.sub("", line.strip())
        if stripped:
            lines.append(stripped)
    return "\n".join(lines).strip()


def parse_frame_range(text):
    if not isinstance(text, str):
        return 0, 0
    for pattern in (START_END_FRAME_RE, FRAME_RANGE_RE, BRACKET_FRAME_RANGE_RE):
        match = pattern.search(text)
        if match:
            start_frame = normalize_frame(match.group(1))
            end_frame = normalize_frame(match.group(2))
            return start_frame, max(start_frame, end_frame)
    return 0, 0


def strip_frame_markers(text):
    if not isinstance(text, str):
        return ""
    text = START_END_FRAME_RE.sub("", text)
    text = FRAME_RANGE_RE.sub("", text)
    text = BRACKET_FRAME_RANGE_RE.sub("", text)
    text = re.sub(r"\s*[:;-]\s*$", "", text.strip())
    text = re.sub(r"^\s*[:;-]\s*", "", text)
    return normalize_description(text)


def extract_segment_items(payload):
    if isinstance(payload, dict):
        for key in ("subTasks", "subtasks", "steps", "actions", "segments", "tasks"):
            value = first_present(payload, (key,))
            if isinstance(value, list):
                return extract_segment_items(value)
        return [payload]

    if isinstance(payload, list):
        items = []
        for item in payload:
            if isinstance(item, dict) and any(
                isinstance(first_present(item, (key,)), list)
                for key in ("subTasks", "subtasks", "steps", "actions", "segments", "tasks")
            ):
                items.extend(extract_segment_items(item))
            else:
                items.append(item)
        return items

    return payload


def extract_task_description(payload, raw_output):
    if isinstance(payload, dict):
        value = first_present(
            payload,
            (
                "taskDescription",
                "task_description",
                "taskName",
                "task_name",
                "description",
                "summary",
            ),
        )
        if isinstance(value, str) and value.strip():
            return normalize_description(value)

        tasks = first_present(payload, ("tasks",))
        if isinstance(tasks, list) and tasks:
            descriptions = [
                extract_task_description(task, "")
                for task in tasks
                if isinstance(task, dict)
            ]
            descriptions = [description for description in descriptions if description != "unknown"]
            if descriptions:
                return normalize_description(" ".join(descriptions))

    if isinstance(payload, list):
        descriptions = [
            extract_task_description(item, "")
            for item in payload
            if isinstance(item, dict)
        ]
        descriptions = [description for description in descriptions if description != "unknown"]
        if descriptions:
            return normalize_description(" ".join(descriptions))

    return normalize_description(clean_output_text(raw_output))


def sentence_tokenize(text):
    try:
        import nltk

        try:
            return nltk.sent_tokenize(text)
        except Exception:
            pass
    except Exception:
        pass

    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def split_text_subtasks(output_text):
    text = clean_output_text(output_text)
    if not text:
        return "unknown", ["unknown"]

    intro_lines = []
    step_lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        match = STEP_LINE_RE.match(stripped)
        if match:
            step_lines.append(match.group(1).strip())
        elif step_lines:
            step_lines[-1] = f"{step_lines[-1]} {stripped}"
        else:
            intro_lines.append(stripped)

    if step_lines:
        task_description = normalize_description(" ".join(intro_lines), normalize_description(text))
        return task_description, step_lines

    sentences = sentence_tokenize(text)
    if len(sentences) > 1:
        return normalize_description(text), sentences
    return normalize_description(text), [text]


def normalize_segments(payload, raw_output):
    payload = extract_segment_items(payload)
    if not isinstance(payload, list):
        payload = []

    segments = []
    for item in payload:
        if isinstance(item, str):
            start_frame, end_frame = parse_frame_range(item)
            segment = {
                "startFrame": start_frame,
                "endFrame": end_frame,
                "subTaskDescription": strip_frame_markers(item),
            }
        elif isinstance(item, dict):
            label = (
                first_present(
                    item,
                    (
                        "subTaskDescription",
                        "sub_task_description",
                        "sub_task",
                        "subtask",
                        "subtask_name",
                        "step",
                        "action",
                        "description",
                        "taskDescription",
                        "task_name",
                    ),
                )
                or "unknown"
            )
            start_frame = normalize_frame(
                first_present(item, ("startFrame", "start_frame", "frame_start", "start"))
            )
            end_frame = normalize_frame(
                first_present(item, ("endFrame", "end_frame", "frame_end", "end"))
            )
            if not start_frame and not end_frame:
                start_frame, end_frame = parse_frame_range(label)
            segment = {
                "startFrame": start_frame,
                "endFrame": max(start_frame, end_frame),
                "subTaskDescription": strip_frame_markers(label),
            }
        else:
            continue
        segments.append(segment)

    if segments:
        return segments

    _, step_texts = split_text_subtasks(raw_output)
    for step_text in step_texts:
        start_frame, end_frame = parse_frame_range(step_text)
        segments.append(
            {
                "startFrame": start_frame,
                "endFrame": end_frame,
                "subTaskDescription": strip_frame_markers(step_text),
            }
        )
    return segments or [{"startFrame": 0, "endFrame": 0, "subTaskDescription": "unknown"}]


def normalize_annotation(raw_output):
    payload = parse_first_json_value(raw_output)
    if payload is None:
        task_description, _ = split_text_subtasks(raw_output)
    else:
        task_description = extract_task_description(payload, raw_output)

    return {
        "taskDescription": task_description,
        "subTasks": normalize_segments(payload, raw_output),
    }


def write_output(annotation, output_json):
    output_path = normalize_output_path(output_json)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(annotation, handle, indent=2)
    return output_path


def main():
    args = parse_args()
    prompt = build_prompt(args.prompt)
    raw_output = run_vila_infer(args, prompt)
    if args.raw_output:
        raw_output_path = os.path.abspath(os.path.expanduser(args.raw_output))
        Path(raw_output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(raw_output_path, "w", encoding="utf-8") as handle:
            handle.write(raw_output)
            if raw_output and not raw_output.endswith("\n"):
                handle.write("\n")

    annotation = normalize_annotation(raw_output)
    output_path = write_output(annotation, args.output_json)
    print(output_path)


if __name__ == "__main__":
    main()
