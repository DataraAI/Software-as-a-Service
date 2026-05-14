from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

import argparse
import json
import os
import re
import tempfile
from pathlib import Path
from urllib.parse import urlparse

import requests
import torch


DEFAULT_PROMPT = """
You are annotating a single frame from a task video.

Return ONLY valid JSON exactly in this schema:
{"sub_task":"short_label"}

Rules:
- The label must describe only the visible sub-task in the current frame.
- Use a short conservative label, 1-4 words.
- Use lowercase.
- Do not include explanations, markdown, or extra keys.
- If the frame is ambiguous, return {"sub_task":"unknown"}.
""".strip()

MODEL_ID = os.path.join(
    os.path.expanduser("~"),
    "models",
    "Qwen-VLM",
    "Qwen2.5-VL-7B-Instruct",
)
DEFAULT_OUTPUT = os.path.join(os.path.expanduser("~"), "sub_task_annotations.json")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".mpeg", ".mpg", ".m4v"}
JSON_OBJECT_RE = re.compile(r"\{.*?\}", re.DOTALL)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompt",
        type=str,
        help="input prompt",
        default=DEFAULT_PROMPT,
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--asset_path",
        type=str,
        help="local video path, local image directory, or remote video URL",
    )
    input_group.add_argument(
        "--asset_paths",
        nargs="+",
        help="ordered image paths or image URLs corresponding to frames",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        help="output JSON path",
        default=DEFAULT_OUTPUT,
    )
    parser.add_argument(
        "--frame_stride",
        type=int,
        help="process every nth frame",
        default=1,
    )
    return parser.parse_args()


def is_url(value):
    return isinstance(value, str) and value.startswith(("http://", "https://"))


def get_path_extension(value):
    parsed = urlparse(value) if is_url(value) else None
    source = parsed.path if parsed else value
    return Path(source).suffix.lower()


def resolve_input_kind(asset_path):
    extension = get_path_extension(asset_path)

    if is_url(asset_path):
        if extension in IMAGE_EXTENSIONS:
            raise SystemExit(
                "Remote single-image inputs are not supported. "
                "Use --asset_paths for ordered image URLs, or a remote video URL."
            )
        if extension not in VIDEO_EXTENSIONS:
            raise SystemExit(
                "Unsupported remote asset. Provide a remote video URL with a known video extension."
            )
        return "remote_video"

    if os.path.isdir(asset_path):
        return "image_dir"

    if os.path.isfile(asset_path):
        if extension in IMAGE_EXTENSIONS:
            raise SystemExit(
                "Local single-image inputs are not supported. "
                "Use --asset_paths for ordered image files, a local video file, "
                "or a local image directory."
            )
        if extension not in VIDEO_EXTENSIONS:
            raise SystemExit(
                "Unsupported local file type. Provide a local video file or a local image directory."
            )
        return "local_video"

    raise SystemExit(f"Asset path not found: {asset_path}")


def normalize_output_path(output_json):
    return os.path.abspath(os.path.expanduser(output_json or DEFAULT_OUTPUT))


def normalize_sub_task(value):
    if not isinstance(value, str):
        return "unknown"
    normalized = value.strip().lower()
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized or "unknown"


def extract_sub_task(output_text):
    if not isinstance(output_text, str) or not output_text.strip():
        return "unknown"

    match = JSON_OBJECT_RE.search(output_text.strip())
    if not match:
        return "unknown"

    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return "unknown"

    return normalize_sub_task(payload.get("sub_task"))


def list_image_files(image_dir):
    image_dir_path = Path(image_dir)
    image_files = sorted(
        str(path)
        for path in image_dir_path.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_files:
        raise SystemExit(f"No supported image files found in directory: {image_dir}")
    return image_files


def validate_image_reference(image_reference):
    extension = get_path_extension(image_reference)

    if is_url(image_reference):
        if extension not in IMAGE_EXTENSIONS:
            raise SystemExit(
                f"Unsupported remote image reference: {image_reference}. "
                "Only image URLs are allowed in --asset_paths."
            )
        return image_reference

    if not os.path.isfile(image_reference):
        raise SystemExit(f"Image reference not found: {image_reference}")

    if extension not in IMAGE_EXTENSIONS:
        raise SystemExit(
            f"Unsupported local image reference: {image_reference}. "
            "Only image files are allowed in --asset_paths."
        )

    return image_reference


def build_batch_image_sources(asset_paths, frame_stride):
    if len(asset_paths) < 2:
        raise SystemExit(
            "--asset_paths requires at least two ordered image references. "
            "Single-image inputs are not supported."
        )

    validated_paths = [validate_image_reference(asset_path) for asset_path in asset_paths]
    return [
        (frame_index, image_reference)
        for frame_index, image_reference in enumerate(validated_paths)
        if frame_index % frame_stride == 0
    ]


def download_remote_video(url, tmp_dir):
    extension = get_path_extension(url)
    local_path = os.path.join(tmp_dir, f"downloaded_video{extension or '.mp4'}")

    try:
        with requests.get(url, stream=True, timeout=60) as response:
            response.raise_for_status()
            with open(local_path, "wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        handle.write(chunk)
    except requests.exceptions.RequestException as error:
        raise SystemExit(f"Failed to download remote video: {error}") from error

    return local_path


def extract_video_frames(video_path, tmp_dir, frame_stride):
    try:
        import cv2
    except ImportError as error:
        raise SystemExit(
            "opencv-python is required for video input. Install it before running this script."
        ) from error

    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise SystemExit(f"Could not open video file: {video_path}")

    frames = []
    frame_index = 0

    try:
        while True:
            success, frame = capture.read()
            if not success:
                break

            if frame_index % frame_stride == 0:
                frame_path = os.path.join(tmp_dir, f"frame_{frame_index:08d}.png")
                if not cv2.imwrite(frame_path, frame):
                    raise SystemExit(f"Failed to write extracted frame: {frame_path}")
                frames.append((frame_index, frame_path))

            frame_index += 1
    finally:
        capture.release()

    if not frames:
        raise SystemExit("No frames were extracted from the input video.")

    return frames


def build_frame_sources(args, tmp_dir):
    if args.asset_paths:
        return build_batch_image_sources(args.asset_paths, args.frame_stride)

    asset_path = args.asset_path
    frame_stride = args.frame_stride
    kind = resolve_input_kind(asset_path)

    if kind == "image_dir":
        image_files = list_image_files(asset_path)
        return [
            (frame_index, image_path)
            for frame_index, image_path in enumerate(image_files)
            if frame_index % frame_stride == 0
        ]

    if kind == "local_video":
        return extract_video_frames(asset_path, tmp_dir, frame_stride)

    if kind == "remote_video":
        local_video_path = download_remote_video(asset_path, tmp_dir)
        return extract_video_frames(local_video_path, tmp_dir, frame_stride)

    raise SystemExit(f"Unsupported asset kind: {kind}")


def load_model():
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype="auto",
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    return model, processor


def infer_sub_task(model, processor, frame_image, prompt):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": frame_image,
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    try:
        image_inputs, video_inputs = process_vision_info(messages)
    except (OSError, requests.exceptions.RequestException):
        return "unknown"

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    if torch.cuda.is_available():
        inputs = inputs.to("cuda")

    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    combined_output = output_text[0] if output_text else ""
    return extract_sub_task(combined_output)


def merge_segments(frame_predictions):
    if not frame_predictions:
        return []

    segments = []
    current_label = frame_predictions[0]["sub_task"]
    start_frame = frame_predictions[0]["frame_index"]
    end_frame = frame_predictions[0]["frame_index"]

    for prediction in frame_predictions[1:]:
        frame_index = prediction["frame_index"]
        label = prediction["sub_task"]

        if label == current_label:
            end_frame = frame_index
            continue

        segments.append(
            {
                "sub_task": current_label,
                "start_frame": start_frame,
                "end_frame": end_frame,
            }
        )
        current_label = label
        start_frame = frame_index
        end_frame = frame_index

    segments.append(
        {
            "sub_task": current_label,
            "start_frame": start_frame,
            "end_frame": end_frame,
        }
    )
    return segments


def write_output(segments, output_json):
    output_path = normalize_output_path(output_json)
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(segments, handle, indent=2)

    return output_path


def main():
    args = parse_args()
    if args.frame_stride < 1:
        raise SystemExit("--frame_stride must be >= 1")

    with tempfile.TemporaryDirectory(prefix="qwen_vlm_segments_") as tmp_dir:
        frame_sources = build_frame_sources(args, tmp_dir)
        model, processor = load_model()

        frame_predictions = []
        for frame_index, frame_path in frame_sources:
            sub_task = infer_sub_task(model, processor, frame_path, args.prompt)
            frame_predictions.append(
                {
                    "frame_index": frame_index,
                    "sub_task": sub_task,
                }
            )

        segments = merge_segments(frame_predictions)
        output_path = write_output(segments, args.output_json)
        print(output_path)


if __name__ == "__main__":
    main()
