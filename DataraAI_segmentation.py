import argparse
import re
import tempfile
from pathlib import Path

import cv2
import numpy as np

from packages.sam3.sam3.model_builder import build_sam3_video_predictor

VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}
DEFAULT_FPS = 30.0


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
        if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS
    ]
    return sorted(images, key=image_sort_key)


def normalize_frame(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    return frame


def build_temp_video(
    image_paths: list[Path],
    output_path: Path,
    fps: float = DEFAULT_FPS,
) -> tuple[Path, tuple[int, int]]:
    if not image_paths:
        raise ValueError("No source images were provided")

    first_frame = cv2.imread(str(image_paths[0]), cv2.IMREAD_UNCHANGED)
    if first_frame is None:
        raise ValueError(f"Could not read {image_paths[0]}")
    first_frame = normalize_frame(first_frame)
    height, width = first_frame.shape[:2]

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

    return output_path, (height, width)


def propagate_in_video(predictor, session_id):
    outputs_per_frame = {}
    for response in predictor.handle_stream_request(
        request=dict(
            type="propagate_in_video",
            session_id=session_id,
        )
    ):
        outputs_per_frame[response["frame_index"]] = response["outputs"]
    return outputs_per_frame


def mask_generation(video_path: Path, segment: str = "humans"):
    predictor = build_sam3_video_predictor()
    response = predictor.handle_request(
        request=dict(
            type="start_session",
            resource_path=str(video_path),
        )
    )
    session_id = response["session_id"]

    try:
        predictor.handle_request(
            request=dict(
                type="add_prompt",
                session_id=session_id,
                frame_index=0,
                text=segment,
            )
        )
        return propagate_in_video(predictor, session_id)
    finally:
        predictor.handle_request(
            request=dict(
                type="close_session",
                session_id=session_id,
            )
        )
        predictor.shutdown()


def infer_mask_arrays(output: dict) -> list[np.ndarray]:
    masks = []
    for raw_mask in output.get("out_binary_masks", []):
        mask = np.asarray(raw_mask).squeeze()
        if mask.size == 0:
            continue
        masks.append(mask.astype(bool))
    return masks


def fallback_frame_name(frame_index: int) -> str:
    return f"frame_{frame_index:04d}.png"


def object_folder_name(object_id: str) -> str:
    return f"object_{object_id}"


def save_mask_image(output_path: Path, mask_frame: np.ndarray) -> None:
    if not cv2.imwrite(str(output_path), mask_frame):
        raise RuntimeError(f"Could not write mask image to {output_path}")


def write_instance_masks(
    outputs_per_frame: dict[int, dict],
    output_dir: Path,
    frame_names: list[str],
    frame_size: tuple[int, int],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    height, width = frame_size

    object_ids: list[str] = []
    for frame_index in sorted(outputs_per_frame):
        output = outputs_per_frame[frame_index]
        mask_arrays = infer_mask_arrays(output)
        obj_ids = list(output.get("out_obj_ids", []))
        for index, _mask in enumerate(mask_arrays):
            object_id = str(int(obj_ids[index])) if index < len(obj_ids) else str(index)
            if object_id not in object_ids:
                object_ids.append(object_id)

    if not object_ids:
        raise ValueError("No mask instances were returned by SAM3")

    blank_mask = np.zeros((height, width), dtype=np.uint8)
    object_dirs = {
        object_id: output_dir / object_folder_name(object_id)
        for object_id in object_ids
    }
    for directory in object_dirs.values():
        directory.mkdir(parents=True, exist_ok=True)

    for frame_index in sorted(outputs_per_frame):
        output = outputs_per_frame[frame_index]
        mask_arrays = infer_mask_arrays(output)
        output_name = frame_names[frame_index] if frame_index < len(frame_names) else fallback_frame_name(frame_index)
        obj_ids = list(output.get("out_obj_ids", []))

        frame_masks_by_object: dict[str, np.ndarray] = {}
        for index, mask in enumerate(mask_arrays):
            object_id = str(int(obj_ids[index])) if index < len(obj_ids) else str(index)
            frame_masks_by_object[object_id] = (mask.astype(np.uint8)) * 255

        for object_id, object_dir in object_dirs.items():
            mask_frame = frame_masks_by_object.get(object_id, blank_mask)
            save_mask_image(object_dir / output_name, mask_frame)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_mode", choices=["folder"], default="folder")
    parser.add_argument("--image_dir", type=Path, required=True)
    parser.add_argument("--segment", type=str, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.input_mode != "folder":
        raise ValueError("Only folder input_mode is supported in this mask-generation flow")
    image_dir = args.image_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    image_paths = collect_images(image_dir)
    if not image_paths:
        raise ValueError(f"No supported image files found in {image_dir}")

    frame_names = [f"{path.stem}.png" for path in image_paths]
    output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="sam3_masks_") as temp_dir:
        video_path, frame_size = build_temp_video(image_paths, Path(temp_dir) / "source.mp4")
        outputs_per_frame = mask_generation(video_path, args.segment)
        write_instance_masks(outputs_per_frame, output_dir, frame_names, frame_size)

    print(output_dir)


if __name__ == "__main__":
    main()
