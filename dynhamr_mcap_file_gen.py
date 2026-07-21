#!/usr/bin/env python3
"""
Convert a single DynHaMR NPZ (both hands over time) and its source MP4 video
into an MCAP for Foxglove Studio (3D panel -> SceneUpdate, Image panel -> CompressedImage).

Dependencies:
  pip install numpy opencv-python mcap-protobuf-support foxglove-schemas-protobuf

The NPZ must contain:
  - joints_world: float array, shape (2, NUM_FRAMES, 21, 3)
  - skeleton_edges: int array, shape (E, 2)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

# Allow `from mcap_file_gen import ...` when run as a script from any cwd
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import cv2
import numpy as np
from google.protobuf.timestamp_pb2 import Timestamp
from mcap_protobuf.writer import Writer

from foxglove_schemas_protobuf.KeyValuePair_pb2 import KeyValuePair
from foxglove_schemas_protobuf.CompressedImage_pb2 import CompressedImage

from mcap_file_gen import build_scene_update


def load_dynhamr_npz(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix.lower() != ".npz":
        raise ValueError(f"Expected a .npz file, got: {path}")

    data = np.load(path, allow_pickle=False)
    if "joints_world" not in data or "skeleton_edges" not in data:
        raise KeyError(
            f"{path} must contain 'joints_world' and 'skeleton_edges' "
            f"(keys present: {sorted(data.files)})"
        )

    joints = np.asarray(data["joints_world"], dtype=np.float64)
    edges = np.asarray(data["skeleton_edges"], dtype=np.int64)

    if joints.ndim != 4 or joints.shape[0] != 2 or joints.shape[2:] != (21, 3):
        raise ValueError(
            "joints_world must have shape (2, NUM_FRAMES, 21, 3); "
            f"got {joints.shape}"
        )
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError(
            f"skeleton_edges must have shape (E, 2); got {edges.shape}"
        )

    return joints, edges


def joints_and_video_to_mcap(
    npz_path: str | Path,
    video_path: str | Path | None,
    output_mcap: str | Path,
    *,
    fps: float = 30.0,
    scene_topic: str = "/scene/hands",
    video_topic: str = "/video/rgb",
    reference_frame: str = "world",
    sphere_diameter: float = 0.02,
    line_thickness: float = 0.004,
) -> Path:
    if fps <= 0:
        raise ValueError("fps must be positive")

    joints, edges = load_dynhamr_npz(npz_path)
    num_frames = int(joints.shape[1])

    # Initialize video capture if video track is provided
    cap = None
    if video_path:
        video_path = Path(video_path).expanduser().resolve()
        if video_path.is_file():
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                print(f"[!] Warning: Could not open video file {video_path}", file=sys.stderr)
                cap = None

    out_path = Path(output_mcap)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dt_ns = int(1e9 / fps)

    with open(out_path, "wb") as f, Writer(f) as writer:
        for t in range(num_frames):
            log_time_ns = t * dt_ns
            ts = Timestamp()
            ts.FromNanoseconds(log_time_ns)

            # --- 1. Write Video Frame if Available ---
            if cap is not None:
                ret, frame = cap.read()
                if ret:
                    # Compress raw image frames into lightweight JPEG payloads
                    success, encoded_img = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    if success:
                        img_msg = CompressedImage()
                        img_msg.timestamp.CopyFrom(ts)
                        img_msg.frame_id = "world"
                        img_msg.format = "jpeg"
                        img_msg.data = encoded_img.tobytes()

                        writer.write_message(
                            topic=video_topic,
                            message=img_msg,
                            log_time=log_time_ns,
                            publish_time=log_time_ns,
                            sequence=t,
                        )
                else:
                    # Video stream finished earlier than tracking points
                    cap.release()
                    cap = None

            # --- 2. Write 3D Scene Update ---
            kp_t = joints[:, t, :, :]
            scene = build_scene_update(
                kp_t,
                edges,
                frame_id=t,
                reference_frame=reference_frame,
                sphere_diameter=sphere_diameter,
                line_thickness=line_thickness,
            )
            if scene.entities:
                ent = scene.entities[0]
                ent.id = "dynhamr_hands"
                ent.metadata.append(KeyValuePair(key="source", value="DynHaMR"))
                ent.timestamp.CopyFrom(ts)

            writer.write_message(
                topic=scene_topic,
                message=scene,
                log_time=log_time_ns,
                publish_time=log_time_ns,
                sequence=t,
            )

    if cap is not None:
        cap.release()

    return out_path


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Build an MCAP containing both 3D joints and source camera video data streams."
    )
    parser.add_argument(
        "npz_path",
        type=str,
        help="Path to the .npz file (joints_world, skeleton_edges)",
    )
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Path to matching source clip mp4 video file (optional)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="towel.mcap",
        help="Output MCAP path (default: towel.mcap)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Playback frame rate (default: 30)",
    )
    parser.add_argument(
        "--scene-topic",
        type=str,
        default="/scene/hands",
        help="MCAP topic for SceneUpdate (default: /scene/hands)",
    )
    parser.add_argument(
        "--video-topic",
        type=str,
        default="/video/rgb",
        help="MCAP topic for embedded video frames (default: /video/rgb)",
    )
    parser.add_argument(
        "--sphere-diameter",
        type=float,
        default=0.02,
        help="Diameter of joint spheres in scene units (default: 0.02)",
    )
    parser.add_argument(
        "--line-thickness",
        type=float,
        default=0.004,
        help="Skeleton line thickness in world units (default: 0.004)",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    out = joints_and_video_to_mcap(
        args.npz_path,
        args.video,
        args.output,
        fps=args.fps,
        scene_topic=args.scene_topic,
        video_topic=args.video_topic,
        sphere_diameter=args.sphere_diameter,
        line_thickness=args.line_thickness,
    )
    print(f"Wrote unified container: {out}")


if __name__ == "__main__":
    main()#!/usr/bin/env python3
"""
Convert a single DynHaMR NPZ (both hands over time) and its source MP4 video
into an MCAP for Foxglove Studio (3D panel -> SceneUpdate, Image panel -> CompressedImage).

Dependencies:
  pip install numpy opencv-python mcap-protobuf-support foxglove-schemas-protobuf

The NPZ must contain:
  - joints_world: float array, shape (2, NUM_FRAMES, 21, 3)
  - skeleton_edges: int array, shape (E, 2)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

# Allow `from mcap_file_gen import ...` when run as a script from any cwd
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import cv2
import numpy as np
from google.protobuf.timestamp_pb2 import Timestamp
from mcap_protobuf.writer import Writer

from foxglove_schemas_protobuf.KeyValuePair_pb2 import KeyValuePair
from foxglove_schemas_protobuf.CompressedImage_pb2 import CompressedImage

from mcap_file_gen import build_scene_update


def load_dynhamr_npz(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix.lower() != ".npz":
        raise ValueError(f"Expected a .npz file, got: {path}")

    data = np.load(path, allow_pickle=False)
    if "joints_world" not in data or "skeleton_edges" not in data:
        raise KeyError(
            f"{path} must contain 'joints_world' and 'skeleton_edges' "
            f"(keys present: {sorted(data.files)})"
        )

    joints = np.asarray(data["joints_world"], dtype=np.float64)
    edges = np.asarray(data["skeleton_edges"], dtype=np.int64)

    if joints.ndim != 4 or joints.shape[0] != 2 or joints.shape[2:] != (21, 3):
        raise ValueError(
            "joints_world must have shape (2, NUM_FRAMES, 21, 3); "
            f"got {joints.shape}"
        )
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError(
            f"skeleton_edges must have shape (E, 2); got {edges.shape}"
        )

    return joints, edges


def joints_and_video_to_mcap(
    npz_path: str | Path,
    video_path: str | Path | None,
    output_mcap: str | Path,
    *,
    fps: float = 30.0,
    scene_topic: str = "/scene/hands",
    video_topic: str = "/video/rgb",
    reference_frame: str = "world",
    sphere_diameter: float = 0.02,
    line_thickness: float = 0.004,
) -> Path:
    if fps <= 0:
        raise ValueError("fps must be positive")

    joints, edges = load_dynhamr_npz(npz_path)
    num_frames = int(joints.shape[1])

    # Initialize video capture if video track is provided
    cap = None
    if video_path:
        video_path = Path(video_path).expanduser().resolve()
        if video_path.is_file():
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                print(f"[!] Warning: Could not open video file {video_path}", file=sys.stderr)
                cap = None

    out_path = Path(output_mcap)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dt_ns = int(1e9 / fps)

    with open(out_path, "wb") as f, Writer(f) as writer:
        for t in range(num_frames):
            log_time_ns = t * dt_ns
            ts = Timestamp()
            ts.FromNanoseconds(log_time_ns)

            # --- 1. Write Video Frame if Available ---
            if cap is not None:
                ret, frame = cap.read()
                if ret:
                    # Compress raw image frames into lightweight JPEG payloads
                    success, encoded_img = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    if success:
                        img_msg = CompressedImage()
                        img_msg.timestamp.CopyFrom(ts)
                        img_msg.frame_id = "world"
                        img_msg.format = "jpeg"
                        img_msg.data = encoded_img.tobytes()

                        writer.write_message(
                            topic=video_topic,
                            message=img_msg,
                            log_time=log_time_ns,
                            publish_time=log_time_ns,
                            sequence=t,
                        )
                else:
                    # Video stream finished earlier than tracking points
                    cap.release()
                    cap = None

            # --- 2. Write 3D Scene Update ---
            kp_t = joints[:, t, :, :]
            scene = build_scene_update(
                kp_t,
                edges,
                frame_id=t,
                reference_frame=reference_frame,
                sphere_diameter=sphere_diameter,
                line_thickness=line_thickness,
            )
            if scene.entities:
                ent = scene.entities[0]
                ent.id = "dynhamr_hands"
                ent.metadata.append(KeyValuePair(key="source", value="DynHaMR"))
                ent.timestamp.CopyFrom(ts)

            writer.write_message(
                topic=scene_topic,
                message=scene,
                log_time=log_time_ns,
                publish_time=log_time_ns,
                sequence=t,
            )

    if cap is not None:
        cap.release()

    return out_path


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Build an MCAP containing both 3D joints and source camera video data streams."
    )
    parser.add_argument(
        "npz_path",
        type=str,
        help="Path to the .npz file (joints_world, skeleton_edges)",
    )
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Path to matching source clip mp4 video file (optional)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="towel.mcap",
        help="Output MCAP path (default: towel.mcap)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Playback frame rate (default: 30)",
    )
    parser.add_argument(
        "--scene-topic",
        type=str,
        default="/scene/hands",
        help="MCAP topic for SceneUpdate (default: /scene/hands)",
    )
    parser.add_argument(
        "--video-topic",
        type=str,
        default="/video/rgb",
        help="MCAP topic for embedded video frames (default: /video/rgb)",
    )
    parser.add_argument(
        "--sphere-diameter",
        type=float,
        default=0.02,
        help="Diameter of joint spheres in scene units (default: 0.02)",
    )
    parser.add_argument(
        "--line-thickness",
        type=float,
        default=0.004,
        help="Skeleton line thickness in world units (default: 0.004)",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    out = joints_and_video_to_mcap(
        args.npz_path,
        args.video,
        args.output,
        fps=args.fps,
        scene_topic=args.scene_topic,
        video_topic=args.video_topic,
        sphere_diameter=args.sphere_diameter,
        line_thickness=args.line_thickness,
    )
    print(f"Wrote unified container: {out}")


if __name__ == "__main__":
    main()