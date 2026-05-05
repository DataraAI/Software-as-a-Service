"""
Convert a single DynHaMR NPZ (both hands over time) into an MCAP for Foxglove Studio
(3D panel → SceneUpdate).

Dependencies:
  pip install numpy mcap-protobuf-support foxglove-schemas-protobuf

The NPZ must contain:
  - joints_world: float array, shape (2, NUM_FRAMES, 21, 3) — left/right hand, frames,
    21 joints, xyz in world space
  - skeleton_edges: int array, shape (E, 2) — undirected edges as index pairs into the
    21 joints (shared topology for both hands)

Metadata on each SceneEntity includes source=DynHaMR so MCAPs are identifiable.
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

import numpy as np
from google.protobuf.timestamp_pb2 import Timestamp
from mcap_protobuf.writer import Writer

from foxglove_schemas_protobuf.KeyValuePair_pb2 import KeyValuePair

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


def joints_npz_to_mcap(
    npz_path: str | Path,
    output_mcap: str | Path,
    *,
    fps: float = 30.0,
    topic: str = "/scene",
    reference_frame: str = "world",
    sphere_diameter: float = 0.02,
    line_thickness: float = 0.004,
) -> Path:
    """
    Write one foxglove.SceneUpdate per frame. Each message uses both hands as two
    colored batches (same layout as mcap_file_gen's multi-batch single frame).
    """
    if fps <= 0:
        raise ValueError("fps must be positive")

    joints, edges = load_dynhamr_npz(npz_path)
    num_frames = int(joints.shape[1])

    out_path = Path(output_mcap)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dt_ns = int(1e9 / fps)

    with open(out_path, "wb") as f, Writer(f) as writer:
        for t in range(num_frames):
            # (2, 21, 3): two hands, same as build_scene_update's (X, 21, 3)
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

            log_time_ns = t * dt_ns
            ts = Timestamp()
            ts.FromNanoseconds(log_time_ns)
            if scene.entities:
                scene.entities[0].timestamp.CopyFrom(ts)

            writer.write_message(
                topic=topic,
                message=scene,
                log_time=log_time_ns,
                publish_time=log_time_ns,
                sequence=t,
            )

    return out_path


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Build an MCAP from a DynHaMR NPZ (joints_world + skeleton_edges) for Foxglove 3D."
    )
    parser.add_argument(
        "npz_path",
        type=str,
        help="Path to the .npz file (joints_world, skeleton_edges)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="dynhamr_scene.mcap",
        help="Output MCAP path (default: dynhamr_scene.mcap)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Playback frame rate: spacing between frames is 1/fps seconds (default: 30)",
    )
    parser.add_argument(
        "--topic",
        type=str,
        default="/scene",
        help="MCAP topic for SceneUpdate (default: /scene)",
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

    out = joints_npz_to_mcap(
        args.npz_path,
        args.output,
        fps=args.fps,
        topic=args.topic,
        sphere_diameter=args.sphere_diameter,
        line_thickness=args.line_thickness,
    )
    print(f"Wrote {out} (DynHaMR, {Path(args.npz_path).name})")


if __name__ == "__main__":
    main()
