"""
Convert a directory of NPZ files (per-frame 3D keypoints + skeleton edges) into an MCAP
for playback in Foxglove Studio (3D panel → SceneUpdate).

Dependencies:
  pip install numpy mcap-protobuf-support foxglove-schemas-protobuf

Each NPZ must contain:
  - keypoints_3d: float array, shape (X, 21, 3) — X batches, 21 joints, xyz
  - skeleton_edges: int array, shape (E, 2) — undirected edges as index pairs into the 21 joints

Frame IDs default to the last integer found in each filename (e.g. pose_00042.npz → 42).
Override with --frame-ids (same order as files sorted alphabetically by path within the directory).
"""

from __future__ import annotations

import argparse
import colorsys
import re
from pathlib import Path
from typing import Sequence

import numpy as np
from google.protobuf.timestamp_pb2 import Timestamp
from mcap_protobuf.writer import Writer

from foxglove_schemas_protobuf.Color_pb2 import Color
from foxglove_schemas_protobuf.KeyValuePair_pb2 import KeyValuePair
from foxglove_schemas_protobuf.LinePrimitive_pb2 import LinePrimitive
from foxglove_schemas_protobuf.Point3_pb2 import Point3
from foxglove_schemas_protobuf.Pose_pb2 import Pose
from foxglove_schemas_protobuf.Quaternion_pb2 import Quaternion
from foxglove_schemas_protobuf.SceneEntity_pb2 import SceneEntity
from foxglove_schemas_protobuf.SceneUpdate_pb2 import SceneUpdate
from foxglove_schemas_protobuf.SpherePrimitive_pb2 import SpherePrimitive
from foxglove_schemas_protobuf.Vector3_pb2 import Vector3


def _batch_colors(num_batches: int) -> list[Color]:
    """Distinct RGBA colors for each batch index."""
    if num_batches <= 0:
        return []
    out: list[Color] = []
    for i in range(num_batches):
        h = i / num_batches
        r, g, b = colorsys.hsv_to_rgb(h, 0.85, 0.95)
        out.append(Color(r=r, g=g, b=b, a=1.0))
    return out


def _identity_pose(x: float, y: float, z: float) -> Pose:
    return Pose(
        position=Vector3(x=float(x), y=float(y), z=float(z)),
        orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
    )


def parse_frame_id_from_filename(path: str | Path) -> int | None:
    """Return the last integer substring in the basename, or None if none."""
    base = Path(path).stem
    matches = re.findall(r"\d+", base)
    if not matches:
        return None
    return int(matches[-1])


def build_scene_update(
    keypoints_3d: np.ndarray,
    skeleton_edges: np.ndarray,
    *,
    frame_id: int,
    reference_frame: str = "world",
    sphere_diameter: float = 0.02,
    line_thickness: float = 0.004,
) -> SceneUpdate:
    """
    keypoints_3d: (X, 21, 3)
    skeleton_edges: (E, 2) indices into the 21 joints per batch
    """
    if keypoints_3d.ndim != 3 or keypoints_3d.shape[1:] != (21, 3):
        raise ValueError(
            f"keypoints_3d must have shape (X, 21, 3); got {keypoints_3d.shape}"
        )
    if skeleton_edges.ndim != 2 or skeleton_edges.shape[1] != 2:
        raise ValueError(
            f"skeleton_edges must have shape (E, 2); got {skeleton_edges.shape}"
        )

    kp = np.asarray(keypoints_3d, dtype=np.float64)
    edges = np.asarray(skeleton_edges, dtype=np.int64)
    num_batches = kp.shape[0]
    colors = _batch_colors(num_batches)

    lines: list[LinePrimitive] = []
    spheres: list[SpherePrimitive] = []

    for b in range(num_batches):
        c = colors[b]
        pts_b = kp[b]
        # Spheres at each joint
        for j in range(21):
            x, y, z = pts_b[j]
            spheres.append(
                SpherePrimitive(
                    pose=_identity_pose(x, y, z),
                    size=Vector3(
                        x=sphere_diameter,
                        y=sphere_diameter,
                        z=sphere_diameter,
                    ),
                    color=c,
                )
            )
        # Line segments for skeleton (LINE_LIST: pairs 0-1, 2-3, ...)
        edge_points: list[Point3] = []
        for e in range(edges.shape[0]):
            i, j = int(edges[e, 0]), int(edges[e, 1])
            if not (0 <= i < 21 and 0 <= j < 21):
                continue
            pi = pts_b[i]
            pj = pts_b[j]
            edge_points.append(Point3(x=float(pi[0]), y=float(pi[1]), z=float(pi[2])))
            edge_points.append(Point3(x=float(pj[0]), y=float(pj[1]), z=float(pj[2])))
        if edge_points:
            lines.append(
                LinePrimitive(
                    type=LinePrimitive.Type.LINE_LIST,
                    pose=Pose(
                        position=Vector3(x=0.0, y=0.0, z=0.0),
                        orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
                    ),
                    thickness=line_thickness,
                    scale_invariant=False,
                    points=edge_points,
                    color=c,
                )
            )

    entity_ts = Timestamp()
    entity_ts.FromNanoseconds(0)

    entity = SceneEntity(
        timestamp=entity_ts,
        frame_id=reference_frame,
        id="skeleton_keypoints",
        metadata=[
            KeyValuePair(key="frame_id", value=str(int(frame_id))),
        ],
        lines=lines,
        spheres=spheres,
    )

    return SceneUpdate(entities=[entity])


def collect_npz_paths(path: str | Path) -> list[Path]:
    """
    If `path` is a directory, return all `*.npz` files directly in that directory only
    (sorted alphabetically for a stable order). Subfolders are not scanned.
    If `path` is a single `.npz` file, return a one-element list.
    """
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(p)
    if p.is_file():
        if p.suffix.lower() != ".npz":
            raise ValueError(f"Expected a .npz file or a directory, got: {p}")
        return [p]
    if p.is_dir():
        found = sorted(p.glob("*.npz"))
        if not found:
            raise ValueError(f"No .npz files found in {p}")
        return found
    raise ValueError(f"Not a file or directory: {p}")


def npz_paths_to_frame_ids(
    npz_paths: Sequence[str | Path],
    frame_ids: Sequence[int] | None,
) -> list[tuple[Path, int]]:
    paths = [Path(p).resolve() for p in npz_paths]
    if frame_ids is not None:
        if len(frame_ids) != len(paths):
            raise ValueError(
                f"frame_ids length ({len(frame_ids)}) must match npz_paths ({len(paths)})"
            )
        return list(zip(paths, map(int, frame_ids)))

    out: list[tuple[Path, int]] = []
    for p in paths:
        fid = parse_frame_id_from_filename(p)
        if fid is None:
            raise ValueError(
                f"Could not infer frame id from filename '{p.name}'. "
                "Pass explicit --frame-ids."
            )
        out.append((p, fid))
    return out


def npz_list_to_mcap(
    npz_paths: Sequence[str | Path],
    output_mcap: str | Path,
    *,
    frame_ids: Sequence[int] | None = None,
    fps: float = 30.0,
    time_from_frame_id: bool = False,
    topic: str = "/scene",
    reference_frame: str = "world",
    sphere_diameter: float = 0.02,
    line_thickness: float = 0.004,
) -> Path:
    """
    Write one foxglove.SceneUpdate per NPZ file, ordered by frame id ascending.

    Timestamps (nanoseconds):
    - If time_from_frame_id is False (default): log_time = sequence_index * (1e9 / fps)
      so consecutive frames in sorted order play at a fixed rate.
    - If True: log_time = frame_id * (1e9 / fps), so timeline gaps match frame id gaps.
    """
    if fps <= 0:
        raise ValueError("fps must be positive")

    pairs = npz_paths_to_frame_ids(npz_paths, frame_ids)
    pairs.sort(key=lambda x: x[1])

    out_path = Path(output_mcap)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    dt_ns = int(1e9 / fps)

    with open(out_path, "wb") as f, Writer(f) as writer:
        for seq_idx, (npz_path, frame_id) in enumerate(pairs):
            data = np.load(npz_path, allow_pickle=False)
            if "keypoints_3d" not in data or "skeleton_edges" not in data:
                raise KeyError(
                    f"{npz_path} must contain 'keypoints_3d' and 'skeleton_edges'"
                )
            kp = data["keypoints_3d"]
            sk = data["skeleton_edges"]
            scene = build_scene_update(
                kp,
                sk,
                frame_id=frame_id,
                reference_frame=reference_frame,
                sphere_diameter=sphere_diameter,
                line_thickness=line_thickness,
            )
            if time_from_frame_id:
                log_time_ns = int(frame_id * dt_ns)
            else:
                log_time_ns = seq_idx * dt_ns
            ts = Timestamp()
            ts.FromNanoseconds(log_time_ns)
            if scene.entities:
                scene.entities[0].timestamp.CopyFrom(ts)

            writer.write_message(
                topic=topic,
                message=scene,
                log_time=log_time_ns,
                publish_time=log_time_ns,
                sequence=seq_idx,
            )

    return out_path


def npz_dir_to_mcap(
    input_dir: str | Path,
    output_mcap: str | Path,
    **kwargs,
) -> Path:
    """Convenience wrapper: `collect_npz_paths` then `npz_list_to_mcap`."""
    paths = collect_npz_paths(input_dir)
    return npz_list_to_mcap(paths, output_mcap, **kwargs)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Build an MCAP from a directory of NPZ keypoint files for Foxglove 3D."
    )
    parser.add_argument(
        "input_dir",
        type=str,
        help="Directory containing .npz files (top-level only; or a single .npz file)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="skeleton.mcap",
        help="Output MCAP path (default: skeleton.mcap)",
    )
    parser.add_argument(
        "--frame-ids",
        type=str,
        default=None,
        help="Comma-separated frame ids in the same order as .npz files sorted by path "
        "within the directory (overrides filename parsing)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Frames per second basis: each frame_id step is 1/fps seconds when using "
        "--time-from-frame-id; otherwise spacing between consecutive sorted frames",
    )
    parser.add_argument(
        "--time-from-frame-id",
        action="store_true",
        help="Set MCAP log_time from frame_id * (1s / fps) instead of consecutive indices",
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

    frame_ids: list[int] | None = None
    if args.frame_ids:
        frame_ids = [int(x.strip()) for x in args.frame_ids.split(",")]

    out = npz_dir_to_mcap(
        args.input_dir,
        args.output,
        frame_ids=frame_ids,
        fps=args.fps,
        time_from_frame_id=args.time_from_frame_id,
        topic=args.topic,
        sphere_diameter=args.sphere_diameter,
        line_thickness=args.line_thickness,
    )
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
