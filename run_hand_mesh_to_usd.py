#!/usr/bin/env python3
"""
run_hand_mesh_to_usd.py
Converts DynHaMR .obj hand mesh outputs into a single animated USD file.

USAGE:
    python run_hand_mesh_to_usd.py \
        --obj-dir  <path to directory containing NNNNNN_0.obj / NNNNNN_1.obj files> \
        --output   <path to write the final hand_animation.usd>
        [--fps     <frames per second, default 30>]
"""

import argparse
import os
import re
import sys
import tempfile
import shutil
import logging
import trimesh
from pathlib import Path
from pxr import Usd, UsdGeom, UsdLux

logging.basicConfig(
    format="[%(asctime)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout,
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert DynHaMR OBJ hand meshes to an animated USD file."
    )
    parser.add_argument("--obj-dir", required=True, help="Directory containing NNNNNN_0/1.obj files")
    parser.add_argument("--output", required=True, help="Output path for the animated .usd file")
    parser.add_argument("--fps", type=float, default=30.0, help="Frames per second (default: 30)")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Step 1: Convert per-frame OBJ pairs → per-frame USD files
# ---------------------------------------------------------------------------

def discover_frame_numbers(obj_dir: str) -> list[int]:
    pattern = re.compile(r'^(\d{6})_[01]\.obj$')
    frame_nums = set()
    for fname in os.listdir(obj_dir):
        match = pattern.match(fname)
        if match:
            frame_nums.add(int(match.group(1)))
    return sorted(frame_nums)


def write_mesh_to_prim(stage, prim_path: str, obj_path: str) -> None:
    mesh = trimesh.load(obj_path, force="mesh")
    mesh_prim = UsdGeom.Mesh.Define(stage, prim_path)
    mesh_prim.GetPointsAttr().Set([tuple(v) for v in mesh.vertices])
    mesh_prim.GetFaceVertexCountsAttr().Set([3] * len(mesh.faces))
    mesh_prim.GetFaceVertexIndicesAttr().Set([int(i) for face in mesh.faces for i in face])
    mesh.fix_normals()
    mesh_prim.GetNormalsAttr().Set([tuple(n) for n in mesh.vertex_normals])


def convert_frames_to_usds(obj_dir: str, usd_dir: str, fps: float) -> list[int]:
    """
    Convert each frame's OBJ pair into a per-frame USD.
    Returns the list of successfully converted frame numbers.
    """
    
    frame_numbers = discover_frame_numbers(obj_dir)
    if not frame_numbers:
        raise RuntimeError(f"No OBJ files found in {obj_dir}")

    log.info("Discovered %d frames (%06d → %06d)", len(frame_numbers), frame_numbers[0], frame_numbers[-1])
    converted = []
    skipped = 0

    for frame_num in frame_numbers:
        left_path  = os.path.join(obj_dir, f"{frame_num:06d}_0.obj")
        right_path = os.path.join(obj_dir, f"{frame_num:06d}_1.obj")
        left_exists  = os.path.exists(left_path)
        right_exists = os.path.exists(right_path)

        if not left_exists and not right_exists:
            skipped += 1
            continue

        out_path = os.path.join(usd_dir, f"hand_frame_{frame_num:06d}.usd")
        try:
            stage = Usd.Stage.CreateNew(out_path)
            UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
            root = stage.DefinePrim("/hands", "Xform")
            stage.SetDefaultPrim(root)

            if left_exists:
                write_mesh_to_prim(stage, "/hands/left", left_path)
            else:
                log.warning("Frame %06d: left hand (_0) missing", frame_num)

            if right_exists:
                write_mesh_to_prim(stage, "/hands/right", right_path)
            else:
                log.warning("Frame %06d: right hand (_1) missing", frame_num)

            stage.GetRootLayer().Save()
            converted.append(frame_num)
        except Exception as exc:
            log.error("Frame %06d: failed — %s", frame_num, exc)
            if os.path.exists(out_path):
                os.remove(out_path)

        if len(converted) % 100 == 0 and converted:
            log.info("Progress: %d/%d frames converted...", len(converted), len(frame_numbers))

    log.info("Converted %d frames, skipped %d (no meshes)", len(converted), skipped)
    return converted


# ---------------------------------------------------------------------------
# Step 2: Assemble per-frame USDs → single animated USD
# ---------------------------------------------------------------------------

def build_animation(usd_dir: str, output_usd: str, frame_numbers: list[int], fps: float) -> None:

    num_frames = len(frame_numbers)
    log.info("Assembling %d frame USDs into %s", num_frames, output_usd)

    stage = Usd.Stage.CreateNew(output_usd)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    stage.SetStartTimeCode(0)
    stage.SetEndTimeCode(num_frames - 1)
    stage.SetFramesPerSecond(fps)

    world = stage.DefinePrim("/World", "Xform")
    stage.SetDefaultPrim(world)

    light = UsdLux.DistantLight.Define(stage, "/World/SunLight")
    light.CreateIntensityAttr().Set(1500.0)
    light.CreateAngleAttr().Set(1.0)
    UsdGeom.Xformable(light.GetPrim()).AddRotateXOp().Set(-45.0)

    for i, frame_num in enumerate(frame_numbers):
        usd_path = os.path.join(usd_dir, f"hand_frame_{frame_num:06d}.usd")
        prim_path = f"/World/HandFrame_{i:06d}"
        prim = stage.OverridePrim(prim_path)
        prim.GetReferences().AddReference(usd_path, "/hands")

        vis_attr = UsdGeom.Imageable(prim).GetVisibilityAttr()
        vis_attr.Set(UsdGeom.Tokens.invisible, Usd.TimeCode(0))
        vis_attr.Set(UsdGeom.Tokens.inherited, Usd.TimeCode(i))
        vis_attr.Set(UsdGeom.Tokens.invisible, Usd.TimeCode(i + 0.5))

        if (i + 1) % 100 == 0:
            log.info("Progress: %d/%d frames written...", i + 1, num_frames)

    stage.GetRootLayer().Save()
    log.info("Saved animated USD: %s", output_usd)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    usd_dir = tempfile.mkdtemp(prefix="hand_frame_usds_")
    try:
        frame_numbers = convert_frames_to_usds(args.obj_dir, usd_dir, args.fps)
        if not frame_numbers:
            log.error("No frames were successfully converted — aborting")
            sys.exit(1)
        build_animation(usd_dir, args.output, frame_numbers, args.fps)
    finally:
        shutil.rmtree(usd_dir, ignore_errors=True)
        log.info("Cleaned up intermediate USD directory")

    print(f"OUTPUT_USD: {args.output}")


if __name__ == "__main__":
    main()