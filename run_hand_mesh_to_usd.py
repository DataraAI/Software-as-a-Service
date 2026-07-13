#!/usr/bin/env python3
"""
run_hand_mesh_to_usd.py
Converts DynHaMR .obj hand mesh outputs into a single self-contained animated USD file.
"""

import argparse
import os
import re
import sys
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

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert DynHaMR OBJ hand meshes to a self-contained animated USD file."
    )
    parser.add_argument("--obj-dir", required=True, help="Directory containing NNNNNN_0/1.obj files")
    parser.add_argument("--output", required=True, help="Output path for the animated .usd or .usdc file")
    parser.add_argument("--fps", type=float, default=30.0, help="Frames per second (default: 30)")
    return parser.parse_args()

def discover_frame_numbers(obj_dir: str) -> list[int]:
    pattern = re.compile(r'^(\d{6})_[01]\.obj$')
    frame_nums = set()
    for fname in os.listdir(obj_dir):
        match = pattern.match(fname)
        if match:
            frame_nums.add(int(match.group(1)))
    return sorted(frame_nums)

def sample_mesh_to_prim(mesh_prim, obj_path: str, time_code: Usd.TimeCode) -> None:
    """Loads an OBJ mesh and appends its structural features to a specific timeline index."""
    mesh = trimesh.load(obj_path, force="mesh")
    mesh.fix_normals()
    
    # Write topology settings at the specified time step
    mesh_prim.GetPointsAttr().Set([tuple(v) for v in mesh.vertices], time_code)
    mesh_prim.GetNormalsAttr().Set([tuple(n) for n in mesh.vertex_normals], time_code)
    
    # Face indices only need to be written once on initial setup
    if not mesh_prim.GetFaceVertexCountsAttr().HasAuthoredValue():
        mesh_prim.GetFaceVertexCountsAttr().Set([3] * len(mesh.faces))
        mesh_prim.GetFaceVertexIndicesAttr().Set([int(i) for face in mesh.faces for i in face])

def main() -> None:
    args = parse_args()
    
    frame_numbers = discover_frame_numbers(args.obj_dir)
    if not frame_numbers:
        log.error("No OBJ files found in directory: %s", args.obj_dir)
        sys.exit(1)

    num_frames = len(frame_numbers)
    log.info("Processing %d frames into self-contained scene: %s", num_frames, args.output)

    # Force binary layout if the user specifies a .usd extension
    stage = Usd.Stage.CreateNew(args.output)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    stage.SetStartTimeCode(0)
    stage.SetEndTimeCode(num_frames - 1)
    stage.SetFramesPerSecond(args.fps)

    # Base world configuration
    world = stage.DefinePrim("/World", "Xform")
    stage.SetDefaultPrim(world)

    # Environment lighting
    light = UsdLux.DistantLight.Define(stage, "/World/SunLight")
    light.CreateIntensityAttr().Set(1500.0)
    UsdGeom.Xformable(light.GetPrim()).AddRotateXOp().Set(-45.0)

    # Define persistent mesh structures across the sequence
    left_hand_prim = UsdGeom.Mesh.Define(stage, "/World/LeftHand")
    right_hand_prim = UsdGeom.Mesh.Define(stage, "/World/RightHand")

    converted_count = 0
    for i, frame_num in enumerate(frame_numbers):
        left_path  = os.path.join(args.obj_dir, f"{frame_num:06d}_0.obj")
        right_path = os.path.join(args.obj_dir, f"{frame_num:06d}_1.obj")
        
        time_code = Usd.TimeCode(i)

        # Handle Left Hand tracking loops
        if os.path.exists(left_path):
            sample_mesh_to_prim(left_hand_prim, left_path, time_code)
            left_hand_prim.GetVisibilityAttr().Set(UsdGeom.Tokens.inherited, time_code)
        else:
            left_hand_prim.GetVisibilityAttr().Set(UsdGeom.Tokens.invisible, time_code)

        # Handle Right Hand tracking loops
        if os.path.exists(right_path):
            sample_mesh_to_prim(right_hand_prim, right_path, time_code)
            right_hand_prim.GetVisibilityAttr().Set(UsdGeom.Tokens.inherited, time_code)
        else:
            right_hand_prim.GetVisibilityAttr().Set(UsdGeom.Tokens.invisible, time_code)

        converted_count += 1
        if converted_count % 100 == 0:
            log.info("Progress: Baked %d/%d animation frames...", converted_count, num_frames)

    stage.GetRootLayer().Save()
    log.info("Successfully saved package: %s", args.output)
    print(f"OUTPUT_USD: {args.output}")

if __name__ == "__main__":
    main()