#!/usr/bin/env python3
"""
run_hand_mesh_to_usd.py
Converts DynHaMR .obj hand mesh outputs into a single self-contained animated USDZ package.
"""

import argparse
import os
import re
import sys
import logging
import tempfile
import shutil
import trimesh
from pathlib import Path
from pxr import Usd, UsdGeom, UsdLux, UsdUtils, Sdf

logging.basicConfig(
    format="[%(asctime)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert DynHaMR OBJ hand meshes to a self-contained animated USDZ file."
    )
    parser.add_argument("--obj-dir", required=True, help="Directory containing NNNNNN_0/1.obj files")
    parser.add_argument("--output", required=True, help="Output path for the animated .usdz package")
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
    mesh = trimesh.load(obj_path, force="mesh")
    mesh.fix_normals()
    
    mesh_prim.GetPointsAttr().Set([tuple(v) for v in mesh.vertices], time_code)
    mesh_prim.GetNormalsAttr().Set([tuple(n) for n in mesh.vertex_normals], time_code)
    
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
    
    final_output_path = Path(args.output).resolve()
    if final_output_path.suffix.lower() != '.usdz':
        final_output_path = final_output_path.with_suffix('.usdz')

    log.info("Processing %d frames into temporary binary cache layer...", num_frames)

    with tempfile.TemporaryDirectory(prefix="usdz_build_") as tmp_dir:
        tmp_usdc_path = os.path.join(tmp_dir, "hand_mesh_cached.usdc")
        
        stage = Usd.Stage.CreateNew(tmp_usdc_path)
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
        stage.SetStartTimeCode(0)
        stage.SetEndTimeCode(num_frames - 1)
        stage.SetFramesPerSecond(args.fps)

        world = stage.DefinePrim("/World", "Xform")
        stage.SetDefaultPrim(world)

        light = UsdLux.DistantLight.Define(stage, "/World/SunLight")
        light.CreateIntensityAttr().Set(1500.0)
        UsdGeom.Xformable(light.GetPrim()).AddRotateXOp().Set(-45.0)

        left_hand_prim = UsdGeom.Mesh.Define(stage, "/World/LeftHand")
        right_hand_prim = UsdGeom.Mesh.Define(stage, "/World/RightHand")

        converted_count = 0
        for i, frame_num in enumerate(frame_numbers):
            left_path  = os.path.join(args.obj_dir, f"{frame_num:06d}_0.obj")
            right_path = os.path.join(args.obj_dir, f"{frame_num:06d}_1.obj")
            
            time_code = Usd.TimeCode(i)

            if os.path.exists(left_path):
                sample_mesh_to_prim(left_hand_prim, left_path, time_code)
                left_hand_prim.GetVisibilityAttr().Set(UsdGeom.Tokens.inherited, time_code)
            else:
                left_hand_prim.GetVisibilityAttr().Set(UsdGeom.Tokens.invisible, time_code)

            if os.path.exists(right_path):
                sample_mesh_to_prim(right_hand_prim, right_path, time_code)
                right_hand_prim.GetVisibilityAttr().Set(UsdGeom.Tokens.inherited, time_code)
            else:
                right_hand_prim.GetVisibilityAttr().Set(UsdGeom.Tokens.invisible, time_code)

            converted_count += 1
            if converted_count % 100 == 0:
                log.info("Progress: Baked %d/%d animation frames...", converted_count, num_frames)

        stage.GetRootLayer().Save()
        
        log.info("Compressing scene graph into target USDZ package asset...")
        os.makedirs(final_output_path.parent, exist_ok=True)
        
        # =====================================================================
        # 🌟 FIXED: Explicitly wrap the input path into an Sdf.AssetPath object
        # =====================================================================
        input_asset_path = Sdf.AssetPath(tmp_usdc_path)
        success = UsdUtils.CreateNewUsdzPackage(input_asset_path, str(final_output_path))
        
        if not success or not final_output_path.is_file():
            log.error("Failed to build package via UsdUtils framework.")
            sys.exit(1)

    log.info("Successfully saved package: %s", final_output_path)
    print(f"OUTPUT_USD: {final_output_path}")

if __name__ == "__main__":
    main()