import argparse
import os
from pxr import Usd, UsdGeom, UsdPhysics, UsdLux, Gf
import trimesh

def build_animation_with_proxy(usd_dir: str, mesh_dir: str,
                                output_usd: str,
                                frames_per_second: float = 6.0) -> None:
    """
    Build an animated Isaac Sim USD stage from per-frame Gaussian splat USDs
    and Poisson collision proxy meshes.

    Args:
        usd_dir: Directory containing per-frame .usd files (frame_NNNN.usd)
        mesh_dir: Directory containing per-frame .obj proxy meshes (proxy_mesh_NNNN.obj)
        output_usd: Output path for the final animated .usd file
        frames_per_second: Playback rate for the animation
    """
    # Auto-discover bullet times from usd_dir
    bullet_times = []
    for f in sorted(os.listdir(usd_dir)):
        if f.startswith("frame_") and f.endswith(".usd"):
            try:
                t = int(f[len("frame_"):-len(".usd")])
                bullet_times.append(t)
            except ValueError:
                continue

    if not bullet_times:
        print(f"No frame_NNNN.usd files found in {usd_dir}")
        return

    print(f"Building animation with {len(bullet_times)} frames: {bullet_times}")

    os.makedirs(os.path.dirname(os.path.abspath(output_usd)), exist_ok=True)

    stage = Usd.Stage.CreateNew(output_usd)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    stage.SetStartTimeCode(0)
    stage.SetEndTimeCode(len(bullet_times) - 1)
    stage.SetFramesPerSecond(frames_per_second)

    world = stage.DefinePrim("/World", "Xform")
    stage.SetDefaultPrim(world)

    # Physics scene
    physics_scene = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
    physics_scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0, -1, 0))
    physics_scene.CreateGravityMagnitudeAttr().Set(9.81)

    # Distant light
    light = UsdLux.DistantLight.Define(stage, "/World/SunLight")
    light.CreateIntensityAttr().Set(1500.0)
    light.CreateAngleAttr().Set(1.0)
    UsdGeom.Xformable(light.GetPrim()).AddRotateXOp().Set(-45.0)

    num_frames = len(bullet_times)
    frames_with_splat = 0
    frames_with_mesh = 0

    for i, t in enumerate(bullet_times):
        usd_path = os.path.join(usd_dir, f"frame_{t:04d}.usd")
        if not os.path.exists(usd_path):
            print(f"[frame {t}] Skipping missing splat USD: {usd_path}")
            continue

        prim_path = f"/World/Frame_{i:04d}"
        prim = stage.OverridePrim(prim_path)
        prim.GetReferences().AddReference(usd_path, "/gaussians_0")
        frames_with_splat += 1

        # Orientation fix (ViPE Y-down vs USD Y-up)
        UsdGeom.Xformable(prim).AddRotateXOp().Set(180.0)

        # Flipbook visibility
        vis_attr = UsdGeom.Imageable(prim).GetVisibilityAttr()
        for frame in range(num_frames):
            value = UsdGeom.Tokens.inherited if frame == i else UsdGeom.Tokens.invisible
            vis_attr.Set(value, Usd.TimeCode(frame))

        # Per-frame collision mesh as child of splat prim
        mesh_path = os.path.join(mesh_dir, f"proxy_mesh_{t:04d}.obj")
        if os.path.exists(mesh_path):
            try:
                tri_mesh = trimesh.load(mesh_path)
                mesh_prim = UsdGeom.Mesh.Define(stage, f"{prim_path}/CollisionProxy")
                mesh_prim.GetPointsAttr().Set([tuple(v) for v in tri_mesh.vertices])
                mesh_prim.GetFaceVertexCountsAttr().Set([3] * len(tri_mesh.faces))
                mesh_prim.GetFaceVertexIndicesAttr().Set(
                    [int(idx) for tri in tri_mesh.faces for idx in tri]
                )
                mesh_prim.GetVisibilityAttr().Set(UsdGeom.Tokens.invisible)

                collision_api = UsdPhysics.CollisionAPI.Apply(mesh_prim.GetPrim())
                enabled_attr = collision_api.CreateCollisionEnabledAttr()
                for frame in range(num_frames):
                    enabled_attr.Set(frame == i, Usd.TimeCode(frame))

                prim.CreateRelationship("proxy").AddTarget(mesh_prim.GetPath())
                frames_with_mesh += 1
            except Exception as e:
                print(f"[frame {t}] Mesh error: {e}")
        else:
            print(f"[frame {t}] No proxy mesh found -- splat only, no collision")

    # Test cube dropped above frame 60 (or first available mesh)
    ref_t = 60 if 60 in bullet_times else bullet_times[len(bullet_times) // 2]
    ref_mesh_path = os.path.join(mesh_dir, f"proxy_mesh_{ref_t:04d}.obj")
    if os.path.exists(ref_mesh_path):
        try:
            ref_mesh = trimesh.load(ref_mesh_path)
            bbox_min, bbox_max = ref_mesh.bounds
            drop_x = (bbox_min[0] + bbox_max[0]) / 2
            drop_z = (bbox_min[2] + bbox_max[2]) / 2
            drop_y = bbox_max[1] + 1.0

            cube = UsdGeom.Cube.Define(stage, "/World/TestCube")
            cube.CreateSizeAttr().Set(0.2)
            UsdGeom.Xformable(cube.GetPrim()).AddTranslateOp().Set(
                Gf.Vec3d(drop_x, drop_y, drop_z)
            )
            UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
            UsdPhysics.RigidBodyAPI.Apply(cube.GetPrim())
            UsdPhysics.MassAPI.Apply(cube.GetPrim()).CreateMassAttr().Set(1.0)
        except Exception as e:
            print(f"Test cube error: {e}")
    else:
        print("No reference mesh found -- skipping test cube")

    stage.GetRootLayer().Save()
    print(f"Saved: {output_usd}")
    print(f"Frames with splat: {frames_with_splat}/{num_frames}")
    print(f"Frames with collision mesh: {frames_with_mesh}/{num_frames}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build animated Isaac Sim USD stage from Lyra outputs"
    )
    parser.add_argument("--usd_dir", required=True,
        help="Directory containing per-frame .usd files")
    parser.add_argument("--mesh_dir", required=True,
        help="Directory containing per-frame .obj proxy meshes")
    parser.add_argument("--output_usd", required=True,
        help="Output path for the final animated .usd file")
    parser.add_argument("--fps", type=float, default=6.0,
        help="Playback rate for the animation (default: 6.0)")
    args = parser.parse_args()
    build_animation_with_proxy(args.usd_dir, args.mesh_dir, args.output_usd, args.fps)