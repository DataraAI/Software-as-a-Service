import argparse
import os
import sys
from poisson_script import gaussian_ply_to_mesh_poisson

def generate_all_proxy_meshes(ply_base: str, mesh_dir: str,
                               opacity_floor: float = 0.05,
                               poisson_depth: int = 9,
                               density_trim_quantile: float = 0.02) -> list:
    """
    Generate Poisson collision proxy meshes for all Lyra bullet-time frames.
    Auto-discovers bullet times from folder structure.

    Args:
        ply_base: Path to lyra_dynamic_demo_generated/ directory
        mesh_dir: Directory to write .obj proxy meshes

    Returns:
        List of bullet times that were successfully processed
    """
    os.makedirs(mesh_dir, exist_ok=True)

    # Auto-discover bullet times
    bullet_times = []
    for entry in sorted(os.scandir(ply_base), key=lambda e: int(e.name) if e.name.isdigit() else -1):
        if entry.is_dir() and entry.name.isdigit():
            ply_path = os.path.join(entry.path, "gaussians_orig", "gaussians_0.ply")
            if os.path.exists(ply_path):
                bullet_times.append(int(entry.name))

    if not bullet_times:
        print(f"No valid bullet time folders found in {ply_base}")
        sys.exit(1)

    print(f"Found {len(bullet_times)} bullet times: {bullet_times}")

    processed = []
    for t in bullet_times:
        ply_path = os.path.join(ply_base, str(t), "gaussians_orig", "gaussians_0.ply")
        out_path = os.path.join(mesh_dir, f"proxy_mesh_{t:04d}.obj")

        if os.path.exists(out_path):
            print(f"[frame {t}] Already exists, skipping: {out_path}")
            processed.append(t)
            continue

        print(f"[frame {t}] Generating proxy mesh...")
        try:
            gaussian_ply_to_mesh_poisson(
                ply_path, out_path,
                opacity_floor=opacity_floor,
                poisson_depth=poisson_depth,
                density_trim_quantile=density_trim_quantile,
            )
            processed.append(t)
        except Exception as e:
            print(f"[frame {t}] ERROR: {e}")

    print(f"Done: {len(processed)}/{len(bullet_times)} proxy meshes generated")
    return processed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate Poisson collision proxy meshes for all Lyra bullet-time frames"
    )
    parser.add_argument("--ply_base", required=True,
        help="Path to lyra_dynamic_demo_generated/ directory")
    parser.add_argument("--mesh_dir", required=True,
        help="Directory to write .obj proxy meshes")
    parser.add_argument("--opacity_floor", type=float, default=0.05)
    parser.add_argument("--poisson_depth", type=int, default=9)
    parser.add_argument("--density_trim_quantile", type=float, default=0.02)
    args = parser.parse_args()
    generate_all_proxy_meshes(
        args.ply_base, args.mesh_dir,
        args.opacity_floor, args.poisson_depth, args.density_trim_quantile
    )