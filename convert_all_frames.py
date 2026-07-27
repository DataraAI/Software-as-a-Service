import argparse
import os
import sys
from py3dgsPlyToUsd import convertPlyUSD

def convert_all_frames(ply_base: str, output_dir: str) -> list:
    """
    Convert all Lyra .ply bullet-time frames to individual .usd files.
    Auto-discovers bullet times from folder structure.

    Args:
        ply_base: Path to lyra_dynamic_demo_generated/ directory
        output_dir: Directory to write per-frame .usd files

    Returns:
        List of bullet times that were successfully converted
    """
    os.makedirs(output_dir, exist_ok=True)

    # Auto-discover bullet times from folder structure
    # Each bullet time is a folder named with an integer inside ply_base
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

    converted = []
    for t in bullet_times:
        ply_path = os.path.join(ply_base, str(t), "gaussians_orig", "gaussians_0.ply")
        usd_path = os.path.join(output_dir, f"frame_{t:04d}.usd")

        if os.path.exists(usd_path):
            print(f"[frame {t}] Already exists, skipping: {usd_path}")
            converted.append(t)
            continue

        print(f"[frame {t}] Converting {ply_path} -> {usd_path}")
        try:
            convertPlyUSD(ply_path, usd_path)
            converted.append(t)
        except Exception as e:
            print(f"[frame {t}] ERROR: {e}")

    print(f"Done: {len(converted)}/{len(bullet_times)} frames converted")
    return converted


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert Lyra .ply bullet-time frames to per-frame .usd files"
    )
    parser.add_argument("--ply_base", required=True,
        help="Path to lyra_dynamic_demo_generated/ directory")
    parser.add_argument("--output_dir", required=True,
        help="Directory to write per-frame .usd files")
    args = parser.parse_args()
    convert_all_frames(args.ply_base, args.output_dir)