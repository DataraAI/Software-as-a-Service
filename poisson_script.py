import numpy as np
from plyfile import PlyData
import open3d as o3d
import argparse

def gaussian_ply_to_mesh_poisson(ply_path, output_obj_path,
                                   opacity_floor=0.05,
                                   poisson_depth=9,
                                   density_trim_quantile=0.02):
    ply = PlyData.read(ply_path)
    verts = ply['vertex']

    x = np.array(verts['x'])
    y = np.array(verts['y'])
    z = np.array(verts['z'])
    positions = np.stack([x, y, z], axis=1)

    opacity_raw = np.array(verts['opacity'])
    opacities = 1 / (1 + np.exp(-opacity_raw))

    scale_0 = np.exp(np.array(verts['scale_0']))
    scale_1 = np.exp(np.array(verts['scale_1']))
    scale_2 = np.exp(np.array(verts['scale_2']))
    scales = np.stack([scale_0, scale_1, scale_2], axis=1).mean(axis=1)

    mask = opacities > opacity_floor
    positions = positions[mask]
    opacities = opacities[mask]
    scales = scales[mask]

    print(f"Using {len(positions):,} Gaussians after opacity filtering")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(positions)

    if scales is not None:
        scale_cutoff = np.quantile(scales, 0.95)
        keep = scales < scale_cutoff
        pcd = pcd.select_by_index(np.where(keep)[0])
        print(f"Kept {len(pcd.points):,} points after trimming largest 5% by scale")

    print("Estimating normals...")
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))
    pcd.orient_normals_consistent_tangent_plane(k=15)

    print(f"Running Poisson reconstruction (depth={poisson_depth})...")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=poisson_depth
    )

    densities = np.asarray(densities)
    density_threshold = np.quantile(densities, density_trim_quantile)
    vertices_to_remove = densities < density_threshold
    mesh.remove_vertices_by_mask(vertices_to_remove)

    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()

    o3d.io.write_triangle_mesh(output_obj_path, mesh)
    print(f"Saved mesh: {len(mesh.vertices):,} vertices, {len(mesh.triangles):,} faces")
    print(f"Output: {output_obj_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Poisson collision proxy mesh from Gaussian splat .ply")
    parser.add_argument("--input", required=True, help="Input .ply file path")
    parser.add_argument("--output", required=True, help="Output .obj file path")
    parser.add_argument("--opacity_floor", type=float, default=0.05)
    parser.add_argument("--poisson_depth", type=int, default=9)
    parser.add_argument("--density_trim_quantile", type=float, default=0.02)
    args = parser.parse_args()
    gaussian_ply_to_mesh_poisson(
        args.input, args.output,
        args.opacity_floor, args.poisson_depth, args.density_trim_quantile
    )