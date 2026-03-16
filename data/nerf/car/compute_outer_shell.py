#!/usr/bin/env python3
"""Compute outer shell benchmark for NeRF point cloud."""

import numpy as np
import open3d as o3d
import json
from scipy.spatial import KDTree
from collections import defaultdict

# Load aligned NeRF and CAD
print("Loading point clouds...")
nerf_pcd = o3d.io.read_point_cloud('mesh/base_final.ply')
cad_pcd = o3d.io.read_point_cloud('mesh/cad_sample.ply')

nerf_points = np.asarray(nerf_pcd.points)
cad_points = np.asarray(cad_pcd.points)

# Extract outer shell using ray-casting from each point to center
print("Extracting outer shell using ray-casting method...")
center = np.mean(nerf_points, axis=0)

# Build angular hash map
print("Building angular hash map...")
from collections import defaultdict

# Compute direction vectors for all points
directions = nerf_points - center
distances = np.linalg.norm(directions, axis=1)
valid_mask = distances > 1e-6
directions[valid_mask] = directions[valid_mask] / distances[valid_mask, np.newaxis]

# Convert to spherical coordinates (theta, phi)
# theta = azimuthal angle (0 to 2*pi), phi = polar angle (0 to pi)
theta = np.arctan2(directions[:, 1], directions[:, 0])  # -pi to pi
phi = np.arccos(np.clip(directions[:, 2], -1, 1))  # 0 to pi

# Discretize angles into bins (1 degree resolution)
theta_bins = 360
phi_bins = 180
theta_indices = ((theta + np.pi) / (2 * np.pi) * theta_bins).astype(int)
phi_indices = (phi / np.pi * phi_bins).astype(int)

# Create hash map: (theta_bin, phi_bin) -> list of point indices
angle_map = defaultdict(list)
for i in range(len(nerf_points)):
    if valid_mask[i]:
        key = (theta_indices[i], phi_indices[i])
        angle_map[key].append(i)

print(f"Created hash map with {len(angle_map)} angular bins")

# For each point, check if there's a farther point in the same angular bin
print(f"Processing {len(nerf_points)} points...")
outer_mask = np.ones(len(nerf_points), dtype=bool)

processed = 0
for i in range(len(nerf_points)):
    if i % 100000 == 0:
        print(f"  Processed {i}/{len(nerf_points)} points...")
    
    if not valid_mask[i]:
        continue
    
    # Get the angular bin for this point
    key = (theta_indices[i], phi_indices[i])
    
    # Get all points in the same angular bin
    candidates = angle_map[key]
    
    # Check if there's a farther point on the same ray
    current_dist = distances[i]
    
    for idx in candidates:
        if idx == i:
            continue
        
        # If there's a farther point, current point is occluded
        if distances[idx] > current_dist:
            outer_mask[i] = False
            break

outer_indices = np.where(outer_mask)[0]
outer_points = nerf_points[outer_indices]

print(f'Total points: {len(nerf_points)}')
print(f'Outer shell points: {len(outer_points)} ({100*len(outer_points)/len(nerf_points):.1f}%)')
print(f'Removed interior points: {len(nerf_points) - len(outer_points)}')

# Create outer shell point cloud
outer_pcd = o3d.geometry.PointCloud()
outer_pcd.points = o3d.utility.Vector3dVector(outer_points)

# Run ICP alignment on outer shell
print("\n=== Running ICP alignment on outer shell ===")
if not cad_pcd.has_normals():
    print("Estimating normals on CAD...")
    cad_pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))

# Multi-scale ICP with translation only
scales = [0.4, 0.2, 0.1]
current_T = np.eye(4)

for i, dist in enumerate(scales):
    result = o3d.pipelines.registration.registration_icp(
        outer_pcd,
        cad_pcd,
        dist,
        current_T,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100),
    )
    current_T = result.transformation
    print(f"ICP scale {i} (dist={dist:.3f}m): fitness={result.fitness:.4f}, inlier_rmse={result.inlier_rmse:.4f}")

# Apply transformation
outer_pcd.transform(current_T)
outer_points_aligned = np.asarray(outer_pcd.points)

print(f"\nTransformation matrix:")
print(current_T)

# Create heat map with normal-based distances
print("\n=== Creating heat map ===")
cad_points = np.asarray(cad_pcd.points)
cad_normals = np.asarray(cad_pcd.normals)
cad_tree = KDTree(cad_points)

distances = np.zeros(len(outer_points_aligned))
k_candidates = 30

for i in range(len(outer_points_aligned)):
    if i % 5000 == 0 and i > 0:
        print(f"  Processed {i}/{len(outer_points_aligned)} points...")
    
    recon_pt = outer_points_aligned[i]
    dists_nn, indices_nn = cad_tree.query(recon_pt, k=min(k_candidates, len(cad_points)))
    
    min_perpendicular_dist = float('inf')
    for cad_idx in indices_nn:
        cad_pt = cad_points[cad_idx]
        cad_normal = cad_normals[cad_idx]
        
        # Vector from CAD to reconstruction point
        vec = recon_pt - cad_pt
        # Project onto normal to get perpendicular distance
        perpendicular_dist = abs(np.dot(vec, cad_normal))
        
        if perpendicular_dist < min_perpendicular_dist:
            min_perpendicular_dist = perpendicular_dist
    
    distances[i] = min_perpendicular_dist

# Create color map: white (close) to red (far)
max_dist = 0.10  # 10cm scale
colors = np.zeros((len(distances), 3))
for i, d in enumerate(distances):
    ratio = min(d / max_dist, 1.0)
    colors[i] = [1.0, 1.0 - ratio, 1.0 - ratio]  # white -> red

outer_pcd.colors = o3d.utility.Vector3dVector(colors)

# Save heat map
o3d.io.write_point_cloud('mesh/base_final_outer_shell_heatmap.ply', outer_pcd)
print(f"Saved heat map to: mesh/base_final_outer_shell_heatmap.ply")

# Compute outer shell metrics
# Compute outer shell metrics
print("\n=== Computing metrics ===")
outer_dists, _ = cad_tree.query(outer_points_aligned)

# Compute completeness metrics (how much of CAD is covered)
nerf_tree = KDTree(outer_points_aligned)
cad_dists, _ = nerf_tree.query(cad_points)

completeness_1cm = np.mean(cad_dists < 0.01) * 100
completeness_2cm = np.mean(cad_dists < 0.02) * 100
completeness_5cm = np.mean(cad_dists < 0.05) * 100

metrics = {
    'num_outer_points': len(outer_points_aligned),
    'num_total_points': len(nerf_points),
    'outer_percentage': float(100*len(outer_points_aligned)/len(nerf_points)),
    'method': 'angular_hash_map',
    'theta_bins': theta_bins,
    'phi_bins': phi_bins,
    'icp_fitness': float(result.fitness),
    'icp_inlier_rmse': float(result.inlier_rmse),
    'accuracy_mean': float(np.mean(outer_dists)),
    'accuracy_median': float(np.median(outer_dists)),
    'accuracy_std': float(np.std(outer_dists)),
    'accuracy_rmse': float(np.sqrt(np.mean(outer_dists**2))),
    'accuracy_90th': float(np.percentile(outer_dists, 90)),
    'accuracy_95th': float(np.percentile(outer_dists, 95)),
    'completeness_1cm': float(completeness_1cm),
    'completeness_2cm': float(completeness_2cm),
    'completeness_5cm': float(completeness_5cm),
}

# Save metrics
with open('mesh/base_final_outer_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

# Save aligned outer shell PLY
aligned_outer_pcd = o3d.geometry.PointCloud()
aligned_outer_pcd.points = o3d.utility.Vector3dVector(outer_points_aligned)
o3d.io.write_point_cloud('mesh/base_final_outer_shell_aligned.ply', aligned_outer_pcd)

print('\n=== Outer Shell Benchmark (After ICP Alignment) ===')
print(f'ICP Fitness:     {metrics["icp_fitness"]:.4f}')
print(f'ICP Inlier RMSE: {metrics["icp_inlier_rmse"]*100:.2f} cm')
print(f'\nAccuracy (NeRF to CAD):')
print(f'Mean:   {metrics["accuracy_mean"]*100:.2f} cm')
print(f'Median: {metrics["accuracy_median"]*100:.2f} cm')
print(f'RMSE:   {metrics["accuracy_rmse"]*100:.2f} cm')
print(f'90th:   {metrics["accuracy_90th"]*100:.2f} cm')
print(f'95th:   {metrics["accuracy_95th"]*100:.2f} cm')
print(f'\nCompleteness (CAD coverage):')
print(f'<1cm:  {metrics["completeness_1cm"]:.2f}%')
print(f'<2cm:  {metrics["completeness_2cm"]:.2f}%')
print(f'<5cm:  {metrics["completeness_5cm"]:.2f}%')
print(f'\nSaved metrics to: mesh/base_final_outer_metrics.json')
print(f'Saved aligned outer shell to: mesh/base_final_outer_shell_aligned.ply')
print(f'Saved heat map to: mesh/base_final_outer_shell_heatmap.ply')
