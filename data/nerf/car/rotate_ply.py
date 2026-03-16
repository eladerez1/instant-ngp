#!/usr/bin/env python3
"""
Rotate a PLY file by -90 degrees around the Z-axis and 60 degrees around the Y-axis.

Usage:
    python rotate_ply.py input.ply output.ply
    python rotate_ply.py input.ply  # outputs to input_rotated.ply
"""

import argparse
import os
import sys
import struct
import numpy as np
import math
import logging
import json
from dataclasses import dataclass, asdict

# Try importing Open3D for PCA/ICP alignment
try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class EvalMetrics:
    """Reconstruction-vs-CAD evaluation metrics."""
    # Accuracy: reconstruction → CAD distances (meters)
    accuracy_mean: float
    accuracy_median: float
    accuracy_std: float
    accuracy_rmse: float
    accuracy_90th: float
    accuracy_95th: float

    # Completeness: CAD → reconstruction distances (meters)
    completeness_mean: float
    completeness_median: float
    completeness_std: float
    completeness_90th: float
    completeness_95th: float

    # Completeness at thresholds: fraction of CAD surface within X meters
    completeness_at_1cm: float
    completeness_at_2cm: float
    completeness_at_5cm: float
    completeness_at_10cm: float

    # Hausdorff (worst-case)
    hausdorff_recon_to_cad: float
    hausdorff_cad_to_recon: float

    num_recon_points: int
    num_cad_points: int


def rotate_point(point):
    """Rotate a point -90 degrees around z-axis, then 60 degrees around y-axis."""
    x, y, z = point
    
    # First: Rotate -90 degrees around z-axis
    # [ 0   1  0]
    # [-1   0  0]
    # [ 0   0  1]
    x1, y1, z1 = y, -x, z
    
    # Second: Rotate 60 degrees around y-axis
    # [cos(60)   0  sin(60)]     [0.5    0  0.866]
    # [   0      1     0   ]  =  [  0    1    0  ]
    # [-sin(60)  0  cos(60)]     [-0.866 0  0.5  ]
    cos60 = 0.5
    sin60 = math.sqrt(3) / 2  # 0.866...
    
    x2 = cos60 * x1 + sin60 * z1
    y2 = y1
    z2 = -sin60 * x1 + cos60 * z1
    
    return [x2, y2, z2]


def parse_ply_header(lines):
    """Parse PLY header and return header lines and properties info."""
    header_lines = []
    vertex_count = 0
    face_count = 0
    in_header = True
    vertex_properties = []
    
    for i, line in enumerate(lines):
        if in_header:
            header_lines.append(line)
            
            if line.startswith('element vertex'):
                vertex_count = int(line.split()[2])
            elif line.startswith('element face'):
                face_count = int(line.split()[2])
            elif line.startswith('property') and vertex_count > 0 and face_count == 0:
                # This is a vertex property
                parts = line.split()
                if len(parts) >= 3:
                    vertex_properties.append(parts[-1])  # property name
            elif line.strip() == 'end_header':
                in_header = False
                return header_lines, vertex_count, face_count, vertex_properties, i + 1
    
    raise ValueError("Invalid PLY file: no end_header found")


def get_ply_center(ply_path):
    """Get the center point of a PLY file (binary or ASCII)."""
    print(f"Reading reference PLY: {ply_path}")
    
    # Try to read header to check format
    with open(ply_path, 'rb') as f:
        header_lines = []
        while True:
            line = f.readline()
            try:
                line_str = line.decode('utf-8').strip()
            except:
                break
            header_lines.append(line_str)
            if line_str == 'end_header':
                break
        
        # Check if binary
        is_binary = any('binary' in line for line in header_lines)
        
        if is_binary:
            # Parse binary PLY - just get vertex count and read positions
            vertex_count = 0
            for line in header_lines:
                if line.startswith('element vertex'):
                    vertex_count = int(line.split()[2])
                    break
            
            print(f"Binary PLY with {vertex_count} vertices")
            
            # Read binary vertex data (assuming double x,y,z,nx,ny,nz based on header)
            vertices = []
            for i in range(vertex_count):
                # Read x, y, z as doubles
                x, y, z = struct.unpack('ddd', f.read(24))
                # Skip normals (nx, ny, nz) - 3 doubles = 24 bytes
                f.read(24)
                vertices.append([x, y, z])
            
            vertices = np.array(vertices)
            center = np.mean(vertices, axis=0)
            print(f"Reference center: {center}")
            return center
    
    # ASCII format - original code
    with open(ply_path, 'r') as f:
        lines = [line.rstrip('\n') for line in f]
    
    print("Parsing header...")
    # Parse header
    header_lines, vertex_count, face_count, vertex_properties, data_start = parse_ply_header(lines)
    
    print(f"Computing center from {vertex_count} vertices...")
    # Compute center
    vertices = []
    for i in range(data_start, data_start + vertex_count):
        parts = lines[i].split()
        x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
        vertices.append([x, y, z])
    
    vertices = np.array(vertices)
    center = np.mean(vertices, axis=0)
    print(f"Reference center: {center}")
    return center


def _center_and_pca(points):
    """Return centroid and PCA axes (3x3, rows = principal components, sorted by variance desc)."""
    centroid = points.mean(axis=0)
    centered = points - centroid
    cov = centered.T @ centered / len(centered)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    # Sort descending by eigenvalue
    order = np.argsort(eigenvalues)[::-1]
    axes = eigenvectors[:, order].T  # rows = principal axes
    return centroid, axes


def _ensure_right_handed(axes):
    """Ensure axes form a right-handed coordinate system."""
    if np.linalg.det(axes) < 0:
        axes[2] *= -1
    return axes


def align_pca_simple(source_points, target_points):
    """Coarse alignment via PCA: align source principal axes to target.
    
    Tries all 4 valid axis flips (signs of first two axes) and picks
    the one with lowest mean distance to target.
    
    Returns:
        4x4 transformation matrix.
    """
    from scipy.spatial import KDTree
    
    src_center, src_axes = _center_and_pca(source_points)
    tgt_center, tgt_axes = _center_and_pca(target_points)
    
    src_axes = _ensure_right_handed(src_axes)
    tgt_axes = _ensure_right_handed(tgt_axes)
    
    # Try 4 sign flips: (++, +-, -+, --) on first two axes
    tgt_tree = KDTree(target_points)
    best_T = np.eye(4)
    best_dist = float("inf")
    
    for s0 in [1, -1]:
        for s1 in [1, -1]:
            flipped = src_axes.copy()
            flipped[0] *= s0
            flipped[1] *= s1
            flipped = _ensure_right_handed(flipped)
            
            R = tgt_axes.T @ flipped
            t = tgt_center - R @ src_center
            
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = t
            
            transformed = (R @ source_points.T).T + t
            # Sample for speed
            sample_idx = np.random.default_rng(42).choice(
                len(transformed), min(5000, len(transformed)), replace=False
            )
            dists, _ = tgt_tree.query(transformed[sample_idx])
            mean_dist = np.mean(dists)
            
            if mean_dist < best_dist:
                best_dist = mean_dist
                best_T = T
    
    logger.info(f"PCA coarse alignment: mean distance = {best_dist:.4f} m")
    return best_T


def align_icp_simple(source_pcd, target_pcd, init_transform=None, max_correspondence_distance=0.1, translation_only=False):
    """Fine alignment via ICP using Open3D.
    
    Args:
        translation_only: If True, only compute translation (no rotation).
    
    Returns:
        4x4 transformation matrix.
    """
    if not HAS_OPEN3D:
        logger.warning("Open3D not available, skipping ICP alignment")
        return np.eye(4) if init_transform is None else init_transform
    
    if init_transform is None:
        init_transform = np.eye(4)
    
    # Estimate normals on target for point-to-plane
    if not target_pcd.has_normals():
        target_pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))
    
    # For translation-only, use point-to-point instead of point-to-plane
    if translation_only:
        estimation_method = o3d.pipelines.registration.TransformationEstimationPointToPoint()
    else:
        estimation_method = o3d.pipelines.registration.TransformationEstimationPointToPlane()
    
    # Multi-scale ICP: coarse → fine
    scales = [
        max_correspondence_distance * 4,
        max_correspondence_distance * 2,
        max_correspondence_distance,
    ]
    current_T = init_transform.copy()
    
    for i, dist in enumerate(scales):
        result = o3d.pipelines.registration.registration_icp(
            source_pcd,
            target_pcd,
            dist,
            current_T,
            estimation_method,
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100),
        )
        
        # Force translation-only if requested - extract only translation
        if translation_only:
            # Keep rotation from init_transform, only update translation
            new_T = init_transform.copy()
            new_T[:3, 3] = result.transformation[:3, 3]
            current_T = new_T
        else:
            current_T = result.transformation
        
        logger.info(
            f"ICP scale {i} (dist={dist:.3f}): fitness={result.fitness:.4f}, inlier_rmse={result.inlier_rmse:.4f}"
        )
    
    return current_T


def create_normal_distance_colored_pointcloud(recon_pcd, cad_pcd):
    """Create a colored point cloud where colors represent normal-based distance to CAD.
    
    White (1,1,1) = closest points (distance 0)
    Red (1,0,0) = farthest points (max distance)
    """
    from scipy.spatial import KDTree
    
    recon_points = np.asarray(recon_pcd.points)
    cad_points = np.asarray(cad_pcd.points)
    cad_normals = np.asarray(cad_pcd.normals)
    
    if len(cad_normals) == 0:
        logger.warning("CAD point cloud has no normals - estimating...")
        cad_pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))
        cad_normals = np.asarray(cad_pcd.normals)
    
    logger.info("Computing normal-based distances to CAD for color mapping...")
    
    # Build KDTree for finding nearby CAD points
    cad_tree = KDTree(cad_points)
    distances = np.zeros(len(recon_points))
    
    k_candidates = 30
    batch_size = 10000
    
    for i in range(0, len(recon_points), batch_size):
        end_i = min(i + batch_size, len(recon_points))
        batch_points = recon_points[i:end_i]
        
        # Query k nearest CAD points
        dists_nn, indices_nn = cad_tree.query(batch_points, k=min(k_candidates, len(cad_points)))
        
        for j in range(len(batch_points)):
            vggt_pt = batch_points[j]
            min_perpendicular_dist = float('inf')
            best_cad_idx = None
            
            candidates = indices_nn[j] if len(indices_nn.shape) > 1 else [indices_nn[j]]
            for cad_idx in candidates:
                cad_pt = cad_points[cad_idx]
                cad_normal = cad_normals[cad_idx]
                
                vec = vggt_pt - cad_pt
                t = np.dot(vec, cad_normal)
                closest_on_ray = cad_pt + t * cad_normal
                perpendicular_dist = np.linalg.norm(vggt_pt - closest_on_ray)
                
                if perpendicular_dist < min_perpendicular_dist:
                    min_perpendicular_dist = perpendicular_dist
                    best_cad_idx = cad_idx
            
            best_cad_pt = cad_points[best_cad_idx]
            best_cad_normal = cad_normals[best_cad_idx]
            vec = vggt_pt - best_cad_pt
            normal_dist = np.abs(np.dot(vec, best_cad_normal))
            distances[i + j] = normal_dist
        
        if (i + batch_size) % 100000 == 0:
            logger.info(f"  Processed {i + batch_size} / {len(recon_points)} points")
    
    min_dist = np.min(distances)
    max_dist = np.max(distances)
    logger.info(f"Normal-based distance range: min={min_dist:.4f}m ({min_dist*100:.1f}cm), max={max_dist:.4f}m ({max_dist*100:.1f}cm)")
    
    # Fixed scale: 0m = 0, 0.1m (10cm) = 1
    max_scale = 0.10
    normalized_dist = np.clip(distances / max_scale, 0.0, 1.0)
    
    above_10cm = np.sum(distances > max_scale)
    logger.info(f"Points above 10cm threshold: {above_10cm} ({100.0 * above_10cm / len(distances):.1f}% clamped to red)")
    
    # Create color map: white (0) -> red (1)
    colors = np.ones((len(recon_points), 3))
    colors[:, 1] = 1.0 - normalized_dist  # Green channel
    colors[:, 2] = 1.0 - normalized_dist  # Blue channel
    
    colored_pcd = o3d.geometry.PointCloud()
    colored_pcd.points = recon_pcd.points
    colored_pcd.colors = o3d.utility.Vector3dVector(colors)
    
    logger.info(f"Created normal-distance-colored point cloud with {len(recon_points)} points")
    return colored_pcd


def extract_outer_points(points, percentile=90):
    """Extract the outer shell of points by keeping only points far from center.
    
    Args:
        points: Nx3 array of points
        percentile: Keep points in the top (100-percentile)% of distances from center
        
    Returns:
        Indices of outer points
    """
    center = np.mean(points, axis=0)
    distances = np.linalg.norm(points - center, axis=1)
    
    threshold = np.percentile(distances, percentile)
    outer_indices = np.where(distances >= threshold)[0]
    
    logger.info(f"Outer shell extraction: {len(outer_indices)} / {len(points)} points ({100*len(outer_indices)/len(points):.1f}%)")
    logger.info(f"  Distance threshold: {threshold:.4f}m (>{percentile}th percentile)")
    logger.info(f"  Distance range: [{np.min(distances):.4f}, {np.max(distances):.4f}]m")
    
    return outer_indices


def compute_metrics(recon_points, cad_points):
    """Compute accuracy and completeness metrics after alignment."""
    from scipy.spatial import KDTree
    
    # Accuracy: reconstruction → CAD
    cad_tree = KDTree(cad_points)
    acc_dists, _ = cad_tree.query(recon_points)
    
    # Completeness: CAD → reconstruction
    recon_tree = KDTree(recon_points)
    comp_dists, _ = recon_tree.query(cad_points)
    
    return EvalMetrics(
        accuracy_mean=float(np.mean(acc_dists)),
        accuracy_median=float(np.median(acc_dists)),
        accuracy_std=float(np.std(acc_dists)),
        accuracy_rmse=float(np.sqrt(np.mean(acc_dists**2))),
        accuracy_90th=float(np.percentile(acc_dists, 90)),
        accuracy_95th=float(np.percentile(acc_dists, 95)),
        completeness_mean=float(np.mean(comp_dists)),
        completeness_median=float(np.median(comp_dists)),
        completeness_std=float(np.std(comp_dists)),
        completeness_90th=float(np.percentile(comp_dists, 90)),
        completeness_95th=float(np.percentile(comp_dists, 95)),
        completeness_at_1cm=float(np.mean(comp_dists < 0.01)),
        completeness_at_2cm=float(np.mean(comp_dists < 0.02)),
        completeness_at_5cm=float(np.mean(comp_dists < 0.05)),
        completeness_at_10cm=float(np.mean(comp_dists < 0.10)),
        hausdorff_recon_to_cad=float(np.max(acc_dists)),
        hausdorff_cad_to_recon=float(np.max(comp_dists)),
        num_recon_points=len(recon_points),
        num_cad_points=len(cad_points),
    )


def save_ply_from_points(points, output_path, colors=None):
    """Save points to PLY file (ASCII format)."""
    header = [
        "ply",
        "format ascii 1.0",
        f"element vertex {len(points)}",
        "property float x",
        "property float y",
        "property float z",
    ]
    
    if colors is not None:
        header.extend([
            "property uchar red",
            "property uchar green",
            "property uchar blue",
        ])
    
    header.append("end_header")
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(header) + '\n')
        
        for i, pt in enumerate(points):
            if colors is not None:
                # Colors should be in 0-255 range
                r, g, b = colors[i]
                f.write(f"{pt[0]} {pt[1]} {pt[2]} {int(r*255)} {int(g*255)} {int(b*255)}\n")
            else:
                f.write(f"{pt[0]} {pt[1]} {pt[2]}\n")
    
    logger.info(f"Saved {len(points)} points to {output_path}")


def compute_metrics(recon_points, cad_points):
    """Compute accuracy and completeness metrics after alignment."""
    from scipy.spatial import KDTree
    
    # Accuracy: reconstruction → CAD
    cad_tree = KDTree(cad_points)
    acc_dists, _ = cad_tree.query(recon_points)
    
    # Completeness: CAD → reconstruction
    recon_tree = KDTree(recon_points)
    comp_dists, _ = recon_tree.query(cad_points)
    
    return EvalMetrics(
        accuracy_mean=float(np.mean(acc_dists)),
        accuracy_median=float(np.median(acc_dists)),
        accuracy_std=float(np.std(acc_dists)),
        accuracy_rmse=float(np.sqrt(np.mean(acc_dists**2))),
        accuracy_90th=float(np.percentile(acc_dists, 90)),
        accuracy_95th=float(np.percentile(acc_dists, 95)),
        completeness_mean=float(np.mean(comp_dists)),
        completeness_median=float(np.median(comp_dists)),
        completeness_std=float(np.std(comp_dists)),
        completeness_90th=float(np.percentile(comp_dists, 90)),
        completeness_95th=float(np.percentile(comp_dists, 95)),
        completeness_at_1cm=float(np.mean(comp_dists < 0.01)),
        completeness_at_2cm=float(np.mean(comp_dists < 0.02)),
        completeness_at_5cm=float(np.mean(comp_dists < 0.05)),
        completeness_at_10cm=float(np.mean(comp_dists < 0.10)),
        hausdorff_recon_to_cad=float(np.max(acc_dists)),
        hausdorff_cad_to_recon=float(np.max(comp_dists)),
        num_recon_points=len(recon_points),
        num_cad_points=len(cad_points),
    )


def rotate_ply(input_path, output_path, reference_ply=None, use_pca=False, use_icp=False, lock_rotation=False, evaluate=False):
    """Rotate PLY file and optionally align it to a reference PLY.
    
    Args:
        input_path: Input PLY file path
        output_path: Output PLY file path
        reference_ply: Reference PLY file for alignment
        use_pca: If True, use PCA for coarse alignment
        use_icp: If True, use ICP for fine alignment (requires Open3D)
        lock_rotation: If True, only apply translation (no rotation) during PCA/ICP
        evaluate: If True, generate evaluation metrics and heat map
    """
    print(f"Reading input PLY: {input_path}")
    with open(input_path, 'r') as f:
        lines = [line.rstrip('\n') for line in f]
    
    print("Parsing header...")
    # Parse header
    header_lines, vertex_count, face_count, vertex_properties, data_start = parse_ply_header(lines)
    
    print(f"Processing {vertex_count} vertices...")
    # Process vertices - rotate and collect
    rotated_vertices = []
    other_data = []
    
    for i in range(data_start, data_start + vertex_count):
        parts = lines[i].split()
        
        # First three values are typically x, y, z
        x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
        new_x, new_y, new_z = rotate_point([x, y, z])
        
        rotated_vertices.append([new_x, new_y, new_z])
        other_data.append(parts[3:])
    
    # If reference PLY is provided, align the rotated mesh to match it
    if reference_ply and os.path.exists(reference_ply):
        rotated_vertices = np.array(rotated_vertices)
        
        if use_pca or use_icp:
            # Load reference PLY
            if HAS_OPEN3D:
                logger.info("Loading reference PLY with Open3D...")
                ref_pcd = o3d.io.read_point_cloud(reference_ply)
                ref_points = np.asarray(ref_pcd.points)
                logger.info(f"Reference has {len(ref_points)} points")
            else:
                logger.warning("Open3D not available, falling back to simple centering")
                ref_points = None
            
            # Apply alignment transformations
            final_transform = np.eye(4)
            
            if use_pca and ref_points is not None and not lock_rotation:
                logger.info("Applying PCA coarse alignment...")
                pca_transform = align_pca_simple(rotated_vertices, ref_points)
                # Apply PCA transform
                ones = np.ones((len(rotated_vertices), 1))
                homogeneous = np.hstack([rotated_vertices, ones])
                rotated_vertices = (pca_transform @ homogeneous.T).T[:, :3]
                final_transform = pca_transform @ final_transform
            elif use_pca and lock_rotation:
                logger.info("PCA skipped (rotation locked)")
            
            if use_icp and HAS_OPEN3D and ref_points is not None:
                logger.info(f"Applying ICP fine alignment (translation_only={lock_rotation})...")
                # Create source point cloud
                src_pcd = o3d.geometry.PointCloud()
                src_pcd.points = o3d.utility.Vector3dVector(rotated_vertices)
                
                # Apply ICP
                icp_transform = align_icp_simple(src_pcd, ref_pcd, init_transform=np.eye(4), translation_only=lock_rotation)
                # Apply ICP transform
                ones = np.ones((len(rotated_vertices), 1))
                homogeneous = np.hstack([rotated_vertices, ones])
                rotated_vertices = (icp_transform @ homogeneous.T).T[:, :3]
                final_transform = icp_transform @ final_transform
                
            logger.info("Final transformation applied")
        else:
            # Simple centering (original behavior)
            reference_center = get_ply_center(reference_ply)
            current_center = np.mean(rotated_vertices, axis=0)
            translation = reference_center - current_center
            
            rotated_vertices += translation
            
            print(f"Current center: {current_center}")
            print(f"Reference center: {reference_center}")
            print(f"Translation applied: {translation}")
    
    # Build output lines
    rotated_lines = header_lines.copy()
    
    for i, (vertex, extra) in enumerate(zip(rotated_vertices, other_data)):
        new_parts = [str(vertex[0]), str(vertex[1]), str(vertex[2])] + extra
        rotated_lines.append(' '.join(new_parts))
    
    # Copy faces and any other data as-is
    for i in range(data_start + vertex_count, len(lines)):
        rotated_lines.append(lines[i])
    
    # Write output
    with open(output_path, 'w') as f:
        for line in rotated_lines:
            f.write(line + '\n')
    
    print(f"Rotated {vertex_count} vertices")
    print(f"Saved to: {output_path}")
    
    # Generate evaluation if requested
    if evaluate and reference_ply and HAS_OPEN3D:
        logger.info("=== Generating Evaluation ===")
        
        # Load reference PLY
        ref_pcd = o3d.io.read_point_cloud(reference_ply)
        ref_points = np.asarray(ref_pcd.points)
        
        # Create point cloud from rotated vertices
        result_pcd = o3d.geometry.PointCloud()
        result_pcd.points = o3d.utility.Vector3dVector(np.array(rotated_vertices))
        
        # Compute metrics
        logger.info("Computing accuracy and completeness metrics...")
        metrics = compute_metrics(np.array(rotated_vertices), ref_points)
        
        # Save metrics to JSON
        output_dir = os.path.dirname(output_path)
        base_name = os.path.splitext(os.path.basename(output_path))[0]
        metrics_path = os.path.join(output_dir, f"{base_name}_metrics.json")
        
        with open(metrics_path, 'w') as f:
            json.dump(asdict(metrics), f, indent=2)
        logger.info(f"Saved metrics to: {metrics_path}")
        
        # Extract outer points and compute outer-only metrics
        logger.info("\n=== Outer Shell Benchmark ===")
        outer_indices = extract_outer_points(np.array(rotated_vertices), percentile=90)
        outer_points = np.array(rotated_vertices)[outer_indices]
        
        # Compute metrics for outer points only
        logger.info("Computing metrics for outer shell only...")
        outer_metrics = compute_metrics(outer_points, ref_points)
        
        # Save outer metrics
        outer_metrics_path = os.path.join(output_dir, f"{base_name}_outer_metrics.json")
        with open(outer_metrics_path, 'w') as f:
            json.dump(asdict(outer_metrics), f, indent=2)
        logger.info(f"Saved outer shell metrics to: {outer_metrics_path}")
        
        # Save outer points to PLY
        outer_ply_path = os.path.join(output_dir, f"{base_name}_outer_shell.ply")
        save_ply_from_points(outer_points, outer_ply_path)
        logger.info(f"Saved outer shell point cloud to: {outer_ply_path}")
        
        # Print summary
        logger.info("=== Evaluation Summary ===")
        logger.info(f"Accuracy (NeRF → CAD):")
        logger.info(f"  Mean:   {metrics.accuracy_mean*100:.2f} cm")
        logger.info(f"  Median: {metrics.accuracy_median*100:.2f} cm")
        logger.info(f"  RMSE:   {metrics.accuracy_rmse*100:.2f} cm")
        logger.info(f"  90th:   {metrics.accuracy_90th*100:.2f} cm")
        logger.info(f"  95th:   {metrics.accuracy_95th*100:.2f} cm")
        logger.info(f"Completeness (CAD → NeRF):")
        logger.info(f"  Mean:   {metrics.completeness_mean*100:.2f} cm")
        logger.info(f"  Median: {metrics.completeness_median*100:.2f} cm")
        logger.info(f"  @ 1cm:  {metrics.completeness_at_1cm*100:.1f}%")
        logger.info(f"  @ 2cm:  {metrics.completeness_at_2cm*100:.1f}%")
        logger.info(f"  @ 5cm:  {metrics.completeness_at_5cm*100:.1f}%")
        logger.info(f"  @ 10cm: {metrics.completeness_at_10cm*100:.1f}%")
        
        logger.info("\n=== Outer Shell Benchmark Summary ===")
        logger.info(f"Outer Shell Accuracy (NeRF outer → CAD):")
        logger.info(f"  Mean:   {outer_metrics.accuracy_mean*100:.2f} cm")
        logger.info(f"  Median: {outer_metrics.accuracy_median*100:.2f} cm")
        logger.info(f"  RMSE:   {outer_metrics.accuracy_rmse*100:.2f} cm")
        logger.info(f"  90th:   {outer_metrics.accuracy_90th*100:.2f} cm")
        logger.info(f"  95th:   {outer_metrics.accuracy_95th*100:.2f} cm")
        
        # Generate normal-based heat map
        logger.info("Generating normal-based distance heat map...")
        colored_pcd = create_normal_distance_colored_pointcloud(result_pcd, ref_pcd)
        
        # Save colored point cloud
        heatmap_path = os.path.join(output_dir, f"{base_name}_heatmap.ply")
        o3d.io.write_point_cloud(heatmap_path, colored_pcd)
        logger.info(f"Saved heat map to: {heatmap_path}")
    elif evaluate and not HAS_OPEN3D:
        logger.warning("Open3D is required for evaluation. Install with: pip install open3d")


def main():
    parser = argparse.ArgumentParser(description='Rotate PLY file -90° around Z-axis and align to reference')
    parser.add_argument('input', help='Input PLY file')
    parser.add_argument('output', nargs='?', help='Output PLY file (default: input_rotated.ply)')
    parser.add_argument('--reference', help='Reference PLY file to match center (default: mesh/cad_sample.ply)')
    parser.add_argument('--pca', action='store_true', help='Use PCA for coarse alignment')
    parser.add_argument('--icp', action='store_true', help='Use ICP for fine alignment (requires Open3D)')
    parser.add_argument('--lock-rotation', action='store_true', help='Lock rotation during PCA/ICP (translation only)')
    parser.add_argument('--evaluate', action='store_true', help='Generate evaluation metrics and heat map')
    
    args = parser.parse_args()
    
    if (args.pca or args.icp or args.evaluate) and not HAS_OPEN3D:
        logger.error("Open3D is required for PCA/ICP/evaluation. Install with: pip install open3d")
        sys.exit(1)
    
    input_path = args.input
    if not os.path.exists(input_path):
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)
    
    if args.output:
        output_path = args.output
    else:
        # Generate output filename
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_rotated{ext}"
    
    # Default reference PLY
    reference_ply = args.reference
    if not reference_ply:
        # Check if cad_sample.ply exists in the same directory or mesh subdirectory
        input_dir = os.path.dirname(input_path)
        potential_refs = [
            os.path.join(input_dir, 'cad_sample.ply'),
            os.path.join(input_dir, '..', 'mesh', 'cad_sample.ply'),
            '/isilon/Automotive/RnD/elad.e/instant-ngp/data/nerf/car/mesh/cad_sample.ply'
        ]
        for ref in potential_refs:
            if os.path.exists(ref):
                reference_ply = ref
                break
    
    rotate_ply(input_path, output_path, reference_ply, use_pca=args.pca, use_icp=args.icp, lock_rotation=args.lock_rotation, evaluate=args.evaluate)


if __name__ == '__main__':
    main()
