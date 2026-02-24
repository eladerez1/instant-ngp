#!/usr/bin/env python3
"""
Build instant-ngp transforms.json from car scan calibration data.

The car scan has 13 cameras that move together along the X axis.
Camera 1 at frame 0 is at world origin (0, 0, 0).
All cameras move -10cm between frames along X (negative direction).

The extrinsics.yaml contains T_cn_cnm1 which is the transform from camera n-1 to camera n.
We chain these to get each camera's pose relative to camera 0.
"""

import json
import yaml
import numpy as np
import math
import argparse
from pathlib import Path


def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def parse_intrinsics(intrinsics_yaml):
    """Parse intrinsics.yaml to get camera parameters."""
    cameras = {}
    for cam_id, cam_data in intrinsics_yaml.items():
        if isinstance(cam_data, dict) and 'cam_name' in cam_data:
            # params: [fx, fy, cx, cy, k1, k2, p1, p2]
            params = cam_data['params']
            cameras[cam_data['cam_name']] = {
                'fx': params[0],
                'fy': params[1],
                'cx': params[2],
                'cy': params[3],
                'k1': params[4],
                'k2': params[5],
                'p1': params[6],
                'p2': params[7],
            }
    return cameras


def parse_extrinsics(extrinsics_yaml):
    """
    Parse extrinsics.yaml to compute each camera's pose relative to cam0.
    
    The yaml contains T_cn_cnm1: transform from camera n-1 to camera n.
    We need to chain these to get T_cn_c0 for each camera.
    """
    # Build ordered list of cameras
    cam_order = []
    cam_data = {}
    
    for key, value in extrinsics_yaml.items():
        if key.startswith('cam') and isinstance(value, dict):
            cam_idx = int(key[3:])
            cam_name = value['cam_name']
            resolution = value['resolution']
            
            # T_cn_cnm1 is the transform from previous camera to this camera
            if 'T_cn_cnm1' in value:
                T = np.array(value['T_cn_cnm1'])
            else:
                # cam0 has no T_cn_cnm1, it's at identity
                T = np.eye(4)
            
            cam_data[cam_idx] = {
                'name': cam_name,
                'T_cn_cnm1': T,
                'resolution': resolution,
            }
            cam_order.append(cam_idx)
    
    cam_order.sort()
    
    # Chain transforms to get T_cn_c0 (transform from cam0 to camN)
    # T_c1_c0 = T_c1_c0 (from yaml, which is T_cn_cnm1 for cam1)
    # T_c2_c0 = T_c2_c1 @ T_c1_c0
    # etc.
    
    T_cumulative = np.eye(4)  # T_c0_c0 = identity
    cam_poses_rel_cam0 = {}
    
    for cam_idx in cam_order:
        if cam_idx == 0:
            cam_poses_rel_cam0[cam_data[cam_idx]['name']] = {
                'T_cn_c0': np.eye(4),
                'resolution': cam_data[cam_idx]['resolution'],
            }
        else:
            T_cn_cnm1 = cam_data[cam_idx]['T_cn_cnm1']
            T_cumulative = T_cn_cnm1 @ T_cumulative
            cam_poses_rel_cam0[cam_data[cam_idx]['name']] = {
                'T_cn_c0': T_cumulative.copy(),
                'resolution': cam_data[cam_idx]['resolution'],
            }
    
    return cam_poses_rel_cam0


def T_to_ngp_transform(T_cam_world):
    """
    Convert camera-to-world transform to instant-ngp format.
    
    instant-ngp expects a 4x4 transform_matrix where:
    - The matrix transforms from camera space to world space
    - Camera looks along -Z in camera space (OpenGL convention)
    - The matrix is [R | t; 0 0 0 1] where R is rotation and t is translation
    
    Our T is world-to-camera, so we need to invert it.
    Also need to apply coordinate system conversion if needed.
    """
    # T_cam_world transforms world points to camera frame
    # We need T_world_cam (camera pose in world)
    T_world_cam = np.linalg.inv(T_cam_world)
    
    # instant-ngp uses OpenGL convention: camera looks along -Z, Y up
    # Our camera convention: camera looks along +Z, Y down
    # Convert: flip Y and Z
    convert = np.array([
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1]
    ], dtype=np.float64)
    
    T_ngp = T_world_cam @ convert
    
    return T_ngp


def main():
    parser = argparse.ArgumentParser(description='Build instant-ngp transforms.json from car scan calibration')
    parser.add_argument('--data_dir', type=str, 
                        default='/isilon/Automotive/RnD/elad.e/mpsfm_data/2024-09-27T16-31-13.000Z_d8350e36-4628-4c39-8164-05521c599810',
                        help='Path to car scan data (contains intrinsics.yaml and extrinsics.yaml)')
    parser.add_argument('--images_dir', type=str,
                        default='/isilon/Automotive/RnD/elad.e/instant-ngp/data/nerf/car/images',
                        help='Directory containing the images (to detect available frames)')
    parser.add_argument('--output_file', type=str,
                        default='/isilon/Automotive/RnD/elad.e/instant-ngp/data/nerf/car/transforms.json',
                        help='Output transforms.json file path')
    parser.add_argument('--motion_per_frame', type=float, default=0.1,
                        help='Motion per frame in meters (default: 0.1m = 10cm)')
    parser.add_argument('--aabb_scale', type=int, default=None,
                        help='AABB scale for instant-ngp (default: auto-calculated)')
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    images_dir = Path(args.images_dir)
    output_file = Path(args.output_file)
    
    print(f"Data directory: {data_dir}")
    print(f"Images directory: {images_dir}")
    print(f"Output file: {output_file}")
    
    # Load calibration
    intrinsics_yaml = load_yaml(data_dir / 'intrinsics.yaml')
    extrinsics_yaml = load_yaml(data_dir / 'extrinsics.yaml')
    
    intrinsics = parse_intrinsics(intrinsics_yaml)
    extrinsics = parse_extrinsics(extrinsics_yaml)
    
    print(f"Loaded {len(intrinsics)} camera intrinsics")
    print(f"Loaded {len(extrinsics)} camera extrinsics")
    
    # Find all images in images_dir
    # Expected format: <camera_name>_<frame_num>.png (e.g., at_cam_01_0033.png)
    image_files = sorted(images_dir.glob('*.png'))
    print(f"Found {len(image_files)} images")
    
    if not image_files:
        print("Error: No images found in images_dir. Run create_masked_frames.py first.")
        return
    
    # Parse image names to get camera and frame info
    frames = []
    
    # Use representative intrinsics (first camera found)
    first_intrinsics = None
    first_resolution = None
    
    for image_path in image_files:
        # Parse filename: <camera_name>_<frame_num>.png
        stem = image_path.stem  # e.g., "at_cam_01_0033"
        parts = stem.rsplit('_', 1)  # Split from right to handle camera names with underscores
        if len(parts) != 2:
            print(f"Warning: Skipping unexpected filename format: {image_path.name}")
            continue
        
        cam_name = parts[0]  # e.g., "at_cam_01" or "at_front_00"
        frame_num_str = parts[1]  # e.g., "0033"
        
        try:
            frame_num = int(frame_num_str)
        except ValueError:
            print(f"Warning: Could not parse frame number from: {image_path.name}")
            continue
        
        if cam_name not in intrinsics or cam_name not in extrinsics:
            print(f"Warning: Camera {cam_name} not found in calibration, skipping")
            continue
        
        cam_intrinsics = intrinsics[cam_name]
        cam_extrinsics = extrinsics[cam_name]
        resolution = cam_extrinsics['resolution']
        
        # Store first intrinsics for global values
        if first_intrinsics is None:
            first_intrinsics = cam_intrinsics
            first_resolution = resolution
        
        # T_cn_c0 is transform from cam0 frame to this camera frame
        T_cam_cam0 = cam_extrinsics['T_cn_c0']
        
        # All cameras are fixed in the world, the car moves through them.
        # From NeRF's perspective (car is stationary), all cameras move together.
        # At frame N, all cameras have moved +N*motion_per_frame in X (positive direction).
        # This will result in cameras moving in -X direction after T_to_ngp_transform inversion.
        T_cam0_world = np.eye(4)
        T_cam0_world[0, 3] = frame_num * args.motion_per_frame
        
        # World to this camera:
        # T_cam_world = T_cam_cam0 @ T_cam0_world
        T_cam_world = T_cam_cam0 @ T_cam0_world
        
        # Convert to ngp format (camera-to-world, OpenGL convention)
        T_ngp = T_to_ngp_transform(T_cam_world)
        
        # Add frame entry
        frame_entry = {
            'file_path': f'images/{image_path.name}',
            'transform_matrix': T_ngp.tolist(),
            # Per-frame intrinsics
            'fl_x': cam_intrinsics['fx'],
            'fl_y': cam_intrinsics['fy'],
            'cx': cam_intrinsics['cx'],
            'cy': cam_intrinsics['cy'],
            'w': resolution[0],
            'h': resolution[1],
            'k1': cam_intrinsics['k1'],
            'k2': cam_intrinsics['k2'],
            'p1': cam_intrinsics['p1'],
            'p2': cam_intrinsics['p2'],
        }
        frames.append(frame_entry)
    
    print(f"Created {len(frames)} frame entries")
    
    if not frames:
        print("Error: No valid frames found")
        return
    
    # Post-process: Center the scene around the origin
    print("\nPost-processing camera positions...")
    
    # 1. Calculate scene bounds and center
    min_x, max_x = float('inf'), float('-inf')
    min_y, max_y = float('inf'), float('-inf')
    min_z, max_z = float('inf'), float('-inf')
    
    for frame in frames:
        m = frame['transform_matrix']
        tx, ty, tz = m[0][3], m[1][3], m[2][3]
        min_x, max_x = min(min_x, tx), max(max_x, tx)
        min_y, max_y = min(min_y, ty), max(max_y, ty)
        min_z, max_z = min(min_z, tz), max(max_z, tz)
    
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    center_z = (min_z + max_z) / 2
    
    # 2. Shift all camera positions to center scene at origin
    for frame in frames:
        frame['transform_matrix'][0][3] -= center_x
        frame['transform_matrix'][1][3] -= center_y
        frame['transform_matrix'][2][3] -= center_z
    
    # 3. Calculate appropriate aabb_scale based on scene extent
    max_extent = max(
        abs(max_x - center_x), abs(min_x - center_x),
        abs(max_y - center_y), abs(min_y - center_y),
        abs(max_z - center_z), abs(min_z - center_z)
    )
    auto_aabb_scale = 2 ** math.ceil(math.log2(max_extent * 2))
    auto_aabb_scale = max(4, auto_aabb_scale)
    
    print(f"  Centered scene at origin (was at {center_x:.2f}, {center_y:.2f}, {center_z:.2f})")
    print(f"  Camera bounds: X=[{min_x-center_x:.2f}, {max_x-center_x:.2f}]")
    print(f"  Auto aabb_scale: {auto_aabb_scale}")
    
    # Use auto-calculated aabb_scale if user didn't specify
    effective_aabb_scale = args.aabb_scale if args.aabb_scale else auto_aabb_scale
    
    # Compute camera_angle_x and camera_angle_y from intrinsics
    w, h = first_resolution
    fx = first_intrinsics['fx']
    fy = first_intrinsics['fy']
    camera_angle_x = 2 * math.atan(w / (2 * fx))
    camera_angle_y = 2 * math.atan(h / (2 * fy))
    
    # Build transforms.json
    transforms = {
        'camera_angle_x': camera_angle_x,
        'camera_angle_y': camera_angle_y,
        'fl_x': fx,
        'fl_y': fy,
        'cx': first_intrinsics['cx'],
        'cy': first_intrinsics['cy'],
        'w': w,
        'h': h,
        'k1': first_intrinsics['k1'],
        'k2': first_intrinsics['k2'],
        'p1': first_intrinsics['p1'],
        'p2': first_intrinsics['p2'],
        'aabb_scale': effective_aabb_scale,
        'frames': frames,
    }
    
    # Write transforms.json
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(transforms, f, indent=2)
    
    print(f"\nSaved transforms.json to: {output_file}")
    print(f"  Total frames: {len(frames)}")
    print(f"  aabb_scale: {effective_aabb_scale}")
    print(f"\nTo train instant-ngp:")
    print(f"  cd /isilon/Automotive/RnD/elad.e/instant-ngp")
    print(f"  ./build/testbed --scene data/nerf/car")


if __name__ == '__main__':
    main()
