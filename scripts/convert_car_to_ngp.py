#!/usr/bin/env python3
"""
Convert car scan data to instant-ngp NeRF format.

The car scan has 13 cameras that move together along the X axis.
Camera 1 at frame 0 is at world origin (0, 0, 0).
All cameras move 10cm between frames along +X.

The extrinsics.yaml contains T_cn_cnm1 which is the transform from camera n-1 to camera n.
We chain these to get each camera's pose relative to camera 0.
"""

import json
import yaml
import numpy as np
import shutil
from pathlib import Path
import math


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
    import argparse
    parser = argparse.ArgumentParser(description='Convert car scan to instant-ngp format')
    parser.add_argument('--data_dir', type=str, 
                        default='/isilon/Automotive/RnD/elad.e/mpsfm_data/2024-09-27T16-31-13.000Z_d8350e36-4628-4c39-8164-05521c599810',
                        help='Path to car scan data')
    parser.add_argument('--output_dir', type=str,
                        default='/isilon/Automotive/RnD/elad.e/instant-ngp/data/nerf/car',
                        help='Output directory for instant-ngp data')
    parser.add_argument('--motion_per_frame', type=float, default=0.1,
                        help='Motion per frame in meters (default: 0.1m = 10cm)')
    parser.add_argument('--frame_step', type=int, default=1,
                        help='Use every Nth frame (default: 1)')
    parser.add_argument('--cameras', nargs='+', default=None,
                        help='Specific cameras to use (default: all)')
    parser.add_argument('--aabb_scale', type=int, default=4,
                        help='AABB scale for instant-ngp (default: 4)')
    parser.add_argument('--symlinks', action='store_true',
                        help='Use symlinks instead of copying images (ignored when using masks)')
    parser.add_argument('--mask_dir', type=str, default=None,
                        help='Directory containing masks (value 255 = car). Will create RGBA images.')
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    mask_dir = Path(args.mask_dir) if args.mask_dir else None
    
    if mask_dir:
        print(f"Using masks from: {mask_dir}")
        # Import PIL for image processing
        from PIL import Image
    
    # Load calibration
    intrinsics_yaml = load_yaml(data_dir / 'intrinsics.yaml')
    extrinsics_yaml = load_yaml(data_dir / 'extrinsics.yaml')
    
    intrinsics = parse_intrinsics(intrinsics_yaml)
    extrinsics = parse_extrinsics(extrinsics_yaml)
    
    print(f"Loaded {len(intrinsics)} camera intrinsics")
    print(f"Loaded {len(extrinsics)} camera extrinsics")
    
    # Determine which cameras to use
    if args.cameras:
        cameras_to_use = args.cameras
    else:
        cameras_to_use = list(intrinsics.keys())
    
    print(f"Using cameras: {cameras_to_use}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / 'images'
    images_dir.mkdir(exist_ok=True)
    
    # Find available frames
    sample_cam = cameras_to_use[0]
    cam_dir = data_dir / sample_cam
    all_frames = sorted([f.stem for f in cam_dir.glob('frame_*.png')])
    print(f"Found {len(all_frames)} frames")
    
    # Build frames list for transforms.json
    frames = []
    image_idx = 0
    
    # Use representative intrinsics (first camera's)
    first_cam = cameras_to_use[0]
    first_intrinsics = intrinsics[first_cam]
    first_resolution = extrinsics[first_cam]['resolution']
    
    for frame_name in all_frames[::args.frame_step]:
        frame_num = int(frame_name.split('_')[1])
        
        # Motion offset for this frame (cameras move along +X)
        motion_offset = np.array([frame_num * args.motion_per_frame, 0, 0, 1])
        
        for cam_name in cameras_to_use:
            if cam_name not in intrinsics or cam_name not in extrinsics:
                print(f"Warning: Camera {cam_name} not found in calibration")
                continue
            
            cam_intrinsics = intrinsics[cam_name]
            cam_extrinsics = extrinsics[cam_name]
            resolution = cam_extrinsics['resolution']
            
            # Source image path
            src_image = data_dir / cam_name / f'{frame_name}.png'
            if not src_image.exists():
                continue
            
            # Check for mask if mask_dir is specified
            if mask_dir:
                mask_path = mask_dir / cam_name / f'{frame_name}_mask.png'
                if not mask_path.exists():
                    # Skip frames without masks
                    continue
            
            # T_cn_c0 is transform from cam0 frame to this camera frame
            T_cam_cam0 = cam_extrinsics['T_cn_c0']
            
            # All cameras are fixed in the world, the car moves through them.
            # From NeRF's perspective (car is stationary), all cameras move together.
            # At frame N, all cameras have moved -N*motion_per_frame in X.
            T_cam0_world = np.eye(4)
            T_cam0_world[0, 3] = -frame_num * args.motion_per_frame
            
            # World to this camera:
            # T_cam_world = T_cam_cam0 @ T_cam0_world
            T_cam_world = T_cam_cam0 @ T_cam0_world
            
            # Convert to ngp format (camera-to-world, OpenGL convention)
            T_ngp = T_to_ngp_transform(T_cam_world)
            
            # Process image (with or without mask)
            # Format: <camera_name>_<frame_index>.png (e.g., at_cam_01_0033.png)
            dst_image_name = f'{cam_name}_{frame_num:04d}.png'
            dst_image = images_dir / dst_image_name
            
            if mask_dir:
                # Load image and mask, create RGBA with mask as alpha
                img = Image.open(src_image).convert('RGB')
                mask = Image.open(mask_path).convert('L')
                
                # Create RGBA image with mask as alpha channel
                # In mask: 255 = car (keep), 0 = background (transparent)
                rgba = Image.new('RGBA', img.size)
                rgba.paste(img, (0, 0))
                rgba.putalpha(mask)
                rgba.save(dst_image)
            elif args.symlinks:
                if dst_image.exists():
                    dst_image.unlink()
                dst_image.symlink_to(src_image.resolve())
            else:
                shutil.copy2(src_image, dst_image)
            
            # Add frame entry
            frame_entry = {
                'file_path': f'images/{dst_image_name}',
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
            image_idx += 1
    
    print(f"Created {len(frames)} frame entries")
    
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
        'aabb_scale': args.aabb_scale,
        'frames': frames,
    }
    
    # Write transforms.json
    transforms_path = output_dir / 'transforms.json'
    with open(transforms_path, 'w') as f:
        json.dump(transforms, f, indent=2)
    
    print(f"\nSaved transforms.json to: {transforms_path}")
    print(f"Saved {image_idx} images to: {images_dir}")
    print(f"\nTo train instant-ngp:")
    print(f"  cd /isilon/Automotive/RnD/elad.e/instant-ngp")
    print(f"  ./build/testbed --scene data/nerf/car")


if __name__ == '__main__':
    main()
