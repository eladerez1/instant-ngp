#!/usr/bin/env python3
"""
Generate transforms.json for NeuS2 from car scan data.

This script uses the SAME pose computation logic as visual_hull.py:
1. Loads camera_rig_transforms (static camera positions in the rig)
2. Loads fused_distances (cumulative distance traveled at each frame)
3. Computes world_from_cam by applying trajectory motion to rig transforms
4. Creates masked images (car only, background removed)
5. Generates transforms.json in NeuS2 format
"""

import json
import re
from pathlib import Path

import cv2
import numpy as np
import yaml


# Configuration
TRAJECTORY_PATH = "/isilon/Automotive/RnD/elad.e/uv-3d/sessions/demo_room_test/run_2026-02-17T09-49-11-416846/02_trajectory/trajectory.npz"
CALIBRATION_PATH = "/isilon/Automotive/RnD/jonathan.s/Demoroom/2024-09-27T16-31-13.000Z_d8350e36-4628-4c39-8164-05521c599810/calibration.yaml"
IMAGES_ROOT = "/isilon/Automotive/Data/CustomersData/Demoroom/20251221/2024-09-27T16-31-13.000Z_d8350e36-4628-4c39-8164-05521c599810"
MASKS_ROOT = "/isilon/Automotive/RnD/elad.e/uv-3d/sessions/demo_room_test/run_2026-02-17T09-49-11-416846/00_segmentation"
OUTPUT_DIR = Path("/isilon/Automotive/RnD/elad.e/NeuS2/car")

# Use masks to remove background
USE_MASKS = True

# Camera list
CAMERAS = [
    "at_cam_01", "at_cam_02", "at_cam_03", "at_cam_04", "at_cam_05",
    "at_cam_06", "at_cam_07", "at_cam_08", "at_cam_09",
    "at_front_00", "at_front_01", "at_rear_00", "at_rear_01"
]

# Frame range (inclusive)
FRAME_START = 42
FRAME_END = 78

# Frame sampling (use every N frames, or specific frames)
FRAME_STEP = 1  # Use every frame for single camera
MAX_FRAMES = None  # Set to limit total frames, or None for all


def load_calibration(calib_path: str) -> dict:
    """Load camera calibration from YAML file."""
    with open(calib_path, 'r') as f:
        calib = yaml.safe_load(f)
    
    cameras = {}
    
    # Build chain of transforms from cam0
    cam_pattern = re.compile(r'^cam(\d+)$')
    cam_indices = []
    for key in calib.keys():
        match = cam_pattern.match(key)
        if match:
            cam_indices.append(int(match.group(1)))
    
    cam_indices.sort()
    
    # Compute world_from_cam for each camera (cam0 is at identity)
    world_from_cam = {0: np.eye(4, dtype=np.float64)}
    
    for idx in cam_indices:
        cam_data = calib[f'cam{idx}']
        cam_name = cam_data['cam_name']
        
        # Get intrinsics [fx, fy, cx, cy]
        intrinsics = cam_data['intrinsics']
        resolution = cam_data['resolution']
        
        # Build intrinsic matrix (4x4 for NeuS2)
        K = np.array([
            [intrinsics[0], 0, intrinsics[2], 0],
            [0, intrinsics[1], intrinsics[3], 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float64)
        
        # Compute world_from_cam by chaining transforms
        if idx == 0:
            T_world_cam = np.eye(4, dtype=np.float64)
        else:
            if 'T_cn_cnm1' in cam_data:
                T_cn_cnm1 = np.array(cam_data['T_cn_cnm1'], dtype=np.float64)
                parent_idx = idx - 1
                if parent_idx in world_from_cam:
                    T_world_cam = world_from_cam[parent_idx] @ np.linalg.inv(T_cn_cnm1)
                else:
                    print(f"Warning: Parent camera {parent_idx} not found for {cam_name}")
                    T_world_cam = np.eye(4)
            else:
                T_world_cam = np.eye(4)
        
        world_from_cam[idx] = T_world_cam
        
        cameras[cam_name] = {
            'intrinsics': K,
            'resolution': resolution,  # [width, height]
            'world_from_cam': T_world_cam,
            'distortion': cam_data.get('distortion_coeffs', [0, 0, 0, 0])
        }
    
    return cameras


def load_trajectory(traj_path: str) -> dict:
    """
    Load trajectory data from NPZ file.
    
    Returns camera_rig_transforms and fused_distances needed to compute poses.
    This matches how TrajectoryLoader works in visual_hull.py.
    """
    data = np.load(traj_path, allow_pickle=True)
    
    # Get camera rig transforms (static transforms in rig coordinate system)
    # These define the position/orientation of each camera relative to the rig origin
    camera_names = [str(n) for n in data['camera_names']]
    rig_transforms = data['camera_rig_transforms']
    
    camera_rig_transforms = {}
    for i, cam_name in enumerate(camera_names):
        camera_rig_transforms[cam_name] = rig_transforms[i].astype(np.float32)
    
    # Get fused distances (cumulative distance traveled at each frame)
    # This represents how far the car has moved from the starting position
    fused_distances = {}
    for frame_idx, dist in zip(
        data['fused_distances_frames'], 
        data['fused_distances_values']
    ):
        fused_distances[int(frame_idx)] = float(dist)
    
    # Get available frames
    unique_frames = sorted(fused_distances.keys())
    
    print(f"  Camera rig transforms: {len(camera_rig_transforms)} cameras")
    print(f"  Fused distances: {len(fused_distances)} frames")
    print(f"  Distance range: {min(fused_distances.values()):.3f}m to {max(fused_distances.values()):.3f}m")
    
    return {
        'camera_rig_transforms': camera_rig_transforms,
        'fused_distances': fused_distances,
        'unique_frames': unique_frames,
        'camera_names': camera_names
    }


def apply_trajectory_motion(
    rig_transform: np.ndarray, 
    tx: float
) -> np.ndarray:
    """
    Apply trajectory motion to rig transform by updating X translation.
    
    This is the SAME logic as in pose_interpolation.py:
    - Keeps rotation and Y/Z translation from rig calibration unchanged
    - Adds trajectory motion to existing camera X offset (preserves stereo baseline)
    
    NOTE: Negates tx to match visual hull coordinate convention:
    - Internal distances are POSITIVE (car moves forward in +X)
    - VH expects NEGATIVE X (camera rig moves backward as car approaches)
    
    Args:
        rig_transform: 4x4 camera transform from rig calibration
        tx: X translation from trajectory estimation (cumulative distance, POSITIVE)
    
    Returns:
        4x4 world_from_cam transformation matrix with updated X translation
    """
    pose = rig_transform.copy()
    # NEGATE and ADD trajectory motion to camera's existing X offset
    # Negation converts from internal (+X forward) to VH convention (-X backward)
    pose[0, 3] += -tx  # Preserve stereo baseline in X direction
    return pose


def interpolate_camera_pose(
    camera_rig_transforms: dict,
    fused_distances: dict,
    camera_name: str,
    frame_num: int
) -> np.ndarray:
    """
    Get world_from_cam pose for a camera at a specific frame.
    
    This is the SAME logic as interpolate_camera_pose in pose_interpolation.py:
    - If exact frame exists, use its distance
    - If frame is between known frames, linearly interpolate the distance
    - If frame is outside range, clamp to nearest known frame
    
    Args:
        camera_rig_transforms: Dict mapping camera names to base rig transforms
        fused_distances: Dict mapping frame numbers to cumulative distances
        camera_name: Camera identifier (e.g., "at_cam_01")
        frame_num: Frame number (0-indexed)
    
    Returns:
        4x4 world_from_cam transformation matrix
    """
    # Check if camera exists
    if camera_name not in camera_rig_transforms:
        raise ValueError(
            f"Camera {camera_name} not found in calibration. "
            f"Available cameras: {list(camera_rig_transforms.keys())}"
        )
    
    # Get camera rig transform from calibration
    rig_transform = camera_rig_transforms[camera_name]
    
    # Get all available frames (sorted)
    available_frames = sorted(fused_distances.keys())
    if not available_frames:
        raise ValueError("No fused distances available")
    
    # Determine the X translation (distance traveled)
    if frame_num <= available_frames[0]:
        # Before first frame: use first distance
        tx = fused_distances[available_frames[0]]
    elif frame_num >= available_frames[-1]:
        # After last frame: use last distance
        tx = fused_distances[available_frames[-1]]
    elif frame_num in fused_distances:
        # Exact frame exists
        tx = fused_distances[frame_num]
    else:
        # Interpolate between frames
        idx_after = next(i for i, f in enumerate(available_frames) if f > frame_num)
        idx_before = idx_after - 1
        
        frame_before = available_frames[idx_before]
        frame_after = available_frames[idx_after]
        dist_before = fused_distances[frame_before]
        dist_after = fused_distances[frame_after]
        
        # Linear interpolation
        alpha = (frame_num - frame_before) / (frame_after - frame_before)
        tx = (1 - alpha) * dist_before + alpha * dist_after
    
    # Apply trajectory motion to rig transform (update X translation only)
    pose = apply_trajectory_motion(rig_transform, tx)
    
    return pose.astype(np.float64)


def get_camera_pose(trajectory: dict, calibration: dict, cam_name: str, frame_num: int) -> np.ndarray:
    """
    Get the world_from_camera transform for a specific camera at a specific frame.
    
    Uses the SAME computation as visual_hull.py / TrajectoryLoader:
    1. Get the static camera rig transform (camera position in rig coordinates)
    2. Get the cumulative distance traveled at this frame
    3. Apply trajectory motion: pose[0,3] += -distance (negate for VH convention)
    
    Args:
        trajectory: Trajectory data with camera_rig_transforms and fused_distances
        calibration: Calibration data (used as fallback)
        cam_name: Camera name
        frame_num: Frame number
    
    Returns:
        4x4 world_from_cam transformation matrix
    """
    return interpolate_camera_pose(
        camera_rig_transforms=trajectory['camera_rig_transforms'],
        fused_distances=trajectory['fused_distances'],
        camera_name=cam_name,
        frame_num=frame_num
    )


def create_transforms_json(
    calibration: dict,
    trajectory: dict,
    output_dir: Path,
    cameras: list[str],
    frame_step: int = 1,
    max_frames: int | None = None
):
    """Generate transforms.json and symlink images."""
    
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    
    frames_data = []
    
    # Get all available frames from trajectory
    all_frames = trajectory['unique_frames']
    
    # Sample frames if needed
    selected_frames = all_frames[::frame_step]
    if max_frames:
        selected_frames = selected_frames[:max_frames]
    
    print(f"Processing {len(selected_frames)} frames x {len(cameras)} cameras")
    print(f"Will include images that have masks available")
    if USE_MASKS:
        print(f"Using masks from: {MASKS_ROOT}")
    
    # Determine image resolution (use first camera's resolution)
    first_cam = cameras[0]
    resolution = calibration[first_cam]['resolution']
    width, height = resolution[0], resolution[1]
    
    # Collect all camera positions to compute scene bounds
    all_positions = []
    
    image_idx = 0
    
    # Process each camera
    for cam_name in cameras:
        for frame_num in selected_frames:
            # Check if image exists
            src_image = Path(IMAGES_ROOT) / cam_name / f"frame_{frame_num:04d}.png"
            if not src_image.exists():
                continue  # Skip silently
            
            # Check if mask exists (required)
            if USE_MASKS:
                mask_path = Path(MASKS_ROOT) / cam_name / f"frame_{frame_num:04d}.png"
                if not mask_path.exists():
                    continue  # Skip if no mask
            
            # Get camera pose using the same logic as visual_hull.py
            world_from_cam = get_camera_pose(trajectory, calibration, cam_name, frame_num)
            
            # Get intrinsics
            intrinsics = calibration[cam_name]['intrinsics']
            
            # Output image path with cam name and frame index
            image_filename = f"{cam_name}_{frame_num:04d}.png"
            dst_image = images_dir / image_filename
            if dst_image.exists():
                dst_image.unlink()
            
            if USE_MASKS:
                # Load original image and mask, apply mask
                mask_path = Path(MASKS_ROOT) / cam_name / f"frame_{frame_num:04d}.png"
                if not mask_path.exists():
                    print(f"  Warning: Mask not found: {mask_path}")
                    continue
                
                # Load image and mask
                img = cv2.imread(str(src_image), cv2.IMREAD_UNCHANGED)
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                
                # Skip if mask has no content (no car pixels)
                if mask is None or np.count_nonzero(mask) == 0:
                    continue
                
                # Convert mask to binary (non-zero = car)
                _, mask_binary = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
                
                # Create RGBA image with alpha channel from mask
                if img.shape[2] == 3:
                    # Add alpha channel
                    img_rgba = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
                else:
                    img_rgba = img
                
                # Set alpha from mask
                img_rgba[:, :, 3] = mask_binary
                
                # Save masked image
                cv2.imwrite(str(dst_image), img_rgba)
            else:
                # Just symlink the original image
                dst_image.symlink_to(src_image)
            
            # Store camera position for bounds calculation
            all_positions.append(world_from_cam[:3, 3])
            
            # Add frame entry
            frame_entry = {
                "file_path": f"images/{image_filename}",
                "transform_matrix": world_from_cam.tolist(),
                "intrinsic_matrix": intrinsics.tolist()
            }
            frames_data.append(frame_entry)
            
            image_idx += 1
    
    print(f"Created {image_idx} masked images")
    
    # Compute scene bounds
    all_positions = np.array(all_positions)
    scene_min = all_positions.min(axis=0)
    scene_max = all_positions.max(axis=0)
    scene_center = (scene_min + scene_max) / 2
    scene_extent = (scene_max - scene_min).max()
    
    print(f"Scene bounds: min={scene_min}, max={scene_max}")
    print(f"Scene center: {scene_center}, extent: {scene_extent}")
    
    # Compute scale and offset to normalize scene to [0, 1]^3
    # NeuS2 expects scene in unit cube with offset
    scale = 0.5 / (scene_extent / 2 + 0.5)  # Add margin
    offset = [0.5 - scene_center[0] * scale,
              0.5 - scene_center[1] * scale, 
              0.5 - scene_center[2] * scale]
    
    # Build transforms.json
    transforms = {
        "w": width,
        "h": height,
        "aabb_scale": 1.0,
        "scale": scale,
        "offset": offset,
        "from_na": True,
        "frames": frames_data
    }
    
    # Write transforms.json
    output_path = output_dir / "transforms.json"
    with open(output_path, 'w') as f:
        json.dump(transforms, f, indent=4)
    
    print(f"\nWrote transforms.json to {output_path}")
    print(f"  Total frames: {len(frames_data)}")
    print(f"  Image size: {width}x{height}")
    print(f"  Scale: {scale}")
    print(f"  Offset: {offset}")
    
    return transforms


def main():
    print("=" * 60)
    print("Generating NeuS2 transforms.json for car scan")
    print("=" * 60)
    print("\nUsing SAME pose computation as visual_hull.py:")
    print("  world_from_cam = rig_transform with X += -fused_distance")
    
    # Load calibration
    print("\nLoading calibration...")
    calibration = load_calibration(CALIBRATION_PATH)
    print(f"  Loaded {len(calibration)} cameras")
    
    # Load trajectory
    print("\nLoading trajectory...")
    trajectory = load_trajectory(TRAJECTORY_PATH)
    print(f"  Available frames: {trajectory['unique_frames'][:5]}...{trajectory['unique_frames'][-3:]}")
    
    # Generate transforms
    print("\nGenerating transforms.json...")
    create_transforms_json(
        calibration=calibration,
        trajectory=trajectory,
        output_dir=OUTPUT_DIR,
        cameras=CAMERAS,
        frame_step=FRAME_STEP,
        max_frames=MAX_FRAMES
    )
    
    print("\nDone!")


if __name__ == '__main__':
    main()
