#!/usr/bin/env python3
"""
Create masked RGBA frames from car scan images and masks.

This script loads images and their corresponding masks, then creates RGBA images
where the mask is used as the alpha channel (255 = car pixels, 0 = background).
"""

import argparse
from pathlib import Path
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description='Create masked RGBA frames from car scan images')
    parser.add_argument('--data_dir', type=str, 
                        default='/isilon/Automotive/RnD/elad.e/mpsfm_data/2024-09-27T16-31-13.000Z_d8350e36-4628-4c39-8164-05521c599810',
                        help='Path to car scan data')
    parser.add_argument('--mask_dir', type=str, required=True,
                        help='Directory containing masks (value 255 = car)')
    parser.add_argument('--output_dir', type=str,
                        default='/isilon/Automotive/RnD/elad.e/instant-ngp/data/nerf/car/images',
                        help='Output directory for masked RGBA images')
    parser.add_argument('--frame_step', type=int, default=1,
                        help='Use every Nth frame (default: 1)')
    parser.add_argument('--cameras', nargs='+', default=None,
                        help='Specific cameras to use (default: all 13 cameras)')
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    mask_dir = Path(args.mask_dir)
    output_dir = Path(args.output_dir)
    
    print(f"Data directory: {data_dir}")
    print(f"Mask directory: {mask_dir}")
    print(f"Output directory: {output_dir}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine which cameras to use
    if args.cameras:
        cameras_to_use = args.cameras
    else:
        # Default: all cameras (side + front + rear)
        side_cameras = [f'at_cam_{i:02d}' for i in range(1, 10)]  # at_cam_01 to at_cam_09
        front_rear_cameras = ['at_front_00', 'at_front_01', 'at_rear_00', 'at_rear_01']
        cameras_to_use = side_cameras + front_rear_cameras
        # Filter to only cameras that exist
        cameras_to_use = [c for c in cameras_to_use if (data_dir / c).exists()]
    
    print(f"Using cameras: {cameras_to_use}")
    
    # Find available frames from first camera
    sample_cam = cameras_to_use[0]
    cam_dir = data_dir / sample_cam
    all_frames = sorted([f.stem for f in cam_dir.glob('frame_*.png')])
    print(f"Found {len(all_frames)} total frames")
    
    # Process frames
    created_count = 0
    skipped_count = 0
    
    for frame_name in all_frames[::args.frame_step]:
        frame_num = int(frame_name.split('_')[1])
        
        for cam_name in cameras_to_use:
            # Source image path
            src_image = data_dir / cam_name / f'{frame_name}.png'
            if not src_image.exists():
                skipped_count += 1
                continue
            
            # Mask path
            mask_path = mask_dir / cam_name / f'{frame_name}_mask.png'
            if not mask_path.exists():
                skipped_count += 1
                continue
            
            # Output image name: <camera_name>_<frame_index>.png
            dst_image_name = f'{cam_name}_{frame_num:04d}.png'
            dst_image = output_dir / dst_image_name
            
            # Load image and mask, create RGBA with mask as alpha
            img = Image.open(src_image).convert('RGB')
            mask = Image.open(mask_path).convert('L')
            
            # Create RGBA image with mask as alpha channel
            # In mask: 255 = car (keep), 0 = background (transparent)
            rgba = Image.new('RGBA', img.size)
            rgba.paste(img, (0, 0))
            rgba.putalpha(mask)
            rgba.save(dst_image)
            
            created_count += 1
    
    print(f"\nCreated {created_count} masked RGBA images")
    print(f"Skipped {skipped_count} frames (missing image or mask)")
    print(f"Output saved to: {output_dir}")


if __name__ == '__main__':
    main()
