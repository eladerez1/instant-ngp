# Car Scan to instant-ngp Pipeline

This directory contains scripts to convert car scan data to instant-ngp format and run NeRF training.

## Overview

The pipeline consists of two main steps:
1. **Create masked frames** - Apply car masks to images (RGBA with alpha channel)
2. **Build transforms.json** - Generate camera poses for instant-ngp

## Prerequisites

- Conda environment with Python 3.9+, PyYAML, and Pillow
- Car scan data with calibration files (intrinsics.yaml, extrinsics.yaml)
- Car masks (grayscale PNG, 255 = car pixels, 0 = background)

```bash
# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate env_instant_ngp
```

## Step 1: Create Masked Frames

The `create_masked_frames.py` script loads source images and masks, then creates RGBA images where the mask becomes the alpha channel.

### Basic Usage

```bash
python scripts/create_masked_frames.py \
    --mask_dir /path/to/masks
```

### Full Options

```bash
python scripts/create_masked_frames.py \
    --data_dir /path/to/car_scan_data \
    --mask_dir /path/to/masks \
    --output_dir /path/to/output/images \
    --frame_step 1 \
    --cameras at_cam_01 at_cam_02 at_cam_03
```

| Option | Default | Description |
|--------|---------|-------------|
| `--data_dir` | Car scan path | Directory containing camera folders with frame_*.png images |
| `--mask_dir` | Required | Directory containing masks (camera_name/frame_XXXX_mask.png) |
| `--output_dir` | data/nerf/car/images | Output directory for RGBA images |
| `--frame_step` | 1 | Use every Nth frame |
| `--cameras` | All 13 cameras | Specific cameras to process |

### Example

```bash
python scripts/create_masked_frames.py \
    --mask_dir /isilon/Automotive/RnD/elad.e/mpsfm_data/2024-09-27T16-31-13.000Z_d8350e36-4628-4c39-8164-05521c599810/masks \
    --frame_step 1
```

## Step 2: Build transforms.json

The `build_transforms_json.py` script generates the camera poses file required by instant-ngp.

### Basic Usage

```bash
python scripts/build_transforms_json.py
```

### Full Options

```bash
python scripts/build_transforms_json.py \
    --data_dir /path/to/car_scan_data \
    --images_dir /path/to/images \
    --output_file /path/to/transforms.json \
    --motion_per_frame 0.1 \
    --aabb_scale 8
```

| Option | Default | Description |
|--------|---------|-------------|
| `--data_dir` | Car scan path | Directory with intrinsics.yaml and extrinsics.yaml |
| `--images_dir` | data/nerf/car/images | Directory with masked images (to detect available frames) |
| `--output_file` | data/nerf/car/transforms.json | Output transforms.json path |
| `--motion_per_frame` | 0.1 | Camera motion per frame in meters (10cm) |
| `--aabb_scale` | Auto | Scene bounding box scale (auto-calculated if not specified) |

### Example

```bash
python scripts/build_transforms_json.py \
    --motion_per_frame 0.1
```

## Step 3: Run instant-ngp

### Option A: With VNC Server (Remote Access)

Start a VNC server and run instant-ngp on that display:

```bash
# Start VNC server (first time or if not running)
vncserver :2 -geometry 1920x1080 -depth 24

# Run instant-ngp on VNC display
cd /isilon/Automotive/RnD/elad.e/instant-ngp
DISPLAY=:2 ./build/instant-ngp --scene data/nerf/car
```

Connect to the VNC server from your local machine:
- **Host**: 10.2.0.35 (or your server IP)
- **Port**: 5902
- **Display**: :2

Using a VNC client (e.g., TigerVNC Viewer, RealVNC):
```bash
vncviewer 10.2.0.35:5902
```

To kill the VNC server when done:
```bash
vncserver -kill :2
```

### Option B: Without VNC (Local Display)

If you have a local display (e.g., connected monitor or X11 forwarding):

```bash
cd /isilon/Automotive/RnD/elad.e/instant-ngp
./build/instant-ngp --scene data/nerf/car
```

For X11 forwarding over SSH:
```bash
ssh -X user@server
cd /isilon/Automotive/RnD/elad.e/instant-ngp
./build/instant-ngp --scene data/nerf/car
```

### Command Line Options for instant-ngp

```bash
./build/instant-ngp --scene data/nerf/car [options]
```

| Option | Description |
|--------|-------------|
| `--scene <path>` | Path to the scene directory containing transforms.json |
| `--width <int>` | Window width |
| `--height <int>` | Window height |
| `--no-gui` | Run without GUI (for headless training) |

### Headless Training (No Display Required)

For training without any display:

```bash
cd /isilon/Automotive/RnD/elad.e/instant-ngp
./build/instant-ngp --scene data/nerf/car --no-gui
```

## Complete Pipeline Example

```bash
# 1. Activate environment
eval "$(conda shell.bash hook)"
conda activate env_instant_ngp

# 2. Create masked frames
cd /isilon/Automotive/RnD/elad.e/instant-ngp
python scripts/create_masked_frames.py \
    --mask_dir /isilon/Automotive/RnD/elad.e/mpsfm_data/2024-09-27T16-31-13.000Z_d8350e36-4628-4c39-8164-05521c599810/masks

# 3. Build transforms.json
python scripts/build_transforms_json.py

# 4. Run instant-ngp (with VNC)
vncserver :2 -geometry 1920x1080 -depth 24
DISPLAY=:2 ./build/instant-ngp --scene data/nerf/car

# Connect via VNC client to 10.2.0.35:5902
```

## Camera Setup

The car scan uses 13 cameras:
- **Side cameras**: at_cam_01 through at_cam_09 (9 cameras)
- **Front cameras**: at_front_00, at_front_01 (2 cameras)
- **Rear cameras**: at_rear_00, at_rear_01 (2 cameras)

All cameras move together at -10cm per frame along the X axis.

## Troubleshooting

### VNC Connection Issues
```bash
# Check if VNC server is running
vncserver -list

# Kill and restart
vncserver -kill :2
vncserver :2 -geometry 1920x1080 -depth 24
```

### Missing Frames
The scripts automatically skip frames without masks. Check mask availability:
```bash
ls /path/to/masks/at_cam_01/ | wc -l
```

### CUDA/GPU Issues
Ensure CUDA is available:
```bash
nvidia-smi
```

### Display Errors
If you get display errors, ensure DISPLAY is set correctly:
```bash
export DISPLAY=:2  # For VNC display 2
# or
export DISPLAY=:0  # For local display
```
