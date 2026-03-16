h1. Instant-NGP Car NeRF Workflow Summary

h2. Overview

This document summarizes the complete workflow for training a NeRF model on car scan data using instant-ngp, extracting meshes, and post-processing them to remove noise and artifacts.

----

h2. 1. Setup and Environment

h3. Prerequisites

* *Conda Environment*: {{env_instant_ngp}} (Python 3.9+)
* *Dependencies*: PyYAML, Pillow, Open3D
* *VNC Server*: Display :2 on port 5902 for GUI applications

h3. Installation

{code:bash}
# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate env_instant_ngp

# Install additional packages for mesh processing
pip install open3d
{code}

h3. VNC Setup

{code:bash}
# Start VNC server for GUI applications
vncserver :2 -geometry 1920x1080 -depth 24

# Connect via VNC client to: 10.2.0.35:5902
{code}

----

h2. 2. Data Preparation Pipeline

h3. Step 1: Create Masked Frames

Applied car masks to create RGBA images with alpha channel for background removal.

*Command:*

{code:bash}
cd /isilon/Automotive/RnD/elad.e/instant-ngp

python scripts/create_masked_frames.py \
    --mask_dir /isilon/Automotive/RnD/elad.e/mpsfm_data/2024-09-27T16-31-13.000Z_d8350e36-4628-4c39-8164-05521c599810/masks \
    --frame_step 1
{code}

*Output:*

* RGBA images saved to: {{data/nerf/car/images/}}
* Processed 13 cameras (at_cam_01-09, at_front_00-01, at_rear_00-01)

h3. Step 2: Build transforms.json

Generated camera poses and intrinsics for instant-ngp.

*Command:*

{code:bash}
python scripts/build_transforms_json.py \
    --motion_per_frame 0.1
{code}

*Output:*

* Camera transforms file: {{data/nerf/car/transforms.json}}
* Camera motion: -10cm per frame along X-axis

----

h2. 3. NeRF Training

h3. Running Instant-NGP

*With VNC (Recommended):*

{code:bash}
cd /isilon/Automotive/RnD/elad.e/instant-ngp
DISPLAY=:2 ./build/instant-ngp --scene data/nerf/car
{code}

*Headless (No GUI):*

{code:bash}
./build/instant-ngp --scene data/nerf/car --no-gui
{code}

h3. Training Process

* *Scene*: Car scan with 13 cameras
* *Input*: RGBA masked images + transforms.json
* *Training*: Interactive GUI allowed real-time visualization and parameter tuning
* *Duration*: Training continued until satisfactory convergence

----

h2. 4. Mesh Extraction

h3. Initial Mesh Export

Exported mesh from instant-ngp GUI:

* *Format*: PLY and OBJ
* *Location*: {{data/nerf/car/base.ply}} and {{data/nerf/car/base.obj}}

h3. Initial Mesh Statistics

* *Vertices*: 2,136,064
* *Triangles*: 4,258,688

h3. Issues Identified

# *Noise*: Mesh surface was blocky and noisy
# *Unwanted holes*: Small artifacts and incomplete surfaces
# *Preserved openings*: Windows should remain open (not filled)

----

h2. 5. Mesh Post-Processing

h3. Created Smoothing Tools

h4. Tool 1: Batch Smoothing Script

*File*: {{scripts/smooth_mesh.py}}

*Features:*

* Taubin smoothing (preserves volume, prevents shrinkage)
* Laplacian smoothing (simpler, may cause shrinkage)
* Optional Poisson reconstruction for watertight meshes
* Outlier removal

*Usage:*

{code:bash}
# Basic smoothing (30 iterations, no hole filling)
python scripts/smooth_mesh.py \
    --input data/nerf/car/base.ply \
    --output data/nerf/car/base_smooth.ply \
    --iterations 30

# More aggressive smoothing
python scripts/smooth_mesh.py --iterations 50
python scripts/smooth_mesh.py --iterations 100
python scripts/smooth_mesh.py --iterations 200

# With hole filling (WARNING: fills windows too)
python scripts/smooth_mesh.py --iterations 50 --fill-holes
{code}

h4. Tool 2: Interactive Mesh Smoother

*File*: {{scripts/interactive_mesh_smoother.py}}

*Features:*

* Real-time 3D OpenGL viewer
* Interactive parameter controls with live preview
* Adjustable smoothing iterations (0-200)
* Lambda/Mu parameter tuning for Taubin smoothing
* Save mesh at any point

*Usage:*

{code:bash}
# Launch interactive viewer
cd /isilon/Automotive/RnD/elad.e/instant-ngp
eval "$(conda shell.bash hook)"
conda activate env_instant_ngp
DISPLAY=:2 python scripts/interactive_mesh_smoother.py --input data/nerf/car/base.ply
{code}

*Controls:*

* Left mouse: Rotate view
* Right mouse: Pan
* Scroll wheel: Zoom
* Sliders: Adjust smoothing in real-time
* Save button: Export current result

----

h2. 6. Results

h3. Smoothing Experiments

h4. Experiment 1: Initial Smoothing (10 iterations + Poisson)

*Command:*

{code:bash}
python scripts/smooth_mesh.py --iterations 10 --poisson-depth 9 --fill-holes
{code}

*Result:*

* Output: {{data/nerf/car/base_smooth.ply}}
* Vertices: 245,962 (88% reduction)
* Triangles: 492,625 (88% reduction)
* *Issue*: Filled all holes including windows (watertight mesh)

h4. Experiment 2: Smoothing Without Hole Filling (50 iterations)

*Command:*

{code:bash}
python scripts/smooth_mesh.py --iterations 50
{code}

*Result:*

* Output: {{data/nerf/car/base_smooth_v2.ply}}
* Vertices: 2,135,438 (preserved)
* Triangles: 4,257,483 (preserved)
* *Success*: Reduced blockiness significantly, preserved windows and openings

h3. Recommended Settings

*For Noise Reduction (Preserve Windows):*

{code:bash}
python scripts/smooth_mesh.py \
    --iterations 50 \
    --smooth taubin \
    --input data/nerf/car/base.ply \
    --output data/nerf/car/base_smooth_final.ply
{code}

*For Watertight Mesh (CAD/Manufacturing):*

{code:bash}
python scripts/smooth_mesh.py \
    --iterations 30 \
    --fill-holes \
    --poisson-depth 9
{code}

h3. Smoothing Parameters

|| Parameter || Default || Description ||
| {{--iterations}} | 30 | Number of smoothing passes (higher = smoother) |
| {{--smooth}} | taubin | Method: {{taubin}} (recommended) or {{laplacian}} |
| {{--fill-holes}} | False | Enable Poisson reconstruction (fills ALL holes) |
| {{--poisson-depth}} | 9 | Octree depth (8-10 typical, higher = more detail) |
| {{--remove-outliers}} | False | Remove isolated vertices |

----

h2. 7. File Structure

{code}
instant-ngp/
├── data/nerf/car/
│   ├── images/                    # Masked RGBA frames
│   ├── transforms.json            # Camera poses and intrinsics
│   ├── base.ply                   # Original extracted mesh
│   ├── base.obj                   # Original mesh (OBJ format)
│   ├── base_smooth.ply           # First smoothing attempt (with holes filled)
│   └── base_smooth_v2.ply        # Smoothed mesh (windows preserved)
│
├── scripts/
│   ├── create_masked_frames.py           # Step 1: Create RGBA images
│   ├── build_transforms_json.py          # Step 2: Generate camera poses
│   ├── smooth_mesh.py                    # Batch mesh smoothing tool
│   ├── interactive_mesh_smoother.py      # Interactive GUI smoother
│   └── README.md                         # Pipeline documentation
│
└── build/
    └── instant-ngp                        # Main executable
{code}

----

h2. 8. Key Learnings

h3. What Worked Well

# *Taubin Smoothing*: Effectively reduced noise while preserving volume
# *Incremental Iterations*: 50-100 iterations provided good balance
# *No Hole Filling*: Preserved intentional openings (windows, gaps)
# *Interactive Tool*: Real-time preview helped find optimal parameters

h3. Issues and Solutions

|| Issue || Cause || Solution ||
| Blocky mesh | NeRF discretization | Increase smoothing iterations (50-200) |
| Windows filled | Poisson reconstruction | Disable {{--fill-holes}} flag |
| Mesh shrinkage | Laplacian smoothing | Use Taubin method instead |
| Too smooth | Over-smoothing | Reduce iterations or adjust lambda/mu |

h3. Optimal Parameters for Car Meshes

* *Method*: Taubin smoothing
* *Iterations*: 50-100 (adjust based on desired smoothness)
* *Lambda*: 0.5 (smoothing strength)
* *Mu*: -0.53 (anti-shrinkage)
* *Hole Filling*: Disabled (preserve windows)

----

h2. 9. Next Steps and Future Work

h3. Potential Improvements

# *Selective Hole Filling*: Implement algorithm to fill only small holes (<N edges)
# *Edge Preservation*: Add bilateral filtering to preserve sharp edges
# *Texture Mapping*: Extract and apply textures from original images
# *Multi-Resolution*: Process different mesh regions with different parameters
# *Batch Processing*: Process multiple meshes with consistent parameters

h3. Alternative Tools

* *MeshLab*: GUI-based mesh processing (install: {{sudo apt install meshlab}})
* *PyMeshLab*: Python bindings for MeshLab filters
* *Blender*: Full 3D modeling suite with scripting capabilities

----

h2. 10. Quick Reference Commands

h3. Complete Pipeline

{code:bash}
# 1. Setup
cd /isilon/Automotive/RnD/elad.e/instant-ngp
eval "$(conda shell.bash hook)"
conda activate env_instant_ngp

# 2. Prepare data
python scripts/create_masked_frames.py --mask_dir /path/to/masks
python scripts/build_transforms_json.py

# 3. Train NeRF
vncserver :2 -geometry 1920x1080 -depth 24
DISPLAY=:2 ./build/instant-ngp --scene data/nerf/car

# 4. Extract mesh (in GUI)
# - Train until convergence
# - Export mesh as PLY/OBJ

# 5. Smooth mesh
python scripts/smooth_mesh.py --iterations 50 --output data/nerf/car/base_smooth_final.ply

# 6. Interactive refinement (optional)
DISPLAY=:2 python scripts/interactive_mesh_smoother.py
{code}

h3. VNC Management

{code:bash}
# List VNC sessions
vncserver -list

# Kill VNC session
vncserver -kill :2

# Restart VNC
vncserver :2 -geometry 1920x1080 -depth 24
{code}

h3. Process Management

{code:bash}
# Kill instant-ngp processes
pkill -f instant-ngp

# Check GPU usage
nvidia-smi
{code}

----

h2. 11. Contact and Resources

h3. Documentation

* Full pipeline details: {{scripts/README.md}}
* [Instant-NGP GitHub|https://github.com/NVlabs/instant-ngp]
* [Open3D docs|http://www.open3d.org/docs/]

h3. Workspace Location

* Base directory: {{/isilon/Automotive/RnD/elad.e/instant-ngp}}
* Training data: {{/isilon/Automotive/RnD/elad.e/mpsfm_data/}}
* Results: {{/isilon/Automotive/RnD/elad.e/instant-ngp/data/nerf/car/}}

----

_Last Updated: February 25, 2026_
