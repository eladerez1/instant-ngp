# NeRF Evaluation Pipeline

Pipeline for evaluating NeRF reconstruction quality against CAD reference using full cloud and outer shell benchmarks.

## Overview

This pipeline processes a raw NeRF mesh export and evaluates its accuracy against a CAD reference model. It automatically generates both full cloud and outer shell benchmarks in a single pass:

1. **Full Cloud Evaluation** - Uses all reconstructed points (1.2M points)
2. **Outer Shell Evaluation** - Uses only visible surface points (48K points, 4%) - **automatically extracted**

## Prerequisites

```bash
conda activate env_instant_ngp
```

**Required Dependencies:**
- Python 3.9+
- open3d 0.19.0
- scipy 1.13.1
- numpy 2.0.2

## Input Files

- `base.obj` - Raw NeRF mesh exported from instant-ngp
- `mesh/cad_sample.ply` - CAD reference (100K points with normals)

## Pipeline: Full Cloud + Outer Shell Evaluation

This pipeline runs both evaluations automatically in a single command.

### Step 1: Convert OBJ to PLY

```bash
cd /isilon/Automotive/RnD/elad.e/instant-ngp/data/nerf/car
python /isilon/Automotive/RnD/elad.e/obj_to_ply.py base.obj base.ply
```

**Output:** `base.ply` (1,198,080 vertices)

### Step 2: Rotate, Align, and Evaluate (Full Cloud + Outer Shell)

```bash
python rotate_and_evaluate.py \
    --input base.ply \
    --output mesh/base_final.ply \
    --reference mesh/cad_sample.ply \
    --icp \
    --lock-rotation \
    --evaluate
```

**What it does:**

**Full Cloud Processing:**
1. Applies initial rotation: -90° Z-axis + 60° Y-axis
2. ICP alignment (translation-only, preserves rotation)
   - Multi-scale: 0.4m → 0.2m → 0.1m
   - Point-to-point method
3. Computes evaluation metrics:
   - Accuracy: NeRF → CAD distances
   - Completeness: CAD → NeRF coverage
4. Generates heat map (white=close, red=far, 10cm scale)

**Outer Shell Processing (Automatic):**
5. Extracts outer shell from aligned cloud:
   - Angular hash map (360×180 bins, 1° resolution)
   - Keeps only farthest point per direction
   - Removes occluded/interior points
6. Re-aligns outer shell with ICP (full 6-DOF)
7. Computes outer shell metrics
8. Generates outer shell heat map

**Outputs:**
- `mesh/base_final.ply` - Aligned point cloud (1.2M points)
- `mesh/base_final_heatmap.ply` - Distance-colored visualization
- `mesh/base_final_metrics.json` - Full evaluation results
- `mesh/base_final_outer_shell_aligned.ply` - Aligned outer shell (48K points)
- `mesh/base_final_outer_shell_heatmap.ply` - Outer shell heat map
- `mesh/base_final_outer_metrics.json` - Outer shell evaluation results

**Full Cloud Results:**
```
Accuracy (NeRF → CAD):
  Mean:   8.88 cm
  Median: 6.57 cm
  RMSE:   11.93 cm
  90th:   20.51 cm

Completeness (CAD coverage):
  @1cm:  13.5%
  @2cm:  26.0%
  @5cm:  46.2%
  @10cm: 66.5%

ICP: fitness=0.714, RMSE=4.86cm
```

**Outer Shell Results:**
```
Accuracy (Outer Shell → CAD):
  Mean:   7.31 cm
  Median: 3.87 cm  ← 41% better than full cloud!
  RMSE:   14.98 cm
  90th:   16.98 cm

Completeness (CAD coverage):
  @1cm:  3.4%
  @2cm:  12.3%
  @5cm:  37.0%

ICP: fitness=0.781, RMSE=4.28cm
Points: 48,085 (4.0% of original)
```

## Benchmark Comparison

### Compare All Methods

```bash
python compare_benchmarks.py
```

This script compares VGGT, NeRF full cloud, and NeRF outer shell benchmarks.

**Output:**
```
================================================================================
BENCHMARK COMPARISON: VGGT vs NeRF vs Outer Shell
================================================================================

--- ACCURACY (Reconstruction → CAD) ---
Method          Mean (cm)    Median (cm)  RMSE (cm)    90th (cm)    95th (cm)
--------------------------------------------------------------------------------
VGGT            7.20         5.21         10.14        14.81        18.91
NeRF (full)     8.88         6.57         11.93        20.51        25.31
Outer Shell     7.31         3.87         14.98        16.98        23.26

--- COMPLETENESS (CAD Coverage) ---
Method          Mean (cm)    Median (cm)  @1cm (%)     @2cm (%)     @5cm (%)
--------------------------------------------------------------------------------
VGGT            6.74         2.74         28.2         42.9         62.1
NeRF (full)     7.75         5.86         13.5         26.0         46.2
Outer Shell     N/A          N/A          3.4          12.3         37.0

--- RELATIVE PERFORMANCE ---
Accuracy (Median):
  • NeRF vs VGGT:        +26.2% (worse)
  • Outer Shell vs VGGT: -25.7% (better)
  • Outer Shell vs NeRF: -41.1% (better)

Completeness (@1cm):
  • NeRF vs VGGT:        -52.0% (worse)
  • Outer Shell vs VGGT: -88.0% (worse)
  • Outer Shell vs NeRF: -75.0% (worse)
```

## Key Findings

### Accuracy
- **Outer Shell achieves best surface accuracy** (3.87cm median)
- 26% better than VGGT (5.21cm)
- 41% better than NeRF full cloud (6.57cm)
- Uses only 4% of points (48K vs 1.2M)

### Completeness
- **VGGT has best CAD coverage** (28.2% @1cm)
- NeRF full cloud: 13.5% @1cm
- Outer shell: 3.4% @1cm (by design - surface-only)

### Trade-offs
- **Full cloud**: Better completeness, includes interior/occluded points
- **Outer shell**: Better surface accuracy, only visible exterior geometry
- **VGGT**: Best overall balance with 6x more points than NeRF

## Pipeline Differences

| Aspect | Full Cloud | Outer Shell |
|--------|-----------|-------------|
| **Input** | base.ply (raw) | base_final.ply (aligned) |
| **Points** | 1,198,080 | 48,085 (4%) |
| **Rotation** | Fixed -90°Z + 60°Y | ICP optimizes |
| **ICP Mode** | Translation-only | Full 6-DOF |
| **ICP Fitness** | 0.714 | 0.781 |
| **Coverage** | All reconstruction | Visible surface only |
| **Median Accuracy** | 6.57 cm | 3.87 cm |

## Algorithm Details

### Full Cloud ICP
- **Method:** Point-to-point (translation-only)
- **Scales:** 0.4m → 0.2m → 0.1m correspondence distance
- **Constraint:** Preserves initial rotation, optimizes only translation vector
- **Purpose:** Align to CAD while maintaining user-specified orientation

### Outer Shell Extraction
- **Method:** Angular hash map + ray-casting
- **Resolution:** 1° angular bins (360 azimuthal × 180 polar)
- **Logic:** For each ray from center, keep only farthest point
- **Effect:** Removes occluded and interior points, keeps visible surface

### Heat Map Generation
- **Method:** Normal-based perpendicular distance to nearest CAD surface
- **K-nearest:** 30 candidate CAD points per reconstruction point
- **Color scale:** White (0cm) → Red (10cm+)
- **Purpose:** Visual quality assessment showing distance errors

## File Structure

```
car/
├── README.md                              # This file
├── base.obj                               # Raw NeRF mesh from instant-ngp
├── base.ply                               # Converted to PLY format
├── rotate_and_evaluate.py                 # Main pipeline: alignment + evaluation + outer shell
├── compare_benchmarks.py                  # Benchmark comparison script
└── mesh/
    ├── cad_sample.ply                     # CAD reference (100K points)
    ├── base_final.ply                     # Aligned full cloud
    ├── base_final_heatmap.ply             # Full cloud heat map
    ├── base_final_metrics.json            # Full cloud results
    ├── base_final_outer_shell_aligned.ply # Aligned outer shell
    ├── base_final_outer_shell_heatmap.ply # Outer shell heat map
    └── base_final_outer_metrics.json      # Outer shell results
```

## Complete Pipeline (One Command)

Run the entire pipeline from raw mesh to comparison:

```bash
cd /isilon/Automotive/RnD/elad.e/instant-ngp/data/nerf/car
conda activate env_instant_ngp

# Step 1: Convert OBJ to PLY
python /isilon/Automotive/RnD/elad.e/obj_to_ply.py base.obj base.ply

# Step 2: Full cloud + outer shell evaluation (automatic)
python rotate_and_evaluate.py \
    --input base.ply \
    --output mesh/base_final.ply \
    --reference mesh/cad_sample.ply \
    --icp \
    --lock-rotation \
    --evaluate

# Step 3: Compare benchmarks
python compare_benchmarks.py
```

**Note:** Outer shell extraction and evaluation now happens automatically when using `--evaluate` flag. No need to run separate scripts.

## Visualization

To view the heat maps in Open3D:

```python
import open3d as o3d

# View full cloud heat map
pcd = o3d.io.read_point_cloud('mesh/base_final_heatmap.ply')
o3d.visualization.draw_geometries([pcd])

# View outer shell heat map
pcd = o3d.io.read_point_cloud('mesh/base_final_outer_shell_heatmap.ply')
o3d.visualization.draw_geometries([pcd])
```

## References

- **VGGT benchmark:** `/isilon/Automotive/RnD/elad.e/uv-3d/sessions/demo_room_test/run_2026-02-17T09-49-11-416846/vggt_poc/cad_eval/metrics.json`
- **Instant-ngp:** https://github.com/NVlabs/instant-ngp
- **CAD reference:** Demo room vehicle model (100K sampled points)

## Notes

- All distances are in meters (displayed as cm in output)
- Heat maps use 10cm color scale (white=0cm, red=10cm+)
- Completeness metrics measure CAD surface coverage by reconstruction
- Accuracy metrics measure reconstruction error to CAD
- ICP convergence: 100 iterations per scale, relative fitness/RMSE convergence
