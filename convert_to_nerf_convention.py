#!/usr/bin/env python3
"""
Convert camera transforms from Colmap/NeuS2 convention to NeRF/instant-ngp convention.

Colmap: Camera looks in +Z, Y down, X right
NeRF:   Camera looks in -Z, Y up, X right
"""
import json
import numpy as np

# Load transforms
with open('data/nerf/car/transforms.json', 'r') as f:
    data = json.load(f)

# Conversion matrix: flip Y and Z axes
# This converts from Colmap camera space to NeRF camera space
conversion = np.array([
    [1,  0,  0, 0],
    [0, -1,  0, 0],
    [0,  0, -1, 0],
    [0,  0,  0, 1]
], dtype=np.float32)

print(f"Converting {len(data['frames'])} camera poses from Colmap to NeRF convention...")

for frame in data['frames']:
    # Get original transform (camera-to-world in Colmap convention)
    transform = np.array(frame['transform_matrix'], dtype=np.float32)
    
    # Convert to NeRF convention: T_nerf = T_colmap @ conversion
    nerf_transform = transform @ conversion
    
    frame['transform_matrix'] = nerf_transform.tolist()

# Save
with open('data/nerf/car/transforms.json', 'w') as f:
    json.dump(data, f, indent=2)

print(f"✓ Converted all camera poses to NeRF convention")
print(f"  Colmap: Camera looks +Z, Y down")
print(f"  NeRF:   Camera looks -Z, Y up")
