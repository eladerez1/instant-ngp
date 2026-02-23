#!/usr/bin/env python3
import json
import numpy as np
import sys

# Load the ORIGINAL transforms.json (backup first if needed)
# First, let's restore from the original NeuS2 data
import shutil

# Read current transforms
with open('data/nerf/car/transforms.json', 'r') as f:
    data = json.load(f)

# Different rotation options to try
rotations = {
    'x': np.array([  # 180° around X-axis
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1]
    ]),
    'y': np.array([  # 180° around Y-axis
        [-1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1]
    ]),
    'z': np.array([  # 180° around Z-axis
        [-1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ]),
}

if len(sys.argv) < 2:
    print("Usage: python fix_camera_rotation.py [x|y|z|local_z]")
    print("  x - rotate 180° around X-axis (pitch flip)")
    print("  y - rotate 180° around Y-axis (yaw flip)")
    print("  z - rotate 180° around Z-axis (roll flip)")
    print("  local_z - rotate 180° around camera's local Z-axis")
    sys.exit(1)

axis = sys.argv[1]

if axis == 'local_z':
    # Rotate in camera's local space (post-multiply)
    rotation = rotations['z']
    for frame in data['frames']:
        transform = np.array(frame['transform_matrix'])
        rotated_transform = transform @ rotation
        frame['transform_matrix'] = rotated_transform.tolist()
    print("Applied 180° rotation around camera's local Z-axis (post-multiply)")
elif axis in rotations:
    # Rotate in world space (pre-multiply)
    rotation = rotations[axis]
    for frame in data['frames']:
        transform = np.array(frame['transform_matrix'])
        rotated_transform = rotation @ transform
        frame['transform_matrix'] = rotated_transform.tolist()
    print(f"Applied 180° rotation around world {axis.upper()}-axis (pre-multiply)")
else:
    print(f"Unknown axis: {axis}")
    sys.exit(1)

# Save
with open('data/nerf/car/transforms.json', 'w') as f:
    json.dump(data, f, indent=2)

print(f"Updated {len(data['frames'])} camera transforms")
