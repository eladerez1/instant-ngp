#!/usr/bin/env python3
import json
import numpy as np

# Load transforms.json
with open('data/nerf/car/transforms.json', 'r') as f:
    data = json.load(f)

# 180 degree rotation around Y-axis
rotation_180_y = np.array([
    [-1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, -1, 0],
    [0, 0, 0, 1]
])

# Apply rotation to each frame
for frame in data['frames']:
    # Get the current transformation matrix
    transform = np.array(frame['transform_matrix'])
    
    # Apply 180 degree rotation: new_transform = rotation_180_y @ transform
    rotated_transform = rotation_180_y @ transform
    
    # Update the frame
    frame['transform_matrix'] = rotated_transform.tolist()

# Save the updated transforms.json
with open('data/nerf/car/transforms.json', 'w') as f:
    json.dump(data, f, indent=2)

print(f"Rotated {len(data['frames'])} camera transforms by 180 degrees around Y-axis")
