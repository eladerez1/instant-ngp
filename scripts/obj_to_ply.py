#!/usr/bin/env python3
"""
Convert OBJ file to PLY format with vertex colors.

Usage:
    python obj_to_ply.py input.obj output.ply
    python obj_to_ply.py input.obj  # outputs to input.ply
"""

import argparse
import sys
from pathlib import Path


def convert_obj_to_ply(input_file: str, output_file: str, binary: bool = False):
    """
    Convert OBJ file to PLY format.
    
    Supports OBJ files with vertex colors in format:
        v x y z r g b
    where r, g, b are in range [0, 1]
    
    Args:
        input_file: Path to input OBJ file
        output_file: Path to output PLY file
        binary: If True, write binary PLY (smaller file size)
    """
    print(f"Reading OBJ file: {input_file}")
    vertices = []
    faces = []
    
    with open(input_file, 'r') as f:
        for i, line in enumerate(f):
            if i % 1000000 == 0 and i > 0:
                print(f"  Read {i:,} lines...")
            
            parts = line.strip().split()
            if not parts:
                continue
            
            if parts[0] == 'v':
                # Vertex: v x y z [r g b]
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                if len(parts) >= 7:
                    # Colors in range [0, 1], convert to [0, 255]
                    r = int(float(parts[4]) * 255)
                    g = int(float(parts[5]) * 255)
                    b = int(float(parts[6]) * 255)
                    # Clamp to valid range
                    r, g, b = max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b))
                else:
                    r, g, b = 128, 128, 128  # Default gray
                vertices.append((x, y, z, r, g, b))
            
            elif parts[0] == 'f':
                # Face: f v1 v2 v3 ... (may include v/vt/vn format)
                face = []
                for p in parts[1:]:
                    # Handle v, v/vt, v/vt/vn, v//vn formats
                    vertex_idx = int(p.split('/')[0])
                    # OBJ is 1-indexed, convert to 0-indexed
                    face.append(vertex_idx - 1)
                faces.append(face)
    
    print(f"Loaded {len(vertices):,} vertices and {len(faces):,} faces")
    
    print(f"Writing PLY file: {output_file}")
    
    if binary:
        _write_binary_ply(output_file, vertices, faces)
    else:
        _write_ascii_ply(output_file, vertices, faces)
    
    print(f"Done! Output: {output_file}")


def _write_ascii_ply(output_file: str, vertices: list, faces: list):
    """Write ASCII PLY file."""
    with open(output_file, 'w') as f:
        # Header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(vertices)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        
        # Vertices
        for i, (x, y, z, r, g, b) in enumerate(vertices):
            if i % 1000000 == 0 and i > 0:
                print(f"  Writing vertex {i:,}...")
            f.write(f"{x} {y} {z} {r} {g} {b}\n")
        
        # Faces
        for i, face in enumerate(faces):
            if i % 1000000 == 0 and i > 0:
                print(f"  Writing face {i:,}...")
            f.write(f"{len(face)} " + " ".join(str(v) for v in face) + "\n")


def _write_binary_ply(output_file: str, vertices: list, faces: list):
    """Write binary PLY file (little-endian)."""
    import struct
    
    with open(output_file, 'wb') as f:
        # Header (ASCII)
        header = "ply\n"
        header += "format binary_little_endian 1.0\n"
        header += f"element vertex {len(vertices)}\n"
        header += "property float x\n"
        header += "property float y\n"
        header += "property float z\n"
        header += "property uchar red\n"
        header += "property uchar green\n"
        header += "property uchar blue\n"
        header += f"element face {len(faces)}\n"
        header += "property list uchar int vertex_indices\n"
        header += "end_header\n"
        f.write(header.encode('ascii'))
        
        # Vertices (binary)
        for i, (x, y, z, r, g, b) in enumerate(vertices):
            if i % 1000000 == 0 and i > 0:
                print(f"  Writing vertex {i:,}...")
            f.write(struct.pack('<fffBBB', x, y, z, r, g, b))
        
        # Faces (binary)
        for i, face in enumerate(faces):
            if i % 1000000 == 0 and i > 0:
                print(f"  Writing face {i:,}...")
            f.write(struct.pack('B', len(face)))
            for v in face:
                f.write(struct.pack('<i', v))


def main():
    parser = argparse.ArgumentParser(
        description='Convert OBJ file to PLY format with vertex colors'
    )
    parser.add_argument('input', type=str, help='Input OBJ file')
    parser.add_argument('output', type=str, nargs='?', default=None,
                        help='Output PLY file (default: same name with .ply extension)')
    parser.add_argument('--binary', '-b', action='store_true',
                        help='Write binary PLY (smaller file size)')
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)
    
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.with_suffix('.ply')
    
    convert_obj_to_ply(str(input_path), str(output_path), binary=args.binary)


if __name__ == '__main__':
    main()
