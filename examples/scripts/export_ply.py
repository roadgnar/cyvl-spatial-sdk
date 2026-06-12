"""Export a colored point cloud of any block to PLY — open it anywhere.

Pulls the LiDAR around a location (COPC range reads, no bulk download) and
writes a binary PLY with RGB that drops straight into CloudCompare, Blender,
MeshLab, or any DCC/CAD tool:

    python export_ply.py --lon -71.1218 --lat 42.3967 --radius 60
"""

import argparse
import struct
from pathlib import Path

import numpy as np

import cyvl


def write_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
    """Write a binary little-endian PLY with positions + colors.

    Args:
        path: Output .ply path.
        xyz: (N, 3) float positions (centered locally for float32 safety).
        rgb: (N, 3) uint8 colors.
    """
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(xyz)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    )
    record = struct.Struct("<fffBBB")
    with open(path, "wb") as f:
        f.write(header.encode())
        for (x, y, z), (r, g, b) in zip(xyz.astype(np.float32), rgb):
            f.write(record.pack(x, y, z, r, g, b))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lon", type=float, default=-71.1218)
    parser.add_argument("--lat", type=float, default=42.3967)
    parser.add_argument("--radius", type=float, default=60.0, help="meters")
    parser.add_argument("--max-points", type=int, default=8_000_000)
    parser.add_argument("--out", type=Path, default=Path("exports/block.ply"))
    args = parser.parse_args()

    scene = cyvl.load_scene("somerville")
    from cyvl.geometry import lonlat_to_utm

    center = lonlat_to_utm(args.lon, args.lat, 0.0, scene.utm_epsg)[0]
    cloud = scene.lidar_around(center[0], center[1], radius_m=args.radius)
    if len(cloud) == 0:
        raise SystemExit("No lidar tiles cover that location.")

    if len(cloud) > args.max_points:
        keep = np.random.default_rng(0).choice(
            len(cloud), args.max_points, replace=False
        )
        cloud = cloud._subset(np.sort(keep))

    # Center coordinates so float32 PLY keeps millimeter precision.
    local = cloud.xyz - cloud.xyz.mean(axis=0)
    rgb = cloud.rgb if cloud.rgb is not None else np.full((len(cloud), 3), 180, np.uint8)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_ply(args.out, local, rgb)
    size_mb = args.out.stat().st_size / 1e6
    print(f"Wrote {len(cloud):,} points ({size_mb:.0f} MB) to {args.out}")
    print(f"UTM origin (EPSG:{scene.utm_epsg}): {cloud.xyz.mean(axis=0).round(2)}")


if __name__ == "__main__":
    main()
