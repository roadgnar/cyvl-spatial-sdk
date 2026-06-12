"""Generate a LiDAR depth map for any street photo.

Projects the frame's own-pass point cloud into the image and rasterizes a
depth map (nearest-point-per-cell). Saves a .npz (depth in meters, NaN where
no return) and a colorized PNG preview. Pair RGB + depth + pose for monocular
depth training, 3D reconstruction seeding, or sensor-fusion experiments:

    python make_depth_map.py --lon -71.1218 --lat 42.3967
    python make_depth_map.py --frame-id GX050786_MP4_1764012224345789_front_3000
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

import cyvl


def rasterize_depth(frame, projected, cell: int = 4) -> np.ndarray:
    """Nearest-depth-per-cell raster at (H/cell, W/cell) resolution."""
    height = frame.camera.height // cell
    width = frame.camera.width // cell
    depth_map = np.full((height, width), np.inf)

    visible = projected.in_view
    u = (projected.pixels[visible, 0] / cell).astype(int).clip(0, width - 1)
    v = (projected.pixels[visible, 1] / cell).astype(int).clip(0, height - 1)
    np.minimum.at(depth_map, (v, u), projected.depth[visible])

    depth_map[np.isinf(depth_map)] = np.nan
    return depth_map


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame-id", default=None)
    parser.add_argument("--lon", type=float, default=-71.093162)
    parser.add_argument("--lat", type=float, default=42.387389)
    parser.add_argument("--cell", type=int, default=4, help="px per depth cell")
    parser.add_argument("--out", type=Path, default=Path("exports/depth"))
    args = parser.parse_args()

    scene = cyvl.load_scene("somerville")
    if args.frame_id:
        frame = scene.frame(args.frame_id)
    else:
        frame = scene.nearest_frame(args.lon, args.lat)

    projected = frame.points_in_view(radius_m=30)
    depth_map = rasterize_depth(frame, projected, cell=args.cell)

    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out / f"{frame.id}_depth.npz",
        depth_m=depth_map,
        cell_px=args.cell,
        pose_camera_to_utm=frame.pose,
        K=frame.camera.K,
    )

    # Colorized preview (turbo, near=red): NaN cells stay black.
    valid = ~np.isnan(depth_map)
    norm = np.zeros_like(depth_map)
    if valid.any():
        d = depth_map[valid]
        norm[valid] = np.clip((d - d.min()) / max(d.max() - d.min(), 1e-9), 0, 1)
    preview = cv2.applyColorMap((255 * (1 - norm)).astype(np.uint8), cv2.COLORMAP_TURBO)
    preview[~valid] = 0
    cv2.imwrite(str(args.out / f"{frame.id}_depth.png"), preview)

    coverage = 100 * valid.mean()
    print(f"{frame.id}: depth map {depth_map.shape}, {coverage:.0f}% coverage -> {args.out}")


if __name__ == "__main__":
    main()
