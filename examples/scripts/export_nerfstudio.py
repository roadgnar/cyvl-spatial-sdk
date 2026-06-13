"""Export posed frames to Nerfstudio format — train NeRFs / Gaussian Splats on real streets.

Writes transforms.json + images/ + a LiDAR seed point cloud for a neighborhood:

    python export_nerfstudio.py --lon -71.0944 --lat 42.3865 --radius 80
    ns-train splatfacto --data exports/nerfstudio_scene

NOTE: expect to tune the training run. A naive default `splatfacto` run
(30k iterations) did not produce a good result on this data — plan on
adjusting things like the iteration count, learning rates, densification
schedule, and the export radius before you get clean geometry.

Two things this does that the usual COLMAP workflow can't:

- Poses and intrinsics come from Cyvl's production SLAM + calibration —
  no structure-from-motion step, no scale ambiguity.
- The splat initializes from dense survey-grade LiDAR (ply_file_path),
  not a sparse SfM cloud — faster convergence, true geometry.

Frames are filtered to the single dominant capture pass (consistent
lighting/weather/parked cars); use --all-passes to disable.
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

import cyvl
from cyvl.cache import cached_download
from cyvl.geometry import lonlat_to_utm


def opencv_to_opengl(camera_to_world: np.ndarray) -> np.ndarray:
    """Convert an OpenCV-convention c2w pose to OpenGL (Nerfstudio) axes."""
    converted = camera_to_world.copy()
    converted[:, 1] *= -1  # y: down -> up
    converted[:, 2] *= -1  # z: forward -> backward
    return converted


def dominant_pass(timestamps_s: np.ndarray, gap_s: float = 120.0) -> np.ndarray:
    """Boolean mask of the largest time-contiguous cluster of frames.

    Frames more than gap_s apart belong to different drive passes (often
    different days — different light, weather, parked cars), which degrades
    photometric consistency for NeRF/3DGS training.
    """
    order = np.argsort(timestamps_s)
    sorted_ts = timestamps_s[order]
    cluster_ids = np.concatenate(([0], np.cumsum(np.diff(sorted_ts) > gap_s)))
    best = np.bincount(cluster_ids).argmax()
    mask = np.zeros(len(timestamps_s), dtype=bool)
    mask[order[cluster_ids == best]] = True
    return mask


def write_seed_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
    """Write a binary PLY (positions + colors) for splat initialization."""
    import struct

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
    parser.add_argument("--lon", type=float, default=-71.094426)
    parser.add_argument("--lat", type=float, default=42.38647)
    parser.add_argument("--radius", type=float, default=80.0, help="meters")
    parser.add_argument("--out", type=Path, default=Path("exports/nerfstudio_scene"))
    parser.add_argument("--max-frames", type=int, default=300)
    parser.add_argument("--seed-points", type=int, default=1_500_000)
    parser.add_argument(
        "--pass-window", type=float, default=60.0,
        help="Half-width (s) of the single LiDAR pass kept around the anchor "
             "frame; smaller = tighter single pass, less memory",
    )
    parser.add_argument(
        "--all-passes", action="store_true",
        help="Keep frames from every capture pass (mixed lighting)",
    )
    args = parser.parse_args()

    scene = cyvl.load_scene("somerville")
    near = scene.frames_near(args.lon, args.lat, radius_m=args.radius)
    if near.empty:
        raise SystemExit("No image-bearing frames in that radius — widen it.")

    # Keep one capture pass: consistent sun, sky, and parked cars.
    if not args.all_passes:
        ts = near["adjusted_timestamp_us"].to_numpy() / 1e6
        keep = dominant_pass(ts)
        dropped = int((~keep).sum())
        near = near[keep]
        if dropped:
            print(f"single-pass filter: kept {len(near)}, dropped {dropped} from other passes")
    frame_ids = near.index[: args.max_frames]

    images_dir = args.out / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # Center everything on the first camera so coordinates stay small.
    origin = None
    frames_json = []
    for frame_id in frame_ids:
        frame = scene.frame(frame_id)
        if origin is None:
            origin = frame.position.copy()
        pose = frame.pose.copy()
        pose[:3, 3] -= origin

        image_path = images_dir / f"{frame.id}.jpg"
        shutil.copyfile(cached_download(frame.image_url, progress=False), image_path)

        camera = frame.camera
        frames_json.append(
            {
                "file_path": f"images/{frame.id}.jpg",
                "transform_matrix": opencv_to_opengl(pose).tolist(),
                "fl_x": camera.fx,
                "fl_y": camera.fy,
                "cx": camera.cx,
                "cy": camera.cy,
                "w": camera.width,
                "h": camera.height,
                "k1": float(camera.dist[0]),
                "k2": float(camera.dist[1]),
                "p1": float(camera.dist[2]),
                "p2": float(camera.dist[3]),
            }
        )

    # LiDAR seed cloud from the SAME pass, in the same shifted world frame —
    # splatfacto initializes gaussians from it instead of random points.
    # Crop to that one pass *during* the fetch: a busy area is scanned by many
    # passes, and merging all of them is both wrong (mixed lighting, ghosting)
    # and an OOM risk — the all-pass cloud here is ~4x the single-pass one.
    center = lonlat_to_utm(args.lon, args.lat, 0.0, scene.utm_epsg)[0]
    anchor = scene.frame(frame_ids[0])
    pass_time_s = anchor.adjusted_timestamp_us / 1e6
    cloud = scene.lidar_around(
        center[0],
        center[1],
        radius_m=args.radius + 15,
        pass_time_s=pass_time_s,
        pass_window_s=args.pass_window,
    )
    if len(cloud) > args.seed_points:
        pick = np.sort(
            np.random.default_rng(0).choice(len(cloud), args.seed_points, replace=False)
        )
        cloud = cloud._subset(pick)
    rgb = (
        cloud.rgb if cloud.rgb is not None
        else np.full((len(cloud), 3), 180, dtype=np.uint8)
    )
    write_seed_ply(args.out / "sparse_pc.ply", cloud.xyz - origin, rgb)

    (args.out / "transforms.json").write_text(
        json.dumps(
            {
                "camera_model": "OPENCV",
                "ply_file_path": "sparse_pc.ply",
                "frames": frames_json,
            },
            indent=2,
        )
    )
    print(f"Wrote {len(frames_json)} frames + {len(cloud):,} LiDAR seed points to {args.out}")
    print(f"Train: ns-train splatfacto --data {args.out}")


if __name__ == "__main__":
    main()
