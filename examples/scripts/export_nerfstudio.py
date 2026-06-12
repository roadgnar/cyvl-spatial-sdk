"""Export posed frames to Nerfstudio format — train NeRFs / Gaussian Splats on real streets.

Writes transforms.json + images/ for a neighborhood of your choice. Point
nerfstudio (or any 3DGS pipeline that reads the format) straight at the
output directory:

    python export_nerfstudio.py --lon -71.1218 --lat 42.3967 --radius 60
    ns-train splatfacto --data exports/nerfstudio_davis_sq

Cyvl poses are production SLAM/calibration output — no COLMAP needed.
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

import cyvl
from cyvl.cache import cached_download


def opencv_to_opengl(camera_to_world: np.ndarray) -> np.ndarray:
    """Convert an OpenCV-convention c2w pose to OpenGL (Nerfstudio) axes."""
    converted = camera_to_world.copy()
    converted[:, 1] *= -1  # y: down -> up
    converted[:, 2] *= -1  # z: forward -> backward
    return converted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lon", type=float, default=-71.1218)
    parser.add_argument("--lat", type=float, default=42.3967)
    parser.add_argument("--radius", type=float, default=60.0, help="meters")
    parser.add_argument("--out", type=Path, default=Path("exports/nerfstudio_scene"))
    parser.add_argument("--max-frames", type=int, default=150)
    args = parser.parse_args()

    scene = cyvl.load_scene("somerville")
    near = scene.frames_near(args.lon, args.lat, radius_m=args.radius)
    frame_ids = near.index[: args.max_frames]
    if len(frame_ids) == 0:
        raise SystemExit("No image-bearing frames in that radius — widen it.")

    images_dir = args.out / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # Center poses on the neighborhood so coordinates stay small.
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

    (args.out / "transforms.json").write_text(
        json.dumps({"camera_model": "OPENCV", "frames": frames_json}, indent=2)
    )
    print(f"Wrote {len(frames_json)} frames to {args.out}")
    print(f"Train: ns-train splatfacto --data {args.out}")


if __name__ == "__main__":
    main()
