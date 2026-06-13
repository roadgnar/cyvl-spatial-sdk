"""Find anything in the physical world: text prompt -> mask -> 3D coordinates.

SAM3 segments whatever you name in a street photo; the frame's LiDAR lifts
each mask to surveyed 3D coordinates. Needs a fal.ai key (see cyvl.segment):

    export FAL_KEY="key-id:key-secret"     # https://fal.ai/dashboard/keys
    python find_anything.py --prompt "traffic sign"
    python find_anything.py --prompt "fire hydrant" --lon -71.0944 --lat 42.3865
"""

import argparse
from pathlib import Path

import cv2

import cyvl
from cyvl.segment import locate, segment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", default="traffic sign")
    parser.add_argument("--lon", type=float, default=-71.094426)
    parser.add_argument("--lat", type=float, default=42.38647)
    parser.add_argument("--frame-id", default=None)
    parser.add_argument("--out", type=Path, default=Path("exports/found"))
    args = parser.parse_args()

    scene = cyvl.load_scene("somerville")
    frame = (
        scene.frame(args.frame_id)
        if args.frame_id
        else scene.nearest_frame(args.lon, args.lat)
    )

    masks = segment(frame, args.prompt)
    hits = locate(frame, args.prompt)
    print(f"{frame.id}: {len(masks)} mask(s), {len(hits)} located in 3D")
    for hit in hits:
        lon, lat = hit.lonlat
        print(
            f"  {hit.prompt!r} score={hit.score:.2f}: "
            f"{lat:.6f}, {lon:.6f} — {hit.distance_m:.1f} m away "
            f"({hit.n_points:,} lidar points)"
        )

    args.out.mkdir(parents=True, exist_ok=True)
    overlay = masks.overlay(frame.image())
    out_path = args.out / f"{frame.id}_{args.prompt.replace(' ', '_')}.jpg"
    cv2.imwrite(str(out_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    print(f"overlay -> {out_path}")


if __name__ == "__main__":
    main()
