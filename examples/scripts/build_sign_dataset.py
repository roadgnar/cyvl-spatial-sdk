"""Build a labeled image dataset of traffic signs — instant CV training data.

For every sign in the city, finds a photo that sees it, projects the sign's
surveyed location into the image, and saves a crop + a labels.csv with the
MUTCD code, category, and condition. A ready-to-train classification or
detection seed set in one command:

    python build_sign_dataset.py --limit 200 --out exports/sign_dataset
"""

import argparse
import csv
from pathlib import Path

import cv2

import cyvl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("exports/sign_dataset"))
    parser.add_argument("--limit", type=int, default=200, help="max signs")
    parser.add_argument("--crop", type=int, default=384, help="crop size px")
    args = parser.parse_args()

    scene = cyvl.load_scene("somerville")
    signs = scene.signs.dropna(subset=["mutcd"])

    crops_dir = args.out / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for _, sign in signs.head(args.limit).iterrows():
        lon, lat = sign.geometry.x, sign.geometry.y
        near = scene.frames_near(lon, lat, radius_m=15, facing="front")
        near = near[near["distance_m"] > 4]  # sign ahead, not under the camera
        if near.empty:
            continue
        frame = scene.frame(near.index[0])

        # Project the sign's surveyed point (signs layer is 2D; use a sign-
        # height estimate above the local road = camera height - 0.4m).
        projected = frame.project(
            lonlat=[[lon, lat]], alt=frame.position[2] - 0.4
        )
        if not projected.in_view[0]:
            continue

        u, v = projected.pixels[0]
        image = frame.image()
        half = args.crop // 2
        top, left = int(max(v - half, 0)), int(max(u - half, 0))
        crop = image[top : top + args.crop, left : left + args.crop]
        if crop.size == 0:
            continue

        name = f"{sign['fid']}_{sign['mutcd']}.jpg".replace("/", "-")
        cv2.imwrite(str(crops_dir / name), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
        rows.append(
            {
                "crop": f"crops/{name}",
                "mutcd": sign["mutcd"],
                "category": sign["category"],
                "condition": sign["condition"],
                "lon": lon,
                "lat": lat,
                "frame_id": frame.id,
                "pixel_u": round(float(u), 1),
                "pixel_v": round(float(v), 1),
                "distance_m": round(float(near["distance_m"].iloc[0]), 1),
                "depth_m": round(float(projected.depth[0]), 1),
            }
        )

    with open(args.out / "labels.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} labeled crops to {args.out}")


if __name__ == "__main__":
    main()
