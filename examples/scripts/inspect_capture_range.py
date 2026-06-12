"""Viewer with a capture-range filter — investigate short-range returns.

Sources points from the raw LAZ tiles (the COPC twins drop the `range`
extra dimension), prints the capture-range distribution, then opens the
viewer where the "hide capture-range <" slider strips points that were
closer than the threshold to the scanner when captured — e.g. to test
whether a floating layer above the road is vehicle self-hits:

    python inspect_capture_range.py --frame-id GX050786_MP4_1764012224345789_front_3000
"""

import argparse
from pathlib import Path

import numpy as np

import cyvl
from cyvl.cache import cache_dir
from cyvl.lidar import LidarSource
from cyvl.viewer import export_viewer_bundle, serve_and_open


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--frame-id", default="GX050786_MP4_1764012224345789_front_3000"
    )
    parser.add_argument("--radius", type=float, default=25.0)
    args = parser.parse_args()

    scene = cyvl.load_scene("somerville")
    frame = scene.frame(args.frame_id)

    # LAZ-only source: blank the copc_url column so the fetch path reads the
    # raw tiles, which retain the per-point capture `range` dimension.
    tiles = scene.tiles.copy()
    tiles["copc_url"] = None
    source = LidarSource(tiles, scene.utm_epsg)
    x, y = frame.position[0], frame.position[1]
    cloud = source.fetch_around(x, y, radius_m=args.radius).crop_time(
        frame.adjusted_timestamp_us / 1e6, 5.0
    )
    if len(cloud) == 0:
        raise SystemExit(f"{frame.id} has no own-pass lidar; pick another frame")
    if cloud.scan_range is None:
        raise SystemExit("Tiles carry no `range` dimension — cannot filter")

    ranges = cloud.scan_range
    print(f"{len(cloud):,} same-pass points | capture-range distribution:")
    for lo, hi in [(0, 1), (1, 2), (2, 3), (3, 5), (5, 10), (10, 25), (25, 1e9)]:
        n = int(((ranges >= lo) & (ranges < hi)).sum())
        label = f"{lo}-{hi if hi < 1e9 else '+'}m"
        print(f"  {label:>7}: {n:>9,} ({100 * n / len(cloud):5.1f}%)")

    bundle = cache_dir() / "viewer" / f"{frame.id}_range"
    export_viewer_bundle(frame, bundle, radius_m=args.radius, cloud=cloud)
    serve_and_open(bundle)


if __name__ == "__main__":
    main()
