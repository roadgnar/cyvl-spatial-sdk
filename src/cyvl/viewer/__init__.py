"""Browser viewer: photo + LiDAR rendered through the frame's real camera."""

import http.server
import json
import logging
import shutil
import socket
import threading
import webbrowser
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from cyvl.frame import Frame

logger = logging.getLogger(__name__)


def export_viewer_bundle(
    frame: "Frame",
    out_dir: Path,
    radius_m: float = 30.0,
    max_points: int = 4_000_000,
    cloud=None,
) -> Path:
    """Write a self-contained viewer bundle for a frame.

    The bundle contains index.html, the frame photo, the surrounding lidar
    (camera-frame float32 xyz + rgb), and data.json with the calibrated
    intrinsics so the 3D render matches the photo pixel-for-pixel.

    Args:
        frame: Frame to visualize (must have a public image).
        out_dir: Output directory (created).
        radius_m: Lidar fetch radius around the camera.
        max_points: Random-subsample cap keeping the bundle browser-friendly
            (~60 MB at the default).

    Returns:
        The bundle directory.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    with resources.files("cyvl.viewer").joinpath("index.html").open("rb") as src:
        with open(out_dir / "index.html", "wb") as dst:
            shutil.copyfileobj(src, dst)

    from cyvl.cache import cached_download

    if frame.image_url is None:
        raise ValueError(f"Frame {frame.id} has no public image to view")
    shutil.copyfile(cached_download(frame.image_url, progress=False), out_dir / "image.jpg")

    # Strict same-pass: a viewer that silently shows another drive's lidar
    # would misrepresent alignment, so refuse rather than fall back.
    # A pre-fetched cloud= overrides (e.g. LAZ-sourced points that retain
    # the capture-range dimension, which the COPC twins lack).
    if cloud is None:
        cloud = frame.lidar(radius_m=radius_m, fallback=False)
    if len(cloud) > max_points:
        keep = np.sort(
            np.random.default_rng(0).choice(len(cloud), max_points, replace=False)
        )
        cloud = cloud._subset(keep)
    # Transform to the camera frame in float64, THEN narrow to float32 —
    # UTM magnitudes (~5e5) would destroy precision if cast first.
    utm_to_camera = np.linalg.inv(frame.pose)
    ones = np.ones((len(cloud), 1))
    points_camera = (utm_to_camera @ np.hstack([cloud.xyz, ones]).T)[:3].T
    xyz32 = points_camera.astype(np.float32)
    rgb = (
        cloud.rgb
        if cloud.rgb is not None
        else np.full((len(cloud), 3), 200, dtype=np.uint8)
    )
    # Percentile-normalized intensity (uint8) drives the viridis ramp in JS.
    if cloud.intensity is not None and len(cloud):
        lo, hi = np.percentile(cloud.intensity, [2, 98])
        intensity_u8 = (
            255 * np.clip((cloud.intensity - lo) / max(hi - lo, 1e-9), 0, 1)
        ).astype(np.uint8)
    else:
        intensity_u8 = np.full(len(cloud), 128, dtype=np.uint8)
    # Capture-time scanner range in 0.1 m units (clamped at 25.5 m): lets the
    # viewer filter short-range returns (e.g. vehicle self-hits) on a slider.
    # True range dim when present, vehicle-path estimate otherwise (COPC
    # tiles lack the dim).
    ranges = None
    if len(cloud):
        if frame._scene is not None:
            path_ts_s, path_xyz = frame._scene.vehicle_path()
            ranges = cloud.capture_range(path_ts_s, path_xyz)
        else:
            ranges = cloud.capture_range()
    if ranges is not None:
        range_u8 = np.clip(ranges * 10.0, 0, 255).astype(np.uint8)
    else:
        range_u8 = np.full(len(cloud), 255, dtype=np.uint8)
    with open(out_dir / "points.bin", "wb") as f:
        f.write(xyz32.tobytes())
        f.write(intensity_u8.tobytes())
        f.write(range_u8.tobytes())
        f.write(rgb.astype(np.uint8).tobytes())

    camera = frame.camera
    (out_dir / "data.json").write_text(
        json.dumps(
            {
                "frame_id": frame.id,
                "point_count": len(cloud),
                "intrinsics": {
                    "fx": camera.fx,
                    "fy": camera.fy,
                    "cx": camera.cx,
                    "cy": camera.cy,
                    "W": camera.width,
                    "H": camera.height,
                },
            }
        )
    )
    logger.info("Viewer bundle at %s (%d points)", out_dir, len(cloud))
    return out_dir


def _interactive_session() -> bool:
    """True when there's a REPL/notebook keeping the process alive."""
    import sys

    if hasattr(sys, "ps1"):
        return True
    ipython = sys.modules.get("IPython")
    return ipython is not None and ipython.get_ipython() is not None


def serve_and_open(bundle_dir: Path, block: bool | None = None) -> None:
    """Serve a viewer bundle on localhost and open the browser.

    Args:
        bundle_dir: Directory containing index.html and assets.
        block: Keep serving in the foreground until Ctrl+C. Default (None)
            blocks in scripts and returns immediately in REPLs/notebooks,
            where the live process keeps the daemon server alive.
    """
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]

    handler = lambda *args: http.server.SimpleHTTPRequestHandler(  # noqa: E731
        *args, directory=str(bundle_dir)
    )
    server = http.server.ThreadingHTTPServer(("127.0.0.1", port), handler)
    url = f"http://127.0.0.1:{port}/index.html"

    if block is None:
        block = not _interactive_session()

    if block:
        webbrowser.open(url)
        print(f"Viewer at {url} — Ctrl+C to stop")  # noqa: T201
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            server.shutdown()
        return

    threading.Thread(target=server.serve_forever, daemon=True).start()
    logger.info("Viewer at %s (server runs until process exit)", url)
    webbrowser.open(url)


def open_viewer(
    frame: "Frame", radius_m: float = 30.0, block: bool | None = None
) -> Path:
    """Build a viewer bundle for a frame and open it in the browser.

    Args:
        frame: Frame to visualize.
        radius_m: Lidar fetch radius.
        block: Serve in the foreground (see serve_and_open).

    Returns:
        The bundle directory (kept for re-serving or sharing).
    """
    from cyvl.cache import cache_dir

    bundle = cache_dir() / "viewer" / frame.id
    export_viewer_bundle(frame, bundle, radius_m=radius_m)
    serve_and_open(bundle, block=block)
    return bundle
