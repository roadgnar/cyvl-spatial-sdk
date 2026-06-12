"""Frame: one posed street-level photo, with full 3D<->2D projection."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import cv2
import numpy as np
import pandas as pd

from cyvl._vendor import create_xyz_point_mask
from cyvl.cache import cached_download
from cyvl.camera import Camera
from cyvl.geometry import lonlat_to_utm

if TYPE_CHECKING:
    from cyvl.lidar import PointCloud
    from cyvl.scene import Scene


@dataclass
class Projected:
    """Result of projecting 3D points into a frame.

    All arrays are aligned with the input points. Pixels of points that fail
    the frustum test are NaN (they have no meaningful image location).
    """

    pixels: np.ndarray  # (N, 2) float64, NaN outside the frustum
    depth: np.ndarray  # (N,) meters from the camera center
    in_view: np.ndarray  # (N,) bool: inside frustum AND image bounds
    points_utm: np.ndarray  # (N, 3) the projected world points
    colors: np.ndarray | None = None  # (N, 3) uint8 when source had RGB
    intensity: np.ndarray | None = None  # (N,) lidar return intensity

    def __len__(self) -> int:
        return len(self.pixels)


class Frame:
    """A single posed image: project world points in, unproject pixels out."""

    def __init__(
        self, row: pd.Series, camera: Camera, scene: "Scene | None" = None
    ) -> None:
        """Build a Frame from a frames-table row.

        Args:
            row: One row of the frames table.
            camera: Calibrated camera for this frame's (dataset, facing).
            scene: Owning scene, used for lidar access (optional).
        """
        self.id: str = row["frame_id"]
        self.facing: str = row["facing"]
        self.lon: float = float(row["lon"])
        self.lat: float = float(row["lat"])
        self.image_url: str | None = (
            row["image_url"] if pd.notna(row["image_url"]) else None
        )
        self.utm_epsg: int = int(row["utm_epsg"])
        self.timestamp_us: int = int(row["timestamp_us"])
        # SLAM-domain capture time: selects this frame's own drive pass
        # out of the multi-pass merged tiles (gps_time is in this domain).
        self.adjusted_timestamp_us: float = float(row["adjusted_timestamp_us"])
        self.camera = camera
        self._scene = scene

        # camera_to_utm: rows pose_00..pose_23 + implied [0,0,0,1].
        pose = np.eye(4)
        for i in range(3):
            for j in range(4):
                pose[i, j] = row[f"pose_{i}{j}"]
        self.pose: np.ndarray = pose

    @property
    def position(self) -> np.ndarray:
        """Camera center in UTM coordinates (easting, northing, altitude)."""
        return self.pose[:3, 3]

    def image(self) -> np.ndarray:
        """Fetch this frame's photo (RGB uint8 array), cached on disk.

        Raises:
            ValueError: If the frame has no public image URL.
        """
        if self.image_url is None:
            raise ValueError(
                f"Frame {self.id} has a pose but no public image. Use "
                "scene.frames_near(..., require_image=True) to pick "
                "image-bearing frames."
            )
        path = cached_download(self.image_url, progress=False)
        bgr = cv2.imread(str(path))
        if bgr is None:
            raise ValueError(f"Could not decode image at {path}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def project(
        self,
        xyz: np.ndarray | None = None,
        *,
        lonlat: np.ndarray | None = None,
        alt: np.ndarray | float = 0.0,
        colors: np.ndarray | None = None,
        intensity: np.ndarray | None = None,
    ) -> Projected:
        """Project world points into this frame's pixels.

        The transform chain and FOV filter are identical to the production
        geolocation pipeline (vendored frustum mask + cv2.projectPoints with
        the same intrinsics/distortion).

        Args:
            xyz: (N, 3) points in this frame's UTM zone.
            lonlat: (N, 2) lon/lat pairs (alternative to xyz).
            alt: Altitude(s) used with lonlat, meters.
            colors: Optional (N, 3) per-point colors carried through.
            intensity: Optional (N,) lidar intensities carried through
                (drives the default overlay color ramp in Frame.show).

        Returns:
            Projected result aligned with the input points.

        Raises:
            ValueError: If neither or both of xyz/lonlat are given.
        """
        if (xyz is None) == (lonlat is None):
            raise ValueError("Pass exactly one of xyz= or lonlat=")
        if lonlat is not None:
            lonlat = np.atleast_2d(np.asarray(lonlat, dtype=np.float64))
            xyz = lonlat_to_utm(lonlat[:, 0], lonlat[:, 1], alt, self.utm_epsg)
        assert xyz is not None
        xyz = np.atleast_2d(np.asarray(xyz, dtype=np.float64))

        # World -> camera frame, then the production FOV mask.
        utm_to_camera = np.linalg.inv(self.pose)
        ones = np.ones((len(xyz), 1))
        points_camera = (utm_to_camera @ np.hstack([xyz, ones]).T)[:3].T
        frustum = create_xyz_point_mask(points_camera)

        pixels = np.full((len(xyz), 2), np.nan)
        if frustum.any():
            pixels[frustum] = self.camera.project(points_camera[frustum])

        depth = np.linalg.norm(points_camera, axis=1)
        in_view = frustum.copy()
        in_view[frustum] &= self.camera.in_image(pixels[frustum])

        return Projected(
            pixels=pixels,
            depth=depth,
            in_view=in_view,
            points_utm=xyz,
            colors=colors,
            intensity=intensity,
        )

    def unproject(
        self,
        px: float,
        py: float,
        *,
        depth: float | None = None,
        lidar: "PointCloud | None" = None,
        pixel_tolerance: float = 12.0,
    ) -> np.ndarray:
        """Lift a pixel back to a 3D world point.

        Depth comes from (in priority order) the explicit `depth` argument,
        or the lidar point whose projection lands nearest the query pixel.

        Args:
            px: Pixel x coordinate.
            py: Pixel y coordinate.
            depth: Known distance along the view ray, meters.
            lidar: Point cloud to source depth from (default: fetch this
                frame's surroundings via the scene).
            pixel_tolerance: Max pixel distance for the lidar depth match.

        Returns:
            (3,) UTM world point.

        Raises:
            ValueError: If no depth source is available or no lidar point
                projects within tolerance of the pixel.
        """
        ray_camera = self.camera.unproject([[px, py]])[0]

        if depth is None:
            if lidar is None:
                lidar = self.lidar()
            projected = self.project(lidar.xyz, colors=lidar.rgb)
            candidates = np.flatnonzero(projected.in_view)
            if candidates.size == 0:
                raise ValueError("No lidar points project into this frame")
            offsets = projected.pixels[candidates] - np.array([px, py])
            distances = np.hypot(offsets[:, 0], offsets[:, 1])
            best = int(np.argmin(distances))
            if distances[best] > pixel_tolerance:
                raise ValueError(
                    f"Nearest lidar point is {distances[best]:.1f}px away "
                    f"(> {pixel_tolerance}px tolerance) — no depth at this pixel"
                )
            depth = float(projected.depth[candidates[best]])

        point_camera = ray_camera * depth
        return (self.pose @ np.append(point_camera, 1.0))[:3]

    def lidar(
        self,
        radius_m: float = 30.0,
        *,
        same_pass: bool = True,
        pass_window_s: float = 5.0,
        fallback: bool = True,
        min_range_m: float | None = 3.0,
    ) -> "PointCloud":
        """Fetch the lidar points surrounding this frame's camera position.

        By default only points captured during this frame's own drive pass
        are returned: the city was scanned by many passes days apart, and
        cross-pass SLAM drift plus moving objects smear projections. Pass
        same_pass=False for the full merged cloud.

        Args:
            radius_m: Horizontal radius around the camera, meters.
            same_pass: Keep only points captured alongside this photo.
            pass_window_s: Half-width of the same-pass window, seconds. The
                default 5 s balances coverage density against moving-object
                ghosts; tighten to ~2 s on busy streets.
            fallback: When the frame's own pass has no delivered lidar
                (has_lidar_pass == False), substitute the densest nearby
                pass with a warning. Set False to raise instead.
            min_range_m: Drop points captured closer than this to the
                scanner (vehicle self-hits form a floating layer above the
                road). Uses the true range dimension when present, else the
                vehicle-path estimate. None disables.

        Returns:
            PointCloud in this frame's UTM zone.

        Raises:
            RuntimeError: If the frame is not attached to a scene.
        """
        if self._scene is None:
            raise RuntimeError("Frame has no scene attached; use scene.frame(id)")
        # min-range filtering happens below, after pass selection (smaller
        # array) — disable it in the scene-level fetch.
        cloud = self._scene.lidar_around(
            self.position[0], self.position[1], radius_m=radius_m, min_range_m=None
        )
        if same_pass:
            capture_s = self.adjusted_timestamp_us / 1e6
            own_pass = cloud.crop_time(capture_s, pass_window_s)
            if len(own_pass) == 0 and not fallback:
                raise ValueError(
                    f"Frame {self.id} has no lidar from its own pass "
                    "(see frames table has_lidar_pass). Pick another frame, "
                    "or pass fallback=True to use the densest nearby pass."
                )
            if len(own_pass) == 0 and len(cloud) and cloud.gps_time is not None:
                # A few workers carry poses but no delivered lidar tiles
                # (see frames table has_lidar_pass). Fall back to the pass
                # with the densest coverage here: bin gps_time coarsely and
                # take the dominant bin (nearest-in-time passes can be ones
                # that merely clipped the edge of the radius).
                bins = np.round(cloud.gps_time / 30.0)
                values, counts = np.unique(bins, return_counts=True)
                dominant_s = float(values[np.argmax(counts)] * 30.0)
                gap_h = abs(dominant_s - capture_s) / 3600
                import logging

                logging.getLogger(__name__).warning(
                    "Frame %s has no lidar from its own pass; using the "
                    "densest nearby pass (%.1f h away). Cross-pass alignment "
                    "may be off by several cm and moving objects will differ.",
                    self.id,
                    gap_h,
                )
                own_pass = cloud.crop_time(dominant_s, pass_window_s + 30.0)
            cloud = own_pass
        if min_range_m is not None and len(cloud):
            path_ts_s, path_xyz = self._scene.vehicle_path()
            cloud = cloud.crop_min_range(min_range_m, path_ts_s, path_xyz)
        return cloud

    def points_in_view(
        self, radius_m: float = 30.0, *, same_pass: bool = True
    ) -> Projected:
        """Project this frame's surrounding lidar into its pixels.

        The one-liner: colored 3D points with exact pixel coordinates,
        ready to overlay on the photo or lift back to 3D.

        Args:
            radius_m: Horizontal lidar fetch radius, meters.
            same_pass: Use only this frame's own drive pass (default True —
                see Frame.lidar for why).

        Returns:
            Projected lidar points (colors carried through when present).
        """
        cloud = self.lidar(radius_m=radius_m, same_pass=same_pass)
        return self.project(cloud.xyz, colors=cloud.rgb, intensity=cloud.intensity)

    def open_viewer(self, radius_m: float = 30.0, block: bool | None = None):
        """Open this frame in the split image/3D browser viewer.

        Renders the surrounding lidar through this frame's exact calibrated
        camera, overlaid on the photo — drag to orbit away from the camera
        viewpoint, double-click to snap back.

        Args:
            radius_m: Lidar fetch radius around the camera, meters.

        Returns:
            Path to the exported viewer bundle.
        """
        from cyvl.viewer import open_viewer

        return open_viewer(self, radius_m=radius_m, block=block)

    def show(self, overlay: Projected | None = None, max_points: int = 200_000):
        """Display the frame (and optionally projected points) with matplotlib.

        Args:
            overlay: Projected points to scatter over the image.
            max_points: Cap on scattered points for responsiveness.

        Returns:
            The matplotlib Axes.
        """
        from cyvl.viz import show_frame  # viz extra; lazy import

        return show_frame(self, overlay=overlay, max_points=max_points)

    def __repr__(self) -> str:
        has_image = "image" if self.image_url else "no public image"
        return (
            f"Frame({self.id!r}, facing={self.facing}, "
            f"lon={self.lon:.6f}, lat={self.lat:.6f}, {has_image})"
        )
