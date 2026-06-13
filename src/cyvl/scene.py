"""Scene: the entry point to one city's worth of Cyvl data."""

import logging

import geopandas as gpd
import numpy as np
import pandas as pd

from cyvl.backends.base import DataBackend
from cyvl.backends.parquet import ParquetBackend
from cyvl.camera import Camera
from cyvl.frame import Frame
from cyvl.geometry import lonlat_to_utm

logger = logging.getLogger(__name__)

# Friendly accessor name -> exported layer name.
LAYER_ALIASES = {
    "signs": "signs",
    "pavements": "pavements",
    "distresses": "distresses",
    "inspection_cells": "distressInspectionCells",
    "markings": "sam",
    "assets": "aboveGroundAssets",
    "pano": "panoramicImagery",
    "imagery": "plainImagery",
    "rollup": "rollup",
    "drive_paths": "streetviewImagePaths",
}


class Scene:
    """One scanned area (e.g. Somerville, MA): frames, layers, and lidar."""

    def __init__(self, backend: DataBackend) -> None:
        """Wrap a data backend.

        Args:
            backend: Any DataBackend implementation (parquet for the
                hackathon; a live spatial-core backend fits the same seam).
        """
        self.backend = backend
        self._frames: gpd.GeoDataFrame | None = None
        self._cameras: dict[str, Camera] | None = None
        self._lidar_source = None

    # ------------------------------------------------------------------ data

    @property
    def frames(self) -> gpd.GeoDataFrame:
        """All posed frames as a GeoDataFrame (point geometry, EPSG:4326)."""
        if self._frames is None:
            df = self.backend.frames()
            self._frames = gpd.GeoDataFrame(
                df,
                geometry=gpd.points_from_xy(df["lon"], df["lat"]),
                crs="EPSG:4326",
            ).set_index("frame_id", drop=False)
        return self._frames

    @property
    def utm_epsg(self) -> int:
        """UTM zone shared by poses and lidar tiles."""
        return int(self.frames["utm_epsg"].iloc[0])

    def __getattr__(self, name: str) -> gpd.GeoDataFrame:
        if name in LAYER_ALIASES:
            return self.backend.layer(LAYER_ALIASES[name])
        raise AttributeError(f"Scene has no attribute {name!r}")

    @property
    def tiles(self) -> gpd.GeoDataFrame:
        """Point-cloud tile index (footprints + URLs)."""
        return self.backend.tiles()

    # ---------------------------------------------------------------- frames

    def camera_for(self, intrinsics_id: str) -> Camera:
        """Calibrated camera for a frames-table intrinsics_id."""
        if self._cameras is None:
            cameras = self.backend.cameras()
            self._cameras = {
                row["intrinsics_id"]: Camera.from_row(row)
                for _, row in cameras.iterrows()
            }
        return self._cameras[intrinsics_id]

    def frame(self, frame_id: str) -> Frame:
        """Look up one frame by id.

        Args:
            frame_id: Frame id (with or without .jpg extension).

        Returns:
            The posed Frame.

        Raises:
            KeyError: If the frame id is unknown (or has no pose).
        """
        key = frame_id.removesuffix(".jpg").removesuffix(".jpeg")
        try:
            row = self.frames.loc[key]
        except KeyError as err:
            raise KeyError(
                f"No posed frame {key!r}. 360 panoramas and frames outside "
                "SLAM coverage have no pose; see scene.pano for the "
                "geo-only panorama layer."
            ) from err
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        return Frame(row, self.camera_for(row["intrinsics_id"]), scene=self)

    def frames_near(
        self,
        lon: float,
        lat: float,
        radius_m: float = 25.0,
        *,
        facing: str | None = None,
        require_image: bool = True,
    ) -> gpd.GeoDataFrame:
        """Find frames whose camera was within a radius of a location.

        Args:
            lon: Query longitude.
            lat: Query latitude.
            radius_m: Search radius in meters.
            facing: Optionally restrict to "front" or "rear".
            require_image: Only frames with a public photo (default True).

        Returns:
            Matching frames sorted nearest-first, with a distance_m column.
        """
        frames = self.frames
        query = lonlat_to_utm(lon, lat, 0.0, self.utm_epsg)[0]
        distance = np.hypot(
            frames["cam_x"].to_numpy() - query[0],
            frames["cam_y"].to_numpy() - query[1],
        )
        mask = distance <= radius_m
        if facing is not None:
            mask &= (frames["facing"] == facing).to_numpy()
        if require_image:
            mask &= frames["has_public_image"].to_numpy()

        result = frames[mask].copy()
        result["distance_m"] = distance[mask]
        return result.sort_values("distance_m")

    def nearest_frame(
        self,
        lon: float,
        lat: float,
        *,
        facing: str | None = None,
        require_image: bool = True,
        require_lidar: bool = True,
    ) -> Frame:
        """The single closest frame to a location, no radius guessing.

        Args:
            lon: Query longitude.
            lat: Query latitude.
            facing: Optionally restrict to "front" or "rear".
            require_image: Only frames with a public photo (default True).
            require_lidar: Only frames whose own drive pass has lidar tiles
                (default True — ~2% of frames lack them and can't drape;
                no-op when the frames table predates the coverage column).

        Returns:
            The nearest matching Frame.

        Raises:
            ValueError: If no frame matches the filters at all.
        """
        frames = self.frames
        mask = np.ones(len(frames), dtype=bool)
        if facing is not None:
            mask &= (frames["facing"] == facing).to_numpy()
        if require_image:
            mask &= frames["has_public_image"].to_numpy()
        if require_lidar and "has_lidar_pass" in frames.columns:
            mask &= frames["has_lidar_pass"].to_numpy()
        if not mask.any():
            raise ValueError(
                f"No frames match facing={facing!r}, require_image={require_image}"
            )

        query = lonlat_to_utm(lon, lat, 0.0, self.utm_epsg)[0]
        candidates = frames[mask]
        distance = np.hypot(
            candidates["cam_x"].to_numpy() - query[0],
            candidates["cam_y"].to_numpy() - query[1],
        )
        return self.frame(candidates.index[int(np.argmin(distance))])

    # ----------------------------------------------------------------- lidar

    def vehicle_path(self) -> tuple[np.ndarray, np.ndarray]:
        """The drive path as (timestamps_s, positions) for range estimation.

        Built from every posed frame, time-sorted. Used to reconstruct
        per-point capture range when the lidar source lacks the range
        dimension (the COPC tiles do).

        Returns:
            Tuple of (N,) seconds-since-epoch and (N, 3) UTM positions.
        """
        if not hasattr(self, "_vehicle_path"):
            ordered = self.frames.sort_values("adjusted_timestamp_us")
            self._vehicle_path = (
                ordered["adjusted_timestamp_us"].to_numpy() / 1e6,
                ordered[["cam_x", "cam_y", "cam_z"]].to_numpy(),
            )
        return self._vehicle_path

    def lidar_around(
        self,
        x_utm: float,
        y_utm: float,
        radius_m: float = 30.0,
        *,
        min_range_m: float | None = 3.0,
        pass_time_s: float | None = None,
        pass_window_s: float = 60.0,
    ):
        """Fetch lidar points around a UTM location (see cyvl.lidar).

        Args:
            x_utm: Easting in the scene's UTM zone.
            y_utm: Northing in the scene's UTM zone.
            radius_m: Horizontal radius, meters.
            min_range_m: Drop points captured closer than this to the
                scanner (vehicle self-hits). None disables.
            pass_time_s: GPS seconds-since-epoch of a single drive pass
                (e.g. a frame's adjusted_timestamp). When given, only points
                captured within pass_window_s of it are loaded — the area is
                cropped to that one pass per-tile during the fetch, so dense
                multi-pass intersections don't blow up memory. None loads
                every pass that covers the area.
            pass_window_s: Half-width of the single-pass time window, seconds.

        Returns:
            PointCloud of the surrounding area.
        """
        from cyvl.lidar import LidarSource

        if self._lidar_source is None:
            self._lidar_source = LidarSource(self.tiles, self.utm_epsg)
        time_window = (
            (pass_time_s, pass_window_s) if pass_time_s is not None else None
        )
        cloud = self._lidar_source.fetch_around(
            x_utm, y_utm, radius_m, time_window=time_window
        )
        if min_range_m is not None and len(cloud):
            path_ts_s, path_xyz = self.vehicle_path()
            cloud = cloud.crop_min_range(min_range_m, path_ts_s, path_xyz)
        return cloud

    def map(self, layers: tuple[str, ...] = ("rollup", "signs")):
        """Quick folium map of selected layers (viz extra).

        Args:
            layers: Scene layer accessor names to render.

        Returns:
            A folium.Map.
        """
        from cyvl.viz import scene_map

        return scene_map(self, layers)

    def __repr__(self) -> str:
        n = len(self.frames)
        return f"Scene({n} posed frames, layers={sorted(LAYER_ALIASES)})"


def load_scene(
    name: str = "somerville", *, backend: DataBackend | None = None
) -> Scene:
    """Load a scene by name (downloads artifacts on first use) or backend.

    Args:
        name: Public scene name. "somerville" is the hackathon dataset.
        backend: Explicit backend override (e.g. ParquetBackend(root=...)
            for offline fixtures, or a future live spatial-core backend).

    Returns:
        A ready Scene.
    """
    import os
    from pathlib import Path

    if backend is None:
        # CYVL_SCENE_DIR points at a local directory holding parquet/ —
        # used for offline work and pre-publication development.
        override = os.environ.get("CYVL_SCENE_DIR")
        if override:
            backend = ParquetBackend(root=Path(override))
        else:
            backend = ParquetBackend.for_scene(name)
    return Scene(backend)
