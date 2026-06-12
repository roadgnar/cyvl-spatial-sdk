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

    # ----------------------------------------------------------------- lidar

    def lidar_around(self, x_utm: float, y_utm: float, radius_m: float = 30.0):
        """Fetch lidar points around a UTM location (see cyvl.lidar).

        Args:
            x_utm: Easting in the scene's UTM zone.
            y_utm: Northing in the scene's UTM zone.
            radius_m: Horizontal radius, meters.

        Returns:
            PointCloud of the surrounding area.
        """
        from cyvl.lidar import LidarSource

        if self._lidar_source is None:
            self._lidar_source = LidarSource(self.tiles, self.utm_epsg)
        return self._lidar_source.fetch_around(x_utm, y_utm, radius_m)

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
    if backend is None:
        backend = ParquetBackend.for_scene(name)
    return Scene(backend)
