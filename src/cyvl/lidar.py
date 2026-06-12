"""Lidar access: tile selection, fetch, and spatial cropping.

Tiles are streamed from Cyvl's public CDN. When COPC twins are available
(copc_url column), only the spatial subset is range-read over HTTP; plain
LAZ tiles are downloaded once and cached.
"""

import logging
from dataclasses import dataclass

import geopandas as gpd
import laspy
import numpy as np
from shapely.geometry import box

from cyvl.cache import cached_download
from cyvl.geometry import utm_to_lonlat_transformer

logger = logging.getLogger(__name__)


@dataclass
class PointCloud:
    """A bag of georeferenced lidar points in a UTM zone."""

    xyz: np.ndarray  # (N, 3) float64, UTM easting/northing/altitude
    rgb: np.ndarray | None  # (N, 3) uint8
    intensity: np.ndarray | None  # (N,) float32
    utm_epsg: int

    def __len__(self) -> int:
        return len(self.xyz)

    def crop_radius(self, x: float, y: float, radius_m: float) -> "PointCloud":
        """Subset points within a horizontal radius of (x, y)."""
        mask = np.hypot(self.xyz[:, 0] - x, self.xyz[:, 1] - y) <= radius_m
        return PointCloud(
            xyz=self.xyz[mask],
            rgb=self.rgb[mask] if self.rgb is not None else None,
            intensity=self.intensity[mask] if self.intensity is not None else None,
            utm_epsg=self.utm_epsg,
        )

    @staticmethod
    def concatenate(clouds: list["PointCloud"]) -> "PointCloud":
        """Merge clouds (assumed same UTM zone and field availability)."""
        clouds = [c for c in clouds if len(c)]
        if not clouds:
            return PointCloud(np.empty((0, 3)), None, None, 0)
        rgb = (
            np.vstack([c.rgb for c in clouds])
            if all(c.rgb is not None for c in clouds)
            else None
        )
        intensity = (
            np.concatenate([c.intensity for c in clouds])
            if all(c.intensity is not None for c in clouds)
            else None
        )
        return PointCloud(
            xyz=np.vstack([c.xyz for c in clouds]),
            rgb=rgb,
            intensity=intensity,
            utm_epsg=clouds[0].utm_epsg,
        )


def _las_to_pointcloud(las: laspy.LasData, utm_epsg: int) -> PointCloud:
    """Convert laspy data to a PointCloud (16-bit LAS color -> 8-bit)."""
    xyz = np.column_stack([np.asarray(las.x), np.asarray(las.y), np.asarray(las.z)])
    rgb = None
    if "red" in las.point_format.dimension_names:
        rgb = np.column_stack(
            [np.asarray(las.red), np.asarray(las.green), np.asarray(las.blue)]
        )
        if rgb.max() > 255:
            rgb = (rgb / 257).astype(np.uint8)
        else:
            rgb = rgb.astype(np.uint8)
    intensity = (
        np.asarray(las.intensity, dtype=np.float32)
        if "intensity" in las.point_format.dimension_names
        else None
    )
    return PointCloud(xyz=xyz, rgb=rgb, intensity=intensity, utm_epsg=utm_epsg)


class LidarSource:
    """Fetches lidar points by location using the exported tile index."""

    def __init__(self, tiles: gpd.GeoDataFrame, utm_epsg: int) -> None:
        """Wrap the tile index.

        Args:
            tiles: tiles.parquet GeoDataFrame (footprints in EPSG:4326).
            utm_epsg: UTM zone of the tile point data.
        """
        self.tiles = tiles
        self.utm_epsg = utm_epsg

    def tiles_for_bounds(
        self, x_min: float, y_min: float, x_max: float, y_max: float
    ) -> gpd.GeoDataFrame:
        """Tiles whose footprint intersects a UTM bounding box."""
        to_lonlat = utm_to_lonlat_transformer(self.utm_epsg)
        lon_min, lat_min = to_lonlat.transform(x_min, y_min)
        lon_max, lat_max = to_lonlat.transform(x_max, y_max)
        query = box(lon_min, lat_min, lon_max, lat_max)
        hits = self.tiles[self.tiles.intersects(query)]
        return hits

    def fetch_around(
        self, x_utm: float, y_utm: float, radius_m: float = 30.0
    ) -> PointCloud:
        """Fetch and crop lidar points around a UTM location.

        Prefers COPC HTTP range reads (only the queried subset moves over
        the network); falls back to whole-tile LAZ download + disk cache.

        Args:
            x_utm: Easting.
            y_utm: Northing.
            radius_m: Horizontal radius, meters.

        Returns:
            Cropped PointCloud (possibly empty if no tiles cover the area).
        """
        hits = self.tiles_for_bounds(
            x_utm - radius_m, y_utm - radius_m, x_utm + radius_m, y_utm + radius_m
        )
        if hits.empty:
            logger.warning("No lidar tiles cover (%.1f, %.1f)", x_utm, y_utm)
            return PointCloud(np.empty((0, 3)), None, None, self.utm_epsg)

        clouds = []
        for _, tile in hits.iterrows():
            cloud = self._fetch_tile_subset(
                tile,
                x_min=x_utm - radius_m,
                y_min=y_utm - radius_m,
                x_max=x_utm + radius_m,
                y_max=y_utm + radius_m,
            )
            clouds.append(cloud)
        merged = PointCloud.concatenate(clouds)
        merged.utm_epsg = self.utm_epsg
        return merged.crop_radius(x_utm, y_utm, radius_m)

    def _fetch_tile_subset(
        self, tile, x_min: float, y_min: float, x_max: float, y_max: float
    ) -> PointCloud:
        """Read one tile's points within a UTM bbox (COPC fast path)."""
        copc_url = tile.get("copc_url")
        if isinstance(copc_url, str) and copc_url:
            bounds = laspy.copc.Bounds(
                mins=np.array([x_min, y_min]), maxs=np.array([x_max, y_max])
            )
            with laspy.copc.CopcReader.open(copc_url) as reader:
                points = reader.query(bounds)
                las = laspy.LasData(header=reader.header, points=points)
            return _las_to_pointcloud(las, self.utm_epsg)

        # Whole-tile fallback: ~hundreds of MB on first touch, cached after.
        logger.info(
            "Tile %s has no COPC twin; downloading full LAZ (cached after)",
            tile.get("tile_id", "?"),
        )
        path = cached_download(tile["laz_url"])
        las = laspy.read(path)
        cloud = _las_to_pointcloud(las, self.utm_epsg)
        mask = (
            (cloud.xyz[:, 0] >= x_min)
            & (cloud.xyz[:, 0] <= x_max)
            & (cloud.xyz[:, 1] >= y_min)
            & (cloud.xyz[:, 1] <= y_max)
        )
        return PointCloud(
            xyz=cloud.xyz[mask],
            rgb=cloud.rgb[mask] if cloud.rgb is not None else None,
            intensity=cloud.intensity[mask] if cloud.intensity is not None else None,
            utm_epsg=self.utm_epsg,
        )
