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
    gps_time: np.ndarray | None = None  # (N,) float64 seconds since epoch
    scan_range: np.ndarray | None = None  # (N,) float32 meters from scanner at capture

    def __len__(self) -> int:
        return len(self.xyz)

    def _subset(self, mask: np.ndarray) -> "PointCloud":
        return PointCloud(
            xyz=self.xyz[mask],
            rgb=self.rgb[mask] if self.rgb is not None else None,
            intensity=self.intensity[mask] if self.intensity is not None else None,
            utm_epsg=self.utm_epsg,
            gps_time=self.gps_time[mask] if self.gps_time is not None else None,
            scan_range=(
                self.scan_range[mask] if self.scan_range is not None else None
            ),
        )

    def crop_radius(self, x: float, y: float, radius_m: float) -> "PointCloud":
        """Subset points within a horizontal radius of (x, y)."""
        mask = np.hypot(self.xyz[:, 0] - x, self.xyz[:, 1] - y) <= radius_m
        return self._subset(mask)

    def capture_range(
        self,
        path_ts_s: np.ndarray | None = None,
        path_xyz: np.ndarray | None = None,
    ) -> np.ndarray | None:
        """Per-point distance from the scanner at capture time.

        Uses the true `range` dimension when the source carried it; otherwise
        estimates it as the distance to the vehicle position interpolated at
        each point's gps_time (validated: 99.95% agreement with the true
        range at a 3 m threshold; ~0.5 m median bias from the camera/scanner
        mount offset).

        Args:
            path_ts_s: Time-sorted vehicle path timestamps, seconds.
            path_xyz: (N, 3) vehicle positions matching path_ts_s.

        Returns:
            (N,) ranges in meters, or None when neither source is available.
        """
        if self.scan_range is not None:
            return self.scan_range
        if self.gps_time is None or path_ts_s is None or path_xyz is None:
            return None
        vx = np.interp(self.gps_time, path_ts_s, path_xyz[:, 0])
        vy = np.interp(self.gps_time, path_ts_s, path_xyz[:, 1])
        vz = np.interp(self.gps_time, path_ts_s, path_xyz[:, 2])
        return np.sqrt(
            (self.xyz[:, 0] - vx) ** 2
            + (self.xyz[:, 1] - vy) ** 2
            + (self.xyz[:, 2] - vz) ** 2
        ).astype(np.float32)

    def crop_min_range(
        self,
        min_range_m: float,
        path_ts_s: np.ndarray | None = None,
        path_xyz: np.ndarray | None = None,
    ) -> "PointCloud":
        """Drop points captured closer than min_range_m to the scanner.

        Short-range returns are dominated by vehicle self-hits (hood, mounts)
        that form a floating carpet above the road along the trajectory.

        Args:
            min_range_m: Minimum capture range to keep, meters.
            path_ts_s: Vehicle path timestamps (for estimation; see
                capture_range).
            path_xyz: Vehicle path positions.

        Returns:
            Filtered cloud (unchanged clone when no range source exists).
        """
        ranges = self.capture_range(path_ts_s, path_xyz)
        if ranges is None:
            logger.warning(
                "No capture-range source (no range dim, gps_time, or path); "
                "min-range filter skipped"
            )
            return self
        return self._subset(ranges >= min_range_m)

    def crop_time(self, center_s: float, window_s: float) -> "PointCloud":
        """Subset points captured within a time window (one drive pass).

        The city is scanned by many passes days apart; centimeter-level
        drapes come from the points captured alongside the photo, not the
        merged multi-pass cloud.

        Args:
            center_s: Center of the window, GPS seconds since epoch.
            window_s: Half-width of the window, seconds.

        Returns:
            The single-pass subset (unchanged clone if gps_time is absent).
        """
        if self.gps_time is None:
            return self
        return self._subset(np.abs(self.gps_time - center_s) <= window_s)

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
        gps_time = (
            np.concatenate([c.gps_time for c in clouds])
            if all(c.gps_time is not None for c in clouds)
            else None
        )
        scan_range = (
            np.concatenate([c.scan_range for c in clouds])
            if all(c.scan_range is not None for c in clouds)
            else None
        )
        return PointCloud(
            xyz=np.vstack([c.xyz for c in clouds]),
            rgb=rgb,
            intensity=intensity,
            utm_epsg=clouds[0].utm_epsg,
            gps_time=gps_time,
            scan_range=scan_range,
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
    gps_time = (
        np.asarray(las.gps_time, dtype=np.float64)
        if "gps_time" in las.point_format.dimension_names
        else None
    )
    scan_range = (
        np.asarray(las["range"], dtype=np.float32)
        if "range" in las.point_format.dimension_names
        else None
    )
    return PointCloud(
        xyz=xyz,
        rgb=rgb,
        intensity=intensity,
        utm_epsg=utm_epsg,
        gps_time=gps_time,
        scan_range=scan_range,
    )


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
        self,
        x_utm: float,
        y_utm: float,
        radius_m: float = 30.0,
        *,
        time_window: tuple[float, float] | None = None,
    ) -> PointCloud:
        """Fetch and crop lidar points around a UTM location.

        Prefers COPC HTTP range reads (only the queried subset moves over
        the network); falls back to whole-tile LAZ download + disk cache.

        Args:
            x_utm: Easting.
            y_utm: Northing.
            radius_m: Horizontal radius, meters.
            time_window: Optional (center_s, half_width_s) GPS-time window.
                Each tile's points are cropped to this single drive pass
                immediately after reading — before tiles are concatenated —
                so a busy intersection scanned by many passes never
                materializes all of them at once (the merged all-pass cloud
                can be 4x the single-pass cloud and blow up memory).

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
            if time_window is not None:
                cloud = cloud.crop_time(*time_window)
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
            from cyvl.copc_flat import flat_spatial_query

            bounds = laspy.copc.Bounds(
                mins=np.array([x_min, y_min]), maxs=np.array([x_max, y_max])
            )
            with laspy.copc.CopcReader.open(copc_url) as reader:
                # flat_spatial_query handles Cyvl's flat hierarchy pages,
                # which stock CopcReader.query silently returns 0 points for.
                points = flat_spatial_query(reader, bounds)
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
        return cloud._subset(mask)
