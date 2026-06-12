"""Coordinate helpers: cached UTM<->WGS84 transformers."""

from functools import lru_cache

import numpy as np
from pyproj import Transformer


@lru_cache(maxsize=8)
def utm_to_lonlat_transformer(utm_epsg: int) -> Transformer:
    """Cached transformer from a UTM zone to WGS84 lon/lat."""
    return Transformer.from_crs(f"EPSG:{utm_epsg}", "EPSG:4326", always_xy=True)


@lru_cache(maxsize=8)
def lonlat_to_utm_transformer(utm_epsg: int) -> Transformer:
    """Cached transformer from WGS84 lon/lat to a UTM zone."""
    return Transformer.from_crs("EPSG:4326", f"EPSG:{utm_epsg}", always_xy=True)


def lonlat_to_utm(
    lon: np.ndarray | float,
    lat: np.ndarray | float,
    alt: np.ndarray | float,
    utm_epsg: int,
) -> np.ndarray:
    """Convert lon/lat/alt arrays to an (N, 3) UTM xyz array.

    Args:
        lon: Longitude(s) in degrees.
        lat: Latitude(s) in degrees.
        alt: Altitude(s) in meters (passed through as z).
        utm_epsg: Target UTM EPSG code.

    Returns:
        (N, 3) float64 array of UTM easting/northing/altitude.
    """
    transformer = lonlat_to_utm_transformer(utm_epsg)
    x, y = transformer.transform(np.atleast_1d(lon), np.atleast_1d(lat))
    z = np.broadcast_to(np.atleast_1d(alt).astype(np.float64), np.shape(x))
    return np.column_stack([x, y, z])


def utm_to_lonlat(xyz: np.ndarray, utm_epsg: int) -> np.ndarray:
    """Convert an (N, 3) UTM xyz array to (N, 3) lon/lat/alt.

    Args:
        xyz: (N, 3) UTM coordinates.
        utm_epsg: Source UTM EPSG code.

    Returns:
        (N, 3) array of lon, lat, alt.
    """
    transformer = utm_to_lonlat_transformer(utm_epsg)
    lon, lat = transformer.transform(xyz[:, 0], xyz[:, 1])
    return np.column_stack([lon, lat, xyz[:, 2]])
