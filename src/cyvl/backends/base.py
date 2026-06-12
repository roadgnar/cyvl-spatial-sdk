"""The data-backend seam the rest of the SDK is written against.

The hackathon ships ParquetBackend (static artifacts over CDN/S3/local disk).
The same protocol can later be served live from Cyvl's spatial-core database
without touching Scene/Frame or any notebook built on them.
"""

from typing import Protocol, runtime_checkable

import geopandas as gpd
import pandas as pd


@runtime_checkable
class DataBackend(Protocol):
    """Everything a Scene needs from a data source — and nothing more."""

    def frames(self) -> pd.DataFrame:
        """Per-frame table: id, facing, lon/lat, camera_to_utm pose columns
        (pose_00..pose_23), utm_epsg, intrinsics_id, image_url."""
        ...

    def cameras(self) -> pd.DataFrame:
        """Per-(dataset, facing) intrinsics: fx fy cx cy k1 k2 p1 p2 k3 w h."""
        ...

    def layer(self, name: str) -> gpd.GeoDataFrame:
        """A named product layer (signs, pavements, distresses, ...)."""
        ...

    def layer_names(self) -> list[str]:
        """Available layer names."""
        ...

    def tiles(self) -> gpd.GeoDataFrame:
        """Point-cloud tile index: footprints + laz/copc URLs."""
        ...
