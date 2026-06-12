"""ParquetBackend: serves a Scene from static parquet artifacts.

Artifacts can live in a local directory (offline / test fixtures) or be
fetched lazily from the public CDN into the local cache.
"""

import logging
from pathlib import Path

import geopandas as gpd
import pandas as pd

from cyvl.download import LAYER_NAMES, SCENES, SceneSource, ensure_artifact

logger = logging.getLogger(__name__)


class ParquetBackend:
    """DataBackend implementation over exported parquet artifacts."""

    def __init__(
        self, root: Path | None = None, source: SceneSource | None = None
    ) -> None:
        """Create a backend from a local directory and/or a remote source.

        Args:
            root: Directory containing parquet/ artifacts. When omitted, the
                scene cache directory is used and artifacts are fetched from
                `source` on first access.
            source: Remote scene source for lazy fetching. Omit for purely
                local/offline use.

        Raises:
            ValueError: If neither root nor source is given.
        """
        if root is None and source is None:
            raise ValueError("ParquetBackend needs a local root, a source, or both")
        self._root = root
        self._source = source
        self._frames: pd.DataFrame | None = None
        self._cameras: pd.DataFrame | None = None
        self._tiles: gpd.GeoDataFrame | None = None
        self._layers: dict[str, gpd.GeoDataFrame] = {}

    @classmethod
    def for_scene(cls, name: str) -> "ParquetBackend":
        """Backend for a named public scene (artifacts fetched on demand)."""
        try:
            source = SCENES[name]
        except KeyError as err:
            raise ValueError(
                f"Unknown scene '{name}'. Available: {sorted(SCENES)}"
            ) from err
        return cls(source=source)

    def _path(self, relative: str) -> Path:
        """Resolve an artifact path locally, fetching from CDN if needed."""
        if self._root is not None:
            local = self._root / relative
            if local.exists():
                return local
            if self._source is None:
                raise FileNotFoundError(f"{local} not found (offline backend)")
        assert self._source is not None
        return ensure_artifact(self._source, relative)

    def frames(self) -> pd.DataFrame:
        if self._frames is None:
            self._frames = pd.read_parquet(self._path("parquet/frames.parquet"))
        return self._frames

    def cameras(self) -> pd.DataFrame:
        if self._cameras is None:
            self._cameras = pd.read_parquet(self._path("parquet/cameras.parquet"))
        return self._cameras

    def layer(self, name: str) -> gpd.GeoDataFrame:
        if name not in self._layers:
            self._layers[name] = gpd.read_parquet(
                self._path(f"parquet/layers/{name}.parquet")
            )
        return self._layers[name]

    def layer_names(self) -> list[str]:
        return list(LAYER_NAMES)

    def tiles(self) -> gpd.GeoDataFrame:
        if self._tiles is None:
            self._tiles = gpd.read_parquet(self._path("parquet/tiles.parquet"))
        return self._tiles
