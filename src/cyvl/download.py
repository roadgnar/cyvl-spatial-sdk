"""Scene artifact discovery and download (CDN-first, no credentials needed)."""

import logging
from dataclasses import dataclass
from pathlib import Path

from cyvl.cache import cache_dir, cached_download

logger = logging.getLogger(__name__)

CDN_DOMAIN = "https://dcygqrjfsypox.cloudfront.net"

# sha256("scan_1185_SOMERVILLE_MA") — the scan's public delivery prefix.
SOMERVILLE_PREFIX = "0e9ad4538ce0a1b7876a5bcfd6275a942b2f92472dfa4b85e93f424eb4a62eda"

LAYER_NAMES = [
    "rollup",
    "pavements",
    "distresses",
    "distressInspectionCells",
    "signs",
    "sam",
    "aboveGroundAssets",
    "panoramicImagery",
    "plainImagery",
    "streetviewImagePaths",
]


@dataclass(frozen=True)
class SceneSource:
    """Where a named scene's parquet artifacts live."""

    name: str
    base_url: str

    def artifact_url(self, relative: str) -> str:
        return f"{self.base_url}/{relative}"


SCENES = {
    "somerville": SceneSource(
        name="somerville",
        base_url=f"{CDN_DOMAIN}/{SOMERVILLE_PREFIX}/hackathon",
    ),
}


def scene_data_dir(name: str) -> Path:
    """Local directory holding a scene's downloaded parquet artifacts."""
    return cache_dir() / "scenes" / name


def ensure_artifact(source: SceneSource, relative: str) -> Path:
    """Download one scene artifact into the scene cache if missing.

    Args:
        source: Scene source descriptor.
        relative: Artifact path relative to the scene base (e.g.
            "parquet/frames.parquet").

    Returns:
        Local path to the artifact.
    """
    destination = scene_data_dir(source.name) / relative
    if destination.exists():
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    downloaded = cached_download(source.artifact_url(relative))
    destination.symlink_to(downloaded)
    logger.info("Fetched %s -> %s", relative, destination)
    return destination
