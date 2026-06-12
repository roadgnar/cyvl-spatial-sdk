"""Shared fixtures: a fully offline mini scene built from the real export.

The fixture data was produced by the hackathon-export validation step from
job 1185 (Somerville) and verified against Cyvl's production geolocation
path (pose parity < 1e-4 m / 1e-5 deg / 1e-3 px).
"""

from pathlib import Path

import pytest

import cyvl
from cyvl.backends import ParquetBackend

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "mini_scene"
FRAME_ID = "GX010763_MP4_1763411255426418_front_3000"


@pytest.fixture()
def scene() -> cyvl.Scene:
    scene = cyvl.load_scene(backend=ParquetBackend(root=FIXTURE_DIR))
    # The fixture tile index stores a repo-relative COPC path; resolve it.
    tiles = scene.tiles
    for column in ("laz_url", "copc_url"):
        tiles[column] = str(FIXTURE_DIR / "mini.copc.laz")
    return scene


@pytest.fixture()
def frame(scene: cyvl.Scene) -> cyvl.Frame:
    return scene.frame(FRAME_ID)
