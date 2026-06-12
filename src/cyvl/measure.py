"""Real-world measurement between pixels of a frame."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from cyvl.frame import Frame
    from cyvl.lidar import PointCloud


@dataclass(frozen=True)
class Measurement:
    """A 3D measurement made between two pixels of one frame."""

    meters: float
    point_a: np.ndarray  # (3,) UTM
    point_b: np.ndarray  # (3,) UTM

    @property
    def feet(self) -> float:
        return self.meters * 3.28084

    def __repr__(self) -> str:
        return f"Measurement({self.meters:.2f} m / {self.feet:.2f} ft)"


def measure(
    frame: "Frame",
    px_a: tuple[float, float],
    px_b: tuple[float, float],
    *,
    lidar: "PointCloud | None" = None,
    pixel_tolerance: float = 12.0,
) -> Measurement:
    """Measure the real-world distance between two pixels of a frame.

    Both pixels are lifted to 3D using lidar depth (the point cloud is
    fetched once and reused for both endpoints).

    Args:
        frame: The frame measured in.
        px_a: First pixel (x, y).
        px_b: Second pixel (x, y).
        lidar: Optional pre-fetched point cloud to source depth from.
        pixel_tolerance: Max pixel distance for lidar depth matching.

    Returns:
        Measurement with distance and both 3D endpoints.
    """
    if lidar is None:
        lidar = frame.lidar()
    point_a = frame.unproject(*px_a, lidar=lidar, pixel_tolerance=pixel_tolerance)
    point_b = frame.unproject(*px_b, lidar=lidar, pixel_tolerance=pixel_tolerance)
    return Measurement(
        meters=float(np.linalg.norm(point_a - point_b)),
        point_a=point_a,
        point_b=point_b,
    )
