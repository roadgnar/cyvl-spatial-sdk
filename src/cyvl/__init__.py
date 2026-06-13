"""cyvl — build on Cyvl's physical-world data.

Street imagery, LiDAR, and infrastructure layers with full 3D<->2D
projection, powered by the same math as Cyvl's production pipeline.

Quick start:
    >>> import cyvl
    >>> scene = cyvl.load_scene("somerville")
    >>> frame = scene.nearest_frame(-71.0995, 42.3876)
    >>> pts = frame.points_in_view()       # lidar projected into the photo
    >>> frame.show(overlay=pts)
"""

from cyvl.camera import Camera
from cyvl.frame import Frame, Projected
from cyvl.lidar import PointCloud
from cyvl.measure import Measurement, measure
from cyvl.scene import Scene, load_scene

__version__ = "0.1.0"

__all__ = [
    "Camera",
    "Frame",
    "Measurement",
    "PointCloud",
    "Projected",
    "Scene",
    "load_scene",
    "measure",
]
