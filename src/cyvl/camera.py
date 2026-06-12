"""Pinhole camera model matching Cyvl's production projection pipeline."""

from dataclasses import dataclass, field

import cv2
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Camera:
    """Calibrated pinhole camera (OpenCV plumb_bob distortion).

    Axes follow the OpenCV convention: +X right, +Y down, +Z into the scene.
    """

    fx: float
    fy: float
    cx: float
    cy: float
    dist: np.ndarray = field(default_factory=lambda: np.zeros(5))
    width: int = 3840
    height: int = 2160

    @classmethod
    def from_row(cls, row: pd.Series) -> "Camera":
        """Build a Camera from a cameras.parquet row."""
        return cls(
            fx=float(row["fx"]),
            fy=float(row["fy"]),
            cx=float(row["cx"]),
            cy=float(row["cy"]),
            dist=np.array(
                [row["k1"], row["k2"], row["p1"], row["p2"], row["k3"]],
                dtype=np.float64,
            ),
            width=int(row["width"]),
            height=int(row["height"]),
        )

    @property
    def K(self) -> np.ndarray:
        """3x3 intrinsic matrix."""
        return np.array(
            [[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]],
            dtype=np.float64,
        )

    def project(self, points_camera: np.ndarray) -> np.ndarray:
        """Project camera-frame 3D points to distorted pixel coordinates.

        Identical math to the production pipeline's cv2.projectPoints call
        (projection.projection.create_point_cloud_data); the rigid transform
        into the camera frame happens before this call, so rvec/tvec are zero.

        Args:
            points_camera: (N, 3) points in the camera frame, z forward.

        Returns:
            (N, 2) float64 pixel coordinates.
        """
        if len(points_camera) == 0:
            return np.empty((0, 2), dtype=np.float64)
        pixels, _ = cv2.projectPoints(
            objectPoints=points_camera.astype(np.float64),
            rvec=np.zeros(3),
            tvec=np.zeros(3),
            cameraMatrix=self.K,
            distCoeffs=self.dist,
        )
        return pixels.reshape(-1, 2)

    def unproject(self, pixels: np.ndarray) -> np.ndarray:
        """Convert pixel coordinates to unit view rays in the camera frame.

        Uses cv2.undistortPoints (iterative distortion inversion), the
        standard inverse of the projection above.

        Args:
            pixels: (N, 2) pixel coordinates.

        Returns:
            (N, 3) unit direction vectors in the camera frame.
        """
        pixels = np.asarray(pixels, dtype=np.float64).reshape(-1, 1, 2)
        normalized = cv2.undistortPoints(pixels, self.K, self.dist).reshape(-1, 2)
        rays = np.column_stack([normalized, np.ones(len(normalized))])
        return rays / np.linalg.norm(rays, axis=1, keepdims=True)

    def in_image(self, pixels: np.ndarray, margin: float = 0.0) -> np.ndarray:
        """Boolean mask of pixels inside the image bounds.

        Args:
            pixels: (N, 2) pixel coordinates.
            margin: Optional inset in pixels.

        Returns:
            (N,) boolean array.
        """
        x, y = pixels[:, 0], pixels[:, 1]
        return (
            (x >= margin)
            & (x < self.width - margin)
            & (y >= margin)
            & (y < self.height - margin)
        )
