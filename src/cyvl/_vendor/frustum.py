# Vendored from cyvl spatial monorepo:
#   packages/projection/src/projection/projection.py (create_xyz_point_mask)
#   @ b9d7eb5757b20fb080a31528a19ef64a06382728
# Do not edit by hand — regenerate with scripts/vendor_sync.py.
import numpy as np


def create_xyz_point_mask(
    xyz_points: np.ndarray,
) -> np.ndarray:
    """Create a mask for the xyz points."""
    # Split xyz points into x, y, z for easier filtering
    x, y, z = xyz_points[:, 0], xyz_points[:, 1], xyz_points[:, 2]

    # Create mask for points in front of the camera (z > 0)
    in_front_mask = z > 0

    # Calculate angles in degrees for points
    vertical_angles = np.abs(np.degrees(np.arctan2(y, z)))
    horizontal_angles = np.abs(np.degrees(np.arctan2(x, z)))

    # Create masks for FOV constraints
    in_vertical_fov_mask = vertical_angles < 80
    in_horizontal_fov_mask = horizontal_angles < 120

    # Combine all masks and return
    return in_front_mask & in_vertical_fov_mask & in_horizontal_fov_mask
