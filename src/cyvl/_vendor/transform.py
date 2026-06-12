# Vendored from cyvl spatial monorepo:
#   packages/projection/src/projection/transform.py @ b9d7eb5757b20fb080a31528a19ef64a06382728
# Do not edit by hand — regenerate with scripts/vendor_sync.py.
import numpy as np


def apply_transform_to_xyz_points(
    xyz_points: np.ndarray,
    transform: np.ndarray,
) -> np.ndarray:
    """Apply a transform to a set of xyz points.

    Args:
        xyz_points: Nx3 array of xyz points
        transform: 4x4 transform matrix

    Returns:
        Nx3 array of transformed xyz points
    """
    # The formula for a transformation matrix is:
    #
    # y = Mx
    #
    # Where:
    # - x is the original point as a column vector
    # - y is the transformed point as a column vector
    # - M is the transformation matrix as a 4x4 matrix

    # Points are Nx3, so we need to pad with 1s and transpose to make 4xN
    ones = np.ones((xyz_points.shape[0], 1), dtype=xyz_points.dtype)
    points_h = np.hstack([xyz_points, ones]).T

    # Apply the transform: y = Mx
    transformed_xyz_points = transform @ points_h

    # Remove the last column, transpose, and return
    return transformed_xyz_points[:3, :].T


def H_to_R_and_T(transform: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Extract rotation and translation from a 4x4 transformation matrix.

    Args:
        transform: 4x4 transformation matrix

    Returns:
        tuple: (rotation matrix, translation vector)
    """
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    return rotation, translation
