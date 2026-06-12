"""Math vendored verbatim from Cyvl's spatial monorepo.

These functions are byte-for-byte the production pipeline's projection math,
so SDK results match Cyvl's internal geolocation outputs exactly. Regenerate
with scripts/vendor_sync.py against a spatial checkout.
"""

from cyvl._vendor.frustum import create_xyz_point_mask
from cyvl._vendor.transform import H_to_R_and_T, apply_transform_to_xyz_points

__all__ = [
    "H_to_R_and_T",
    "apply_transform_to_xyz_points",
    "create_xyz_point_mask",
]
