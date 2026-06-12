"""Hierarchy-agnostic COPC spatial queries.

Cyvl's production COPC tiles store a flat hierarchy page: every entry is a
depth-6 leaf, with no D0-0-0-0 root node. laspy's CopcReader.query traverses
the octree from the root key, finds nothing, and returns zero points. PDAL
tolerates the layout, which is how it went unnoticed.

This module selects nodes by intersecting EVERY hierarchy entry's voxel cube
with the query bounds — no traversal, no assumption about which levels exist.
It works identically for flat Cyvl tiles and spec-proper octrees, and reuses
laspy's grouped range reads + lazrs decompression for the heavy lifting.
"""

import numpy as np
from laspy.copc import Bounds, CopcReader, HierarchyPage, OctreeNode, VoxelKey


def voxel_cube(key: VoxelKey, copc_info) -> tuple[np.ndarray, np.ndarray]:
    """Compute the world-space cube of a COPC voxel key.

    The COPC root cube is centered on copc_info.center with half-size
    copc_info.halfsize; level L splits it into 2^L cubes per axis.

    Args:
        key: Voxel key (level, x, y, z).
        copc_info: The file's CopcInfo VLR.

    Returns:
        (mins, maxs) arrays of the voxel cube.
    """
    center = np.asarray(copc_info.center, dtype=np.float64)
    halfsize = float(copc_info.halfsize)
    size = (2.0 * halfsize) / (2.0**key.level)
    root_min = center - halfsize
    index = np.array([key.x, key.y, key.z], dtype=np.float64)
    mins = root_min + index * size
    return mins, mins + size


def _overlaps(a_mins, a_maxs, b_mins, b_maxs) -> bool:
    return bool(np.all(a_mins <= b_maxs) and np.all(b_mins <= a_maxs))


def _collect_entries(reader: CopcReader) -> dict:
    """All point-bearing entries, following nested hierarchy pages if any.

    Entries with point_count < 0 reference child hierarchy pages; Cyvl's
    flat tiles have none, but spec-proper files may — fetch and merge them
    so selection always sees the full node set.
    """
    entries = dict(reader.root_page.entries)
    pending = [e for e in entries.values() if e.point_count < 0]
    while pending:
        page_entry = pending.pop()
        del entries[page_entry.key]
        reader.source.seek(page_entry.offset)
        page = HierarchyPage.from_bytes(reader.source.read(page_entry.byte_size))
        for key, entry in page.entries.items():
            entries[key] = entry
            if entry.point_count < 0:
                pending.append(entry)
    return entries


def flat_spatial_query(reader: CopcReader, bounds: Bounds):
    """Spatial query that works regardless of hierarchy shape.

    Args:
        reader: An open laspy CopcReader (local file or HTTP source).
        bounds: 2D or 3D query bounds in file coordinates.

    Returns:
        laspy ScaleAwarePointRecord of points inside the bounds — same
        return type and exact-filter semantics as CopcReader.query.
    """
    bounds = bounds.ensure_3d(reader.header.mins, reader.header.maxs)

    nodes = []
    for key, entry in _collect_entries(reader).items():
        if entry.point_count <= 0:
            continue
        mins, maxs = voxel_cube(key, reader.copc_info)
        if not _overlaps(mins, maxs, bounds.mins, bounds.maxs):
            continue
        node = OctreeNode()
        node.key = key
        node.bounds = Bounds(mins, maxs)
        node.offset = entry.offset
        node.byte_size = entry.byte_size
        node.point_count = entry.point_count
        nodes.append(node)

    points = reader._fetch_and_decompress_points_of_nodes(nodes)

    # Exact in-bounds filter on integer coordinates (same as laspy.query).
    mins_i = np.round(
        (bounds.mins - reader.header.offsets) / reader.header.scales
    ).astype(np.int64)
    maxs_i = np.round(
        (bounds.maxs - reader.header.offsets) / reader.header.scales
    ).astype(np.int64)
    keep = (
        (mins_i[0] <= points.X)
        & (points.X <= maxs_i[0])
        & (mins_i[1] <= points.Y)
        & (points.Y <= maxs_i[1])
        & (mins_i[2] <= points.Z)
        & (points.Z <= maxs_i[2])
    )
    points.array = points.array[keep].copy()
    return points
