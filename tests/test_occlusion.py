"""Occlusion-aware depth selection in Frame.unproject / cyvl.measure.

Fully synthetic and offline: a pinhole camera with identity orientation at a
UTM-sized position, and point clouds placed directly in its frame, so every
expected depth is exact. Query pixels sit between lidar samples, as real
clicks do. Each occluder test also checks that the plain nearest projection
(`occlusion_cell_px=None`, the previous behavior) gets the same queries
wrong, so the tests fail without the fix.
"""

import numpy as np
import pandas as pd
import pytest

import cyvl
from cyvl.camera import Camera
from cyvl.frame import Frame, _nearest_visible

ORIGIN = np.array([327000.0, 4694000.0, 10.0])
FX, CX, CY = 1840.0, 1920.0, 1080.0


@pytest.fixture()
def cam_frame() -> Frame:
    row = {
        "frame_id": "synthetic",
        "facing": "front",
        "lon": -71.1,
        "lat": 42.39,
        "image_url": None,
        "utm_epsg": 32619,
        "timestamp_us": 0,
        "adjusted_timestamp_us": 0.0,
    }
    pose = np.eye(4)
    pose[:3, 3] = ORIGIN  # camera axes = world axes: +x right, +y down, +z ahead
    for i in range(3):
        for j in range(4):
            row[f"pose_{i}{j}"] = pose[i, j]
    return Frame(pd.Series(row), Camera(fx=FX, fy=FX, cx=CX, cy=CY))


def _plane(x0, x1, y0, y1, step, z):
    gx, gy = np.meshgrid(np.arange(x0, x1 + 1e-9, step), np.arange(y0, y1 + 1e-9, step))
    return np.column_stack([gx.ravel(), gy.ravel(), np.full(gx.size, z)])


def _cloud(*parts):
    xyz = np.vstack(parts) + ORIGIN
    return cyvl.PointCloud(xyz=xyz, rgb=None, intensity=None, utm_epsg=32619)


def _pixels(frame, points_camera):
    return frame.project(np.asarray(points_camera, dtype=float) + ORIGIN).pixels


def _at_pixel(u, v, z):
    """Camera-frame point at depth z that projects to pixel (u, v)."""
    return [(u - CX) * z / FX, (v - CY) * z / FX, z]


def _ranges(frame, pixels, cloud, **kwargs):
    return np.array(
        [
            np.linalg.norm(frame.unproject(px, py, lidar=cloud, **kwargs) - ORIGIN)
            for px, py in pixels
        ]
    )


# A 1 m plate 8 m away sampled every 1 cm (~2.3 px) in front of a wall 20 m
# away sampled every 3 cm (~2.8 px): the wall points behind the plate are what
# a scanner driving past records and the camera cannot see.
PLATE = _plane(-0.5, 0.5, -0.5, 0.5, 0.01, 8.0)
WALL = _plane(-1.5, 1.5, -1.5, 1.5, 0.03, 20.0)


def _plate_queries(frame):
    return _pixels(frame, _plane(-0.395, 0.4, -0.395, 0.4, 0.1, 8.0))


class TestOccluders:
    def test_occluder_pixels_get_the_occluders_depth(self, cam_frame) -> None:
        cloud, queries = _cloud(PLATE, WALL), _plate_queries(cam_frame)
        assert np.any(_ranges(cam_frame, queries, cloud, occlusion_cell_px=None) > 19.0)
        assert _ranges(cam_frame, queries, cloud).max() < 8.1  # plate: 8.00-8.03 m

    def test_measure_on_an_occluder(self, cam_frame) -> None:
        """A 0.6 m span on the plate measures 0.6 m."""
        cloud = _cloud(PLATE, WALL)
        pa, pb = _pixels(cam_frame, [[-0.295, 0.005, 8.0], [0.305, 0.005, 8.0]])
        legacy = cyvl.measure(
            cam_frame, tuple(pa), tuple(pb), lidar=cloud, occlusion_cell_px=None
        )
        assert legacy.meters > 1.0
        m = cyvl.measure(cam_frame, tuple(pa), tuple(pb), lidar=cloud)
        assert m.meters == pytest.approx(0.6, abs=0.03)

    @pytest.mark.parametrize(
        "u_post", [1940.0, 1944.0]
    )  # centred in / straddling a cell
    def test_post_narrower_than_a_cell(self, cam_frame, u_post) -> None:
        """A 3 cm post (~7 px at 8 m) in front of the wall."""
        x0 = (u_post - CX) * 8.0 / FX
        post = _plane(x0 - 0.015, x0 + 0.015, -0.5, 0.5, 0.01, 8.0)
        queries = _pixels(
            cam_frame, [[x0 + 0.005, y, 8.0] for y in np.arange(-0.395, 0.4, 0.05)]
        )
        cloud = _cloud(post, WALL)
        assert np.any(_ranges(cam_frame, queries, cloud, occlusion_cell_px=None) > 19.0)
        assert _ranges(cam_frame, queries, cloud).max() < 8.1

    def test_sparse_occluder_is_never_worse(self, cam_frame) -> None:
        """Plate sampled every 5 cm (~11.5 px, sparser than a cell): partly fixed."""
        cloud = _cloud(_plane(-0.5, 0.5, -0.5, 0.5, 0.05, 8.0), WALL)
        queries = _plate_queries(cam_frame)
        wrong_old = np.sum(
            _ranges(cam_frame, queries, cloud, occlusion_cell_px=None) > 19.0
        )
        wrong_new = np.sum(_ranges(cam_frame, queries, cloud) > 19.0)
        assert wrong_old > 0 and wrong_new <= wrong_old


class TestUnoccludedSurfacesUnchanged:
    def _assert_same_as_plain(self, frame, cloud, queries) -> None:
        new = np.array([frame.unproject(px, py, lidar=cloud) for px, py in queries])
        old = np.array(
            [
                frame.unproject(px, py, lidar=cloud, occlusion_cell_px=None)
                for px, py in queries
            ]
        )
        np.testing.assert_array_equal(new, old)

    def test_facing_wall(self, cam_frame) -> None:
        queries = _pixels(
            cam_frame, _plane(-1.2, 1.2, -1.2, 1.2, 0.4, 20.0) + [0.015, 0.015, 0.0]
        )
        self._assert_same_as_plain(cam_frame, _cloud(WALL), queries)

    def test_grazing_road_to_40m(self, cam_frame) -> None:
        """Ground 2.1 m below the camera at 4-40 m (down to a 3 deg grazing angle)."""
        x, z = np.meshgrid(
            np.arange(-0.9, 0.9 + 1e-9, 0.03), np.arange(4.0, 40.0, 0.03)
        )
        road = np.column_stack([x.ravel(), np.full(x.size, 2.1), z.ravel()])
        qx, qz = np.meshgrid([-0.8, 0.011, 0.8], np.arange(5.0, 39.6, 4.3))
        queries = _pixels(
            cam_frame, np.column_stack([qx.ravel(), np.full(qx.size, 2.1), qz.ravel()])
        )
        self._assert_same_as_plain(cam_frame, _cloud(road), queries)

    def test_wall_beyond_one_cell_of_an_outline(self, cam_frame) -> None:
        """Wall pixels more than one cell from the plate's edge stay on the wall."""
        edge_u = CX + FX * 0.5 / 8.0  # right edge of the plate, 2035 px
        cloud = _cloud(PLATE, WALL)
        queries = [
            (edge_u + d, CY + 0.5 + dv)
            for d in (10.5, 13.5, 16.5)
            for dv in (-40, 0, 40)
        ]
        assert _ranges(cam_frame, queries, cloud).min() > 19.0

    def test_layers_within_the_margin_are_not_separated(self, cam_frame) -> None:
        """A plate 6% in front of a wall is one surface to the test, as before."""
        near_wall = _plane(-0.7, 0.7, -0.7, 0.7, 0.03, 8.5)
        self._assert_same_as_plain(
            cam_frame, _cloud(PLATE, near_wall), _plate_queries(cam_frame)
        )

    def test_single_stray_point_does_not_hide_a_surface(self, cam_frame) -> None:
        """One return in front of a wall (dust, an edge return) changes nothing."""
        stray = np.array(
            [_at_pixel(2000.3, 1100.3, 10.0)]
        )  # cell [2000,2008) x [1096,1104)
        wall = _plane(-1.0, 1.0, -0.5, 0.5, 0.03, 20.0)
        queries = [(2000.5 + i, 1096.5 + j) for i in range(8) for j in range(8)]
        self._assert_same_as_plain(cam_frame, _cloud(stray, wall), queries)


class TestEdgeCases:
    def test_window_with_only_hidden_points(self) -> None:
        """Nothing visible within 12 px of the query: no answer."""
        # Cell [1896, 1904) holds two near points 13+ px from the query at
        # x=1890 and a far point 7 px from it.
        pixels = np.array([[1903.0, 1080.0], [1903.5, 1080.0], [1897.0, 1080.0]])
        depth = np.array([8.0, 8.0, 20.0])
        distances = np.hypot(pixels[:, 0] - 1890.0, pixels[:, 1] - 1080.0)
        assert (
            _nearest_visible(pixels, depth, distances, 1890.0, 1080.0, 12.0, 8.0)
            is None
        )
        # With 4 px cells the far point has a cell of its own.
        assert (
            _nearest_visible(pixels, depth, distances, 1890.0, 1080.0, 12.0, 4.0) == 2
        )

    def test_only_hidden_points_raises_through_unproject(self, cam_frame) -> None:
        near = [_at_pixel(1903.0, CY, 8.0), _at_pixel(1903.5, CY, 8.0)]
        far = [_at_pixel(1897.0, CY, 20.0)]
        cloud = _cloud(np.array(near), np.array(far))
        with pytest.raises(ValueError, match="hidden behind a nearer surface"):
            cam_frame.unproject(1890.0, CY, lidar=cloud)
        legacy = cam_frame.unproject(1890.0, CY, lidar=cloud, occlusion_cell_px=None)
        np.testing.assert_allclose(
            np.linalg.norm(legacy - ORIGIN), np.linalg.norm(far[0])
        )

    @pytest.mark.parametrize("cell", [0.0, -8.0, float("nan"), float("inf")])
    def test_invalid_cell_size(self, cam_frame, cell) -> None:
        with pytest.raises(ValueError, match="occlusion_cell_px"):
            cam_frame.unproject(CX, CY, lidar=_cloud(WALL), occlusion_cell_px=cell)

    def test_non_finite_pixel(self, cam_frame) -> None:
        with pytest.raises(ValueError, match="not finite"):
            cam_frame.unproject(float("nan"), CY, lidar=_cloud(WALL))
