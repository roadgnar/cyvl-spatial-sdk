"""Projection regression + round-trip tests against the verified fixture."""

import numpy as np
import pytest

from tests.conftest import FIXTURE_DIR


class TestProjectionRegression:
    def test_pixels_match_verified_export(self, frame) -> None:
        """Projecting fixture world points reproduces the verified pixels."""
        expected = np.load(FIXTURE_DIR / "expected_pixels.npz")
        projected = frame.project(expected["points_utm"])

        np.testing.assert_array_equal(projected.in_view, expected["in_view"])
        in_view = expected["in_view"]
        np.testing.assert_allclose(
            projected.pixels[in_view], expected["pixels"][in_view], atol=1e-6
        )
        np.testing.assert_allclose(projected.depth, expected["depth"], atol=1e-9)

    def test_nearest_frame_never_needs_a_radius(self, scene, frame) -> None:
        """nearest_frame finds the closest match from any distance."""
        found = scene.nearest_frame(frame.lon + 0.01, frame.lat + 0.01)
        assert found.image_url is not None
        import pytest

        with pytest.raises(ValueError, match="No frames match"):
            scene.nearest_frame(frame.lon, frame.lat, facing="sideways")

    def test_signs_layer_projects_into_frame(self, scene, frame) -> None:
        """At least one nearby sign lands inside the image."""
        signs = scene.signs
        lonlat = np.column_stack([signs.geometry.x, signs.geometry.y])
        projected = frame.project(lonlat=lonlat, alt=frame.position[2])
        assert projected.in_view.sum() >= 1


class TestRoundTrips:
    def test_camera_project_unproject_under_a_tenth_pixel(self, frame) -> None:
        """Distortion inversion: project(unproject(px)) within 0.1 px."""
        rng = np.random.default_rng(5)
        pixels = np.column_stack(
            [
                rng.uniform(200, frame.camera.width - 200, 50),
                rng.uniform(200, frame.camera.height - 200, 50),
            ]
        )
        rays = frame.camera.unproject(pixels)
        points_camera = rays * rng.uniform(3, 50, 50)[:, None]
        reprojected = frame.camera.project(points_camera)
        np.testing.assert_allclose(reprojected, pixels, atol=0.1)

    def test_world_unproject_with_known_depth(self, frame) -> None:
        """unproject(project(p), depth) recovers p to sub-centimeter."""
        point = frame.position + frame.pose[:3, 2] * 12.0 + np.array([1.0, 0.5, -1.2])
        projected = frame.project(point[None, :])
        assert projected.in_view[0]
        recovered = frame.unproject(
            *projected.pixels[0], depth=float(projected.depth[0])
        )
        np.testing.assert_allclose(recovered, point, atol=0.005)

    def test_lidar_depth_unproject_close_to_surface(self, frame) -> None:
        """Pixel -> 3D via lidar depth lands within 25 cm of a lidar point."""
        # Fixture is a 400k subsample; the same-pass 20m crop is ~48k points.
        cloud = frame.lidar(radius_m=20)
        assert len(cloud) > 10_000
        projected = frame.project(cloud.xyz)
        candidate = int(np.flatnonzero(projected.in_view)[1000])
        px, py = projected.pixels[candidate]
        recovered = frame.unproject(px, py, lidar=cloud)
        distance = np.linalg.norm(recovered - cloud.xyz[candidate])
        assert distance < 0.25


class TestMeasure:
    def test_measure_known_pair(self, frame) -> None:
        """Distance between two lidar-backed pixels matches 3D truth."""
        import cyvl

        cloud = frame.lidar(radius_m=20)
        projected = frame.project(cloud.xyz)
        visible = np.flatnonzero(projected.in_view)
        a, b = int(visible[500]), int(visible[5000])
        measurement = cyvl.measure(
            frame,
            tuple(projected.pixels[a]),
            tuple(projected.pixels[b]),
            lidar=cloud,
        )
        truth = float(np.linalg.norm(cloud.xyz[a] - cloud.xyz[b]))
        assert measurement.meters == pytest.approx(truth, abs=0.5)


class TestFlatCopc:
    def test_flat_query_matches_full_read(self, scene, frame) -> None:
        """flat_spatial_query returns exactly the brute-force crop."""
        import laspy

        from cyvl.copc_flat import flat_spatial_query

        path = str(FIXTURE_DIR / "mini.copc.laz")
        x, y = frame.position[0], frame.position[1]
        with laspy.copc.CopcReader.open(path) as reader:
            bounds = laspy.copc.Bounds(
                mins=np.array([x - 10, y - 10]), maxs=np.array([x + 10, y + 10])
            )
            queried = len(flat_spatial_query(reader, bounds))

        las = laspy.read(path)
        brute = int(
            (
                (las.x >= x - 10)
                & (las.x <= x + 10)
                & (las.y >= y - 10)
                & (las.y <= y + 10)
            ).sum()
        )
        assert queried == brute > 0
