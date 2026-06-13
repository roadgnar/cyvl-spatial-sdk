"""Visualization helpers (requires the [viz] extra)."""

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from cyvl.frame import Frame, Projected
    from cyvl.scene import Scene


def show_frame(
    frame: "Frame", overlay: "Projected | None" = None, max_points: int = 200_000
):
    """Show a frame image with optional projected-point overlay.

    Points are colored by their RGB when available, else by depth.

    Args:
        frame: Frame to display.
        overlay: Projected points to scatter.
        max_points: Cap on scattered points.

    Returns:
        The matplotlib Axes.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as err:
        raise ImportError("Install the viz extra: pip install 'cyvl[viz]'") from err

    _, ax = plt.subplots(figsize=(14, 8))
    ax.imshow(frame.image())
    ax.set_title(frame.id)
    ax.axis("off")

    if overlay is not None:
        visible = np.flatnonzero(overlay.in_view)
        if visible.size > max_points:
            # Screen-space-uniform subsample: perspective spreads near points
            # over ~1/depth^2 more pixels, so a uniform 3D sample looks
            # artificially sparse in the foreground. Weight by 1/depth^2 to
            # keep perceived density even across the frame.
            weights = 1.0 / np.maximum(overlay.depth[visible], 2.0) ** 2
            weights /= weights.sum()
            visible = np.random.default_rng(0).choice(
                visible, max_points, replace=False, p=weights
            )
        pixels = overlay.pixels[visible]
        if overlay.intensity is not None:
            # Intensity ramp (percentile-normalized): structure pops far
            # better than raw RGB — markings, signs, and retroreflectors
            # read clearly against pavement.
            values = overlay.intensity[visible].astype(float)
            lo, hi = np.percentile(values, [2, 98])
            scatter = ax.scatter(
                pixels[:, 0],
                pixels[:, 1],
                c=np.clip((values - lo) / max(hi - lo, 1e-9), 0, 1),
                cmap="turbo",
                s=1.5,
                alpha=0.85,
            )
            plt.colorbar(scatter, ax=ax, label="intensity (normalized)", shrink=0.7)
        elif overlay.colors is not None:
            colors = overlay.colors[visible] / 255.0
            ax.scatter(pixels[:, 0], pixels[:, 1], c=colors, s=1.5, alpha=0.8)
        else:
            scatter = ax.scatter(
                pixels[:, 0],
                pixels[:, 1],
                c=overlay.depth[visible],
                cmap="turbo",
                s=1.5,
                alpha=0.8,
            )
            plt.colorbar(scatter, ax=ax, label="depth (m)", shrink=0.7)
    return ax


def scene_map(scene: "Scene", layers: tuple[str, ...] = ("rollup", "signs")):
    """Build a quick folium map of scene layers.

    Args:
        scene: Scene to map.
        layers: Scene layer accessor names.

    Returns:
        A folium.Map centered on the data.
    """
    try:
        import folium
    except ImportError as err:
        raise ImportError("Install the viz extra: pip install 'cyvl[viz]'") from err

    center = [scene.frames["lat"].mean(), scene.frames["lon"].mean()]
    fmap = folium.Map(location=center, zoom_start=14, tiles="cartodbpositron")
    for name in layers:
        gdf = getattr(scene, name)
        # Sample large layers so the map stays responsive in a notebook.
        if len(gdf) > 5000:
            gdf = gdf.sample(5000, random_state=0)
        # Point layers as small circle markers — default Leaflet pins are
        # huge and thousands of them bury every other layer.
        is_points = gdf.geometry.geom_type.isin(["Point"]).all()
        marker = (
            folium.CircleMarker(
                radius=3, weight=0, fill=True, fill_opacity=0.7, color="#c0392b"
            )
            if is_points
            else None
        )
        folium.GeoJson(gdf.to_json(), name=name, marker=marker).add_to(fmap)
    folium.LayerControl().add_to(fmap)
    return fmap
