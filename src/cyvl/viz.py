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
            visible = np.random.default_rng(0).choice(
                visible, max_points, replace=False
            )
        pixels = overlay.pixels[visible]
        if overlay.colors is not None:
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
        folium.GeoJson(gdf.to_json(), name=name).add_to(fmap)
    folium.LayerControl().add_to(fmap)
    return fmap
