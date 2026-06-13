"""SAM3 segmentation over Cyvl imagery — text prompt in, masks (and 3D) out.

Powered by fal.ai's hosted SAM3 endpoint. Needs an API key:

1. Create one at https://fal.ai/dashboard/keys
2. export FAL_KEY="key-id:key-secret"   (or pass api_key= explicitly)

Install the extra:  pip install "cyvl[sam]"

    import cyvl
    from cyvl.segment import segment, locate

    scene = cyvl.load_scene("somerville")
    frame = scene.nearest_frame(-71.0944, 42.3865)

    masks = segment(frame, "traffic sign")          # pixel masks
    hits = locate(frame, "traffic sign")            # masks lifted to 3D
    print(hits[0].lonlat, hits[0].distance_m)
"""

import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from cyvl.frame import Frame

logger = logging.getLogger(__name__)

SIGNUP_URL = "https://fal.ai/dashboard/keys"
ENDPOINT = "fal-ai/sam-3/image-rle"


def _resolve_api_key(api_key: str | None) -> str:
    """Find the fal.ai key: explicit argument, else the FAL_KEY env var.

    Raises:
        RuntimeError: With setup instructions when no key is found.
    """
    if api_key:
        return api_key
    if os.environ.get("FAL_KEY"):
        return os.environ["FAL_KEY"]
    raise RuntimeError(
        f"No fal.ai API key found. Create one at {SIGNUP_URL}, then "
        "`export FAL_KEY=...` before launching (or pass api_key=...)."
    )


def decode_rle(rle_data: "list[int] | str", height: int, width: int) -> np.ndarray:
    """Decode an RLE mask (absolute-start, run-length pairs, row-major).

    Args:
        rle_data: Whitespace-separated string or flat list of int pairs.
        height: Image height in pixels.
        width: Image width in pixels.

    Returns:
        Boolean mask of shape (height, width).
    """
    if isinstance(rle_data, str):
        rle_data = list(map(int, rle_data.split()))
    total = height * width
    flat = np.zeros(total, dtype=bool)
    for i in range(0, len(rle_data) - 1, 2):
        start, length = rle_data[i], rle_data[i + 1]
        if start < total:
            flat[start : min(start + length, total)] = True
    return flat.reshape((height, width))


@dataclass
class Segmentation:
    """SAM3 result: masks aligned to the queried image."""

    masks: np.ndarray  # (N, H, W) bool
    scores: list[float]
    boxes: list[list[float]]
    prompt: str

    def __len__(self) -> int:
        return len(self.masks)

    def overlay(self, image: np.ndarray, opacity: float = 0.45) -> np.ndarray:
        """Burn colored masks onto an RGB image (returns a copy)."""
        colors = [(230, 25, 75), (60, 180, 75), (0, 130, 200), (255, 225, 25)]
        out = image.astype(np.float64)
        for i, mask in enumerate(self.masks):
            color = np.array(colors[i % len(colors)], dtype=np.float64)
            region = mask[..., np.newaxis]
            out = np.where(region, out * (1 - opacity) + color * opacity, out)
        return out.astype(np.uint8)


def segment(
    image: "Frame | np.ndarray | str | Path",
    prompt: str,
    *,
    max_masks: int = 3,
    api_key: str | None = None,
) -> Segmentation:
    """Run SAM3 text-prompted segmentation on a frame or image.

    Args:
        image: A cyvl Frame, an RGB ndarray, or an image path.
        prompt: What to segment, in plain text ("traffic sign", "pothole").
        max_masks: Maximum masks to return (1-32).
        api_key: fal.ai key; defaults to the FAL_KEY env var.

    Returns:
        Segmentation with (N, H, W) boolean masks, scores, and boxes.
    """
    os.environ["FAL_KEY"] = _resolve_api_key(api_key)
    import fal_client  # lazy: requires the [sam] extra

    # Normalize input to a local image file for upload.
    cleanup = None
    if hasattr(image, "image_url"):  # Frame
        from cyvl.cache import cached_download

        path = str(cached_download(image.image_url, progress=False))
        height, width = image.camera.height, image.camera.width
    elif isinstance(image, (str, Path)):
        import cv2

        decoded = cv2.imread(str(image))
        if decoded is None:
            raise ValueError(f"Could not read image at {image}")
        height, width = decoded.shape[:2]
        path = str(image)
    else:
        import cv2

        height, width = image.shape[:2]
        tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        cv2.imwrite(tmp.name, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        path = tmp.name
        cleanup = tmp.name

    try:
        image_url = fal_client.upload_file(path)
        result = fal_client.subscribe(
            ENDPOINT,
            arguments={
                "image_url": image_url,
                "prompt": prompt,
                "return_multiple_masks": max_masks > 1,
                "max_masks": max_masks,
                "include_scores": True,
                "include_boxes": True,
            },
        )
    finally:
        if cleanup:
            os.unlink(cleanup)

    rle_data = result.get("rle", [])
    if not isinstance(rle_data, list):
        rle_data = [rle_data]
    masks = []
    for entry in rle_data:
        if isinstance(entry, dict):  # COCO-style {"counts": ..., "size": [h, w]}
            size = entry.get("size", [height, width])
            masks.append(decode_rle(entry.get("counts", []), size[0], size[1]))
        else:
            masks.append(decode_rle(entry, height, width))

    return Segmentation(
        masks=np.array(masks) if masks else np.zeros((0, height, width), dtype=bool),
        scores=list(result.get("scores", [])),
        boxes=list(result.get("boxes", [])),
        prompt=prompt,
    )


@dataclass
class Located:
    """A segmented object lifted to 3D via the frame's LiDAR."""

    prompt: str
    score: float
    mask: np.ndarray  # (H, W) bool
    xyz_utm: np.ndarray  # (3,) object position (median of masked lidar)
    lonlat: tuple[float, float]
    distance_m: float
    n_points: int


def locate(
    frame: "Frame",
    prompt: str,
    *,
    max_masks: int = 3,
    radius_m: float = 30.0,
    min_points: int = 25,
    api_key: str | None = None,
) -> list[Located]:
    """Text prompt -> 3D world coordinates, via SAM3 + LiDAR.

    Segments the frame with SAM3, selects the same-pass lidar points whose
    projections fall inside each mask, and reports the median 3D position —
    the same mask-to-coordinate idea Cyvl's production geolocation uses.

    Args:
        frame: A posed cyvl Frame (with a public image and own-pass lidar).
        prompt: What to find ("traffic sign", "fire hydrant", "bench").
        max_masks: Maximum objects to return.
        radius_m: LiDAR fetch radius around the camera.
        min_points: Minimum masked lidar points for a confident result.
        api_key: fal.ai key; defaults to the FAL_KEY env var.

    Returns:
        Located objects (may be fewer than the mask count when masks have
        too little lidar support), sorted by score.
    """
    from cyvl.geometry import utm_to_lonlat

    result = segment(frame, prompt, max_masks=max_masks, api_key=api_key)
    if not len(result):
        return []

    projected = frame.points_in_view(radius_m=radius_m)
    visible = np.flatnonzero(projected.in_view)
    pixel_u = projected.pixels[visible, 0].astype(int)
    pixel_v = projected.pixels[visible, 1].astype(int)

    located = []
    for i, mask in enumerate(result.masks):
        inside = mask[
            np.clip(pixel_v, 0, mask.shape[0] - 1),
            np.clip(pixel_u, 0, mask.shape[1] - 1),
        ]
        indices = visible[inside]
        if len(indices) < min_points:
            logger.warning(
                "Mask %d (%s): only %d lidar points inside — skipping",
                i,
                prompt,
                len(indices),
            )
            continue
        # Median is robust to mask bleed onto background points.
        xyz = np.median(projected.points_utm[indices], axis=0)
        lon, lat, _ = utm_to_lonlat(xyz[None, :], frame.utm_epsg)[0]
        located.append(
            Located(
                prompt=prompt,
                score=float(result.scores[i]) if i < len(result.scores) else 0.0,
                mask=mask,
                xyz_utm=xyz,
                lonlat=(float(lon), float(lat)),
                distance_m=float(np.linalg.norm(xyz - frame.position)),
                n_points=int(len(indices)),
            )
        )
    return sorted(located, key=lambda located_obj: -located_obj.score)
