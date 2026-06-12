"""Local disk cache for downloaded artifacts (parquet, images, lidar crops)."""

import hashlib
import os
import tempfile
from pathlib import Path

import requests
from tqdm import tqdm


def cache_dir() -> Path:
    """Resolve the cache root (override with CYVL_CACHE_DIR)."""
    root = os.environ.get("CYVL_CACHE_DIR")
    if root:
        return Path(root)
    return Path.home() / ".cache" / "cyvl"


def cached_path(url: str, suffix: str = "") -> Path:
    """Deterministic cache location for a URL."""
    key = hashlib.sha256(url.encode()).hexdigest()[:24]
    name = Path(url.split("?")[0]).name or "artifact"
    return cache_dir() / "downloads" / f"{key}_{name}{suffix}"


def cached_download(url: str, *, progress: bool = True, timeout: int = 120) -> Path:
    """Download a URL once; subsequent calls return the cached file.

    Args:
        url: HTTP(S) URL to fetch.
        progress: Show a tqdm progress bar for large files.
        timeout: Per-request timeout in seconds.

    Returns:
        Path to the cached file.

    Raises:
        requests.HTTPError: On a non-2xx response.
    """
    destination = cached_path(url)
    if destination.exists():
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)

    response = requests.get(url, stream=True, timeout=timeout)
    response.raise_for_status()
    total = int(response.headers.get("content-length", 0))

    # Write to a temp file in the same directory, then atomically rename, so
    # a interrupted download never leaves a truncated cache entry behind.
    with tempfile.NamedTemporaryFile(
        dir=destination.parent, delete=False, suffix=".part"
    ) as tmp:
        with tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            desc=destination.name,
            disable=not progress or total < 5_000_000,
        ) as bar:
            for chunk in response.iter_content(chunk_size=1 << 20):
                tmp.write(chunk)
                bar.update(len(chunk))
        tmp_path = Path(tmp.name)
    tmp_path.rename(destination)
    return destination
