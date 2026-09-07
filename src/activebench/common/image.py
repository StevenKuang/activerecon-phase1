"""Small image writing helpers without optional image dependencies."""

import struct
import tempfile
import zlib
from pathlib import Path
from typing import Optional

import numpy as np

from .io import ensure_dir


def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    body = chunk_type + data
    checksum = zlib.crc32(body) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + body + struct.pack(">I", checksum)


def save_rgb_png(path: Path, image: np.ndarray) -> Path:
    """Write an RGB/RGBA uint8 image as PNG."""

    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] not in (3, 4):
        raise ValueError("image must have shape HxWx3 or HxWx4")
    rgb = array[:, :, :3]
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return _write_png_rgb(path, rgb)


def depth_to_grayscale(depth: np.ndarray, max_depth: Optional[float] = None) -> np.ndarray:
    """Convert a depth image to uint8 grayscale for visualization."""

    depth_array = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(depth_array) & (depth_array > 0.0)
    gray = np.zeros(depth_array.shape, dtype=np.uint8)
    if not np.any(valid):
        return gray
    if max_depth is None:
        max_depth = float(np.percentile(depth_array[valid], 95.0))
    scale = max(float(max_depth), 1e-6)
    normalized = np.clip(depth_array / scale, 0.0, 1.0)
    gray[valid] = ((1.0 - normalized[valid]) * 255.0).astype(np.uint8)
    return gray


def scalar_to_colormap(values: np.ndarray) -> np.ndarray:
    """Map scalar values in [0, 1] to a readable false-color RGB image."""

    clipped = np.clip(np.asarray(values, dtype=np.float32), 0.0, 1.0)
    anchors = np.asarray(
        [
            [35, 23, 74],
            [42, 117, 142],
            [38, 173, 129],
            [126, 211, 79],
            [253, 231, 37],
            [220, 60, 45],
        ],
        dtype=np.float32,
    )
    scaled = clipped * float(len(anchors) - 1)
    lower = np.floor(scaled).astype(np.int64)
    upper = np.clip(lower + 1, 0, len(anchors) - 1)
    alpha = scaled - lower.astype(np.float32)
    rgb = (1.0 - alpha[..., None]) * anchors[lower] + alpha[..., None] * anchors[upper]
    return np.clip(rgb, 0, 255).astype(np.uint8)


def depth_to_color(depth: np.ndarray, max_depth: Optional[float] = None) -> np.ndarray:
    """Convert metric depth to a false-color visualization.

    Invalid and zero depth are black. Valid depth is normalized by `max_depth`
    or by the 95th percentile of valid values when `max_depth` is omitted.
    """

    depth_array = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(depth_array) & (depth_array > 0.0)
    color = np.zeros(depth_array.shape + (3,), dtype=np.uint8)
    if not np.any(valid):
        return color
    if max_depth is None:
        max_depth = float(np.percentile(depth_array[valid], 95.0))
    scale = max(float(max_depth), 1e-6)
    normalized = np.clip(depth_array / scale, 0.0, 1.0)
    color[valid] = scalar_to_colormap(normalized[valid])
    return color


def save_depth_png(path: Path, depth: np.ndarray, max_depth: Optional[float] = None) -> Path:
    """Write a depth image as a normalized grayscale PNG."""

    gray = depth_to_grayscale(depth, max_depth=max_depth)
    rgb = np.repeat(gray[:, :, None], 3, axis=2)
    return _write_png_rgb(path, rgb)


def save_depth_color_png(path: Path, depth: np.ndarray, max_depth: Optional[float] = None) -> Path:
    """Write a depth image as a false-color PNG."""

    return _write_png_rgb(path, depth_to_color(depth, max_depth=max_depth))


def save_rgb_depth_preview(
    path: Path,
    rgb: np.ndarray,
    depth: np.ndarray,
    max_depth: Optional[float] = None,
) -> Path:
    """Write a side-by-side RGB/depth preview PNG."""

    rgb_array = np.asarray(rgb)[:, :, :3]
    if rgb_array.dtype != np.uint8:
        rgb_array = np.clip(rgb_array, 0, 255).astype(np.uint8)
    depth_gray = depth_to_grayscale(depth, max_depth=max_depth)
    depth_rgb = np.repeat(depth_gray[:, :, None], 3, axis=2)
    divider = np.full((rgb_array.shape[0], 4, 3), 255, dtype=np.uint8)
    preview = np.concatenate([rgb_array, divider, depth_rgb], axis=1)
    return _write_png_rgb(path, preview)


def save_rgb_depth_color_preview(
    path: Path,
    rgb: np.ndarray,
    depth: np.ndarray,
    max_depth: Optional[float] = None,
) -> Path:
    """Write a side-by-side RGB/false-color-depth preview PNG."""

    rgb_array = np.asarray(rgb)[:, :, :3]
    if rgb_array.dtype != np.uint8:
        rgb_array = np.clip(rgb_array, 0, 255).astype(np.uint8)
    depth_rgb = depth_to_color(depth, max_depth=max_depth)
    divider = np.full((rgb_array.shape[0], 4, 3), 255, dtype=np.uint8)
    preview = np.concatenate([rgb_array, divider, depth_rgb], axis=1)
    return _write_png_rgb(path, preview)


def _write_png_rgb(path: Path, rgb: np.ndarray) -> Path:
    ensure_dir(path.parent)
    height, width, channels = rgb.shape
    if channels != 3:
        raise ValueError("rgb must have exactly 3 channels")
    raw_rows = []
    for row in rgb:
        raw_rows.append(b"\x00" + row.tobytes())
    payload = b"".join(raw_rows)
    png = b"\x89PNG\r\n\x1a\n"
    png += _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    png += _png_chunk(b"IDAT", zlib.compress(payload))
    png += _png_chunk(b"IEND", b"")
    _write_bytes_atomic_verified(path, png)
    return path


def _write_bytes_atomic_verified(path: Path, payload: bytes, attempts: int = 3) -> None:
    """Write ``payload`` to ``path`` atomically with durable fsync and read-back.

    Catches silent write-time corruption (bad SSD blocks, torn writes) by
    re-reading the renamed file and comparing bytes to ``payload``. Retries
    the full temp-file + fsync + rename cycle up to ``attempts`` times so a
    fresh block allocation can sidestep a transiently bad region. Raises
    IOError on persistent verification failure.
    """

    import os as _os

    ensure_dir(path.parent)
    parent_fd = None
    try:
        parent_fd = _os.open(str(path.parent), _os.O_RDONLY | _os.O_DIRECTORY)
    except (OSError, NotImplementedError):
        parent_fd = None

    last_error: Optional[Exception] = None
    try:
        for attempt in range(attempts):
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="wb",
                    dir=path.parent,
                    prefix=".%s." % path.name,
                    suffix=".tmp",
                    delete=False,
                ) as h:
                    h.write(payload)
                    h.flush()
                    _os.fsync(h.fileno())
                temporary = Path(h.name)
                temporary.replace(path)
                if parent_fd is not None:
                    _os.fsync(parent_fd)
                with open(path, "rb") as check:
                    read_back = check.read()
                if read_back == payload:
                    return
                last_error = IOError(
                    "verified write returned mismatched bytes for %s (attempt %d, "
                    "wrote %d, read %d)" % (path, attempt + 1, len(payload), len(read_back))
                )
            except (OSError, IOError) as exc:
                last_error = exc
            finally:
                if temporary is not None and temporary.exists():
                    try:
                        temporary.unlink()
                    except OSError:
                        pass
        raise IOError(
            "verified write of %s failed after %d attempts: %s"
            % (path, attempts, last_error)
        )
    finally:
        if parent_fd is not None:
            try:
                _os.close(parent_fd)
            except OSError:
                pass
