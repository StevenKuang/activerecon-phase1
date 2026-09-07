"""Checks for the post-campaign frame integrity scan."""

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from verify_stream_integrity import check_depth, check_image, collect  # noqa: E402


def _write_png(path: Path, value: int = 128) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((16, 16, 3), value, dtype=np.uint8)).save(path)


def test_healthy_png_passes(tmp_path):
    p = tmp_path / "frame_00000.png"
    _write_png(p)
    assert check_image(p) is None


def test_interior_corruption_is_caught(tmp_path):
    # The real failure mode: header and trailer intact, bytes flipped inside
    # the compressed stream. Opening alone would not notice.
    p = tmp_path / "frame_00001.png"
    _write_png(p)
    raw = bytearray(p.read_bytes())
    middle = len(raw) // 2
    raw[middle:middle + 8] = b"\xff" * 8
    p.write_bytes(bytes(raw))
    assert p.read_bytes()[:8].hex() == "89504e470d0a1a0a"  # header still valid
    assert check_image(p) is not None


def test_truncated_png_is_caught(tmp_path):
    p = tmp_path / "frame_00002.png"
    _write_png(p)
    raw = p.read_bytes()
    p.write_bytes(raw[: len(raw) // 2])
    assert check_image(p) is not None


def test_depth_non_finite_is_caught(tmp_path):
    good, bad = tmp_path / "d0.npy", tmp_path / "d1.npy"
    np.save(good, np.ones((4, 4), dtype=np.float32))
    np.save(bad, np.array([[np.nan, 1.0], [1.0, 1.0]], dtype=np.float32))
    assert check_depth(good) is None
    assert "non-finite" in check_depth(bad)


def test_collect_walks_the_campaign_layout(tmp_path):
    _write_png(tmp_path / "scene__d0__s0/method/stream/frame_00000.png")
    _write_png(tmp_path / "scene__d0__s0/method/frames/frame_00000.png")
    np.save(tmp_path / "scene__d0__s0/method/stream/depth_00000.npy",
            np.ones((2, 2), dtype=np.float32))

    images = collect([tmp_path], include_depth=False)
    assert len(images) == 2 and {kind for _, kind in images} == {"image"}
    assert len(collect([tmp_path], include_depth=True)) == 3
