from __future__ import annotations

import os

import numpy as np

from activebench.common.image import save_rgb_png


def test_verified_png_writes_do_not_leak_parent_file_descriptors(tmp_path):
    """Long benchmark runs write thousands of RGB and mask images."""

    baseline = len(os.listdir("/proc/self/fd"))
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    for index in range(64):
        save_rgb_png(tmp_path / ("frame_%03d.png" % index), image)
    assert len(os.listdir("/proc/self/fd")) <= baseline + 2
