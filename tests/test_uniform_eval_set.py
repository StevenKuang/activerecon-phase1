"""Uniform-panorama eval-set construction: view validity and candidate ranking.

The defect these guard against is recorded in
``docs/results/2026-07-31-calibration-label-audit.md``: two views whose camera
escaped the scene returned all-black GT with no valid depth, so ``psnr()`` hit
its ``99.0`` identical-images sentinel and added a fixed 16.5 dB to every score
computed against that 12-view set.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))


def _load_builder():
    """Import the script by path; it is not an installed module."""

    path = _REPO_ROOT / "scripts" / "build_uniform_eval_set.py"
    spec = importlib.util.spec_from_file_location("build_uniform_eval_set", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


builder = _load_builder()


def test_view_with_no_valid_depth_is_rejected():
    """The exact failure that produced the 99.0 sentinel."""

    escaped = np.zeros((8, 8), dtype=np.float32)
    assert not builder.view_is_valid(escaped, 0.5)
    # and it stays rejected however permissive the threshold, short of zero
    assert not builder.view_is_valid(escaped, 0.01)


def test_mostly_escaping_view_is_rejected_before_it_goes_fully_black():
    """Partial escape inflates PSNR too: both sides render nothing there."""

    depth = np.zeros((10, 10), dtype=np.float32)
    depth[:2, :] = 3.0  # 20% valid, matching the worst real apartment azimuth
    assert not builder.view_is_valid(depth, 0.5)


def test_fully_observed_view_passes():
    assert builder.view_is_valid(np.full((8, 8), 2.5, dtype=np.float32), 0.5)


def test_threshold_separates_the_measured_populations():
    """Observed fractions: legitimate 0.57-1.00, escaping 0.00-0.20."""

    legitimate = [0.57, 0.61, 0.67, 0.83, 0.87, 0.95, 0.96, 0.99, 1.00]
    escaping = [0.00, 0.00, 0.10, 0.15, 0.20]
    shape = (100, 100)

    def depth_with(fraction):
        depth = np.zeros(shape, dtype=np.float32)
        rows = int(round(fraction * shape[0]))
        depth[:rows, :] = 1.0
        return depth

    assert all(builder.view_is_valid(depth_with(f), 0.5) for f in legitimate)
    assert not any(builder.view_is_valid(depth_with(f), 0.5) for f in escaping)


def test_farthest_point_ranking_is_prefix_stable():
    """Oversampling candidates must not change the points a clean scene picks.

    ``standing_points`` asks for ``max(count, pool)`` candidates so rejected
    points can be replaced. That is only safe if the greedy order's first
    ``count`` entries are independent of how many were requested -- otherwise
    adding the validity check would silently move every existing eval set's
    points.
    """

    rng = np.random.default_rng(0)
    points = rng.uniform(-5.0, 5.0, size=(400, 3))
    short = builder.farthest_points(points, 6)
    long = builder.farthest_points(points, 48)
    assert list(short) == list(long[:6])


def test_ranking_degrades_gracefully_when_pool_exceeds_the_population():
    points = np.random.default_rng(1).uniform(0.0, 1.0, size=(5, 3))
    assert len(builder.farthest_points(points, 50)) == 5
