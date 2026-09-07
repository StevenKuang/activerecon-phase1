"""Headless checks for the gsplat backend's pure helpers (no GPU/gsplat)."""

import numpy as np
import pytest

from activebench.eval.gsplat_backend import (
    GsplatVanilla3DGSBackend,
    _camera_extent,
    _focal,
    _initial_point_cloud,
    _knn_mean_sq_dist,
)
from activebench.eval.reconstruction import ReconstructionDataset


def make_dataset(count=2):
    poses = []
    for index in range(count):
        pose = np.eye(4)
        pose[0, 3] = float(index)
        poses.append(pose)
    return ReconstructionDataset(
        images=[np.full((4, 4, 3), 0.5, dtype=np.float32)] * count,
        poses_c2w_cv=poses,
        depths=[np.full((4, 4), 2.0, dtype=np.float32)] * count,
        alpha_masks=None,
        width=4,
        height=4,
        fovx=np.pi / 2,
        fovy=np.pi / 2,
        aabb=np.array([[0, 2], [0, 1], [0, 1]]),
    )


def test_focal_matches_pinhole_model():
    assert _focal(np.pi / 2, 640) == pytest.approx(320.0)


def test_camera_extent_matches_graphdeco_normalization():
    poses = [np.eye(4), np.eye(4)]
    poses[1][0, 3] = 2.0
    dataset = make_dataset()
    dataset.poses_c2w_cv[1][0, 3] = 2.0
    assert _camera_extent(dataset) == pytest.approx(1.1)


def test_initial_point_cloud_is_deterministic_and_posed():
    dataset = make_dataset()
    points_a, colors_a = _initial_point_cloud(dataset, per_view=8, seed=3)
    points_b, colors_b = _initial_point_cloud(dataset, per_view=8, seed=3)
    np.testing.assert_array_equal(points_a, points_b)
    np.testing.assert_array_equal(colors_a, colors_b)
    assert points_a.shape == (16, 3)
    # Depth 2.0 along +z in OpenCV camera frame; identity pose keeps z=2.
    assert np.allclose(points_a[:8, 2], 2.0)
    # Second camera is translated +1 in x.
    assert points_a[8:, 0].mean() > points_a[:8, 0].mean() + 0.5


def test_knn_scale_init_reflects_local_spacing():
    sparse = np.array([[0.0, 0, 0], [1.0, 0, 0], [2.0, 0, 0], [3.0, 0, 0]])
    dense = sparse * 0.1
    assert _knn_mean_sq_dist(dense).mean() < _knn_mean_sq_dist(sparse).mean()
    assert (_knn_mean_sq_dist(sparse) >= 1e-7).all()


def test_backend_rejects_unknown_options_and_bad_iterations():
    backend = GsplatVanilla3DGSBackend()
    with pytest.raises(ValueError, match="unknown backend options"):
        backend.reconstruct(
            make_dataset(), iterations=10, seed=0, options={"nope": 1}
        )
    with pytest.raises(ValueError, match="iterations"):
        backend.reconstruct(make_dataset(), iterations=0, seed=0, options={})


def test_registry_returns_gsplat_backend_without_gavis_repo():
    from activebench.eval.reconstruction import create_reconstruction_backend

    backend = create_reconstruction_backend("gsplat")
    assert backend.name == "gsplat"
    with pytest.raises(ValueError, match="unknown reconstruction backend"):
        create_reconstruction_backend("vanilla-3dgs")  # removed gavis-vendored port
