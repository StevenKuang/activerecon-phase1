import numpy as np
import pytest

from activebench.eval.reconstruction import (
    GaussianSplats,
    ReconstructionCamera,
    ReconstructionDataset,
    reconstruction_artifacts,
    resolve_reconstruction_eval,
)
from activebench.eval.retrain import ACCURACY_CAP, geometry_metrics, psnr


class TestPsnr:
    def test_identical_images(self):
        img = np.random.default_rng(0).random((8, 8, 3))
        assert psnr(img, img) == 99.0

    def test_known_mse(self):
        gt = np.zeros((4, 4, 3))
        pred = np.full((4, 4, 3), 0.1)
        assert psnr(pred, gt) == pytest.approx(20.0, abs=1e-6)


def test_reconstruction_dataset_validates_stream_lengths():
    image = np.zeros((2, 2, 3), dtype=np.float32)
    depth = np.ones((2, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="same non-zero length"):
        ReconstructionDataset(
            images=[image],
            poses_c2w_cv=[],
            depths=[depth],
            alpha_masks=None,
            width=2,
            height=2,
            fovx=1.0,
            fovy=1.0,
            aabb=np.array([[0, 1], [0, 1], [0, 1]]),
        )


def test_reconstruction_camera_derives_fov_and_validates():
    camera = ReconstructionCamera(1280, 960, 640.0, 640.0, 640.0, 480.0)
    assert camera.fovx == pytest.approx(np.pi / 2)
    assert camera.fovy == pytest.approx(2.0 * np.arctan(0.75))
    with pytest.raises(ValueError, match="positive"):
        ReconstructionCamera(0, 960, 640.0, 640.0, 0.0, 0.0)


def test_reconstruction_artifacts_are_named_and_confined(tmp_path):
    artifacts = reconstruction_artifacts(tmp_path, "vanilla-3dgs-30k")
    assert artifacts.eval_path == (
        tmp_path / "reconstructions/vanilla-3dgs-30k/eval.json"
    )
    assert artifacts.gaussians_path.name == "gaussians.npz"
    with pytest.raises(ValueError, match="run name"):
        reconstruction_artifacts(tmp_path, "../escape")


def test_reconstruction_eval_resolution_prefers_vanilla_then_legacy(tmp_path):
    legacy = tmp_path / "retrain_eval.json"
    legacy.write_text("{}")
    assert resolve_reconstruction_eval(tmp_path) == legacy

    named = reconstruction_artifacts(tmp_path, "vanilla-3dgs").eval_path
    named.parent.mkdir(parents=True)
    named.write_text("{}")
    assert resolve_reconstruction_eval(tmp_path) == named
    # The standard gsplat backend outranks every other named run under auto,
    # even when additional named runs (e.g. anysplat) coexist.
    anysplat = reconstruction_artifacts(tmp_path, "anysplat").eval_path
    anysplat.parent.mkdir(parents=True)
    anysplat.write_text("{}")
    standard = reconstruction_artifacts(tmp_path, "vanilla-3dgs-gsplat").eval_path
    standard.parent.mkdir(parents=True)
    standard.write_text("{}")
    assert resolve_reconstruction_eval(tmp_path) == standard
    assert resolve_reconstruction_eval(tmp_path, "vanilla-3dgs") == named
    assert resolve_reconstruction_eval(tmp_path, "legacy") == legacy
    assert resolve_reconstruction_eval(tmp_path, "missing") is None
    assert resolve_reconstruction_eval(tmp_path, "") is None


def test_gaussian_splats_validate_all_parameter_shapes():
    with pytest.raises(ValueError, match="scales"):
        GaussianSplats(
            centers=np.zeros((2, 3)),
            scales=np.zeros((2, 1)),
            quats_wxyz=np.zeros((2, 4)),
            opacities=np.zeros(2),
            colors=np.zeros((2, 3)),
        )


def test_gaussian_splats_sh_rest_validates_shape_and_degree():
    n = 3
    # sh_degree=3 needs 15 coefficients per channel
    good = np.zeros((n, 15, 3), dtype=np.float32)
    ok = GaussianSplats(
        centers=np.zeros((n, 3)),
        scales=np.ones((n, 3)),
        quats_wxyz=np.tile([1.0, 0, 0, 0], (n, 1)),
        opacities=np.full(n, 0.5),
        colors=np.zeros((n, 3)),
        sh_rest=good, sh_degree=3,
    )
    assert ok.sh_degree == 3 and ok.sh_rest is not None

    # wrong coefficient count for the declared degree
    with pytest.raises(ValueError, match="sh_rest must have shape"):
        GaussianSplats(
            centers=np.zeros((n, 3)),
            scales=np.ones((n, 3)),
            quats_wxyz=np.tile([1.0, 0, 0, 0], (n, 1)),
            opacities=np.full(n, 0.5),
            colors=np.zeros((n, 3)),
            sh_rest=np.zeros((n, 8, 3), dtype=np.float32), sh_degree=3,
        )

    # sh_degree > 0 without sh_rest is not allowed
    with pytest.raises(ValueError, match="sh_degree > 0 requires sh_rest"):
        GaussianSplats(
            centers=np.zeros((n, 3)),
            scales=np.ones((n, 3)),
            quats_wxyz=np.tile([1.0, 0, 0, 0], (n, 1)),
            opacities=np.full(n, 0.5),
            colors=np.zeros((n, 3)),
            sh_degree=3,
        )

    # negative degree is not allowed
    with pytest.raises(ValueError, match="sh_degree must be non-negative"):
        GaussianSplats(
            centers=np.zeros((n, 3)),
            scales=np.ones((n, 3)),
            quats_wxyz=np.tile([1.0, 0, 0, 0], (n, 1)),
            opacities=np.full(n, 0.5),
            colors=np.zeros((n, 3)),
            sh_degree=-1,
        )


def make_gt():
    rng = np.random.default_rng(0)
    floor = np.column_stack([rng.uniform(0, 1, 200), np.zeros(200), rng.uniform(0, 1, 200)])
    ceil = floor + np.array([0.0, 3.0, 0.0])
    # Wall starts 0.5 above the floor so no wall point is within tau of it.
    wall = np.column_stack([np.zeros(200), rng.uniform(0.5, 2.5, 200), rng.uniform(0, 1, 200)])
    points = np.vstack([floor, ceil, wall])
    normals = np.vstack(
        [
            np.tile([0.0, 1.0, 0.0], (200, 1)),
            np.tile([0.0, -1.0, 0.0], (200, 1)),
            np.tile([1.0, 0.0, 0.0], (200, 1)),
        ]
    )
    return points, normals


class TestGeometryMetrics:
    def test_perfect_reconstruction(self):
        points, normals = make_gt()
        result = geometry_metrics(points.copy(), points, normals)
        for name in ("up", "down", "side"):
            assert result["bins"][name]["completeness@0.05"] == 1.0
        assert result["accuracy"]["median"] == pytest.approx(0.0, abs=1e-12)

    def test_partial_reconstruction_binned(self):
        points, normals = make_gt()
        # Reconstruct only the floor: up-bin complete, down/side empty.
        result = geometry_metrics(points[:200], points, normals)
        assert result["bins"]["up"]["completeness@0.05"] == 1.0
        assert result["bins"]["down"]["completeness@0.05"] == 0.0
        assert result["bins"]["side"]["completeness@0.05"] == 0.0
        assert result["completeness@0.05"] == pytest.approx(1 / 3, abs=1e-9)

    def test_offset_respects_tau(self):
        points, normals = make_gt()
        shifted = points + np.array([0.07, 0.0, 0.0])
        result = geometry_metrics(shifted, points, normals)
        # 7cm offset: fails tau=5cm, passes tau=10cm (nearest neighbor may be
        # closer than the offset for dense same-plane points, so >=).
        assert result["bins"]["down"]["completeness@0.10"] >= 0.99
        assert result["accuracy"]["median"] < 0.075

    def test_floaters_penalize_accuracy_not_completeness(self):
        points, normals = make_gt()
        floaters = np.tile([50.0, 50.0, 50.0], (100, 1))
        recon = np.vstack([points, floaters])
        result = geometry_metrics(recon, points, normals)
        assert result["completeness@0.05"] == 1.0
        # 100 of 700 recon points are capped-distance floaters.
        assert result["accuracy"]["mean"] == pytest.approx(100 / 700 * ACCURACY_CAP, rel=0.01)

    def test_empty_reconstruction(self):
        points, normals = make_gt()
        result = geometry_metrics(np.zeros((0, 3)), points, normals)
        assert result["bins"]["up"]["completeness@0.05"] == 0.0
        assert result["accuracy"]["median"] == ACCURACY_CAP
