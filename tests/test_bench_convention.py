import numpy as np
import pytest

from activebench.convention import c2w_cv_to_pose, pose_to_c2w_cv
from activebench.common.camera import CameraPose


def test_round_trip_preserves_pose():
    pose = CameraPose.from_xyz_yaw_pitch([1.0, 2.0, -3.0], yaw=0.7, pitch=-0.25)
    restored = c2w_cv_to_pose(pose_to_c2w_cv(pose))
    np.testing.assert_allclose(restored.position, pose.position, atol=1e-12)
    assert restored.yaw == pytest.approx(pose.yaw)
    assert restored.pitch == pytest.approx(pose.pitch)


def test_cv_convention_axes():
    # At yaw=0/pitch=0 the benchmark camera views along -Z with +Y image-up.
    # In OpenCV convention the same camera has +Z forward and +Y image-down,
    # i.e. the c2w rotation flips the Y and Z columns.
    pose = CameraPose.from_xyz_yaw_pitch([0.0, 0.0, 0.0])
    c2w = pose_to_c2w_cv(pose)
    view_dir_world = c2w[:3, :3] @ np.array([0.0, 0.0, 1.0])  # cv camera +Z
    np.testing.assert_allclose(view_dir_world, [0.0, 0.0, -1.0], atol=1e-12)
    image_down_world = c2w[:3, :3] @ np.array([0.0, 1.0, 0.0])  # cv camera +Y
    np.testing.assert_allclose(image_down_world, [0.0, -1.0, 0.0], atol=1e-12)


def test_scene_candidate_pool_stays_in_bounds():
    from activebench.registry import _scene_candidate_pool

    options = {
        "scene_bbox": [[-1.0, -2.0, -3.0], [5.0, 1.0, 8.0]],
        "start_pose": [0.0, 0.5, 0.0, 0.0, 0.0],
        "max_captures": 10,
    }
    rng = np.random.default_rng(3)
    pool = _scene_candidate_pool(options, rng)
    assert len(pool) == 30
    positions = np.stack([p.position for p in pool])
    assert (positions >= np.array([-1.0, -2.0, -3.0])).all()
    assert (positions <= np.array([5.0, 1.0, 8.0])).all()
    # Deterministic under the same seed.
    pool2 = _scene_candidate_pool(options, np.random.default_rng(3))
    np.testing.assert_allclose(positions, np.stack([p.position for p in pool2]))
