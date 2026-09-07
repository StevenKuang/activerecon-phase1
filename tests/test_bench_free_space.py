import numpy as np
import pytest

from activebench.api import CaptureRecord, Observation
from activebench.baselines import PoolNBVAgent
from activebench.free_space import FreeSpaceTracker, sample_view_directions
from activebench.common.camera import CameraIntrinsics, CameraPose

INTR = CameraIntrinsics.from_hfov(64, 48, 90.0)


def wall_depth(distance: float) -> np.ndarray:
    return np.full((48, 64), distance, dtype=np.float32)


class TestFreeSpaceTracker:
    def make_tracker(self):
        return FreeSpaceTracker(bbox=np.array([[-5.0, -1.0, -5.0], [5.0, 3.0, 5.0]]))

    def test_free_marked_between_camera_and_wall(self):
        tracker = self.make_tracker()
        pose = CameraPose.from_xyz_yaw_pitch([0.0, 1.0, 0.0])
        tracker.update(wall_depth(3.0), pose, INTR)
        assert tracker.num_free_voxels > 0
        rng = np.random.default_rng(0)
        positions = tracker.candidate_positions(pose.position, radius=2.0, count=40, rng=rng)
        assert len(positions) == 40
        # All candidates within radius and inside the observed cone in front
        # of the camera (view direction is -Z at yaw 0).
        dists = np.linalg.norm(positions - pose.position, axis=1)
        assert dists.max() <= 2.0 + 1e-9

    def test_margin_erodes_near_wall(self):
        tracker = self.make_tracker()
        pose = CameraPose.from_xyz_yaw_pitch([0.0, 1.0, 0.0])
        tracker.update(wall_depth(3.0), pose, INTR)
        rng = np.random.default_rng(0)
        positions = tracker.candidate_positions(pose.position, radius=4.0, count=200, rng=rng)
        # Wall plane is at z = -3 (camera looks along -Z); with a 0.3 margin
        # plus voxel quantization no candidate may sit within 0.3 of it.
        assert positions[:, 2].min() > -3.0 + 0.3 - 1e-9

    def test_no_hit_rays_mark_nothing(self):
        tracker = self.make_tracker()
        pose = CameraPose.from_xyz_yaw_pitch([0.0, 1.0, 0.0])
        tracker.update(np.zeros((48, 64), dtype=np.float32), pose, INTR)
        assert tracker.num_free_voxels == 0

    def test_empty_before_updates(self):
        tracker = self.make_tracker()
        rng = np.random.default_rng(0)
        assert len(tracker.candidate_positions(np.zeros(3), 2.0, 10, rng)) == 0

    def test_deterministic_under_seed(self):
        tracker = self.make_tracker()
        pose = CameraPose.from_xyz_yaw_pitch([0.0, 1.0, 0.0])
        tracker.update(wall_depth(3.0), pose, INTR)
        a = tracker.candidate_positions(pose.position, 2.0, 20, np.random.default_rng(7))
        b = tracker.candidate_positions(pose.position, 2.0, 20, np.random.default_rng(7))
        np.testing.assert_allclose(a, b)


class TestViewDirections:
    def test_isotropic_pitch_spread_and_round_trip(self):
        rng = np.random.default_rng(0)
        yaws, pitches = sample_view_directions(2000, rng)
        # arcsin-distributed pitch: a quarter of directions are steeper than 30 deg.
        steep = np.abs(pitches) > np.deg2rad(30.0)
        assert 0.35 < steep.mean() < 0.65
        assert pitches.max() > np.deg2rad(60.0) and pitches.min() < -np.deg2rad(60.0)
        # Round trip: the pose's viewing direction matches (yaw, pitch).
        for i in range(0, 2000, 400):
            pose = CameraPose.from_xyz_yaw_pitch([0, 0, 0], yaw=float(yaws[i]), pitch=float(pitches[i]))
            view = -pose.forward()
            assert view[1] == pytest.approx(np.sin(pitches[i]), abs=1e-9)
            assert np.arctan2(-view[0], -view[2]) == pytest.approx(yaws[i], abs=1e-9)


class _FirstSelector:
    def __init__(self):
        self.seen = []

    def select(self, history, candidates):
        self.seen.append(list(candidates))
        return 0


def _obs(pose, depth):
    return Observation(
        rgb=np.zeros((48, 64, 3), dtype=np.uint8),
        depth=depth,
        pose=pose,
        time=0.0,
        intrinsics=INTR,
    )


class TestLocalPoolAgent:
    def make_agent(self, selector):
        return PoolNBVAgent(
            selector=selector,
            name="test-local",
            pool_mode="local",
            radius=2.0,
            sample_num=12,
            scene_bbox=np.array([[-5.0, -1.0, -5.0], [5.0, 3.0, 5.0]]),
        )

    def test_local_mode_requires_depth_and_bbox(self):
        assert self.make_agent(_FirstSelector()).info().needs_depth is True
        with pytest.raises(ValueError):
            PoolNBVAgent(selector=_FirstSelector(), pool_mode="local")

    def test_first_round_falls_back_to_in_place(self):
        agent = self.make_agent(_FirstSelector())
        agent.reset(seed=3)
        pose = CameraPose.from_xyz_yaw_pitch([0.0, 1.0, 0.0])
        # No-hit depth: nothing observed free yet -> in-place rotations.
        action = agent.act(_obs(pose, np.zeros((48, 64), dtype=np.float32)))
        np.testing.assert_allclose(action.target.position, pose.position)

    def test_candidates_come_from_observed_free_space(self):
        selector = _FirstSelector()
        agent = self.make_agent(selector)
        agent.reset(seed=3)
        pose = CameraPose.from_xyz_yaw_pitch([0.0, 1.0, 0.0])
        action = agent.act(_obs(pose, wall_depth(3.0)))
        candidates = selector.seen[-1]
        assert len(candidates) == 12
        positions = np.stack([c.position for c in candidates])
        assert np.linalg.norm(positions - pose.position, axis=1).max() <= 2.0 + 1e-9
        pitches = np.array([c.pitch for c in candidates])
        assert np.abs(pitches).max() > np.deg2rad(20.0)  # pitch diversity
        assert action.kind == "move_to"

    def test_reproducible_under_seed(self):
        s1, s2 = _FirstSelector(), _FirstSelector()
        a1, a2 = self.make_agent(s1), self.make_agent(s2)
        pose = CameraPose.from_xyz_yaw_pitch([0.0, 1.0, 0.0])
        for agent in (a1, a2):
            agent.reset(seed=9)
            agent.act(_obs(pose, wall_depth(3.0)))
        p1 = np.stack([c.position for c in s1.seen[-1]])
        p2 = np.stack([c.position for c in s2.seen[-1]])
        np.testing.assert_allclose(p1, p2)
