"""Protocol v5 navmesh collision: routed motion, routed charging, valid pools."""

import numpy as np
import pytest

from activebench.runner import SegmentPath
from activebench.common.camera import CameraPose

from test_bench_capture_mode import _FakeSim


def pose(x=0.0, z=0.0, y=1.5, yaw=0.0, pitch=0.0):
    return CameraPose.from_xyz_yaw_pitch([x, y, z], yaw=yaw, pitch=pitch)


class TestSegmentPath:
    def test_straight_segment_matches_linear_interpolation(self):
        path = SegmentPath(pose(0.0), pose(2.0, yaw=1.0))
        assert path.length == pytest.approx(2.0)
        mid = path.pose_at(0.5)
        np.testing.assert_allclose(mid.position, [1.0, 1.5, 0.0])
        assert mid.yaw == pytest.approx(0.5)

    def test_l_shaped_route_follows_the_polyline(self):
        # Floor route detours through a corner: (0,0)->(2,0)->(2,2), length 4.
        route = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, 0.0, 2.0]])
        path = SegmentPath(pose(0.0, 0.0), pose(2.0, 2.0), route)
        assert path.length == pytest.approx(4.0)
        # Half way along the arc is exactly the corner.
        np.testing.assert_allclose(path.pose_at(0.5).position, [2.0, 1.5, 0.0])
        # Straight-line midpoint (through the wall) is never visited.
        quarter = path.pose_at(0.25)
        np.testing.assert_allclose(quarter.position, [1.0, 1.5, 0.0])

    def test_camera_height_blends_over_floor_route(self):
        route = np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
        path = SegmentPath(pose(0.0, y=1.0), pose(4.0, y=2.0), route)
        assert path.pose_at(0.5).position[1] == pytest.approx(1.5)
        np.testing.assert_allclose(path.pose_at(0.0).position, [0.0, 1.0, 0.0])
        np.testing.assert_allclose(path.pose_at(1.0).position, [4.0, 2.0, 0.0])

    def test_pure_rotation_has_zero_length(self):
        path = SegmentPath(pose(1.0, yaw=0.0), pose(1.0, yaw=1.2))
        assert path.length == 0.0
        assert path.pose_at(0.5).yaw == pytest.approx(0.6)
        np.testing.assert_allclose(path.pose_at(0.7).position, [1.0, 1.5, 0.0])


class _RoutedFakeSim(_FakeSim):
    """FakeSim with an L-shaped navmesh route and one unreachable pocket."""

    def set_navigation_anchor(self, xyz):
        pass

    def navmesh_route(self, start_xyz, end_xyz):
        start = np.asarray(start_xyz, dtype=np.float64)
        end = np.asarray(end_xyz, dtype=np.float64)
        if end[0] > 90.0:  # the unreachable pocket
            return None
        floor = lambda p: np.array([p[0], 0.0, p[2]])
        corner = np.array([end[0], 0.0, start[2]])
        points = np.stack([floor(start), corner, floor(end)])
        length = float(np.linalg.norm(corner - floor(start)) + np.linalg.norm(floor(end) - corner))
        return points, length


class _OneHopAgent:
    """Moves once to (3, z=4) — straight 5 m, routed 7 m — then stops."""

    def __init__(self, target):
        from activebench.api import AgentAction

        self.actions = [AgentAction.move_to(target), AgentAction.done()]

    def info(self):
        from activebench.api import MethodInfo

        return MethodInfo(name="hop")

    def reset(self, seed, task=None):
        pass

    def act(self, observation):
        return self.actions.pop(0)


def _navmesh_spec(**episode_overrides):
    from activebench.episode import EpisodeSpec

    episode = {"max_captures": 10, "max_sim_time": 100.0, "capture_cost": 0.0,
               "collision": "navmesh", "reconstruction_interval": 1.0}
    episode.update(episode_overrides)
    return EpisodeSpec.from_dict({
        "habitat": {"scene_path": "/tmp/fake.glb"},
        "motion": {"speed": 1.0},
        "start_pose": [0.0, 1.5, 0.0, 0.0, 0.0],
        "episode": episode,
        "eval": {"num_poses": 1},
    })


def test_runner_routes_charges_and_samples_on_the_route(tmp_path):
    from activebench.runner import run_episode

    target = pose(3.0, 4.0)  # straight 5 m; L-route 3 + 4 = 7 m
    manifest = run_episode(
        _navmesh_spec(), _OneHopAgent(target), tmp_path / "routed", sim=_RoutedFakeSim())

    assert manifest["collision"] == "navmesh"
    # Charged for the routed length at 1 m/s, not the straight line.
    assert manifest["clock"]["path_length_m"] == pytest.approx(7.0)
    assert manifest["clock"]["final_sim_time"] == pytest.approx(7.0)
    assert manifest["clock"]["unreachable_targets"] == 0
    # Stream samples ride the polyline: at t=2 the camera is still on the
    # first leg (x=2, z=0), not on the straight line to the target.
    frames = manifest["reconstruction"]["frames"]
    at_2s = next(f for f in frames if abs(f["time"] - 2.0) < 1e-6)
    np.testing.assert_allclose(at_2s["pose"][:3], [2.0, 1.5, 0.0], atol=1e-9)


def test_runner_unreachable_target_rotates_in_place(tmp_path):
    from activebench.runner import run_episode

    target = pose(95.0, 0.0, yaw=1.0)  # inside the unreachable pocket
    manifest = run_episode(
        _navmesh_spec(), _OneHopAgent(target), tmp_path / "unreachable", sim=_RoutedFakeSim())

    assert manifest["clock"]["unreachable_targets"] == 1
    assert manifest["clock"]["path_length_m"] == pytest.approx(0.0)
    # The rotation was still executed and charged.
    final = manifest["captures"][-1]["pose"]
    np.testing.assert_allclose(final[:3], [0.0, 1.5, 0.0], atol=1e-9)
    assert final[3] == pytest.approx(1.0)
    assert manifest["clock"]["final_sim_time"] > 0.0


def test_free_flight_default_is_unchanged(tmp_path):
    from activebench.runner import run_episode

    spec = _navmesh_spec()
    spec.collision = "none"
    target = pose(3.0, 4.0)
    manifest = run_episode(
        spec, _OneHopAgent(target), tmp_path / "free", sim=_RoutedFakeSim())
    assert manifest["clock"]["path_length_m"] == pytest.approx(5.0)


def test_episode_spec_rejects_unknown_collision():
    from activebench.episode import EpisodeSpec

    with pytest.raises(ValueError):
        EpisodeSpec.from_dict({
            "habitat": {"scene_path": "/tmp/fake.glb"},
            "episode": {"collision": "solid"},
        })


def test_pool_uses_provided_navmesh_positions():
    from activebench.registry import _scene_candidate_pool

    rng = np.random.default_rng(0)
    provided = [[1.0, 1.6, 2.0], [3.0, 1.4, -1.0]]
    pool = _scene_candidate_pool(
        {"scene_bbox": [[-5, 0, -5], [5, 3, 5]], "candidate_positions": provided}, rng)
    assert len(pool) == 2
    np.testing.assert_allclose(pool[0].position, provided[0])
    np.testing.assert_allclose(pool[1].position, provided[1])
    # Orientations are still drawn from the pool policy rng.
    assert -np.pi <= pool[0].yaw <= np.pi
