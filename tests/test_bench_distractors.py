import numpy as np
import pytest

from activebench.distractors import (
    CircleOrbit,
    DistractorSpec,
    WaypointPatrol,
    make_trajectory,
    random_walk_patrol,
)

SQUARE = np.array(
    [
        [0.0, 1.0, 0.0],
        [2.0, 1.0, 0.0],
        [2.0, 1.0, 2.0],
        [0.0, 1.0, 2.0],
    ]
)


def test_waypoint_patrol_is_deterministic_and_periodic():
    patrol = WaypointPatrol(waypoints=SQUARE, speed=1.0, mode="loop")
    p0, _ = patrol.pose_at(0.0)
    p_again, _ = patrol.pose_at(0.0)
    np.testing.assert_allclose(p0, p_again)
    # Path length is 8m at 1 m/s: one full loop returns to the start.
    p_loop, _ = patrol.pose_at(8.0)
    np.testing.assert_allclose(p_loop, p0, atol=1e-9)
    p_half, _ = patrol.pose_at(1.0)
    np.testing.assert_allclose(p_half, [1.0, 1.0, 0.0], atol=1e-9)


def test_waypoint_patrol_pingpong_reverses():
    line = np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
    patrol = WaypointPatrol(waypoints=line, speed=1.0, mode="pingpong")
    forward, _ = patrol.pose_at(3.0)
    np.testing.assert_allclose(forward, [3.0, 0.0, 0.0], atol=1e-9)
    backward, _ = patrol.pose_at(5.0)
    np.testing.assert_allclose(backward, [3.0, 0.0, 0.0], atol=1e-9)
    at_end, _ = patrol.pose_at(4.0)
    np.testing.assert_allclose(at_end, [4.0, 0.0, 0.0], atol=1e-9)


def test_patrol_faces_motion_direction():
    line = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 4.0]])
    patrol = WaypointPatrol(waypoints=line, speed=1.0, mode="pingpong")
    _, yaw = patrol.pose_at(1.0)
    assert yaw == pytest.approx(0.0)  # moving along +Z
    _, yaw_back = patrol.pose_at(6.0)
    assert abs(yaw_back) == pytest.approx(np.pi)


def test_circle_orbit_radius_and_period():
    orbit = CircleOrbit(center=np.array([1.0, 2.0, 3.0]), radius=2.0, angular_speed=np.pi)
    for t in (0.0, 0.25, 1.3):
        position, _ = orbit.pose_at(t)
        assert np.linalg.norm(position - orbit.center) == pytest.approx(2.0)
    p0, _ = orbit.pose_at(0.0)
    p_period, _ = orbit.pose_at(2.0)
    np.testing.assert_allclose(p0, p_period, atol=1e-9)


def test_random_walk_is_seed_deterministic():
    kwargs = dict(
        bounds_min=np.zeros(3),
        bounds_max=np.ones(3),
        num_waypoints=6,
        speed=0.5,
    )
    a = random_walk_patrol(seed=3, **kwargs)
    b = random_walk_patrol(seed=3, **kwargs)
    c = random_walk_patrol(seed=4, **kwargs)
    for t in (0.0, 1.7, 9.2):
        pa, _ = a.pose_at(t)
        pb, _ = b.pose_at(t)
        np.testing.assert_allclose(pa, pb)
    pc, _ = c.pose_at(1.7)
    assert not np.allclose(a.pose_at(1.7)[0], pc)


def test_make_trajectory_dispatch_and_spec_cache():
    trajectory = make_trajectory(
        {"type": "waypoint_patrol", "waypoints": SQUARE.tolist(), "speed": 2.0}
    )
    assert isinstance(trajectory, WaypointPatrol)
    with pytest.raises(ValueError):
        make_trajectory({"type": "teleport"})

    spec = DistractorSpec(
        name="d",
        object_template="chefcan",
        trajectory={"type": "circle", "center": [0, 0, 0], "radius": 1.0},
    )
    assert spec.build_trajectory(seed=1) is spec.build_trajectory(seed=2)
