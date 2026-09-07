"""Headless checks for the GLEAM adapter (no Isaac Gym, no checkpoint).

Covers the pieces this adapter had to rebuild outside GLEAM's simulator: the
y-up/z-up boundary, the raycast that drives their probabilistic map, the
egocentric crop and frontier semantics their policy reads, goal decoding, and
the spin-while-flying trajectory. The policy forward pass needs the GLEAM
checkout and is skipped without it; scoring a real episode needs their
checkpoint and is not a unit test.
"""

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from activebench.adapters.gleam import (  # noqa: E402
    EGO_CELL_SIZE,
    GRID_SIZE,
    POSE_BUFFER,
    POSE_SIZE,
    STAY_INDEX,
    GleamAgent,
    _bresenham_rays,
    _reachable_mask,
    _SceneGrid,
    compute_frontier_mask,
    discretize_prob_map,
    extract_ego_map,
    gleam_to_hab,
    hab_to_gleam,
)
from activebench.api import ActionKind, Observation  # noqa: E402
from activebench.common.camera import CameraIntrinsics, CameraPose  # noqa: E402

REPO_ROOT = Path("/home/steven/Projects/GLEAM")
SCENE_BBOX = np.array([[-6.0, 0.0, -4.0], [6.0, 3.0, 4.0]])


# -- coordinate boundary ----------------------------------------------------


def test_frame_conversion_is_a_right_handed_involution():
    points = np.array([[1.0, 2.0, 3.0], [-4.0, 0.5, 7.0], [0.0, 0.0, 0.0]])
    assert np.allclose(gleam_to_hab(hab_to_gleam(points)), points)
    basis = hab_to_gleam(np.eye(3))
    # A mirrored map would flip left/right under a policy trained z-up.
    assert np.isclose(np.linalg.det(basis), 1.0)
    # Habitat's up axis (+y) must become GLEAM's up axis (+z).
    assert np.allclose(hab_to_gleam(np.array([0.0, 1.0, 0.0])), [0.0, 0.0, 1.0])


def test_scene_grid_spans_the_bbox_and_indexes_like_gleam():
    grid = _SceneGrid.from_scene_bbox(SCENE_BBOX)
    # Habitat x/z extents (12 x 8 m) become GLEAM's map plane; y (3 m) is up.
    assert np.allclose(grid.upper - grid.lower, [12.0, 8.0, 3.0])
    assert np.allclose(grid.voxel, np.array([12.0, 8.0, 3.0]) / GRID_SIZE)

    # pose_coord_to_2d_idx: floor((p - (min - half voxel)) / voxel), clipped.
    xy = np.array([1.3, -2.7])
    origin = grid.lower[:2] - 0.5 * grid.voxel[:2]
    expected = np.floor((xy - origin) / grid.voxel[:2]).astype(np.int64)
    assert np.array_equal(grid.cell_of(xy), expected)
    # Out-of-bounds queries clip rather than wrap.
    assert np.array_equal(grid.cell_of(np.array([1e6, -1e6])), [GRID_SIZE - 1, 0])


# -- raycast ----------------------------------------------------------------


def _bresenham_reference(x0, y0, x1, y1, map_size):
    """Line-by-line transcription of gleam/utils/utils.py::bresenham_2d's kernel.

    Written from their CUDA source rather than reusing the adapter's rewrite,
    so this is an independent oracle. Returns the set of in-bounds cells.
    """

    dx, dy = abs(x1 - x0), abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    x, y, err = x0, y0, dx - dy
    cells, idx = set(), 0
    inside = lambda a, b: 0 <= a < map_size and 0 <= b < map_size  # noqa: E731
    if inside(x, y):
        cells.add((x, y))
        idx = 1
    while idx < map_size * 2:
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy
        if inside(x, y):
            cells.add((x, y))
            idx += 1
    return cells


def test_bresenham_matches_their_kernel():
    rng = np.random.default_rng(0)
    map_size = 32
    source = np.array([13, 7])
    targets = rng.integers(0, map_size, size=(64, 2))
    # Degenerate and axis-aligned rays are the ones an integer DDA gets wrong.
    targets = np.vstack([targets, source, [source[0], 0], [0, source[1]], [31, 31]])

    got = _bresenham_rays(
        torch.as_tensor(source), torch.as_tensor(targets), map_size
    )
    got_set = {(int(a), int(b)) for a, b in got.tolist()}
    expected = set()
    for tx, ty in targets.tolist():
        expected |= _bresenham_reference(int(source[0]), int(source[1]), tx, ty, map_size)
    assert got_set == expected


def test_bresenham_output_stays_in_bounds():
    cells = _bresenham_rays(
        torch.as_tensor(np.array([0, 0])),
        torch.as_tensor(np.array([[GRID_SIZE - 1, GRID_SIZE - 1]])),
        GRID_SIZE,
    )
    assert cells.min() >= 0 and cells.max() <= GRID_SIZE - 1


# -- map semantics ----------------------------------------------------------


def test_discretize_matches_their_thresholds():
    prob = torch.tensor([[-0.10, 0.0, 0.25, 0.6, 1.0]])
    occupancy, tri = discretize_prob_map(prob)
    # >0.5 occupied, <0.0 free, everything else unknown.
    assert occupancy.tolist() == [[0.0, 0.0, 0.0, 1.0, 1.0]]
    assert tri.tolist() == [[-1.0, 0.0, 0.0, 1.0, 1.0]]


def test_frontier_marks_free_cells_touching_unknown():
    ego = torch.zeros(GRID_SIZE, GRID_SIZE)  # all unknown
    ego[:, :10] = -1.0  # a free slab; its last column touches unknown
    kernel = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]
    )[None, None]
    frontier = compute_frontier_mask(ego, kernel)
    # Their max-pool denoise dilates occupied/unknown by one cell first, so
    # the boundary sits just inside the free region, never in the interior.
    assert frontier.any()
    assert not frontier[:, :6].any()
    assert frontier[:, 6:10].any()


def test_ego_crop_is_centred_and_resampled_to_ten_centimetre_cells():
    grid = _SceneGrid.from_scene_bbox(SCENE_BBOX)
    global_map = torch.zeros(GRID_SIZE, GRID_SIZE)
    centre = np.array([70, 40])
    global_map[centre[0] - 2:centre[0] + 3, centre[1] - 2:centre[1] + 3] = 1.0

    ego = extract_ego_map(
        global_map,
        grid.voxel[:2],
        torch.as_tensor(centre, dtype=torch.float32),
        ego_cell=EGO_CELL_SIZE,
    )
    assert ego.shape == (GRID_SIZE, GRID_SIZE)
    # The agent sits at the crop's centre, so the marked block lands there.
    hot = torch.nonzero(ego == 1.0)
    assert hot.numel() > 0
    assert abs(float(hot[:, 0].float().mean()) - GRID_SIZE // 2) <= 1.0
    assert abs(float(hot[:, 1].float().mean()) - GRID_SIZE // 2) <= 1.0


def test_ego_crop_resamples_when_scene_cells_are_finer_than_ten_centimetres():
    """The crop is a resample, not a copy — and here it downsamples.

    A 12 x 8 m scene gives 9.4 x 6.3 cm global cells, so the 12.8 m ego patch
    spans ~137 x 205 global cells squeezed into 128 x 128. Isolated cells can
    therefore be skipped by the nearest-neighbour sampling; the benchmark's
    consumers only ever see contiguous structure, but it is worth pinning.
    """

    grid = _SceneGrid.from_scene_bbox(SCENE_BBOX)
    assert grid.voxel[0] < EGO_CELL_SIZE and grid.voxel[1] < EGO_CELL_SIZE
    global_map = torch.zeros(GRID_SIZE, GRID_SIZE)
    global_map[70, 40] = 1.0
    ego = extract_ego_map(
        global_map, grid.voxel[:2], torch.as_tensor([70, 40], dtype=torch.float32)
    )
    assert not bool((ego == 1.0).any())


def test_reachability_respects_walls():
    free = torch.zeros(16, 16, dtype=torch.bool)
    free[:, :] = True
    free[:, 8] = False  # a wall splitting the room in two
    reachable = _reachable_mask(free, torch.as_tensor([2, 2]))
    assert bool(reachable[5, 5])
    assert not bool(reachable[5, 12])
    # A gap in the wall reconnects the halves.
    free[3, 8] = True
    reachable = _reachable_mask(free, torch.as_tensor([2, 2]))
    assert bool(reachable[5, 12])


def test_reachability_needs_a_free_start():
    free = torch.ones(8, 8, dtype=torch.bool)
    free[4, 4] = False
    assert not _reachable_mask(free, torch.as_tensor([4, 4])).any()


# -- action decoding --------------------------------------------------------


def _agent(**kwargs):
    return GleamAgent(scene_bbox=SCENE_BBOX, repo_root=str(REPO_ROOT),
                      device="cpu", **kwargs)


def test_stay_index_decodes_to_no_displacement():
    agent = _agent()
    position = np.array([1.0, 2.0, 1.5])
    action = np.array([STAY_INDEX, STAY_INDEX, 0, 0, 0, 0])
    assert np.allclose(agent._goal_from_action(action, position), position)


def test_goal_offset_is_cells_times_voxel_size():
    agent = _agent()
    position = np.array([0.0, 0.0, 1.5])
    action = np.array([STAY_INDEX + 10, STAY_INDEX - 4, 0, 0, 0, 0])
    goal = agent._goal_from_action(action, position)
    assert np.isclose(goal[0], 10 * agent.grid.voxel[0])
    assert np.isclose(goal[1], -4 * agent.grid.voxel[1])
    assert np.isclose(goal[2], position[2])


def test_goal_is_clipped_to_the_scene():
    agent = _agent()
    position = np.array([5.0, 3.0, 1.5])
    action = np.array([128, 128, 0, 0, 0, 0])  # the largest legal index
    goal = agent._goal_from_action(action, position)
    assert np.all(goal <= agent.grid.upper + 1e-9)
    assert np.all(goal >= agent.grid.lower - 1e-9)


def test_only_an_occupied_target_cell_blocks_a_goal():
    """Their pre-move gate is the target cell alone — not connectivity.

    Aiming into unknown space is what exploration is; GLEAM's A*/BFS check
    runs after the move and only raises a collision flag. Gating on it here
    once cost the agent two thirds of its mission clock.
    """

    agent = _agent()
    agent._occ_map = torch.zeros(GRID_SIZE, GRID_SIZE)
    agent._tri_map = torch.zeros(GRID_SIZE, GRID_SIZE)  # everything unknown
    here, there = np.array([64, 64]), np.array([20, 90])
    agent._tri_map[here[0], here[1]] = -1.0  # the agent stands on free space

    # Unknown, unreachable through free space, and still a legal goal.
    assert agent._goal_rejection(here, there) is None
    assert "unconnected" in agent._goal_connectivity(here, there)

    agent._occ_map[there[0], there[1]] = 1.0
    assert agent._goal_rejection(here, there) == "goal cell occupied"


def test_spin_trajectory_turns_a_full_circle_and_lands_on_the_goal():
    agent = _agent(spin_legs=4)
    start = CameraPose.from_xyz_yaw_pitch([0.0, 1.2, 0.0], yaw=0.0, pitch=0.3)
    goal = np.array([2.0, 1.2, 1.0])
    action = agent._spin_trajectory(start, goal)

    assert action.kind == ActionKind.TRAJECTORY
    # Travel-only waypoints: only the final pose is a decision capture.
    assert action.capture_mode == "last"
    assert len(action.waypoints) == 4
    assert np.allclose(action.waypoints[-1].position, goal)
    # Each leg turns a quarter turn, so the legs together sweep 360 degrees
    # and the benchmark charges max(translation, rotation) per leg.
    yaws = [w.yaw for w in action.waypoints]
    for index, yaw in enumerate(yaws):
        expected = (start.yaw + (index + 1) * np.pi / 2 + np.pi) % (2 * np.pi) - np.pi
        assert np.isclose(yaw, expected)
    # Pitch is levelled: GLEAM only ever maps the layer at flight height.
    assert all(np.isclose(w.pitch, 0.0) for w in action.waypoints)


def test_state_vector_is_relative_to_the_current_pose():
    agent = _agent()
    agent._pose_history = [
        np.array([2.0, 1.0, 1.5, 0.0, 0.0, 0.0]),
        np.array([1.0, 1.0, 1.5, 0.0, 0.0, 0.0]),
    ]
    state = agent._state_vector().reshape(POSE_BUFFER, POSE_SIZE)
    assert state.shape == (POSE_BUFFER, POSE_SIZE)
    assert np.allclose(state[0], 0.0)  # index 0 is always the current pose
    assert np.isclose(state[1][0], -1.0)
    assert np.allclose(state[2:], 0.0)  # unfilled history stays zeroed


# -- registry + end to end --------------------------------------------------


def test_registry_exposes_gleam_in_its_own_env():
    from activebench import registry

    assert "gleam" in registry.available_agents()
    assert registry.default_env("gleam") == "gleam"


def _observation(pose, intrinsics, depth_value=2.0):
    depth = np.full((intrinsics.height, intrinsics.width), depth_value, dtype=np.float32)
    rgb = np.zeros((intrinsics.height, intrinsics.width, 3), dtype=np.uint8)
    return Observation(
        rgb=rgb, depth=depth, pose=pose, time=0.0, intrinsics=intrinsics, step=0
    )


@pytest.mark.skipif(not REPO_ROOT.exists(), reason="GLEAM checkout not present")
def test_act_runs_end_to_end_with_an_untrained_policy():
    """The full act() path: map update, ego crop, policy, action emission.

    Weights are random without their checkpoint, so this asserts shape and
    protocol conformance, not behaviour.
    """

    # The campaign camera. Vertical sampling matters here: GLEAM only maps the
    # voxel layer at flight height, and a coarse camera can miss it entirely.
    intrinsics = CameraIntrinsics.from_hfov(640, 480, 75.1781789379499)
    agent = _agent()
    agent.reset(seed=0)
    pose = CameraPose.from_xyz_yaw_pitch([0.0, 1.5, 0.0], yaw=0.0, pitch=0.0)

    first = agent.act(_observation(pose, intrinsics))
    # Their env forces the first action of an episode to "stay", which here
    # means spinning on the spot rather than translating.
    assert first.kind == ActionKind.TRAJECTORY
    assert all(np.allclose(w.position, pose.position) for w in first.waypoints)
    # Depth at 2 m must have written occupied cells into the map.
    assert bool((agent._prob_map >= 1.0).any())

    second = agent.act(_observation(pose, intrinsics))
    assert second.kind in (ActionKind.TRAJECTORY, ActionKind.DONE)
    if second.kind == ActionKind.TRAJECTORY:
        assert second.capture_mode == "last"
        assert len(second.waypoints) == agent.spin_legs


GLEAM_PYTHON = Path.home() / "miniconda3/envs/gleam/bin/python"


@pytest.mark.skipif(
    not (REPO_ROOT.exists() and GLEAM_PYTHON.exists()),
    reason="gleam env or GLEAM checkout not present",
)
def test_proxy_round_trip_in_the_gleam_env(tmp_path):
    """The adapter must survive the RPC boundary it actually runs behind.

    GLEAM is a CUDA method, so the runner never builds it in the simulator
    process; this drives the real worker in the real env.
    """

    import sys

    from activebench.rpc import AgentProcessProxy
    from activebench.common.image import save_rgb_png

    intrinsics = CameraIntrinsics.from_hfov(640, 480, 75.1781789379499)
    rgb = np.zeros((intrinsics.height, intrinsics.width, 3), dtype=np.uint8)
    depth = np.full((intrinsics.height, intrinsics.width), 2.0, dtype=np.float32)
    rgb_path, depth_path = tmp_path / "rgb.png", tmp_path / "depth.npy"
    save_rgb_png(rgb_path, rgb)
    np.save(depth_path, depth)
    observation = Observation(
        rgb=rgb, depth=depth,
        pose=CameraPose.from_xyz_yaw_pitch([0.0, 1.5, 0.0], yaw=0.0),
        time=0.0, intrinsics=intrinsics, step=0,
        rgb_path=str(rgb_path), depth_path=str(depth_path),
    )

    proxy = AgentProcessProxy(
        agent_name="gleam",
        options={"scene_bbox": SCENE_BBOX.tolist(), "repo_root": str(REPO_ROOT)},
        python_exe=str(GLEAM_PYTHON),
        repo_src=str(Path(__file__).resolve().parent.parent / "src"),
        stderr_log=tmp_path / "worker.log",
    )
    try:
        info = proxy.info()
        assert info.name == "gleam"
        assert info.needs_depth is True
        assert info.conda_env == "gleam"
        proxy.reset(seed=0)
        action = proxy.act(observation)
        # A spin trajectory has to cross the JSON transport intact.
        assert action.kind == ActionKind.TRAJECTORY
        assert action.capture_mode == "last"
        assert len(action.waypoints) == 4
    finally:
        proxy.close()
    assert "Traceback" not in (tmp_path / "worker.log").read_text()


@pytest.mark.skipif(not REPO_ROOT.exists(), reason="GLEAM checkout not present")
def test_repeated_unreachable_goals_end_the_episode():
    # The campaign camera. Vertical sampling matters here: GLEAM only maps the
    # voxel layer at flight height, and a coarse camera can miss it entirely.
    intrinsics = CameraIntrinsics.from_hfov(640, 480, 75.1781789379499)
    agent = _agent(max_consecutive_stays=2)
    agent.reset(seed=0)
    pose = CameraPose.from_xyz_yaw_pitch([0.0, 1.5, 0.0], yaw=0.0, pitch=0.0)
    agent.act(_observation(pose, intrinsics))

    # Force every goal to look unreachable, as an enclosed agent would see.
    agent._goal_rejection = lambda *_: "forced"
    kinds = [agent.act(_observation(pose, intrinsics)).kind for _ in range(4)]
    assert ActionKind.DONE in kinds
