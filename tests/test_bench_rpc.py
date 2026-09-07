import sys
from pathlib import Path

import numpy as np
import pytest

from activebench.api import ActionKind, Observation
from activebench.rpc import AgentProcessProxy
from activebench.common.camera import CameraIntrinsics, CameraPose
from activebench.common.image import save_rgb_png


@pytest.fixture
def observation(tmp_path):
    rgb = np.random.default_rng(0).integers(0, 255, size=(48, 64, 3), dtype=np.uint8)
    depth = np.full((48, 64), 2.0, dtype=np.float32)
    rgb_path = tmp_path / "rgb.png"
    depth_path = tmp_path / "depth.npy"
    save_rgb_png(rgb_path, rgb)
    np.save(depth_path, depth)
    return Observation(
        rgb=rgb,
        depth=depth,
        pose=CameraPose.from_xyz_yaw_pitch([0.0, 1.5, 0.0], yaw=0.1),
        time=3.5,
        intrinsics=CameraIntrinsics.from_hfov(64, 48, 90.0),
        step=0,
        rgb_path=str(rgb_path),
        depth_path=str(depth_path),
    )


def test_png_write_is_atomic_and_readable(tmp_path):
    from PIL import Image

    path = tmp_path / "rgb.png"
    rgb = np.random.default_rng(1).integers(0, 255, size=(48, 64, 3), dtype=np.uint8)
    save_rgb_png(path, rgb)

    with Image.open(path) as image:
        np.testing.assert_array_equal(np.asarray(image), rgb)
    assert list(tmp_path.glob(".rgb.png.*.tmp")) == []


def test_proxy_round_trip_with_random_agent(tmp_path, observation):
    proxy = AgentProcessProxy(
        agent_name="random",
        options={},
        python_exe=sys.executable,
        repo_src=str(Path(__file__).resolve().parent.parent / "src"),
        stderr_log=tmp_path / "worker.log",
    )
    try:
        info = proxy.info()
        assert info.name == "random"
        assert info.needs_depth is True
        proxy.reset(seed=7)
        action = proxy.act(observation)
        assert action.kind == ActionKind.MOVE_TO
        assert proxy.decision_diagnostics() is None
        assert action.target is not None
        # Same seed in-process must give the identical decision: the proxy
        # is a transport, not a different method.
        from activebench.baselines import RandomAgent

        local = RandomAgent()
        local.reset(seed=7)
        expected = local.act(observation)
        np.testing.assert_allclose(action.target.position, expected.target.position)
        assert action.target.yaw == pytest.approx(expected.target.yaw)
    finally:
        proxy.close()


def test_proxy_round_trip_pose_free_wander(tmp_path, observation):
    from activebench.api import mask_observation_pose

    proxy = AgentProcessProxy(
        agent_name="wander",
        options={},
        python_exe=sys.executable,
        repo_src=str(Path(__file__).resolve().parent.parent / "src"),
        stderr_log=tmp_path / "worker.log",
    )
    try:
        info = proxy.info()
        assert info.pose_access == "none"
        proxy.reset(seed=7)
        # A pose-masked observation must survive the JSON transport, and the
        # camera-frame action must come back with its frame intact.
        action = proxy.act(mask_observation_pose(observation))
        assert action.kind == ActionKind.MOVE_TO
        assert action.frame == "camera"
    finally:
        proxy.close()


def test_proxy_survives_agent_stdout_noise(tmp_path, observation):
    # The worker redirects prints to stderr; protocol frames must stay clean.
    proxy = AgentProcessProxy(
        agent_name="random",
        options={},
        python_exe=sys.executable,
        repo_src=str(Path(__file__).resolve().parent.parent / "src"),
        stderr_log=tmp_path / "worker.log",
        env={"PYTHONSTARTUP": ""},
    )
    try:
        proxy.reset(seed=1)
        for _ in range(3):
            action = proxy.act(observation)
            assert action.kind in (ActionKind.MOVE_TO, ActionKind.DONE)
    finally:
        proxy.close()


def test_proxy_reports_worker_errors(tmp_path, observation):
    proxy = AgentProcessProxy(
        agent_name="random",
        options={},
        python_exe=sys.executable,
        repo_src=str(Path(__file__).resolve().parent.parent / "src"),
        stderr_log=tmp_path / "worker.log",
    )
    try:
        broken = Observation(
            rgb=observation.rgb,
            depth=None,
            pose=observation.pose,
            time=0.0,
            intrinsics=observation.intrinsics,
            rgb_path="/nonexistent/rgb.png",
        )
        with pytest.raises(RuntimeError, match="agent worker error"):
            proxy.act(broken)
        # The worker keeps serving after a request-level error.
        proxy.reset(seed=2)
        assert proxy.act(observation) is not None
    finally:
        proxy.close()


def test_unknown_agent_name_fails_fast():
    from activebench.registry import build_agent

    with pytest.raises(KeyError):
        build_agent("does-not-exist", {})
