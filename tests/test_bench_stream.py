"""Protocol v3: the reconstruction stream is delivered to the agent."""

from pathlib import Path

import numpy as np
import pytest

from activebench.api import AgentAction, MethodInfo, Observation, StreamFrame
from activebench.common.camera import CameraIntrinsics, CameraPose


def pose(x=0.0):
    return CameraPose.from_xyz_yaw_pitch([x, 0.0, 0.0])


class _FakeSim:
    def __init__(self):
        self.intrinsics = CameraIntrinsics.from_hfov(8, 6, 90.0)

    def observe(self, pose, t, include_clean=False, include_mask=False):
        return {
            "rgb": np.zeros((6, 8, 3), dtype=np.uint8),
            "depth": np.ones((6, 8), dtype=np.float32),
            "distractor_mask": np.zeros((6, 8), dtype=bool),
        }

    def render_clean(self, pose):
        return {"rgb": np.zeros((6, 8, 3), dtype=np.uint8), "depth": np.ones((6, 8), dtype=np.float32)}

    def scene_aabb(self):
        return np.array([[-5.0, -1.0, -5.0], [5.0, 3.0, 5.0]])

    def start_pose(self):
        return pose(0.0)

    def close(self):
        pass


class _RecordingAgent:
    """One three-waypoint trajectory (capture at the end), then done."""

    def __init__(self, needs_depth=True):
        self.needs_depth = needs_depth
        self.observations = []

    def info(self):
        return MethodInfo(name="recorder", needs_depth=self.needs_depth)

    def reset(self, seed, task=None):
        pass

    def act(self, observation):
        self.observations.append(observation)
        if len(self.observations) > 1:
            return AgentAction.done()
        return AgentAction.trajectory(
            [pose(1.0), pose(2.0), pose(3.0)], capture_mode="last"
        )


def stream_spec(**episode_overrides):
    from activebench.episode import EpisodeSpec

    episode = {
        "max_captures": 10,
        "max_sim_time": 10.0,
        "capture_cost": 0.0,
        "reconstruction_interval": 0.5,
        "stream_observations": True,
    }
    episode.update(episode_overrides)
    return EpisodeSpec.from_dict({
        "habitat": {"scene_path": "/tmp/fake.glb"},
        "motion": {"speed": 1.0},
        "episode": episode,
        "eval": {"num_poses": 1},
    })


class TestRunnerStreamDelivery:
    def test_frames_delivered_once_and_cover_the_stream(self, tmp_path):
        from activebench.runner import run_episode

        agent = _RecordingAgent()
        manifest = run_episode(stream_spec(), agent, tmp_path / "s", sim=_FakeSim())

        first, second = agent.observations
        assert first.stream_frames == []  # nothing recorded before the move
        times = [frame.time for frame in second.stream_frames]
        assert times == [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
        for frame in second.stream_frames:
            assert Path(frame.rgb_path).exists()
            assert frame.depth_path is not None and Path(frame.depth_path).exists()
            assert frame.load_rgb().shape == (6, 8, 3)
        # The t=0 sample stays in the eval stream but is never re-delivered
        # (it duplicates the first decision frame).
        assert manifest["reconstruction"]["num_frames"] == 7
        assert manifest["reconstruction"]["streamed_to_agent"] is True

    def test_depth_paths_follow_provide_depth(self, tmp_path):
        from activebench.runner import run_episode

        agent = _RecordingAgent(needs_depth=False)
        run_episode(stream_spec(), agent, tmp_path / "nodepth", sim=_FakeSim())
        frames = agent.observations[1].stream_frames
        assert frames and all(frame.depth_path is None for frame in frames)

    def test_v2_specs_deliver_no_stream(self, tmp_path):
        from activebench.runner import run_episode

        agent = _RecordingAgent()
        manifest = run_episode(
            stream_spec(stream_observations=False), agent, tmp_path / "v2", sim=_FakeSim()
        )
        assert all(not obs.stream_frames for obs in agent.observations)
        assert manifest["reconstruction"]["streamed_to_agent"] is False


class TestSpecValidation:
    def test_stream_requires_reconstruction_interval(self):
        with pytest.raises(ValueError, match="reconstruction_interval"):
            stream_spec(reconstruction_interval=None)


class TestObservationRoundTrip:
    def test_stream_frames_survive_rpc_serialization(self, tmp_path):
        from PIL import Image

        rgb_path = tmp_path / "decision.png"
        Image.fromarray(np.zeros((6, 8, 3), dtype=np.uint8)).save(rgb_path)
        observation = Observation(
            rgb=np.zeros((6, 8, 3), dtype=np.uint8),
            pose=pose(1.0),
            time=2.0,
            intrinsics=CameraIntrinsics.from_hfov(8, 6, 90.0),
            rgb_path=str(rgb_path),
            stream_frames=[
                StreamFrame(pose=pose(0.5), time=0.5, rgb_path="/nowhere/a.png"),
                StreamFrame(
                    pose=pose(0.8), time=1.0,
                    rgb_path="/nowhere/b.png", depth_path="/nowhere/b.npy",
                ),
            ],
        )
        restored = Observation.from_dict(observation.to_dict())
        assert [f.time for f in restored.stream_frames] == [0.5, 1.0]
        # Lazy by design: paths are carried verbatim, pixels are not loaded.
        assert restored.stream_frames[0].rgb_path == "/nowhere/a.png"
        assert restored.stream_frames[0].depth_path is None
        assert restored.stream_frames[1].depth_path == "/nowhere/b.npy"
        np.testing.assert_allclose(
            restored.stream_frames[1].pose.position, [0.8, 0.0, 0.0]
        )

    def test_v2_payloads_without_stream_field_still_parse(self, tmp_path):
        from PIL import Image

        rgb_path = tmp_path / "old.png"
        Image.fromarray(np.zeros((6, 8, 3), dtype=np.uint8)).save(rgb_path)
        payload = Observation(
            rgb=np.zeros((6, 8, 3), dtype=np.uint8),
            pose=pose(0.0), time=0.0,
            intrinsics=CameraIntrinsics.from_hfov(8, 6, 90.0),
            rgb_path=str(rgb_path),
        ).to_dict()
        del payload["stream_frames"]
        assert Observation.from_dict(payload).stream_frames == []


class TestPoolNBVIngestion:
    class _CountingSelector:
        def __init__(self):
            self.history_sizes = []

        def select(self, history, candidates):
            self.history_sizes.append(len(history))
            return 0

    def observation(self, stream_count=0):
        return Observation(
            rgb=np.zeros((6, 8, 3), dtype=np.uint8),
            pose=pose(0.0),
            time=float(stream_count),
            intrinsics=CameraIntrinsics.from_hfov(8, 6, 90.0),
            rgb_path="/nowhere/obs.png",
            stream_frames=[
                StreamFrame(pose=pose(0.1 * i), time=0.1 * i, rgb_path="/nowhere/%d.png" % i)
                for i in range(stream_count)
            ],
        )

    def test_stream_frames_join_selector_history(self):
        from activebench.baselines import PoolNBVAgent

        selector = self._CountingSelector()
        agent = PoolNBVAgent(selector=selector, pool=[pose(1.0), pose(2.0), pose(3.0)])
        agent.reset(seed=0)
        agent.act(self.observation(stream_count=0))
        agent.act(self.observation(stream_count=2))
        # 1 decision record; then 2 streamed + 1 decision = 4 total.
        assert selector.history_sizes == [1, 4]
