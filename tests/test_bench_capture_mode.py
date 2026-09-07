import numpy as np
import pytest

from activebench.api import AgentAction

from activebench.common.camera import CameraPose


def pose(x=0.0):
    return CameraPose.from_xyz_yaw_pitch([x, 0.0, 0.0])


def test_capture_mode_default_and_round_trip():
    action = AgentAction.trajectory([pose(1.0), pose(2.0)], capture_mode="last")
    restored = AgentAction.from_dict(action.to_dict())
    assert restored.capture_mode == "last"
    assert AgentAction.trajectory([pose(1.0)]).capture_mode == "all"
    # Older serialized actions without the field default to "all".
    payload = action.to_dict()
    del payload["capture_mode"]
    assert AgentAction.from_dict(payload).capture_mode == "all"


def test_capture_mode_validation():
    with pytest.raises(ValueError):
        AgentAction.trajectory([pose(1.0)], capture_mode="every-other")


class _WaypointScriptAgent:
    """Emits one trajectory action, then done."""

    def __init__(self, capture_mode):
        self.capture_mode = capture_mode
        self.calls = 0

    def info(self):
        from activebench.api import MethodInfo

        return MethodInfo(name="script", needs_depth=False)

    def reset(self, seed, task=None):
        pass

    def act(self, observation):
        self.calls += 1
        if self.calls > 1:
            return AgentAction.done()
        return AgentAction.trajectory(
            [pose(1.0), pose(2.0), pose(3.0)], capture_mode=self.capture_mode
        )


class _FakeSim:
    """Minimal DynamicSceneSim stand-in for runner logic tests."""

    def __init__(self):
        from activebench.common.camera import CameraIntrinsics

        self.intrinsics = CameraIntrinsics.from_hfov(8, 6, 90.0)

    def observe(self, pose, t, include_clean=False, include_mask=False):
        rgb = np.zeros((6, 8, 3), dtype=np.uint8)
        depth = np.ones((6, 8), dtype=np.float32)
        return {"rgb": rgb, "depth": depth, "distractor_mask": np.zeros((6, 8), dtype=bool)}

    def render_clean(self, pose):
        return {"rgb": np.zeros((6, 8, 3), dtype=np.uint8), "depth": np.ones((6, 8), dtype=np.float32)}

    def scene_aabb(self):
        return np.array([[-5.0, -1.0, -5.0], [5.0, 3.0, 5.0]])

    def start_pose(self):
        return pose(0.0)

    def close(self):
        pass


@pytest.mark.parametrize("capture_mode,expected", [("all", 4), ("last", 2)])
def test_runner_honors_capture_mode(tmp_path, capture_mode, expected):
    from activebench.episode import EpisodeSpec
    from activebench.runner import run_episode

    spec = EpisodeSpec.from_dict(
        {
            "habitat": {"scene_path": "/tmp/fake.glb"},
            "episode": {"max_captures": 10, "capture_cost": 0.0},
            "eval": {"num_poses": 1},
        }
    )
    agent = _WaypointScriptAgent(capture_mode)
    manifest = run_episode(spec, agent, tmp_path / capture_mode, sim=_FakeSim())
    # "all": initial capture + one per waypoint = 4. "last": initial + final = 2.
    assert manifest["num_captures"] == expected


def test_uniform_reconstruction_frames_are_separate_from_agent_observations(tmp_path):
    from activebench.episode import EpisodeSpec
    from activebench.runner import run_episode

    spec = EpisodeSpec.from_dict(
        {
            "habitat": {"scene_path": "/tmp/fake.glb"},
            "motion": {"speed": 1.0},
            "episode": {
                "max_captures": 10,
                "max_sim_time": 10.0,
                "capture_cost": 0.0,
                "reconstruction_interval": 0.5,
            },
            "eval": {"num_poses": 1},
        }
    )
    out = tmp_path / "uniform"
    manifest = run_episode(spec, _WaypointScriptAgent("last"), out, sim=_FakeSim())

    assert manifest["num_captures"] == 2
    assert manifest["reconstruction"]["num_frames"] == 7
    assert manifest["clock"]["path_length_m"] == pytest.approx(3.0)
    assert manifest["clock"]["planning_wall_time_s"] >= 0.0
    assert [f["time"] for f in manifest["reconstruction"]["frames"]] == pytest.approx(
        [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    )
    assert (out / "transforms.json").exists()
    assert (out / "transforms_stream.json").exists()
    assert (out / "stream").is_dir()
    # Readers resolve the stream through the shared helper (name recorded in
    # the manifest keeps v2-era episodes readable too).
    from activebench.eval.dataset import training_transforms_path

    assert training_transforms_path(out).name == "transforms_stream.json"
    assert manifest["reconstruction"]["transforms"] == "transforms_stream.json"


def test_motion_is_clipped_at_mission_time_budget(tmp_path):
    from activebench.episode import EpisodeSpec
    from activebench.runner import run_episode

    spec = EpisodeSpec.from_dict(
        {
            "habitat": {"scene_path": "/tmp/fake.glb"},
            "motion": {"speed": 1.0},
            "episode": {
                "max_captures": 10,
                "max_sim_time": 1.25,
                "capture_cost": 0.0,
                "reconstruction_interval": 0.5,
            },
            "eval": {"num_poses": 1},
        }
    )
    manifest = run_episode(
        spec, _WaypointScriptAgent("last"), tmp_path / "clipped", sim=_FakeSim()
    )

    assert manifest["clock"]["final_sim_time"] == pytest.approx(1.25)
    assert manifest["num_captures"] == 1
    assert manifest["clock"]["path_length_m"] == pytest.approx(1.25)
    assert manifest["reconstruction"]["num_frames"] == 3
    assert manifest["reconstruction"]["frames"][-1]["time"] == pytest.approx(1.0)
