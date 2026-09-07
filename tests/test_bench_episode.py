import numpy as np
import pytest

from activebench.api import ActionKind, AgentAction, Observation
from activebench.episode import AgentMotionModel, EpisodeSpec, interpolate_pose
from activebench.common.camera import CameraIntrinsics, CameraPose


def pose(x=0.0, y=0.0, z=0.0, yaw=0.0, pitch=0.0):
    return CameraPose.from_xyz_yaw_pitch([x, y, z], yaw=yaw, pitch=pitch)


class TestMotionModel:
    def test_translation_time(self):
        motion = AgentMotionModel(speed=2.0)
        assert motion.travel_time(pose(), pose(x=4.0)) == pytest.approx(2.0)

    def test_rotation_dominates_when_slower(self):
        motion = AgentMotionModel(speed=100.0, yaw_rate=np.deg2rad(90.0))
        t = motion.travel_time(pose(), pose(x=0.1, yaw=np.pi / 2.0))
        assert t == pytest.approx(1.0)

    def test_yaw_wraps_shortest_arc(self):
        motion = AgentMotionModel(yaw_rate=np.deg2rad(90.0))
        t = motion.travel_time(pose(yaw=np.deg2rad(170.0)), pose(yaw=np.deg2rad(-170.0)))
        assert t == pytest.approx(20.0 / 90.0)

    def test_zero_motion_costs_nothing(self):
        assert AgentMotionModel().travel_time(pose(), pose()) == 0.0


class TestInterpolatePose:
    def test_midpoint(self):
        mid = interpolate_pose(pose(), pose(x=2.0, yaw=1.0), 0.5)
        np.testing.assert_allclose(mid.position, [1.0, 0.0, 0.0])
        assert mid.yaw == pytest.approx(0.5)

    def test_yaw_shortest_arc(self):
        mid = interpolate_pose(
            pose(yaw=np.deg2rad(170.0)), pose(yaw=np.deg2rad(-170.0)), 0.5
        )
        assert abs(mid.yaw) == pytest.approx(np.pi)


class TestActionSerialization:
    def test_move_to_round_trip(self):
        action = AgentAction.move_to(pose(x=1.0, yaw=0.3))
        restored = AgentAction.from_dict(action.to_dict())
        assert restored.kind == ActionKind.MOVE_TO
        np.testing.assert_allclose(restored.target.position, [1.0, 0.0, 0.0])
        assert restored.target.yaw == pytest.approx(0.3)

    def test_trajectory_round_trip(self):
        action = AgentAction.trajectory([pose(x=1.0), pose(x=2.0, pitch=0.1)])
        restored = AgentAction.from_dict(action.to_dict())
        assert len(restored.waypoints) == 2
        assert restored.waypoints[1].pitch == pytest.approx(0.1)

    def test_invalid_actions_rejected(self):
        with pytest.raises(ValueError):
            AgentAction(kind="move_to")
        with pytest.raises(ValueError):
            AgentAction(kind="trajectory")
        with pytest.raises(ValueError):
            AgentAction(kind="teleport")


class TestObservation:
    def test_to_dict_references_arrays_by_path(self):
        observation = Observation(
            rgb=np.zeros((4, 4, 3), dtype=np.uint8),
            pose=pose(x=1.0),
            time=2.5,
            intrinsics=CameraIntrinsics.from_hfov(320, 240, 90.0),
            step=3,
            rgb_path="frames/frame_00003.png",
        )
        payload = observation.to_dict()
        assert payload["time"] == 2.5
        assert payload["rgb_path"] == "frames/frame_00003.png"
        assert payload["depth_path"] is None
        assert payload["intrinsics"]["width"] == 320


class TestEpisodeSpec:
    def test_from_dict_defaults_and_overrides(self):
        spec = EpisodeSpec.from_dict(
            {
                "habitat": {"scene_path": "/tmp/scene.glb"},
                "seed": 5,
                "distractors": [
                    {
                        "name": "d0",
                        "object_template": "chefcan",
                        "trajectory": {"type": "circle", "center": [0, 0, 0], "radius": 1.0},
                    }
                ],
                "motion": {"speed": 1.5, "yaw_rate_deg": 90.0},
                "episode": {
                    "max_captures": 5,
                    "capture_cost": 0.0,
                },
                "eval": {"num_poses": 4},
            }
        )
        assert spec.seed == 5
        assert spec.scene.trajectory_seed == 5
        assert spec.scene.habitat.enable_physics is True
        assert len(spec.scene.distractors) == 1
        assert spec.motion.speed == 1.5
        assert spec.motion.yaw_rate == pytest.approx(np.deg2rad(90.0))
        assert spec.max_captures == 5
        assert spec.capture_cost == 0.0
        assert spec.reconstruction_interval is None


    def test_uniform_reconstruction_requires_a_continuous_clock(self):
        payload = {
            "habitat": {"scene_path": "/tmp/scene.glb"},
            "episode": {"capture_cost": 0.5, "reconstruction_interval": 1.0},
        }
        with pytest.raises(ValueError, match="capture_cost=0"):
            EpisodeSpec.from_dict(payload)

    def test_resolved_start_pose(self):
        spec = EpisodeSpec.from_dict(
            {"habitat": {"scene_path": "/tmp/scene.glb"}, "start_pose": [1, 2, 3, 0.5, 0.1]}
        )
        start = spec.resolved_start_pose(pose())
        np.testing.assert_allclose(start.position, [1.0, 2.0, 3.0])
        assert start.yaw == pytest.approx(0.5)
        default = EpisodeSpec.from_dict({"habitat": {"scene_path": "/tmp/scene.glb"}})
        assert default.resolved_start_pose(pose(x=9.0)).position[0] == 9.0
