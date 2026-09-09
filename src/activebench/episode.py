"""Episode specification and the 4D clock: agent motion model and time accounting.

The clock is ``motion-only`` by default: sim time advances with agent motion
(and a fixed per-capture cost), while planning is free. A ``realtime`` mode
that also charges planning wall-time to the clock is planned for agentic VLM
methods.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

from activebench.distractors import DistractorSpec
from activebench.sim import DynamicSceneConfig
from activebench.common.camera import CameraPose
from activebench.common.habitat_env import HabitatEnvConfig


def _wrap_angle(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


@dataclass
class AgentMotionModel:
    """Constant-rate motion model that converts pose changes into sim time."""

    speed: float = 0.5  # m/s
    yaw_rate: float = np.deg2rad(60.0)  # rad/s
    pitch_rate: float = np.deg2rad(60.0)  # rad/s

    def travel_time(self, start: CameraPose, end: CameraPose) -> float:
        """Time to move between poses; translation and rotation overlap."""

        distance = float(np.linalg.norm(end.position - start.position))
        return self.travel_time_along(distance, start, end)

    def travel_time_along(self, path_length: float, start: CameraPose, end: CameraPose) -> float:
        """Time to traverse a routed path of ``path_length`` between poses."""

        dyaw = abs(_wrap_angle(end.yaw - start.yaw))
        dpitch = abs(end.pitch - start.pitch)
        return max(path_length / self.speed, dyaw / self.yaw_rate, dpitch / self.pitch_rate)


def interpolate_pose(start: CameraPose, end: CameraPose, alpha: float) -> CameraPose:
    """Linear position interpolation with shortest-arc yaw."""

    alpha = float(np.clip(alpha, 0.0, 1.0))
    position = (1.0 - alpha) * start.position + alpha * end.position
    yaw = start.yaw + alpha * _wrap_angle(end.yaw - start.yaw)
    pitch = start.pitch + alpha * (end.pitch - start.pitch)
    return CameraPose.from_xyz_yaw_pitch(position, yaw=_wrap_angle(yaw), pitch=pitch)


@dataclass
class EpisodeSpec:
    """Everything that determines one benchmark episode besides the agent."""

    scene: DynamicSceneConfig
    seed: int = 0
    task: Optional[str] = None
    start_pose: Optional[List[float]] = None  # [x, y, z, yaw, pitch]; None = scene default
    motion: AgentMotionModel = field(default_factory=AgentMotionModel)
    max_captures: int = 20
    max_sim_time: float = 600.0  # seconds of scene time
    capture_cost: float = 0.5  # seconds charged per capture (shutter/hover)
    capture_interval: Optional[float] = None  # legacy: en-route frames share the agent budget
    reconstruction_interval: Optional[float] = None  # independent trajectory samples for evaluation
    # Protocol v3: deliver the reconstruction stream to the agent too, so its
    # planning inputs match what Tier-2 evaluation retrains on.
    stream_observations: bool = False
    # Protocol v5 collision model: "none" = free-flight camera charged for
    # straight lines (v2-v4); "navmesh" = all motion follows the scene's
    # shortest navigable route and is charged for the routed length —
    # through-wall travel becomes physically impossible for every method.
    collision: str = "none"
    provide_depth: bool = True  # ignored for methods whose manifest declines depth

    def __post_init__(self) -> None:
        if self.collision not in ("none", "navmesh"):
            raise ValueError("collision must be 'none' or 'navmesh'")
        if self.capture_interval is not None and self.reconstruction_interval is not None:
            raise ValueError("capture_interval and reconstruction_interval are mutually exclusive")
        if self.reconstruction_interval is not None:
            if self.reconstruction_interval <= 0.0:
                raise ValueError("reconstruction_interval must be positive")
            if self.capture_cost != 0.0:
                raise ValueError(
                    "uniform-time reconstruction requires capture_cost=0 so its clock is continuous"
                )
        if self.stream_observations and self.reconstruction_interval is None:
            raise ValueError(
                "stream_observations delivers the reconstruction stream, so "
                "reconstruction_interval must be set"
            )

    @classmethod
    def from_yaml(cls, path: Path) -> "EpisodeSpec":
        from activebench.configuration import load_yaml
        payload = load_yaml(path)
        return cls.from_dict(payload)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "EpisodeSpec":
        habitat = HabitatEnvConfig(**payload["habitat"])
        distractors = [DistractorSpec.from_dict(d) for d in payload.get("distractors", [])]
        scene = DynamicSceneConfig(
            habitat=habitat,
            distractors=distractors,
            trajectory_seed=int(payload.get("trajectory_seed", payload.get("seed", 0))),
        )
        motion_cfg = payload.get("motion", {})
        motion = AgentMotionModel(
            speed=float(motion_cfg.get("speed", 0.5)),
            yaw_rate=np.deg2rad(float(motion_cfg.get("yaw_rate_deg", 60.0))),
            pitch_rate=np.deg2rad(float(motion_cfg.get("pitch_rate_deg", 60.0))),
        )
        episode = payload.get("episode", {})
        # Old configs carry an "eval" block for the removed per-episode eval
        # sets; it is deliberately ignored (shared eval sets replaced them).
        return cls(
            scene=scene,
            seed=int(payload.get("seed", 0)),
            task=payload.get("task"),
            start_pose=payload.get("start_pose"),
            motion=motion,
            max_captures=int(episode.get("max_captures", 20)),
            max_sim_time=float(episode.get("max_sim_time", 600.0)),
            capture_cost=float(episode.get("capture_cost", 0.5)),
            capture_interval=episode.get("capture_interval"),
            reconstruction_interval=episode.get("reconstruction_interval"),
            stream_observations=bool(episode.get("stream_observations", False)),
            collision=str(episode.get("collision", "none")),
            provide_depth=bool(episode.get("provide_depth", True)),
        )

    def resolved_start_pose(self, default: CameraPose) -> CameraPose:
        if self.start_pose is None:
            return default
        values = list(self.start_pose)
        return CameraPose.from_xyz_yaw_pitch(values[:3], yaw=float(values[3]), pitch=float(values[4]))
