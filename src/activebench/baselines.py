"""Reference agents: a random explorer and the NBV pool wrapper."""

from dataclasses import dataclass, field
from typing import Any, List, Optional

import numpy as np

from activebench.api import (
    ActionKind,
    AgentAction,
    CaptureRecord,
    MethodInfo,
    Observation,
    ViewSelector,
)
from activebench.common.camera import CameraPose


def forward_clear(observation: Observation, min_clearance: float) -> bool:
    """True when the central depth patch is farther than ``min_clearance``."""

    depth = observation.depth
    if depth is None:
        return False
    h, w = depth.shape[:2]
    center = depth[h // 3 : 2 * h // 3, w // 3 : 2 * w // 3]
    valid = center[center > 0.0]
    if valid.size == 0:
        return False
    return float(valid.min()) > min_clearance


@dataclass
class RandomAgent:
    """Depth-aware random walker: turn freely, step forward when clear."""

    step_size: float = 0.4
    min_clearance: float = 0.8
    max_pitch: float = np.deg2rad(30.0)
    _rng: np.random.Generator = field(default_factory=np.random.default_rng, repr=False)

    def info(self) -> MethodInfo:
        return MethodInfo(name="random", needs_depth=True, action_space="free")

    def reset(self, seed: int, task: Optional[str] = None) -> None:
        self._rng = np.random.default_rng(seed)

    def _forward_clear(self, observation: Observation) -> bool:
        return forward_clear(observation, self.min_clearance)

    def act(self, observation: Observation) -> AgentAction:
        pose = observation.pose
        yaw = pose.yaw + float(self._rng.uniform(-np.pi / 3.0, np.pi / 3.0))
        pitch = float(np.clip(pose.pitch + self._rng.uniform(-0.15, 0.15), -self.max_pitch, self.max_pitch))
        position = pose.position.copy()
        if self._forward_clear(observation):
            heading = CameraPose.from_xyz_yaw_pitch(position, yaw=yaw, pitch=0.0)
            # Cameras look along -Z of the pose frame in this convention, so
            # step against the forward vector to move where the camera faces.
            position = position - heading.forward() * self.step_size
        return AgentAction.move_to(CameraPose.from_xyz_yaw_pitch(position, yaw=yaw, pitch=pitch))


@dataclass
class WanderAgent:
    """Pose-free random walker: camera-frame hops, no world coordinates.

    Floor baseline for ``pose_access="none"``. It receives pose-masked
    observations and emits only camera-frame actions (turn, then hop forward
    when the central depth patch is clear). Clipped motion and collisions
    surface only through the next image — the regime a pose-free method
    lives in. Pitch is kept near level by dead-reckoning its own commands,
    which uses no ground truth (commanded ≠ executed after clipping, and
    that drift is part of the baseline's honesty).
    """

    step_range: tuple = (0.3, 0.9)
    max_turn: float = np.deg2rad(60.0)
    max_pitch_step: float = np.deg2rad(10.0)
    max_pitch_estimate: float = np.deg2rad(20.0)
    min_clearance: float = 0.8
    _rng: np.random.Generator = field(default_factory=np.random.default_rng, repr=False)
    _pitch_estimate: float = field(default=0.0, repr=False)

    def info(self) -> MethodInfo:
        return MethodInfo(
            name="wander", needs_depth=True, action_space="free", pose_access="none"
        )

    def reset(self, seed: int, task: Optional[str] = None) -> None:
        self._rng = np.random.default_rng(seed)
        self._pitch_estimate = 0.0

    def act(self, observation: Observation) -> AgentAction:
        dyaw = float(self._rng.uniform(-self.max_turn, self.max_turn))
        dpitch = float(np.clip(
            self._rng.uniform(-self.max_pitch_step, self.max_pitch_step),
            -self.max_pitch_estimate - self._pitch_estimate,
            self.max_pitch_estimate - self._pitch_estimate,
        ))
        self._pitch_estimate += dpitch
        forward = 0.0
        if forward_clear(observation, self.min_clearance):
            forward = float(self._rng.uniform(*self.step_range))
        # Hop level, not along the tilted view axis: cancel the commanded
        # pitch in the camera frame (+y is down in OpenCV axes) so looking
        # up does not accumulate altitude. Uses dead-reckoned commands only.
        level = self._pitch_estimate
        offset = [0.0, forward * float(np.sin(level)), forward * float(np.cos(level))]
        return AgentAction.move_rel(offset, dyaw=dyaw, dpitch=dpitch)


@dataclass
class RandomSelector:
    """Trivial ViewSelector: uniform choice over unvisited candidates."""

    seed: int = 0

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)

    def select(self, history: List[CaptureRecord], candidates: List[CameraPose]) -> int:
        return int(self._rng.integers(len(candidates)))


@dataclass
class PoolNBVAgent:
    """Lift a ViewSelector into an ActiveAgent over a candidate pool.

    Two pool modes:

    - ``fixed``: a pre-built global pool; chosen candidates are removed,
      matching the classic batch-NBV setting.
    - ``local``: R3CON-style embodied candidates, regenerated every round —
      positions drawn from *observed* free space (harness-side
      ``FreeSpaceTracker`` fed by the captured depth maps) within ``radius``
      of the current pose, each with one uniform-random view direction on
      the sphere. Until free space is observed (or when none is within
      reach) candidates fall back to in-place rotations at the current
      position.

    In ``local`` mode the agent requests depth for the harness tracker even
    if the wrapped selector is RGB-only; the selector still only reads what
    it wants from the capture records.
    """

    selector: ViewSelector
    pool: List[CameraPose] = field(default_factory=list)
    name: str = "pool-nbv"
    needs_depth: bool = False
    pool_mode: str = "fixed"  # "fixed" | "local"
    radius: float = 2.0  # local mode: R3CON confidence_pano action radius
    sample_num: int = 50  # local mode: R3CON sample_num
    scene_bbox: Optional[Any] = None  # local mode: tracker bounds, (2, 3)
    seed: int = 0
    _history: List[CaptureRecord] = field(default_factory=list, repr=False)
    _remaining: List[CameraPose] = field(default_factory=list, repr=False)
    _tracker: Optional[Any] = field(default=None, repr=False)
    _rng: Optional[np.random.Generator] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.pool_mode not in ("fixed", "local"):
            raise ValueError("pool_mode must be 'fixed' or 'local'")
        if self.pool_mode == "local" and self.scene_bbox is None:
            raise ValueError("local pool mode needs scene_bbox for the tracker")

    def info(self) -> MethodInfo:
        needs_depth = self.needs_depth or self.pool_mode == "local"
        return MethodInfo(name=self.name, needs_depth=needs_depth, action_space="pool")

    def reset(self, seed: int, task: Optional[str] = None) -> None:
        self._history = []
        self._remaining = [pose.copy() for pose in self.pool]
        self._rng = np.random.default_rng(seed)
        if self.pool_mode == "local":
            from activebench.free_space import FreeSpaceTracker

            self._tracker = FreeSpaceTracker(bbox=np.asarray(self.scene_bbox))

    def _local_candidates(self, observation: Observation) -> List[CameraPose]:
        from activebench.free_space import sample_view_directions

        positions = self._tracker.candidate_positions(
            observation.pose.position, self.radius, self.sample_num, self._rng
        )
        if len(positions) == 0:
            positions = np.tile(observation.pose.position, (self.sample_num, 1))
        yaws, pitches = sample_view_directions(len(positions), self._rng)
        return [
            CameraPose.from_xyz_yaw_pitch(positions[i], yaw=float(yaws[i]), pitch=float(pitches[i]))
            for i in range(len(positions))
        ]

    def act(self, observation: Observation) -> AgentAction:
        # Protocol v3: fold passively-streamed trajectory frames into the
        # capture history, so the wrapped selector trains/ranks on the same
        # stream Tier-2 evaluation retrains on.
        for frame in observation.stream_frames:
            self._history.append(
                CaptureRecord(
                    pose=frame.pose,
                    time=frame.time,
                    rgb_path=frame.rgb_path,
                    depth_path=frame.depth_path,
                )
            )
            if self.pool_mode == "local" and frame.depth_path:
                self._tracker.update(
                    frame.load_depth(), frame.pose, observation.intrinsics
                )
        self._history.append(
            CaptureRecord(
                pose=observation.pose,
                time=observation.time,
                rgb_path=observation.rgb_path,
                depth_path=observation.depth_path,
            )
        )
        if self.pool_mode == "local":
            if observation.depth is None:
                raise ValueError("local pool mode requires depth observations")
            self._tracker.update(observation.depth, observation.pose, observation.intrinsics)
            candidates = self._local_candidates(observation)
            index = self.selector.select(self._history, candidates)
            return AgentAction.move_to(candidates[index])
        if not self._remaining:
            return AgentAction.done()
        index = self.selector.select(self._history, self._remaining)
        target = self._remaining.pop(index)
        return AgentAction.move_to(target)
