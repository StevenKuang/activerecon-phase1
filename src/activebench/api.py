"""Benchmark-facing interfaces: observations, actions, and method protocols.

Two adapter tiers share one runner interface:

- ``ActiveAgent`` is the native interface (full policies: MAGICIAN-style
  agents and other full-policy implementations).
- ``ViewSelector`` covers next-best-view methods (FisherRF, GAVIS) that rank a
  candidate pose pool; ``activebench.baselines.PoolNBVAgent`` lifts a selector
  into an ``ActiveAgent``.

All payloads have dict round-trips so the same interface runs in-process or
across the subprocess RPC boundary (arrays travel as file references there).
"""

from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Protocol, Sequence, runtime_checkable

import numpy as np

from activebench.common.camera import CameraIntrinsics, CameraPose


def pose_to_list(pose: CameraPose) -> List[float]:
    return pose.as_list()


def pose_from_list(values: Sequence[float]) -> CameraPose:
    return CameraPose.from_xyz_yaw_pitch(values[:3], yaw=float(values[3]), pitch=float(values[4]))


@dataclass
class StreamFrame:
    """One passively-recorded trajectory frame delivered with an Observation.

    Protocol v3: the runner samples the executed trajectory at the
    reconstruction interval (1 Hz) and hands those frames to the agent in the
    next Observation, so every method sees the same stream its Tier-2
    evaluation will be trained on. Frames are persisted as files; agents load
    pixels lazily, and RPC sends file paths.
    """

    pose: Optional[CameraPose]
    time: float
    rgb_path: str
    depth_path: Optional[str] = None

    def load_rgb(self) -> np.ndarray:
        from PIL import Image

        return np.asarray(Image.open(self.rgb_path))

    def load_depth(self) -> Optional[np.ndarray]:
        return np.load(self.depth_path) if self.depth_path else None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pose": pose_to_list(self.pose) if self.pose is not None else None,
            "time": float(self.time),
            "rgb_path": self.rgb_path,
            "depth_path": self.depth_path,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "StreamFrame":
        pose = payload["pose"]
        return cls(
            pose=pose_from_list(pose) if pose is not None else None,
            time=float(payload["time"]),
            rgb_path=payload["rgb_path"],
            depth_path=payload.get("depth_path"),
        )


@dataclass
class Observation:
    """What an agent sees at one capture instant.

    ``pose`` (and the poses inside ``stream_frames``) is ground truth and is
    set to ``None`` for methods declaring ``pose_access="none"``; see
    :func:`mask_observation_pose`.
    """

    rgb: np.ndarray
    pose: Optional[CameraPose]
    time: float
    intrinsics: CameraIntrinsics
    depth: Optional[np.ndarray] = None
    task: Optional[str] = None
    step: int = 0
    # Absolute paths of the persisted arrays, set by the runner so RPC proxies
    # can send file references.
    rgb_path: Optional[str] = None
    depth_path: Optional[str] = None
    # Protocol v3: trajectory frames recorded since the previous decision
    # (empty under the v2 decision-frames-only protocol).
    stream_frames: List[StreamFrame] = field(default_factory=list)
    # Benchmark-owned safe action chunks for the candidate-matched UAV macro
    # probe. Every item is JSON data ``{"name": str, "chunk": [[...]]}`` in
    # the camera frame, so it never exposes a world pose or evaluator truth.
    action_candidates: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize metadata; arrays are referenced by path for RPC transport."""

        return {
            "pose": pose_to_list(self.pose) if self.pose is not None else None,
            "time": float(self.time),
            "step": int(self.step),
            "task": self.task,
            "rgb_path": self.rgb_path,
            "depth_path": self.depth_path if self.depth is not None else None,
            "stream_frames": [frame.to_dict() for frame in self.stream_frames],
            "action_candidates": self.action_candidates,
            "intrinsics": {
                "width": self.intrinsics.width,
                "height": self.intrinsics.height,
                "fx": self.intrinsics.fx,
                "fy": self.intrinsics.fy,
                "cx": self.intrinsics.cx,
                "cy": self.intrinsics.cy,
            },
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "Observation":
        """Rebuild an observation on the far side of an RPC boundary.

        Pixel data is loaded from the referenced files (PNG for RGB, npy for
        depth); a missing depth path yields ``depth=None``.
        """

        from PIL import Image

        rgb = np.asarray(Image.open(payload["rgb_path"]))
        depth_path = payload.get("depth_path")
        depth = np.load(depth_path) if depth_path else None
        intr = payload["intrinsics"]
        pose = payload["pose"]
        return cls(
            rgb=rgb,
            depth=depth,
            pose=pose_from_list(pose) if pose is not None else None,
            time=float(payload["time"]),
            intrinsics=CameraIntrinsics(
                width=int(intr["width"]),
                height=int(intr["height"]),
                fx=float(intr["fx"]),
                fy=float(intr["fy"]),
                cx=float(intr["cx"]),
                cy=float(intr["cy"]),
            ),
            task=payload.get("task"),
            step=int(payload.get("step", 0)),
            rgb_path=payload["rgb_path"],
            depth_path=depth_path,
            stream_frames=[
                StreamFrame.from_dict(frame)
                for frame in payload.get("stream_frames", [])
            ],
            action_candidates=list(payload.get("action_candidates", [])),
        )


class ActionKind:
    MOVE_TO = "move_to"
    TRAJECTORY = "trajectory"
    CAPTURE = "capture"
    DONE = "done"


@dataclass
class AgentAction:
    """One agent decision.

    - ``move_to``: travel to ``target`` and capture on arrival.
    - ``trajectory``: traverse ``waypoints`` in order. ``capture_mode``
      controls where frames are taken: ``"all"`` captures at every waypoint
      (continuous-control methods emitting trajectory chunks), ``"last"``
      treats intermediate waypoints as travel-only path and captures at the
      final pose only (classic NBV planners returning a collision-free path).
    - ``capture``: capture again at the current pose (time advances by
      ``capture_cost`` from the episode spec).
    - ``done``: end the episode early.

    ``frame`` selects the coordinate frame of ``target``/``waypoints``:
    ``"world"`` (default) or ``"camera"``. Camera-frame poses are relative
    to the camera at the decision instant (OpenCV axes: +x right, +y down, +z forward;
    yaw/pitch are deltas on the current values). Camera-frame actions are the
    action space of pose-free methods; the runner resolves them against the
    true pose via :func:`resolve_action_frame`, so agents never need world
    coordinates.
    """

    kind: str
    target: Optional[CameraPose] = None
    waypoints: List[CameraPose] = field(default_factory=list)
    capture_mode: str = "all"
    frame: str = "world"

    def __post_init__(self) -> None:
        if self.kind not in (
            ActionKind.MOVE_TO,
            ActionKind.TRAJECTORY,
            ActionKind.CAPTURE,
            ActionKind.DONE,
        ):
            raise ValueError("unknown action kind: %r" % self.kind)
        if self.kind == ActionKind.MOVE_TO and self.target is None:
            raise ValueError("move_to action requires a target pose")
        if self.kind == ActionKind.TRAJECTORY and not self.waypoints:
            raise ValueError("trajectory action requires waypoints")
        if self.capture_mode not in ("all", "last"):
            raise ValueError("capture_mode must be 'all' or 'last'")
        if self.frame not in ("world", "camera"):
            raise ValueError("frame must be 'world' or 'camera'")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "target": pose_to_list(self.target) if self.target is not None else None,
            "waypoints": [pose_to_list(p) for p in self.waypoints],
            "capture_mode": self.capture_mode,
            "frame": self.frame,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "AgentAction":
        target = payload.get("target")
        return cls(
            kind=str(payload["kind"]),
            target=pose_from_list(target) if target is not None else None,
            waypoints=[pose_from_list(p) for p in payload.get("waypoints", [])],
            capture_mode=str(payload.get("capture_mode", "all")),
            frame=str(payload.get("frame", "world")),
        )

    @classmethod
    def move_to(cls, target: CameraPose) -> "AgentAction":
        return cls(kind=ActionKind.MOVE_TO, target=target)

    @classmethod
    def move_rel(cls, offset: Sequence[float], dyaw: float = 0.0, dpitch: float = 0.0) -> "AgentAction":
        """Camera-frame move: ``offset`` = (right, down, forward) in meters."""

        target = CameraPose.from_xyz_yaw_pitch(offset, yaw=float(dyaw), pitch=float(dpitch))
        return cls(kind=ActionKind.MOVE_TO, target=target, frame="camera")

    @classmethod
    def trajectory(
        cls, waypoints: Sequence[CameraPose], capture_mode: str = "all", frame: str = "world"
    ) -> "AgentAction":
        return cls(
            kind=ActionKind.TRAJECTORY, waypoints=list(waypoints),
            capture_mode=capture_mode, frame=frame,
        )

    @classmethod
    def capture(cls) -> "AgentAction":
        return cls(kind=ActionKind.CAPTURE)

    @classmethod
    def done(cls) -> "AgentAction":
        return cls(kind=ActionKind.DONE)


@dataclass
class MethodInfo:
    """Manifest a method declares so the runner can provision it fairly."""

    name: str
    needs_depth: bool = False
    needs_task_text: bool = False
    action_space: str = "free"  # "free" | "pool"
    conda_env: Optional[str] = None
    # Ego-pose privilege: "gt" methods receive ground-truth poses; "none"
    # methods get pose-masked observations and act in the camera frame.
    # A noisy "odometry" pose tier is planned for future implementation.
    pose_access: str = "gt"
    # The optional decision_diagnostics() callback is invoked only when this
    # is true. Diagnostics are written after action selection and never become
    # policy input or evaluator data.
    emits_decision_diagnostics: bool = False

    def __post_init__(self) -> None:
        if self.pose_access not in ("gt", "none"):
            raise ValueError("pose_access must be 'gt' or 'none'")


@runtime_checkable
class ActiveAgent(Protocol):
    """Full active-reconstruction policy."""

    def info(self) -> MethodInfo:
        ...

    def reset(self, seed: int, task: Optional[str] = None) -> None:
        ...

    def act(self, observation: Observation) -> AgentAction:
        ...


@dataclass
class CaptureRecord:
    """Pose/time bookkeeping for one captured frame."""

    pose: CameraPose
    time: float
    rgb_path: Optional[str] = None
    depth_path: Optional[str] = None


@runtime_checkable
class ViewSelector(Protocol):
    """Next-best-view method: pick the next capture from a candidate pool."""

    def select(self, history: List[CaptureRecord], candidates: List[CameraPose]) -> int:
        """Return the index of the chosen candidate pose."""
        ...


def mask_observation_pose(observation: Observation) -> Observation:
    """Copy of ``observation`` with every ground-truth pose withheld.

    The single enforcement point for ``pose_access="none"``: both the ego
    pose and the poses of the delivered stream frames are blanked (a stream
    pose would otherwise expose localization). Pixel arrays retain shared
    references.
    """

    return replace(
        observation,
        pose=None,
        stream_frames=[replace(f, pose=None) for f in observation.stream_frames],
    )


def resolve_action_frame(action: AgentAction, current: CameraPose) -> AgentAction:
    """Return a world-frame equivalent of ``action`` decided at ``current``.

    World-frame actions pass through untouched. Camera-frame poses are
    offsets in the decision camera's OpenCV axes (+x right, +y down,
    +z forward) with yaw/pitch deltas; every waypoint of a trajectory is
    relative to the same decision pose.
    """

    if action.frame == "world":
        return action
    from activebench.convention import pose_to_c2w_cv

    rotation = pose_to_c2w_cv(current)[:3, :3]

    def to_world(rel: CameraPose) -> CameraPose:
        return CameraPose.from_xyz_yaw_pitch(
            np.asarray(current.position) + rotation @ np.asarray(rel.position),
            yaw=float(current.yaw + rel.yaw),
            pitch=float(current.pitch + rel.pitch),
        )

    return AgentAction(
        kind=action.kind,
        target=to_world(action.target) if action.target is not None else None,
        waypoints=[to_world(p) for p in action.waypoints],
        capture_mode=action.capture_mode,
    )
