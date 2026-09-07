"""Distractor trajectories: deterministic rigid-object pose as a function of sim time.

Every trajectory is a pure function of (config, seed, t) so episodes replay
exactly and clean ground-truth renders can be produced at any capture time.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Tuple

import numpy as np


class Trajectory(Protocol):
    """Continuous rigid-object trajectory over sim time."""

    def pose_at(self, t: float) -> Tuple[np.ndarray, float]:
        """Return (position xyz, yaw radians) at sim time ``t`` seconds."""
        ...


def _segment_lengths(waypoints: np.ndarray, closed: bool) -> np.ndarray:
    points = waypoints
    if closed:
        points = np.vstack([waypoints, waypoints[:1]])
    return np.linalg.norm(np.diff(points, axis=0), axis=1)


@dataclass
class WaypointPatrol:
    """Constant-speed patrol along a waypoint polyline.

    ``mode`` is ``loop`` (closed circuit) or ``pingpong`` (reverse at the ends).
    ``phase`` shifts the start position along the path, in seconds.
    """

    waypoints: np.ndarray
    speed: float = 0.5
    mode: str = "loop"
    phase: float = 0.0
    face_motion: bool = True

    def __post_init__(self) -> None:
        self.waypoints = np.asarray(self.waypoints, dtype=np.float64)
        if self.waypoints.ndim != 2 or self.waypoints.shape[1] != 3:
            raise ValueError("waypoints must have shape (N, 3)")
        if len(self.waypoints) < 2:
            raise ValueError("waypoints must contain at least two points")
        if self.mode not in ("loop", "pingpong"):
            raise ValueError("mode must be 'loop' or 'pingpong'")
        if self.speed <= 0.0:
            raise ValueError("speed must be positive")
        self._lengths = _segment_lengths(self.waypoints, closed=self.mode == "loop")
        self._cum = np.concatenate([[0.0], np.cumsum(self._lengths)])
        self._path_length = float(self._cum[-1])
        if self._path_length <= 0.0:
            raise ValueError("waypoints are degenerate (zero path length)")

    def _arc_position(self, s: float) -> Tuple[np.ndarray, np.ndarray]:
        """Return (position, unit tangent) at arc length ``s`` along the path."""

        points = self.waypoints
        if self.mode == "loop":
            points = np.vstack([self.waypoints, self.waypoints[:1]])
        s = float(np.clip(s, 0.0, self._path_length))
        idx = int(np.searchsorted(self._cum, s, side="right") - 1)
        idx = min(idx, len(self._lengths) - 1)
        seg_len = self._lengths[idx]
        alpha = 0.0 if seg_len <= 0.0 else (s - self._cum[idx]) / seg_len
        a, b = points[idx], points[idx + 1]
        direction = b - a
        norm = np.linalg.norm(direction)
        tangent = direction / norm if norm > 0.0 else np.array([1.0, 0.0, 0.0])
        return a + alpha * direction, tangent

    def pose_at(self, t: float) -> Tuple[np.ndarray, float]:
        distance = (float(t) + self.phase) * self.speed
        reverse = False
        if self.mode == "loop":
            s = distance % self._path_length
        else:
            period = 2.0 * self._path_length
            s = distance % period
            if s > self._path_length:
                s = period - s
                reverse = True
        position, tangent = self._arc_position(s)
        if reverse:
            tangent = -tangent
        yaw = float(np.arctan2(tangent[0], tangent[2])) if self.face_motion else 0.0
        return position, yaw


@dataclass
class CircleOrbit:
    """Constant angular-speed orbit in a horizontal circle."""

    center: np.ndarray
    radius: float
    angular_speed: float = 0.5
    phase: float = 0.0
    face_motion: bool = True

    def __post_init__(self) -> None:
        self.center = np.asarray(self.center, dtype=np.float64)
        if self.center.shape != (3,):
            raise ValueError("center must be an xyz point")
        if self.radius <= 0.0:
            raise ValueError("radius must be positive")

    def pose_at(self, t: float) -> Tuple[np.ndarray, float]:
        angle = self.phase + self.angular_speed * float(t)
        offset = np.array([np.cos(angle), 0.0, np.sin(angle)]) * self.radius
        position = self.center + offset
        if not self.face_motion:
            return position, 0.0
        tangent = np.array([-np.sin(angle), 0.0, np.cos(angle)]) * np.sign(self.angular_speed)
        yaw = float(np.arctan2(tangent[0], tangent[2]))
        return position, yaw


def random_walk_patrol(
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
    num_waypoints: int,
    speed: float,
    seed: int,
    mode: str = "pingpong",
) -> WaypointPatrol:
    """Build a seeded random-walk patrol inside an axis-aligned box.

    Waypoints are precomputed from the seed, so the trajectory is a
    deterministic function of time like every other trajectory.
    """

    if num_waypoints < 2:
        raise ValueError("num_waypoints must be at least 2")
    rng = np.random.default_rng(seed)
    lo = np.asarray(bounds_min, dtype=np.float64)
    hi = np.asarray(bounds_max, dtype=np.float64)
    waypoints = rng.uniform(lo, hi, size=(num_waypoints, 3))
    return WaypointPatrol(waypoints=waypoints, speed=speed, mode=mode)


@dataclass
class DistractorSpec:
    """One distractor: a Habitat object template plus a trajectory config."""

    name: str
    object_template: str
    trajectory: Dict[str, Any]
    scale: Optional[float] = None
    _built: Optional[Trajectory] = field(default=None, repr=False, compare=False)

    def build_trajectory(self, seed: int = 0) -> Trajectory:
        """Instantiate (and cache) the trajectory from its config dict."""

        if self._built is None:
            self._built = make_trajectory(self.trajectory, seed=seed)
        return self._built

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "DistractorSpec":
        return cls(
            name=str(payload["name"]),
            object_template=str(payload["object_template"]),
            trajectory=dict(payload["trajectory"]),
            scale=payload.get("scale"),
        )


def make_trajectory(config: Dict[str, Any], seed: int = 0) -> Trajectory:
    """Build a trajectory from a YAML/JSON-friendly config dict."""

    kind = config.get("type")
    params = {k: v for k, v in config.items() if k != "type"}
    if kind == "waypoint_patrol":
        return WaypointPatrol(
            waypoints=np.asarray(params["waypoints"], dtype=np.float64),
            speed=float(params.get("speed", 0.5)),
            mode=str(params.get("mode", "loop")),
            phase=float(params.get("phase", 0.0)),
            face_motion=bool(params.get("face_motion", True)),
        )
    if kind == "circle":
        return CircleOrbit(
            center=np.asarray(params["center"], dtype=np.float64),
            radius=float(params["radius"]),
            angular_speed=float(params.get("angular_speed", 0.5)),
            phase=float(params.get("phase", 0.0)),
            face_motion=bool(params.get("face_motion", True)),
        )
    if kind == "random_walk":
        return random_walk_patrol(
            bounds_min=np.asarray(params["bounds_min"], dtype=np.float64),
            bounds_max=np.asarray(params["bounds_max"], dtype=np.float64),
            num_waypoints=int(params.get("num_waypoints", 8)),
            speed=float(params.get("speed", 0.5)),
            seed=int(params.get("seed", seed)),
            mode=str(params.get("mode", "pingpong")),
        )
    raise ValueError("unknown trajectory type: %r" % kind)


def available_trajectory_types() -> List[str]:
    return ["waypoint_patrol", "circle", "random_walk"]
