"""Episode runner: drives an ActiveAgent through a dynamic scene and logs data.

Output layout (one directory per episode):

- ``frames/`` — captured RGB (png), GT depth (npy), distractor masks (png)
- ``transforms.json`` — planner observations at method decision poses
- ``stream/`` + ``transforms_stream.json`` — the recorded video stream
  (uniform-time trajectory samples); input to coverage and Tier-2. Distinct
  from ``reconstructions/``, which holds models built FROM it.
- ``manifest.json`` — episode spec summary, per-capture clock log, totals
"""

from dataclasses import dataclass, field, replace
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional

import numpy as np

from activebench.api import (
    ActiveAgent,
    AgentAction,
    ActionKind,
    MethodInfo,
    Observation,
    mask_observation_pose,
    resolve_action_frame,
)
from activebench.episode import EpisodeSpec, interpolate_pose
from activebench.sim import DynamicSceneSim
from activebench.common.transforms_export import export_transforms_json, frame_from_pose
from activebench.common.camera import CameraPose
from activebench.common.image import save_rgb_png
from activebench.common.io import ensure_dir, save_json


@dataclass
class EpisodeState:
    """Mutable episode progress shared by the capture/motion helpers."""

    pose: CameraPose
    time: float = 0.0
    captures: int = 0
    frames: List[Dict[str, Any]] = field(default_factory=list)
    capture_log: List[Dict[str, Any]] = field(default_factory=list)
    # Every policy command is retained both before and after the runner alone
    # resolves a camera-frame action. This makes pose-free/VLA action codecs
    # auditable without leaking the world pose back to the agent.
    action_log: List[Dict[str, Any]] = field(default_factory=list)
    reconstruction_frames: List[Dict[str, Any]] = field(default_factory=list)
    reconstruction_log: List[Dict[str, Any]] = field(default_factory=list)
    next_reconstruction_time: Optional[float] = None
    # Protocol v3: reconstruction frames not yet delivered to the agent.
    pending_stream: List[Any] = field(default_factory=list)
    path_length: float = 0.0
    planning_wall_time: float = 0.0
    # v5 navmesh collision: targets no route could reach (rotation-only).
    unreachable_targets: int = 0

    def budget_left(self, spec: EpisodeSpec) -> bool:
        return self.captures < spec.max_captures and self.time < spec.max_sim_time


def _save_mask_png(path: Path, mask: np.ndarray) -> None:
    gray = mask.astype(np.uint8) * 255
    save_rgb_png(path, np.repeat(gray[..., None], 3, axis=-1))


class SegmentPath:
    """One motion segment as a piecewise-linear path with endpoint rotations.

    Free flight is the straight line start→target. Under navmesh collision
    the position rides the routed floor polyline lifted to camera height
    (the endpoints' heights above their floor points, blended by arc
    length), while yaw/pitch interpolate exactly as in free flight. The
    charged length is the 3D length of the traversed polyline.
    """

    def __init__(self, start: CameraPose, target: CameraPose, route_points=None):
        self.start = start.copy()
        self.target = target.copy()
        verts = None
        if route_points is not None and len(route_points) >= 2:
            floor = np.asarray(route_points, dtype=np.float64)
            seg = np.linalg.norm(np.diff(floor, axis=0), axis=1)
            cum = np.concatenate([[0.0], np.cumsum(seg)])
            if cum[-1] > 0.0:
                blend = cum / cum[-1]
                height_start = start.position[1] - floor[0, 1]
                height_end = target.position[1] - floor[-1, 1]
                verts = floor.copy()
                verts[:, 1] += (1.0 - blend) * height_start + blend * height_end
                verts[0] = start.position
                verts[-1] = target.position
        if verts is None:
            verts = np.stack([start.position, target.position]).astype(np.float64)
        self._verts = verts
        seg = np.linalg.norm(np.diff(verts, axis=0), axis=1)
        self._cum = np.concatenate([[0.0], np.cumsum(seg)])
        self.length = float(self._cum[-1])

    def pose_at(self, alpha: float) -> CameraPose:
        alpha = float(np.clip(alpha, 0.0, 1.0))
        if self.length <= 0.0:
            position = self.start.position
        else:
            distance = alpha * self.length
            index = int(np.searchsorted(self._cum, distance, side="right"))
            index = min(max(index, 1), len(self._verts) - 1)
            span = self._cum[index] - self._cum[index - 1]
            local = 0.0 if span <= 0.0 else (distance - self._cum[index - 1]) / span
            position = (1.0 - local) * self._verts[index - 1] + local * self._verts[index]
        # Rotation semantics identical to interpolate_pose (shortest-arc yaw).
        rotated = interpolate_pose(self.start, self.target, alpha)
        return CameraPose.from_xyz_yaw_pitch(position, yaw=rotated.yaw, pitch=rotated.pitch)


class EpisodeRunner:
    def __init__(self, spec: EpisodeSpec, sim: DynamicSceneSim, out_dir: Path) -> None:
        self.spec = spec
        self.sim = sim
        # Resolved so Observation/StreamFrame path references stay valid in
        # worker processes that chdir (e.g. the MAGICIAN adapter) — the api
        # contract promises absolute paths.
        self.out_dir = ensure_dir(Path(out_dir)).resolve()
        self.frames_dir = ensure_dir(self.out_dir / "frames")

    def _capture(self, state: EpisodeState, provide_depth: bool) -> Observation:
        """Render at (pose, time), persist the frame, and charge capture cost."""

        index = state.captures
        result = self.sim.observe(state.pose, state.time, include_mask=True)
        rgb_rel = "frames/frame_%05d.png" % index
        depth_rel = "frames/depth_%05d.npy" % index
        mask_rel = "frames/mask_%05d.png" % index
        save_rgb_png(self.out_dir / rgb_rel, result["rgb"])
        np.save(self.out_dir / depth_rel, result["depth"])
        _save_mask_png(self.out_dir / mask_rel, result["distractor_mask"])

        frame = frame_from_pose(state.pose, rgb_rel, depth_path=depth_rel)
        frame["time"] = float(state.time)
        frame["mask_path"] = mask_rel
        state.frames.append(frame)
        state.capture_log.append(
            {
                "index": index,
                "time": float(state.time),
                "pose": state.pose.as_list(),
                "distractor_pixel_fraction": float(result["distractor_mask"].mean()),
            }
        )

        observation = Observation(
            rgb=result["rgb"],
            depth=result["depth"] if provide_depth else None,
            pose=state.pose.copy(),
            time=state.time,
            intrinsics=self.sim.intrinsics,
            task=self.spec.task,
            step=index,
            rgb_path=str(self.out_dir / rgb_rel),
            depth_path=str(self.out_dir / depth_rel) if provide_depth else None,
            stream_frames=list(state.pending_stream),
        )
        state.pending_stream.clear()
        state.captures += 1
        state.time += self.spec.capture_cost
        return observation

    def _capture_reconstruction(
        self,
        state: EpisodeState,
        pose: CameraPose,
        capture_time: float,
        provide_depth: bool = False,
    ) -> None:
        """Record an evaluation frame; under protocol v3 it is also queued
        for delivery to the agent with the next observation."""

        index = len(state.reconstruction_frames)
        result = self.sim.observe(pose, capture_time, include_mask=True)
        rgb_rel = "stream/frame_%05d.png" % index
        depth_rel = "stream/depth_%05d.npy" % index
        mask_rel = "stream/mask_%05d.png" % index
        ensure_dir(self.out_dir / "stream")
        save_rgb_png(self.out_dir / rgb_rel, result["rgb"])
        np.save(self.out_dir / depth_rel, result["depth"])
        _save_mask_png(self.out_dir / mask_rel, result["distractor_mask"])

        frame = frame_from_pose(pose, rgb_rel, depth_path=depth_rel)
        frame["time"] = float(capture_time)
        frame["mask_path"] = mask_rel
        state.reconstruction_frames.append(frame)
        state.reconstruction_log.append(
            {
                "index": index,
                "time": float(capture_time),
                "pose": pose.as_list(),
                "distractor_pixel_fraction": float(result["distractor_mask"].mean()),
            }
        )
        if self.spec.stream_observations:
            from activebench.api import StreamFrame

            state.pending_stream.append(StreamFrame(
                pose=pose.copy(),
                time=float(capture_time),
                rgb_path=str(self.out_dir / rgb_rel),
                depth_path=str(self.out_dir / depth_rel) if provide_depth else None,
            ))

    def _move_segment(self, state: EpisodeState, target: CameraPose, provide_depth: bool) -> None:
        """Advance the clock along one motion segment, capturing en route.

        Under ``collision="navmesh"`` the segment follows the scene's
        shortest navigable route and is charged for the routed length; an
        unreachable target translates nothing (rotation only) — the method
        discovers the outcome in its next observation.
        """

        if self.spec.collision == "navmesh":
            found = self.sim.navmesh_route(state.pose.position, target.position)
            if found is None:
                state.unreachable_targets += 1
                target = CameraPose.from_xyz_yaw_pitch(
                    state.pose.position, yaw=target.yaw, pitch=target.pitch)
                path = SegmentPath(state.pose, target)
            else:
                path = SegmentPath(state.pose, target, found[0])
        else:
            path = SegmentPath(state.pose, target)

        full_duration = self.spec.motion.travel_time_along(path.length, state.pose, target)
        start_time = state.time
        duration = min(full_duration, max(0.0, self.spec.max_sim_time - start_time))
        alpha_end = 1.0 if full_duration <= 0.0 else duration / full_duration
        end_pose = path.pose_at(alpha_end)

        interval = self.spec.capture_interval
        if interval is not None and duration > 0.0:
            elapsed = interval
            while elapsed < duration and state.budget_left(self.spec):
                state.pose = path.pose_at(elapsed / full_duration)
                state.time = start_time + elapsed
                self._capture(state, provide_depth)
                # Capture cost pushes the clock forward; keep en-route samples
                # on the original traversal schedule.
                elapsed = (state.time - start_time) + interval

        reconstruction_interval = self.spec.reconstruction_interval
        if reconstruction_interval is not None and duration > 0.0:
            next_time = state.next_reconstruction_time
            while next_time is not None and next_time <= start_time + duration + 1e-9:
                sample_pose = path.pose_at((next_time - start_time) / full_duration)
                self._capture_reconstruction(state, sample_pose, next_time, provide_depth)
                next_time += reconstruction_interval
            state.next_reconstruction_time = next_time

        state.path_length += alpha_end * path.length
        state.pose = end_pose
        state.time = start_time + duration




    def run(self, agent: ActiveAgent) -> Dict[str, Any]:
        info = agent.info()
        provide_depth = self.spec.provide_depth and info.needs_depth
        provide_pose = info.pose_access != "none"
        agent.reset(seed=self.spec.seed, task=self.spec.task)

        state = EpisodeState(pose=self.spec.resolved_start_pose(self.sim.start_pose()))
        if self.spec.collision == "navmesh":
            self.sim.set_navigation_anchor(state.pose.position)
        observation = self._capture(state, provide_depth)
        if self.spec.reconstruction_interval is not None:
            self._capture_reconstruction(state, state.pose, 0.0, provide_depth)
            # The t=0 sample duplicates the first decision frame exactly.
            state.pending_stream.clear()
            state.next_reconstruction_time = self.spec.reconstruction_interval

        while state.budget_left(self.spec):
            candidate_actions: List[Dict[str, Any]] = []
            selected_candidate_name: Optional[str] = None
            action_observation = observation
            planning_started = time.perf_counter()
            raw_action = agent.act(
                action_observation
                if provide_pose
                else mask_observation_pose(action_observation)
            )
            planning_elapsed = time.perf_counter() - planning_started
            state.planning_wall_time += planning_elapsed
            decision_diagnostics = None
            if info.emits_decision_diagnostics:
                callback = getattr(agent, "decision_diagnostics", None)
                if not callable(callback):
                    raise RuntimeError(
                        "method declares decision diagnostics but implements no callback"
                    )
                decision_diagnostics = callback()
                if not isinstance(decision_diagnostics, dict):
                    raise RuntimeError("decision diagnostics must be a JSON object")
                try:
                    json.dumps(decision_diagnostics)
                except (TypeError, ValueError) as exc:
                    raise RuntimeError("decision diagnostics must be JSON serializable") from exc
            # Camera-frame actions (pose-free methods) become world targets
            # here; only the runner ever holds the true pose.
            action = resolve_action_frame(raw_action, state.pose)
            state.action_log.append(
                {
                    "step": int(observation.step),
                    "time": float(state.time),
                    "decision_pose": state.pose.as_list(),
                    "planning_wall_time_s": float(planning_elapsed),
                    "candidate_actions": candidate_actions,
                    "selected_candidate_name": selected_candidate_name,
                    "raw_action": raw_action.to_dict(),
                    "resolved_action": action.to_dict(),
                    **(
                        {"agent_diagnostics": decision_diagnostics}
                        if decision_diagnostics is not None
                        else {}
                    ),
                }
            )
            if action.kind == ActionKind.DONE:
                break
            if action.kind == ActionKind.CAPTURE:
                observation = self._capture(state, provide_depth)
                continue
            targets = [action.target] if action.kind == ActionKind.MOVE_TO else action.waypoints
            for index, target in enumerate(targets):
                self._move_segment(state, target, provide_depth)
                if not state.budget_left(self.spec):
                    break
                if action.capture_mode == "all" or index == len(targets) - 1:
                    observation = self._capture(state, provide_depth)

        return self.finalize(state, info)

    def finalize(self, state: EpisodeState, info: MethodInfo) -> Dict[str, Any]:
        """Persist one partially or fully executed episode.

        The normal agent loop calls this when it reaches ``done`` or a budget.
        The stepwise RL pilot uses the same finalizer after its terminal action,
        so it produces exactly the stream and manifest consumed by the ordinary
        reconstruction evaluation rather than a parallel data format.
        """

        transforms_path = export_transforms_json(self.out_dir, self.sim.intrinsics, state.frames)
        reconstruction = None
        if state.reconstruction_frames:
            reconstruction_path = export_transforms_json(
                self.out_dir,
                self.sim.intrinsics,
                state.reconstruction_frames,
                filename="transforms_stream.json",
            )
            reconstruction = {
                "policy": "uniform-time",
                "interval_s": self.spec.reconstruction_interval,
                "streamed_to_agent": self.spec.stream_observations,
                "num_frames": len(state.reconstruction_frames),
                "frames": state.reconstruction_log,
                "transforms": reconstruction_path.name,
            }

        manifest = {
            "method": {
                "name": info.name,
                "needs_depth": info.needs_depth,
                "action_space": info.action_space,
                "pose_access": info.pose_access,
            },
            "seed": self.spec.seed,
            "task": self.spec.task,
            "collision": self.spec.collision,
            "clock": {
                "final_sim_time": float(state.time),
                "max_sim_time": self.spec.max_sim_time,
                "capture_cost": self.spec.capture_cost,
                "max_agent_captures": self.spec.max_captures,
                "path_length_m": float(state.path_length),
                "planning_wall_time_s": float(state.planning_wall_time),
                "unreachable_targets": state.unreachable_targets,
            },
            "captures": state.capture_log,
            "actions": state.action_log,
            "num_captures": state.captures,
            "reconstruction": reconstruction,
            "distractors": [
                {"name": d.name, "object_template": d.object_template, "trajectory": d.trajectory}
                for d in self.spec.scene.distractors
            ],
            "transforms": transforms_path.name,
        }
        save_json(self.out_dir / "manifest.json", manifest)
        return manifest

def run_episode(
    spec: EpisodeSpec,
    agent: ActiveAgent,
    out_dir: Path,
    sim: Optional[DynamicSceneSim] = None,
) -> Dict[str, Any]:
    """Run one episode end to end; creates (and closes) the sim if not given."""

    owns_sim = sim is None
    if sim is None:
        sim = DynamicSceneSim(spec.scene)
    try:
        return EpisodeRunner(spec, sim, out_dir).run(agent)
    finally:
        if owns_sim:
            sim.close()
