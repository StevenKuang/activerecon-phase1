"""Dynamic scene simulator: Habitat rendering plus time-scripted distractors.

Wraps :class:`activebench.common.habitat_env.HabitatSimWrapper` and adds
kinematic rigid objects whose poses are deterministic functions of sim time.
Any (pose, t) can be rendered with distractors present or parked out of the
scene, which yields clean ground truth and exact distractor masks.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from activebench.distractors import DistractorSpec
from activebench.common.camera import CameraPose
from activebench.common.habitat_env import HabitatEnvConfig, HabitatSimWrapper

# Depth difference (meters) above which a pixel is attributed to a distractor.
MASK_DEPTH_EPS = 0.005

# Parked distractors live far below the scene so they cannot appear in renders.
_PARK_BASE = np.array([0.0, -1000.0, 0.0])
_PARK_SPACING = np.array([10.0, 0.0, 0.0])


@dataclass
class DynamicSceneConfig:
    """Scene, distractor set, and trajectory seed for one benchmark scene."""

    habitat: HabitatEnvConfig
    distractors: List[DistractorSpec] = field(default_factory=list)
    trajectory_seed: int = 0

    def __post_init__(self) -> None:
        # Kinematic rigid objects require physics support in the simulator.
        self.habitat.enable_physics = True


def _ensure_nvidia_egl_vendor() -> None:
    """Prefer NVIDIA's EGL vendor when a conda env ships mesa EGL libraries.

    Conda CUDA toolchains pull in a mesa-based glvnd stack whose device list
    hides the NVIDIA GPU from Habitat's windowless EGL context. Pinning the
    vendor file restores direct rendering; users can override by setting the
    variable themselves.
    """

    if "__EGL_VENDOR_LIBRARY_FILENAMES" in os.environ:
        return
    prefix = os.environ.get("CONDA_PREFIX", "")
    nvidia_json = Path("/usr/share/glvnd/egl_vendor.d/10_nvidia.json")
    if prefix and nvidia_json.exists():
        if list(Path(prefix, "lib").glob("libEGL_anaconda_mesa*")):
            os.environ["__EGL_VENDOR_LIBRARY_FILENAMES"] = str(nvidia_json)


class DynamicSceneSim:
    """Habitat renderer over a 4D scene: static stage + scripted distractors."""

    def __init__(self, config: DynamicSceneConfig) -> None:
        _ensure_nvidia_egl_vendor()
        self.config = config
        self.env = HabitatSimWrapper(config.habitat)
        self.intrinsics = config.habitat.intrinsics
        self._objects: List[Any] = []
        self._spawn_distractors()

    def _spawn_distractors(self) -> None:
        if not self.config.distractors:
            return
        import habitat_sim  # noqa: F401  (available because env imported it)

        sim = self.env.sim
        template_manager = sim.get_object_template_manager()
        object_manager = sim.get_rigid_object_manager()
        for index, spec in enumerate(self.config.distractors):
            handles = self._resolve_template(template_manager, spec.object_template)
            if spec.scale is not None:
                template = template_manager.get_template_by_handle(handles)
                template.scale = np.full(3, float(spec.scale))
                template_manager.register_template(template, handles)
            obj = object_manager.add_object_by_template_handle(handles)
            if obj is None:
                raise RuntimeError(
                    "failed to spawn distractor %r from template %r" % (spec.name, handles)
                )
            obj.motion_type = self.env.habitat_sim.physics.MotionType.KINEMATIC
            obj.translation = (_PARK_BASE + index * _PARK_SPACING).tolist()
            self._objects.append(obj)
            spec.build_trajectory(seed=self.config.trajectory_seed + index)

    @staticmethod
    def _resolve_template(template_manager: Any, template: str) -> str:
        """Load/lookup an object template, accepting a path or a handle substring."""

        path = Path(template).expanduser()
        if path.suffix == ".json" and path.exists():
            template_manager.load_configs(str(path.parent))
            return str(path)
        handles = template_manager.get_template_handles(template)
        if not handles:
            raise FileNotFoundError(
                "no object template matches %r; pass an .object_config.json path "
                "or a handle substring of an already-loaded template" % template
            )
        return handles[0]

    def _pose_distractors(self, t: float, clean: bool) -> None:
        import magnum as mn

        for index, (spec, obj) in enumerate(zip(self.config.distractors, self._objects)):
            if clean:
                obj.translation = (_PARK_BASE + index * _PARK_SPACING).tolist()
                continue
            position, yaw = spec.build_trajectory().pose_at(t)
            obj.translation = position.tolist()
            obj.rotation = mn.Quaternion.rotation(mn.Rad(yaw), mn.Vector3.y_axis())

    def observe(
        self,
        pose: CameraPose,
        t: float,
        include_clean: bool = False,
        include_mask: bool = False,
    ) -> Dict[str, np.ndarray]:
        """Render the scene at (pose, t).

        Returns ``rgb`` and ``depth`` with distractors posed at time ``t``.
        With ``include_clean``/``include_mask``, adds ``rgb_clean``,
        ``depth_clean``, and a boolean ``distractor_mask`` (True on pixels the
        distractors changed, judged by depth difference).
        """

        self._pose_distractors(t, clean=False)
        full = self.env.render(pose)
        result: Dict[str, np.ndarray] = {"rgb": full["rgb"], "depth": full["depth"]}
        if include_clean or include_mask:
            self._pose_distractors(t, clean=True)
            clean = self.env.render(pose)
            result["rgb_clean"] = clean["rgb"]
            result["depth_clean"] = clean["depth"]
            if include_mask:
                result["distractor_mask"] = (
                    np.abs(full["depth"] - clean["depth"]) > MASK_DEPTH_EPS
                )
        return result

    def render_clean(self, pose: CameraPose) -> Dict[str, np.ndarray]:
        """Render distractor-free ground truth at a pose (time-invariant)."""

        self._pose_distractors(0.0, clean=True)
        return self.env.render(pose)

    def scene_aabb(self) -> np.ndarray:
        """Return the stage bounding box in world coordinates, shape (2, 3).

        Mesh stages report their geometry AABB from the scene graph. Gaussian-
        splat (``.gs.ply``) stages render via CUDA and carry no mesh scene-graph
        geometry, so that cumulative bounding box is degenerate (all zeros); in
        that case fall back to the navmesh bounds, which bracket the walkable
        extent. The navmesh only covers the floor, so pad the vertical span to
        bracket plausible room height (methods size their free-space grid from
        this box).
        """

        bb = self.env.sim.get_active_scene_graph().get_root_node().cumulative_bb
        aabb = np.array([list(bb.min), list(bb.max)], dtype=np.float64)
        degenerate = (
            not np.all(np.isfinite(aabb))
            or float(np.prod(aabb[1] - aabb[0])) <= 0.0
        )
        if degenerate:
            pathfinder = self.env.sim.pathfinder
            if getattr(pathfinder, "is_loaded", False):
                lo, hi = pathfinder.get_bounds()
                aabb = np.array([list(lo), list(hi)], dtype=np.float64)
                aabb[0, 1] -= 0.5  # a little below the floor
                aabb[1, 1] += 3.0  # bracket a room's height above the floor
        return aabb

    def start_pose(self) -> CameraPose:
        """Return Habitat's default camera pose for the scene."""

        return self.env.current_camera_pose()

    def set_navigation_anchor(self, xyz) -> None:
        """Pin routing/snapping to the navmesh island containing ``xyz``.

        A flying camera's 3D-nearest navmesh point can belong to a
        disconnected island (window sills, outdoor patches), which would
        make every route from it "unreachable". Anchoring on the episode
        start island removes that snap ambiguity.
        """

        self._nav_island = -1
        pathfinder = self.env.sim.pathfinder
        if not pathfinder.is_loaded:
            return
        snapped = np.asarray(pathfinder.snap_point(np.asarray(xyz, dtype=np.float32)))
        if np.isfinite(snapped).all():
            self._nav_island = int(pathfinder.get_island(snapped.astype(np.float32)))

    def snap_navigable(self, xyz) -> Optional[np.ndarray]:
        """Nearest navmesh (floor) point to ``xyz``, or None if unresolvable.

        Honors the island pinned by :meth:`set_navigation_anchor`.
        """

        pathfinder = self.env.sim.pathfinder
        if not pathfinder.is_loaded:
            return None
        point = np.asarray(pathfinder.snap_point(
            np.asarray(xyz, dtype=np.float32),
            island_index=getattr(self, "_nav_island", -1),
        ))
        return point.astype(np.float64) if np.isfinite(point).all() else None

    def navmesh_route(self, start_xyz, end_xyz):
        """Shortest navigable floor route between two points.

        Returns ``(points, length)`` — a polyline of at least the two snapped
        endpoints and its geodesic length — or None when either endpoint
        cannot be snapped to the navmesh or no path connects them.
        """

        import habitat_sim

        start = self.snap_navigable(start_xyz)
        end = self.snap_navigable(end_xyz)
        if start is None or end is None:
            return None
        path = habitat_sim.ShortestPath()
        path.requested_start = start.astype(np.float32)
        path.requested_end = end.astype(np.float32)
        pathfinder = self.env.sim.pathfinder
        if not pathfinder.find_path(path) or not np.isfinite(path.geodesic_distance):
            return None
        points = np.asarray(path.points, dtype=np.float64)
        if len(points) < 2:
            points = np.stack([start, end])
        return points, float(path.geodesic_distance)

    def sample_navigable_points(
        self, count: int, seed: int, anchor_xyz=None
    ) -> Optional[np.ndarray]:
        """Seeded navmesh floor samples for benchmark-owned candidate pools.

        With ``anchor_xyz`` the samples are kept on the anchor's navmesh
        island (scenes expose disconnected patches, e.g. terrain seen
        through windows, which would be unreachable pool candidates). If the
        island cannot be filled, remaining slots fall back to unfiltered
        samples rather than shrinking the pool.
        """

        pathfinder = self.env.sim.pathfinder
        if not pathfinder.is_loaded:
            return None
        pathfinder.seed(int(seed))
        island = None
        if anchor_xyz is not None:
            snapped = self.snap_navigable(anchor_xyz)
            if snapped is not None:
                island = int(pathfinder.get_island(snapped.astype(np.float32)))
        points = []
        tries = 0
        while len(points) < count and tries < 50 * count:
            tries += 1
            point = np.asarray(pathfinder.get_random_navigable_point(), dtype=np.float64)
            if not np.isfinite(point).all():
                continue
            if island is not None and island >= 0:
                if int(pathfinder.get_island(point.astype(np.float32))) != island:
                    continue
            points.append(point)
        while len(points) < count:
            point = np.asarray(pathfinder.get_random_navigable_point(), dtype=np.float64)
            if np.isfinite(point).all():
                points.append(point)
        return np.stack(points) if points else None

    def close(self) -> None:
        self.env.close()
