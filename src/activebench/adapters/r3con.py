"""ActiveAgent adapter for the official R3CON implementation (ECCV 2026).

Wraps the planner + mapping stack from https://github.com/jkff00/R3CON
(checked out at ``repo_root``) without touching their simulator or mission
recorder: the benchmark supplies observations, R3CON's own voxel map,
renderability metric map, and planner produce the next camera path.

Faithfulness notes:
- Planner configs are loaded from the official ``config/planner/*.yaml``.
- The per-frame map update mirrors ``IncrementalMapper.run()`` for the
  renderability planners (``use_confidence: false``); the GaussianMap branch is
  not needed by those planners and is skipped.
- R3CON plans in OpenCV camera convention (c2w, +Z forward); the benchmark
  uses OpenGL-style poses (-Z view). Conversion happens only at this boundary.
- Their planner returns a dense interpolated camera path where only the final
  NBV pose becomes a keyframe, so actions use ``capture_mode="last"``.

Runtime requirements (the ``r3con`` conda env): torch + CUDA, the two custom
rasterizers (``diff_gauss``, ``diff_gaussian_rasterization_2d``), open3d, cv2,
einops, omegaconf. Habitat itself is not required by the adapter.
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from activebench.api import AgentAction, MethodInfo, Observation
from activebench.convention import c2w_cv_to_pose, pose_to_c2w_cv
from activebench.common.camera import CameraPose

from activebench.runtime import external_repo

DEFAULT_REPO_ROOT = external_repo("R3CON")


@dataclass
class _SimulatorShim:
    """The subset of R3CON's HabitatSimulator the planner actually touches."""

    resolution: np.ndarray
    intrinsic: Any  # torch tensor, normalized like their compute_camera_intrinsic
    depth_range: List[float]
    has_missing_surface: bool = False

    def simulate(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError(
            "R3CON planner asked the simulator for a valid mask; this only "
            "happens when has_missing_surface is set, which the adapter does "
            "not enable."
        )


@dataclass
class R3ConAgent:
    """Official R3CON planner as a benchmark ActiveAgent."""

    # Scene bounds in benchmark (Habitat) world coordinates, shape (2, 3).
    # Prefer DynamicSceneSim.scene_aabb(); trimesh-loaded glTF bounds can be
    # rotated relative to the coordinates Habitat renders in.
    scene_bbox: Optional[Any] = None
    scene_mesh_path: Optional[str] = None
    repo_root: str = DEFAULT_REPO_ROOT
    planner_type: str = "confidence_pano"  # or "confidence_perspective"
    depth_range: List[float] = field(default_factory=lambda: [0.0, 5.0])
    voxel_size: float = 0.05
    planner_overrides: Dict[str, Any] = field(default_factory=dict)
    path_stride: int = 1
    device: str = "cuda"

    def __post_init__(self) -> None:
        repo = str(Path(self.repo_root).expanduser())
        if repo not in sys.path:
            sys.path.insert(0, repo)
        self._initialized = False
        self._planner = None

    def info(self) -> MethodInfo:
        return MethodInfo(
            name="r3con-%s" % self.planner_type.replace("confidence_", ""),
            needs_depth=True,
            action_space="free",
            conda_env="r3con",
        )

    def reset(self, seed: int, task: Optional[str] = None) -> None:
        import torch

        torch.manual_seed(seed)
        np.random.seed(seed)
        self._initialized = False
        self._planner = None

    # -- lazy construction of the official stack ---------------------------

    def _scene_bbox(self) -> np.ndarray:
        if self.scene_bbox is not None:
            bbox = np.asarray(self.scene_bbox, dtype=np.float64)
            if bbox.shape != (2, 3):
                raise ValueError("scene_bbox must have shape (2, 3)")
            return bbox
        if self.scene_mesh_path is None:
            raise ValueError("R3ConAgent needs scene_bbox or scene_mesh_path")
        import trimesh

        mesh = trimesh.load(self.scene_mesh_path)
        return np.array(mesh.bounding_box.bounds)

    def _init_stack(self, observation: Observation) -> None:
        import torch
        from omegaconf import OmegaConf

        from mapping.camera_new_utils import SimpleCamera
        from mapping.renderability import PointMetrics, VoxelHashMap
        from mapping.voxel_map import VoxelMap
        from planning import get_planner
        from utils.operations import sample_image_grid

        intr = observation.intrinsics
        h, w = intr.height, intr.width
        # Their compute_camera_intrinsic with normalize=true, built from the
        # benchmark's actual pinhole parameters.
        k = np.array(
            [
                [intr.fx / w, 0.0, intr.cx / w],
                [0.0, intr.fy / h, intr.cy / h],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        self._intrinsic = torch.from_numpy(k)
        self._resolution = np.array([h, w])
        xy_ray, _ = sample_image_grid((h, w))
        self._xy_ray = xy_ray.reshape(h * w, 1, 2)

        self._shim = _SimulatorShim(
            resolution=self._resolution,
            intrinsic=self._intrinsic,
            depth_range=list(self.depth_range),
        )

        repo = Path(self.repo_root)
        mapper_cfg = OmegaConf.load(repo / "config" / "mapper" / "incremental.yaml")
        planner_cfg = OmegaConf.load(
            repo / "config" / "planner" / ("%s.yaml" % self.planner_type)
        )
        planner_cfg.init_pose = pose_to_c2w_cv(observation.pose).tolist()
        for key, value in self.planner_overrides.items():
            planner_cfg[key] = value

        device = torch.device(self.device)
        bbox = self._scene_bbox()
        self.voxel_map = VoxelMap(mapper_cfg.voxel_map, bbox, device, None)
        self.hash_map = VoxelHashMap(self.voxel_size)
        self.metric_map = PointMetrics(Fx=self._intrinsic[0, 0])
        self.observer = SimpleCamera(
            width=w,
            height=h,
            fx=float(self._intrinsic[0, 0]),
            fy=float(self._intrinsic[1, 1]),
            scale_val=self.hash_map.scale_val,
            tag="perspective",
        )
        # Mirrors IncrementalMapper.load_metric_processor: the spherical
        # sampler is square at the image width.
        self.sampler = SimpleCamera(
            width=w, height=w, scale_val=self.hash_map.scale_val, tag="spherical"
        )
        self._planner = get_planner(OmegaConf.create({"planner": planner_cfg}), device)
        self._device = device
        self._initialized = True

    # -- per-step conversion + update ---------------------------------------

    def _dataframe(self, observation: Observation) -> Dict[str, Any]:
        if observation.depth is None:
            raise ValueError("R3CON requires depth observations (needs_depth=True)")
        return self._dataframe_arrays(observation.rgb, observation.depth, observation.pose)

    def _dataframe_arrays(
        self, rgb_array: np.ndarray, depth_m: np.ndarray, pose: CameraPose
    ) -> Dict[str, Any]:
        import torch

        rgb = torch.from_numpy(
            (rgb_array[..., :3] / 255.0).astype(np.float32)
        ).permute(2, 0, 1)
        depth = depth_m.astype(np.float32).copy()
        lo, hi = self.depth_range
        invalid = ~((depth > lo) & (depth < hi) & (depth > 0.0))
        depth[invalid] = -1.0
        depth_t = torch.from_numpy(depth).unsqueeze(0)
        extrinsic = torch.from_numpy(
            pose_to_c2w_cv(pose).astype(np.float32)
        )
        frame = {
            "extrinsic": extrinsic,
            "intrinsic": self._intrinsic,
            "uv": self._xy_ray,
            "rgb": rgb,
            "depth": depth_t,
            "depth_range": torch.tensor(list(self.depth_range)),
        }
        return {k: v.to(self._device) for k, v in frame.items()}

    def _update_maps(self, dataframe: Dict[str, Any]) -> None:
        """Per-frame update, mirroring IncrementalMapper.run() (renderability)."""

        from utils.operations import compute_valid_point_cloud, get_visible_points

        new_pcd_world = compute_valid_point_cloud(dataframe)
        self.hash_map.update(new_pcd_world)
        self.metric_map.add_new_points(self.hash_map.new_num_voxels)
        out = self.observer.render_simple(
            self.hash_map.points,
            self.hash_map.opacities,
            self.hash_map.scales,
            self.hash_map.rotations,
            dataframe["extrinsic"],
            self.hash_map.colors,
        )
        data_dict = get_visible_points(
            out, dataframe["rgb"], dataframe["extrinsic"], self.hash_map.points
        )
        self.metric_map.update_all(data_dict)
        self.voxel_map.update(dataframe)

    def update_from_observation(self, observation: Observation) -> None:
        """Fold one decision observation and its pending stream into R3 state.

        This is intentionally public for Track-A teacher policies that reuse
        the released R3 map but supply a benchmark-owned action candidate set.
        It performs no R3 planning and never receives evaluator renders.
        """

        import torch

        if not self._initialized:
            self._init_stack(observation)
        with torch.no_grad():
            for frame in observation.stream_frames:
                depth = frame.load_depth()
                if depth is None:
                    raise ValueError("R3CON stream ingestion requires depth")
                self._update_maps(
                    self._dataframe_arrays(frame.load_rgb(), depth, frame.pose)
                )
            self._update_maps(self._dataframe(observation))

    def map_evidence(self) -> Dict[str, float]:
        """Return a bounded-size Track-A summary of released R3 map state.

        These are online mapper statistics, never evaluator images or PSNR.
        They mirror the evidence logged by ``replay_r3_track_a.py`` so a
        candidate policy and its calibration use identical definitions.
        """

        if not self._initialized:
            return {
                "map_points": 0.0,
                "observed_points": 0.0,
                "mean_observations_per_point": 0.0,
                "directional_bin_entries": 0.0,
            }
        counts = self.metric_map.counts
        if not counts.numel():
            return {
                "map_points": 0.0,
                "observed_points": 0.0,
                "mean_observations_per_point": 0.0,
                "directional_bin_entries": 0.0,
            }
        return {
            "map_points": float(counts.numel()),
            "observed_points": float((counts > 0).sum().detach().cpu()),
            "mean_observations_per_point": float(counts.mean().detach().cpu()),
            "directional_bin_entries": float(
                (self.metric_map.keys_exist >= 0).sum().detach().cpu()
            ),
        }

    def act(self, observation: Observation) -> AgentAction:
        import torch

        self.update_from_observation(observation)
        with torch.no_grad():
            projector = self.observer if self._planner.use_perspective else self.sampler
            camera_path = self._planner.plan(
                (None, self.voxel_map),
                self._shim,
                None,
                [projector, self.hash_map, self.metric_map],
            )
        matrices = [m.detach().cpu().numpy() for m in camera_path]
        if self.path_stride > 1:
            kept = matrices[:: self.path_stride]
            if not np.allclose(kept[-1], matrices[-1]):
                kept.append(matrices[-1])
            matrices = kept
        waypoints = [c2w_cv_to_pose(m) for m in matrices]
        return AgentAction.trajectory(waypoints, capture_mode="last")
