"""Ground-truth surface sampling: points + normals in benchmark coordinates.

Rather than trusting scene mesh files (whose axes may differ from what
Habitat renders — glTF up-axis conventions bit us before), surfaces are
sampled from Habitat itself: clean (distractor-free) depth renders from a
seeded grid of viewpoints are unprojected to world points with
depth-gradient normals, then deduplicated on a voxel grid. The resulting
set covers exactly the renderable surface, in exactly the benchmark's
coordinates, and includes down-facing surfaces (ceilings, furniture
undersides) because the sampling views look steeply up and down.
"""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from activebench.convention import pose_to_c2w_cv
from activebench.eval.geometry import normals_from_depth_cv, unproject_depth_cv
from activebench.sim import DynamicSceneSim
from activebench.common.camera import CameraPose


@dataclass
class SurfaceSampleConfig:
    grid_spacing: float = 2.0
    margin: float = 0.4
    num_heights: int = 2
    yaw_bins: int = 4
    pitches_deg: Tuple[float, ...] = (-65.0, -20.0, 20.0, 65.0)
    min_depth: float = 0.5
    max_depth: float = 8.0
    points_per_frame: int = 4000
    voxel_size: float = 0.04
    min_view_cos: float = 0.2
    seed: int = 0


@dataclass
class SurfaceSamples:
    points: np.ndarray  # (N, 3) float32, world
    normals: np.ndarray  # (N, 3) float32, unit, oriented toward observing camera
    config: Dict = field(default_factory=dict)

    def save(self, path: Path) -> None:
        np.savez_compressed(
            path, points=self.points, normals=self.normals, config=repr(self.config)
        )

    @classmethod
    def load(cls, path: Path) -> "SurfaceSamples":
        data = np.load(path, allow_pickle=False)
        return cls(points=data["points"], normals=data["normals"], config={})


def _sampling_poses(sim: DynamicSceneSim, cfg: SurfaceSampleConfig) -> List[CameraPose]:
    bbox = sim.scene_aabb()
    lo = bbox[0] + cfg.margin
    hi = bbox[1] - cfg.margin
    xs = np.arange(lo[0], hi[0] + 1e-6, cfg.grid_spacing)
    zs = np.arange(lo[2], hi[2] + 1e-6, cfg.grid_spacing)
    ys = np.linspace(lo[1] + 0.2 * (hi[1] - lo[1]), hi[1] - 0.1 * (hi[1] - lo[1]), cfg.num_heights)
    yaws = np.linspace(-np.pi, np.pi, cfg.yaw_bins, endpoint=False)
    poses = []
    for x in xs:
        for z in zs:
            for y in ys:
                for yaw in yaws:
                    for pitch_deg in cfg.pitches_deg:
                        poses.append(
                            CameraPose.from_xyz_yaw_pitch(
                                [float(x), float(y), float(z)],
                                yaw=float(yaw),
                                pitch=float(np.deg2rad(pitch_deg)),
                            )
                        )
    return poses


def _voxel_dedupe(points: np.ndarray, voxel: float) -> np.ndarray:
    """Return indices keeping the first point in each voxel cell."""

    keys = np.floor(points / voxel).astype(np.int64)
    # Lexicographic unique over rows; stable to keep deterministic ordering.
    _, first = np.unique(keys, axis=0, return_index=True)
    return np.sort(first)


def build_surface_samples(
    sim: DynamicSceneSim, cfg: SurfaceSampleConfig = SurfaceSampleConfig()
) -> SurfaceSamples:
    rng = np.random.default_rng(cfg.seed)
    intr = sim.intrinsics
    all_points: List[np.ndarray] = []
    all_normals: List[np.ndarray] = []
    poses = _sampling_poses(sim, cfg)
    for pose in poses:
        depth = sim.render_clean(pose)["depth"].astype(np.float64)
        normals_cam, valid = normals_from_depth_cv(depth, intr)
        valid &= (depth > cfg.min_depth) & (depth < cfg.max_depth)
        points_cam = unproject_depth_cv(depth, intr)
        # Reject grazing observations: unreliable depth-gradient normals.
        ray = points_cam / np.maximum(np.linalg.norm(points_cam, axis=-1, keepdims=True), 1e-12)
        view_cos = -(normals_cam * ray).sum(-1)
        valid &= view_cos > cfg.min_view_cos

        idx = np.flatnonzero(valid.reshape(-1))
        if idx.size == 0:
            continue
        if idx.size > cfg.points_per_frame:
            idx = rng.choice(idx, size=cfg.points_per_frame, replace=False)
        c2w = pose_to_c2w_cv(pose)
        p = points_cam.reshape(-1, 3)[idx] @ c2w[:3, :3].T + c2w[:3, 3]
        n = normals_cam.reshape(-1, 3)[idx] @ c2w[:3, :3].T
        all_points.append(p)
        all_normals.append(n)

    points = np.concatenate(all_points, axis=0)
    normals = np.concatenate(all_normals, axis=0)
    keep = _voxel_dedupe(points, cfg.voxel_size)
    return SurfaceSamples(
        points=points[keep].astype(np.float32),
        normals=normals[keep].astype(np.float32),
        config={**asdict(cfg), "num_sampling_views": len(poses)},
    )
