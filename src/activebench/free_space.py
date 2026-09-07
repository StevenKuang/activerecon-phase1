"""Observed free-space tracking for harness-side candidate sampling.

Mirrors the semantics of R3CON's voxel-map free mask (``free_mask_w_margin``):
a voxel is *free* once a depth ray from a captured frame passed through it,
*occupied* once a depth ray terminated in it, and a candidate position must be
a free voxel with no occupied voxel within the safety margin. Positions are
therefore restricted to space the agent has actually observed — no
teleporting into unseen rooms, and no ground-truth scene knowledge.

Conservative deviation from R3CON (documented): rays with no depth return
(holes in scan geometry, e.g. apartment_1's missing ceiling) mark nothing —
carving free space along no-hit rays would place candidates outside the
scene through those holes.
"""

from dataclasses import dataclass, field
from typing import Tuple

import numpy as np

from activebench.convention import pose_to_c2w_cv
from activebench.eval.geometry import unproject_depth_cv
from activebench.common.camera import CameraIntrinsics, CameraPose


@dataclass
class FreeSpaceTracker:
    """Voxel grid of observed free / occupied space in world coordinates."""

    bbox: np.ndarray  # (2, 3) world-space min/max
    resolution: float = 0.2  # R3CON map_resolution
    safety_margin: float = 0.3  # R3CON safety_margin
    max_depth: float = 6.0
    pixel_stride: int = 8
    _free: np.ndarray = field(init=False, repr=False)
    _occupied: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.bbox = np.asarray(self.bbox, dtype=np.float64)
        if self.bbox.shape != (2, 3):
            raise ValueError("bbox must have shape (2, 3)")
        extent = self.bbox[1] - self.bbox[0]
        self._dims = np.maximum(np.ceil(extent / self.resolution).astype(int), 1)
        self._free = np.zeros(tuple(self._dims), dtype=bool)
        self._occupied = np.zeros(tuple(self._dims), dtype=bool)
        margin_cells = int(np.ceil(self.safety_margin / self.resolution))
        offsets = []
        rng_range = range(-margin_cells, margin_cells + 1)
        for i in rng_range:
            for j in rng_range:
                for k in rng_range:
                    if np.linalg.norm([i, j, k]) * self.resolution <= self.safety_margin + 1e-9:
                        offsets.append((i, j, k))
        self._margin_offsets = np.asarray(offsets, dtype=np.int64)

    # -- indexing helpers ------------------------------------------------------

    def _to_indices(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        idx = np.floor((points - self.bbox[0]) / self.resolution).astype(np.int64)
        inside = np.all((idx >= 0) & (idx < self._dims), axis=-1)
        return idx, inside

    def _mark(self, grid: np.ndarray, points: np.ndarray) -> None:
        idx, inside = self._to_indices(points.reshape(-1, 3))
        idx = idx[inside]
        if idx.size:
            grid[idx[:, 0], idx[:, 1], idx[:, 2]] = True

    # -- updates ---------------------------------------------------------------

    def update(self, depth: np.ndarray, pose: CameraPose, intrinsics: CameraIntrinsics) -> None:
        """Integrate one captured depth map (planar-z, benchmark pose)."""

        stride = self.pixel_stride
        d = depth[::stride, ::stride].astype(np.float64)
        sub_intr = CameraIntrinsics(
            width=d.shape[1],
            height=d.shape[0],
            fx=intrinsics.fx / stride,
            fy=intrinsics.fy / stride,
            cx=intrinsics.cx / stride,
            cy=intrinsics.cy / stride,
        )
        valid = (d > 0.0) & (d < self.max_depth)
        if not valid.any():
            return
        points_cam = unproject_depth_cv(d, sub_intr)[valid]
        c2w = pose_to_c2w_cv(pose)
        endpoints = points_cam @ c2w[:3, :3].T + c2w[:3, 3]
        origin = c2w[:3, 3]

        self._mark(self._occupied, endpoints)

        # Sample free space along each ray, stopping one voxel short of the
        # surface so surface voxels never get marked free.
        rays = endpoints - origin
        lengths = np.linalg.norm(rays, axis=1)
        keep = lengths > self.resolution
        rays, lengths = rays[keep], lengths[keep]
        if not len(rays):
            return
        step = 0.9 * self.resolution
        max_steps = int(np.ceil((lengths.max() - self.resolution) / step))
        for i in range(max_steps):
            t = (i + 0.5) * step
            active = lengths - self.resolution > t
            if not active.any():
                break
            pts = origin + rays[active] * (t / lengths[active])[:, None]
            self._mark(self._free, pts)

    # -- queries -----------------------------------------------------------------

    def _free_with_margin(self) -> np.ndarray:
        occupied_idx = np.argwhere(self._occupied)
        blocked = np.zeros_like(self._occupied)
        if occupied_idx.size:
            for offset in self._margin_offsets:
                shifted = occupied_idx + offset
                inside = np.all((shifted >= 0) & (shifted < self._dims), axis=1)
                s = shifted[inside]
                blocked[s[:, 0], s[:, 1], s[:, 2]] = True
        return self._free & ~blocked

    def candidate_positions(
        self, center: np.ndarray, radius: float, count: int, rng: np.random.Generator
    ) -> np.ndarray:
        """Sample voxel-center positions from observed free space near ``center``.

        Mirrors R3CON's ``generate_random_candidates``: uniform choice (with
        replacement) among free-with-margin voxel centers within ``radius``.
        Returns an empty array when nothing qualifies yet.
        """

        mask = self._free_with_margin()
        idx = np.argwhere(mask)
        if idx.size == 0:
            return np.empty((0, 3))
        centers = self.bbox[0] + (idx + 0.5) * self.resolution
        within = np.linalg.norm(centers - np.asarray(center, dtype=np.float64), axis=1) <= radius
        centers = centers[within]
        if len(centers) == 0:
            return np.empty((0, 3))
        choice = rng.integers(0, len(centers), size=count)
        return centers[choice]

    @property
    def num_free_voxels(self) -> int:
        return int(self._free.sum())


def sample_view_directions(count: int, rng: np.random.Generator, max_pitch_deg: float = 85.0):
    """Uniform-random view directions on the sphere as (yaw, pitch) pairs.

    Matches R3CON's ``random_rotation`` with ``pitch_angle=None``: direction
    drawn isotropically (so pitch is arcsin-distributed over ±90°), roll zero.
    Pitch is clamped to ``±max_pitch_deg`` for numerical safety of the
    roll-free yaw/pitch parameterization (documented deviation).
    """

    dirs = rng.normal(size=(count, 3))
    dirs /= np.maximum(np.linalg.norm(dirs, axis=1, keepdims=True), 1e-12)
    limit = np.deg2rad(max_pitch_deg)
    pitch = np.clip(np.arcsin(dirs[:, 1]), -limit, limit)
    # Benchmark convention: view = -forward, forward = R @ [0,0,1] with
    # yaw = atan2(f_x, f_z); solving for the view direction v gives
    # yaw = atan2(-v_x, -v_z).
    yaw = np.arctan2(-dirs[:, 0], -dirs[:, 2])
    return yaw, pitch
