"""Pinhole camera math shared by the evaluation tools.

Conventions (validated by a cross-view consistency check, see
docs/results/2026-07-09-r3con-smoke.md and tests/test_bench_coverage.py):

- Habitat depth maps store planar z-depth (distance along the optical axis),
  not euclidean ray length.
- All matrices here are OpenCV/COLMAP-convention camera-to-world (+Z forward,
  +Y image-down); convert benchmark poses with
  :func:`activebench.convention.pose_to_c2w_cv`.
"""

from typing import Tuple

import numpy as np

from activebench.common.camera import CameraIntrinsics


def unproject_depth_cv(depth: np.ndarray, intrinsics: CameraIntrinsics) -> np.ndarray:
    """Unproject a planar-z depth map to camera-space points, shape (H, W, 3)."""

    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    x = (u - intrinsics.cx) / intrinsics.fx * depth
    y = (v - intrinsics.cy) / intrinsics.fy * depth
    return np.stack([x, y, depth.astype(np.float64)], axis=-1)


def project_points_cv(
    points_world: np.ndarray,
    c2w_cv: np.ndarray,
    intrinsics: CameraIntrinsics,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project world points into a camera.

    Returns integer pixel coordinates ``(u, v)`` and planar depth ``z``; callers
    must mask with ``z > 0`` and image bounds before indexing.
    """

    w2c = np.linalg.inv(c2w_cv)
    cam = points_world @ w2c[:3, :3].T + w2c[:3, 3]
    z = cam[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = np.round(cam[:, 0] / z * intrinsics.fx + intrinsics.cx).astype(np.int64)
        v = np.round(cam[:, 1] / z * intrinsics.fy + intrinsics.cy).astype(np.int64)
    return u, v, z


def normals_from_depth_cv(
    depth: np.ndarray,
    intrinsics: CameraIntrinsics,
    max_rel_neighbor_gap: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate camera-space surface normals from a planar-z depth map.

    Uses central differences of the unprojected point grid. Returns
    ``(normals, valid)`` of shapes (H, W, 3) and (H, W); pixels near depth
    discontinuities (relative neighbor gap above ``max_rel_neighbor_gap``)
    are marked invalid. Normals are oriented to face the camera.
    """

    points = unproject_depth_cv(depth, intrinsics)
    du = np.zeros_like(points)
    dv = np.zeros_like(points)
    du[:, 1:-1] = points[:, 2:] - points[:, :-2]
    dv[1:-1, :] = points[2:, :] - points[:-2, :]
    normals = np.cross(dv, du)
    norm = np.linalg.norm(normals, axis=-1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        normals = np.where(norm > 1e-12, normals / norm, 0.0)

    # Orient toward the camera: a visible surface has n · p < 0 (the normal
    # points back along the viewing ray).
    flip = (normals * points).sum(-1) > 0.0
    normals[flip] *= -1.0

    valid = depth > 0.0
    gap_u = np.zeros_like(depth)
    gap_v = np.zeros_like(depth)
    gap_u[:, 1:-1] = np.maximum(
        np.abs(depth[:, 2:] - depth[:, 1:-1]), np.abs(depth[:, 1:-1] - depth[:, :-2])
    )
    gap_v[1:-1, :] = np.maximum(
        np.abs(depth[2:, :] - depth[1:-1, :]), np.abs(depth[1:-1, :] - depth[:-2, :])
    )
    smooth = np.maximum(gap_u, gap_v) < max_rel_neighbor_gap * np.maximum(depth, 0.5)
    valid &= smooth
    valid &= norm[..., 0] > 1e-12
    valid[:1] = valid[-1:] = False
    valid[:, :1] = valid[:, -1:] = False
    return normals, valid
