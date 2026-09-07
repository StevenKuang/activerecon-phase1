"""Tier-1 metric: observation coverage of GT surfaces, binned by orientation.

A GT surface point counts as *observed* by a captured frame when it projects
into the image, passes a planar-depth visibility test against the frame's
stored depth map (which includes distractors, so distractor occlusion
correctly suppresses observations), lies within range, and is seen at a
non-grazing incidence angle. Results are stratified by surface-normal
elevation:

- ``up``:   n_y >  bin_cos   (floors, table tops)
- ``down``: n_y < -bin_cos   (ceilings, undersides of tables/shelves)
- ``side``: everything else  (walls, furniture sides)

Down-facing coverage is the discriminator for 5-DoF viewpoint control: a
ground-robot-like policy scores near zero on it regardless of visited area.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from activebench.eval.dataset import training_transforms_path
from activebench.eval.geometry import project_points_cv
from activebench.common.camera import CameraIntrinsics

_GL_CV_FLIP = np.diag([1.0, -1.0, -1.0, 1.0])


@dataclass
class CoverageParams:
    max_range: float = 6.0
    min_range: float = 0.2
    depth_tol_abs: float = 0.03
    depth_tol_rel: float = 0.02
    min_cos_incidence: float = 0.1
    well_observed_cos: float = 0.5
    bin_cos: float = 0.7


@dataclass
class FrameObservation:
    """One captured frame: benchmark-convention (OpenGL) c2w + planar depth."""

    c2w_gl: np.ndarray
    depth: np.ndarray


def coverage_from_frames(
    frames: List[FrameObservation],
    points: np.ndarray,
    normals: np.ndarray,
    intrinsics: CameraIntrinsics,
    params: CoverageParams = CoverageParams(),
) -> Dict[str, Any]:
    """Compute best observation incidence per GT point over all frames."""

    n_points = points.shape[0]
    best_cos = np.zeros(n_points, dtype=np.float64)
    observed = np.zeros(n_points, dtype=bool)

    for frame in frames:
        c2w_cv = np.asarray(frame.c2w_gl, dtype=np.float64) @ _GL_CV_FLIP
        u, v, z = project_points_cv(points, c2w_cv, intrinsics)
        h, w = frame.depth.shape
        valid = (
            (z > params.min_range)
            & (z < params.max_range)
            & (u >= 0)
            & (u < w)
            & (v >= 0)
            & (v < h)
        )
        idx = np.flatnonzero(valid)
        if idx.size == 0:
            continue
        depth_at = frame.depth[v[idx], u[idx]].astype(np.float64)
        tol = np.maximum(params.depth_tol_abs, params.depth_tol_rel * z[idx])
        visible = (depth_at > 0.0) & (np.abs(depth_at - z[idx]) < tol)
        idx = idx[visible]
        if idx.size == 0:
            continue
        to_cam = c2w_cv[:3, 3] - points[idx]
        to_cam /= np.maximum(np.linalg.norm(to_cam, axis=1, keepdims=True), 1e-12)
        cos_inc = (normals[idx] * to_cam).sum(axis=1)
        seen = cos_inc > params.min_cos_incidence
        idx = idx[seen]
        observed[idx] = True
        best_cos[idx] = np.maximum(best_cos[idx], cos_inc[seen])

    n_y = normals[:, 1]
    bins = {
        "up": n_y > params.bin_cos,
        "down": n_y < -params.bin_cos,
        "side": np.abs(n_y) <= params.bin_cos,
    }
    result: Dict[str, Any] = {"params": asdict(params), "num_frames": len(frames), "bins": {}}
    for name, mask in bins.items():
        count = int(mask.sum())
        obs = observed & mask
        well = obs & (best_cos > params.well_observed_cos)
        result["bins"][name] = {
            "num_points": count,
            "observed_frac": float(obs.sum() / count) if count else 0.0,
            "well_observed_frac": float(well.sum() / count) if count else 0.0,
            "mean_best_cos_observed": float(best_cos[obs].mean()) if obs.any() else 0.0,
        }
    total = points.shape[0]
    result["overall"] = {
        "num_points": total,
        "observed_frac": float(observed.sum() / total) if total else 0.0,
        "well_observed_frac": float(((best_cos > params.well_observed_cos)).sum() / total)
        if total
        else 0.0,
    }
    return result


def load_episode_frames(episode_dir: Path) -> tuple:
    """Load frame poses/depths and intrinsics from an episode directory."""

    episode_dir = Path(episode_dir)
    payload = json.loads(training_transforms_path(episode_dir).read_text())
    intrinsics = CameraIntrinsics(
        width=int(payload["w"]),
        height=int(payload["h"]),
        fx=float(payload["fl_x"]),
        fy=float(payload["fl_y"]),
        cx=float(payload["cx"]),
        cy=float(payload["cy"]),
    )
    frames = []
    for entry in payload["frames"]:
        depth_path = entry.get("depth_path")
        if depth_path is None:
            raise ValueError("episode frames lack depth maps; coverage needs them")
        frames.append(
            FrameObservation(
                c2w_gl=np.asarray(entry["transform_matrix"], dtype=np.float64),
                depth=np.load(episode_dir / depth_path),
            )
        )
    return frames, intrinsics


def evaluate_episode_dir(
    episode_dir: Path,
    points: np.ndarray,
    normals: np.ndarray,
    params: CoverageParams = CoverageParams(),
    output_name: Optional[str] = "coverage.json",
) -> Dict[str, Any]:
    frames, intrinsics = load_episode_frames(episode_dir)
    result = coverage_from_frames(frames, points, normals, intrinsics, params)
    result["episode_dir"] = str(episode_dir)
    if output_name:
        (Path(episode_dir) / output_name).write_text(json.dumps(result, indent=2))
    return result
