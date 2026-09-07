"""Export selected views to a NeRF/3DGS-style transforms.json file."""

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from .camera import CameraIntrinsics, CameraPose
from .io import ensure_dir, load_json, save_json

# TODO: Add native gsplat and WildGS-SLAM export adapters once their offline
# training data contracts are selected. This MVP only writes transforms.json.


def frame_from_pose(
    pose: CameraPose,
    file_path: str,
    depth_path: Optional[str] = None,
    stratum: Optional[str] = None,
) -> Dict[str, Any]:
    """Create one transforms.json frame from a camera pose.

    ``stratum`` labels the view so the evaluator reports it as its own group
    (``appearance_per_stratum``) instead of pooling everything. Any eval set
    whose views differ in kind should set it: a pooled mean over heterogeneous
    view classes hides exactly the differences worth seeing.
    """

    frame: Dict[str, Any] = {
        "file_path": file_path,
        "transform_matrix": pose.as_matrix().tolist(),
    }
    if depth_path is not None:
        frame["depth_path"] = depth_path
    if stratum is not None:
        frame["stratum"] = stratum
    return frame


def export_transforms_json(
    output_dir: Path,
    intrinsics: CameraIntrinsics,
    frames: Iterable[Dict[str, Any]],
    filename: str = "transforms.json",
) -> Path:
    """Write selected camera views for later offline 3DGS training."""

    ensure_dir(output_dir)
    payload = {
        "w": int(intrinsics.width),
        "h": int(intrinsics.height),
        "fl_x": float(intrinsics.fx),
        "fl_y": float(intrinsics.fy),
        "cx": float(intrinsics.cx),
        "cy": float(intrinsics.cy),
        "camera_model": "OPENCV",
        "frames": list(frames),
    }
    path = output_dir / filename
    save_json(path, payload)
    return path


def export_from_episode(
    episode_path: Path,
    output_dir: Path,
    intrinsics: CameraIntrinsics,
    every_n: int = 1,
) -> Path:
    """Export frames from an episode manifest."""

    episode = load_json(episode_path)
    frames: List[Dict[str, Any]] = []
    for step in episode.get("steps", []):
        if int(step["step"]) % max(1, every_n) != 0:
            continue
        pose = CameraPose.from_xyz_yaw_pitch(step["pose"][:3], yaw=step["pose"][3], pitch=step["pose"][4])
        rgb_path = step.get("rgb_path") or ("observations/rgb_%05d.npy" % int(step["step"]))
        frames.append(frame_from_pose(pose, rgb_path, depth_path=step.get("depth_path")))
    return export_transforms_json(output_dir, intrinsics, frames)
