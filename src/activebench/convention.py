"""Camera convention conversions at method-adapter boundaries.

The benchmark's `CameraPose` renders with the camera looking along -Z of the
pose frame with +Y image-up (OpenGL style), and `pose.as_matrix()` is the
camera-to-world transform in that convention. Most reconstruction codebases
(R3CON, GAVIS/3DGS, COLMAP pipelines) use OpenCV/COLMAP convention instead:
+Z forward, +Y image-down. The two differ by a sign flip of the Y and Z
camera axes.
"""

import numpy as np

from activebench.common.camera import CameraPose
from activebench.common.transforms import rotation_to_yaw_pitch

_GL_CV_FLIP = np.diag([1.0, -1.0, -1.0, 1.0])


def pose_to_c2w_cv(pose: CameraPose) -> np.ndarray:
    """Benchmark pose → OpenCV/COLMAP-convention camera-to-world matrix."""

    return pose.as_matrix() @ _GL_CV_FLIP


def c2w_cv_to_pose(c2w_cv: np.ndarray) -> CameraPose:
    """OpenCV/COLMAP camera-to-world matrix → roll-free benchmark pose."""

    c2w_gl = np.asarray(c2w_cv, dtype=np.float64) @ _GL_CV_FLIP
    yaw, pitch = rotation_to_yaw_pitch(c2w_gl[:3, :3])
    return CameraPose.from_xyz_yaw_pitch(c2w_gl[:3, 3], yaw=yaw, pitch=pitch)
