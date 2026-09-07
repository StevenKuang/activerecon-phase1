"""Coordinate conversion helpers for Habitat-Sim camera control.

The project-level `CameraPose5D` convention is:
- right-handed world coordinates
- +Y is up
- at yaw=0 and pitch=0, the camera looks along +Z
- the stored transform is camera-to-world

Habitat-Sim agent and sensor states also live in world coordinates, but the
Python API expects orientation as a `numpy.quaternion` object. Keeping this
conversion in one module makes the Habitat boundary explicit and easy to audit.
"""

from typing import Any

import numpy as np

from .camera import CameraPose


def rotation_matrix_to_wxyz(rotation: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to a normalized quaternion [w, x, y, z]."""

    matrix = np.asarray(rotation, dtype=np.float64)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (matrix[2, 1] - matrix[1, 2]) / scale
        y = (matrix[0, 2] - matrix[2, 0]) / scale
        z = (matrix[1, 0] - matrix[0, 1]) / scale
    else:
        diag = np.diag(matrix)
        index = int(np.argmax(diag))
        if index == 0:
            scale = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            w = (matrix[2, 1] - matrix[1, 2]) / scale
            x = 0.25 * scale
            y = (matrix[0, 1] + matrix[1, 0]) / scale
            z = (matrix[0, 2] + matrix[2, 0]) / scale
        elif index == 1:
            scale = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            w = (matrix[0, 2] - matrix[2, 0]) / scale
            x = (matrix[0, 1] + matrix[1, 0]) / scale
            y = 0.25 * scale
            z = (matrix[1, 2] + matrix[2, 1]) / scale
        else:
            scale = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            w = (matrix[1, 0] - matrix[0, 1]) / scale
            x = (matrix[0, 2] + matrix[2, 0]) / scale
            y = (matrix[1, 2] + matrix[2, 1]) / scale
            z = 0.25 * scale
    quat = np.asarray([w, x, y, z], dtype=np.float64)
    norm = np.linalg.norm(quat)
    if norm == 0.0:
        return np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return quat / norm


def wxyz_to_rotation_matrix(wxyz: np.ndarray) -> np.ndarray:
    """Convert a normalized quaternion [w, x, y, z] to a 3x3 rotation matrix."""

    quat = np.asarray(wxyz, dtype=np.float64)
    norm = float(np.linalg.norm(quat))
    if norm == 0.0:
        return np.eye(3, dtype=np.float64)
    w, x, y, z = quat / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def habitat_quaternion_to_rotation_matrix(rotation: Any) -> np.ndarray:
    """Return a rotation matrix from Habitat's numpy-quaternion object."""

    return wxyz_to_rotation_matrix(
        np.asarray(
            [
                float(rotation.w),
                float(rotation.x),
                float(rotation.y),
                float(rotation.z),
            ],
            dtype=np.float64,
        )
    )


def pose_to_habitat_quaternion(pose: CameraPose) -> Any:
    """Return Habitat's quaternion object for a `CameraPose5D` orientation."""

    wxyz = rotation_matrix_to_wxyz(pose.rotation_matrix())
    try:
        import quaternion  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "Habitat pose control requires numpy-quaternion, but it is not "
            "installed in this Python environment."
        ) from exc
    return quaternion.quaternion(float(wxyz[0]), float(wxyz[1]), float(wxyz[2]), float(wxyz[3]))
