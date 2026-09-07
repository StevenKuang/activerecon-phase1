"""Rigid transform helpers."""

from typing import Tuple

import numpy as np


def yaw_pitch_to_rotation(yaw: float, pitch: float) -> np.ndarray:
    """Return camera-to-world rotation for yaw about Y and pitch about X."""

    cy = float(np.cos(yaw))
    sy = float(np.sin(yaw))
    cp = float(np.cos(pitch))
    sp = float(np.sin(pitch))

    yaw_matrix = np.array(
        [
            [cy, 0.0, sy],
            [0.0, 1.0, 0.0],
            [-sy, 0.0, cy],
        ],
        dtype=np.float64,
    )
    pitch_matrix = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, cp, -sp],
            [0.0, sp, cp],
        ],
        dtype=np.float64,
    )
    return yaw_matrix @ pitch_matrix


def rotation_to_yaw_pitch(rotation: np.ndarray) -> Tuple[float, float]:
    """Project a rotation matrix to the project's roll-free yaw/pitch angles."""

    matrix = np.asarray(rotation, dtype=np.float64)
    forward = matrix @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
    norm = float(np.linalg.norm(forward))
    if norm == 0.0:
        return 0.0, 0.0
    forward = forward / norm
    pitch = float(np.arcsin(np.clip(-forward[1], -1.0, 1.0)))
    yaw = float(np.arctan2(forward[0], forward[2]))
    return yaw, pitch


def make_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """Create a 4x4 transform from rotation and translation."""

    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.asarray(rotation, dtype=np.float64)
    transform[:3, 3] = np.asarray(translation, dtype=np.float64)
    return transform


def invert_transform(transform: np.ndarray) -> np.ndarray:
    """Invert a rigid 4x4 transform."""

    matrix = np.asarray(transform, dtype=np.float64)
    rotation = matrix[:3, :3]
    translation = matrix[:3, 3]
    inverse = np.eye(4, dtype=np.float64)
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -rotation.T @ translation
    return inverse
