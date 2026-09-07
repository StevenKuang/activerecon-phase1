"""Camera pose and intrinsics utilities for 5-DoF active reconstruction."""

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

from .transforms import make_transform, yaw_pitch_to_rotation


@dataclass
class CameraIntrinsics:
    """Pinhole camera intrinsics."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    @classmethod
    def from_hfov(
        cls,
        width: int,
        height: int,
        hfov_deg: float,
    ) -> "CameraIntrinsics":
        """Create intrinsics from image size and horizontal field of view."""

        hfov = np.deg2rad(hfov_deg)
        fx = 0.5 * float(width) / np.tan(0.5 * hfov)
        fy = fx
        cx = (float(width) - 1.0) * 0.5
        cy = (float(height) - 1.0) * 0.5
        return cls(width=width, height=height, fx=fx, fy=fy, cx=cx, cy=cy)

    def as_matrix(self) -> np.ndarray:
        """Return the 3x3 calibration matrix."""

        return np.array(
            [
                [self.fx, 0.0, self.cx],
                [0.0, self.fy, self.cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )


@dataclass
class CameraPose:
    """5-DoF camera pose with roll fixed to zero.

    The coordinate convention is right-handed with +Y up. At yaw=0 and pitch=0,
    the camera looks along +Z.
    """

    position: np.ndarray
    yaw: float = 0.0
    pitch: float = 0.0

    @classmethod
    def from_xyz_yaw_pitch(
        cls,
        xyz: Sequence[float],
        yaw: float = 0.0,
        pitch: float = 0.0,
    ) -> "CameraPose":
        """Build a pose from an XYZ sequence and radians angles."""

        return cls(position=np.asarray(xyz, dtype=np.float64), yaw=float(yaw), pitch=float(pitch))

    def copy(self) -> "CameraPose":
        """Return a deep copy of the pose."""

        return CameraPose(position=self.position.copy(), yaw=self.yaw, pitch=self.pitch)

    def rotation_matrix(self) -> np.ndarray:
        """Return camera-to-world rotation for the yaw/pitch orientation."""

        return yaw_pitch_to_rotation(self.yaw, self.pitch)

    def forward(self) -> np.ndarray:
        """Return the camera forward vector in world coordinates."""

        return self.rotation_matrix() @ np.array([0.0, 0.0, 1.0], dtype=np.float64)

    def right(self) -> np.ndarray:
        """Return the camera right vector in world coordinates."""

        return self.rotation_matrix() @ np.array([1.0, 0.0, 0.0], dtype=np.float64)

    def up(self) -> np.ndarray:
        """Return the camera up vector in world coordinates."""

        return self.rotation_matrix() @ np.array([0.0, 1.0, 0.0], dtype=np.float64)

    def as_matrix(self) -> np.ndarray:
        """Return a 4x4 camera-to-world transform."""

        return make_transform(self.rotation_matrix(), self.position)

    def as_list(self) -> List[float]:
        """Return a JSON-friendly pose vector."""

        return [
            float(self.position[0]),
            float(self.position[1]),
            float(self.position[2]),
            float(self.yaw),
            float(self.pitch),
        ]

    def xyz_tuple(self) -> Tuple[float, float, float]:
        """Return position as a tuple."""

        return (float(self.position[0]), float(self.position[1]), float(self.position[2]))


CameraPose5D = CameraPose
