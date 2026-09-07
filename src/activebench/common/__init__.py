"""Shared benchmark primitives owned by ActiveBench.

This package contains camera, Habitat, transform, image, and artifact I/O
helpers that used to live under ``revisit_recon``. Keeping them here makes the
benchmark independently extractable from the research method.
"""

from .camera import CameraIntrinsics, CameraPose, CameraPose5D

__all__ = ["CameraIntrinsics", "CameraPose", "CameraPose5D"]
