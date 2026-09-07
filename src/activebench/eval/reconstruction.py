"""Reconstruction backend contract for standardized Tier-2 evaluation."""

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Dict, Optional, Protocol, Sequence

import numpy as np


_RUN_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


@dataclass(frozen=True)
class ReconstructionArtifacts:
    """Named output paths for one reconstruction backend run."""

    root: Path

    @property
    def eval_path(self) -> Path:
        return self.root / "eval.json"

    @property
    def gaussians_path(self) -> Path:
        return self.root / "gaussians.npz"

    @property
    def points_path(self) -> Path:
        return self.root / "points.npz"

    @property
    def renders_dir(self) -> Path:
        return self.root / "renders"


def reconstruction_artifacts(episode_dir: Path, run_name: str) -> ReconstructionArtifacts:
    if not _RUN_NAME_RE.fullmatch(run_name):
        raise ValueError(
            "reconstruction run name must contain only letters, numbers, '.', '_' or '-'"
        )
    return ReconstructionArtifacts(Path(episode_dir) / "reconstructions" / run_name)


def resolve_reconstruction_eval(
    episode_dir: Path, run_name: Optional[str] = None
) -> Optional[Path]:
    """Resolve one result without silently mixing multiple named backends."""

    episode_dir = Path(episode_dir)
    if run_name in ("", "none"):
        return None
    if run_name == "legacy":
        legacy = episode_dir / "retrain_eval.json"
        return legacy if legacy.exists() else None
    if run_name not in (None, "auto"):
        selected = reconstruction_artifacts(episode_dir, run_name).eval_path
        return selected if selected.exists() else None
    # Standard backend first (current + archived names), then a unique run.
    for preferred in ("gsplat", "vanilla-3dgs-gsplat", "vanilla-3dgs"):
        candidate = reconstruction_artifacts(episode_dir, preferred).eval_path
        if candidate.exists():
            return candidate
    legacy = episode_dir / "retrain_eval.json"
    if legacy.exists():
        return legacy
    named = sorted((episode_dir / "reconstructions").glob("*/eval.json"))
    if len(named) <= 1:
        return named[0] if named else None
    raise ValueError("multiple reconstruction results found; select a run name")


def resolve_shared_eval(
    episode_dir: Path, run_name: Optional[str] = None
) -> Optional[Path]:
    """Locate the shared-eval class-PSNR result matching a reconstruction run.

    ``eval_shared.json`` lives next to a named run's ``eval.json``, or at the
    episode root for the legacy artifacts.
    """

    eval_path = resolve_reconstruction_eval(episode_dir, run_name)
    if eval_path is None:
        return None
    if eval_path.name == "retrain_eval.json":
        shared = Path(episode_dir) / "eval_shared.json"
    else:
        shared = eval_path.parent / "eval_shared.json"
    return shared if shared.exists() else None


@dataclass(frozen=True)
class ReconstructionDataset:
    """Framework-neutral posed RGB-D observations for one episode.

    ``images``/``depths`` may be lazy sequences (decode on access) when a
    backend streams frames from disk rather than training on a resident set;
    ``image_paths``/``depth_paths`` then carry the on-disk source frames so such
    a backend can reference them without ever materializing the arrays in RAM.
    """

    images: Sequence[np.ndarray]
    poses_c2w_cv: Sequence[np.ndarray]
    depths: Sequence[np.ndarray]
    alpha_masks: Optional[Sequence[np.ndarray]]
    width: int
    height: int
    fovx: float
    fovy: float
    aabb: np.ndarray
    image_paths: Optional[Sequence[Any]] = None
    depth_paths: Optional[Sequence[Any]] = None

    def __post_init__(self) -> None:
        count = len(self.images)
        if count == 0 or len(self.poses_c2w_cv) != count or len(self.depths) != count:
            raise ValueError("images, poses, and depths must have the same non-zero length")
        if self.alpha_masks is not None and len(self.alpha_masks) != count:
            raise ValueError("alpha_masks must match the number of images")
        if self.image_paths is not None and len(self.image_paths) != count:
            raise ValueError("image_paths must match the number of images")
        if np.asarray(self.aabb).shape != (3, 2):
            raise ValueError("aabb must have shape (3, 2)")


@dataclass(frozen=True)
class ReconstructionCamera:
    """Pinhole camera used for held-out rendering.

    Training and evaluation normally share one resolution. Keeping the
    evaluation camera explicit lets resolution ablations render every fitted
    model against one common target without changing its training images.
    """

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("camera width and height must be positive")
        if self.fx <= 0.0 or self.fy <= 0.0:
            raise ValueError("camera focal lengths must be positive")

    @property
    def fovx(self) -> float:
        return 2.0 * float(np.arctan(self.width / (2.0 * self.fx)))

    @property
    def fovy(self) -> float:
        return 2.0 * float(np.arctan(self.height / (2.0 * self.fy)))

    @classmethod
    def from_dataset(cls, dataset: "ReconstructionDataset") -> "ReconstructionCamera":
        fx = dataset.width / (2.0 * np.tan(dataset.fovx / 2.0))
        fy = dataset.height / (2.0 * np.tan(dataset.fovy / 2.0))
        return cls(
            width=dataset.width,
            height=dataset.height,
            fx=float(fx),
            fy=float(fy),
            cx=dataset.width / 2.0,
            cy=dataset.height / 2.0,
        )


@dataclass(frozen=True)
class ReconstructionRender:
    """Rendered RGB and optional metric depth."""

    rgb: np.ndarray
    depth: Optional[np.ndarray]


@dataclass(frozen=True)
class GaussianSplats:
    """Renderer-independent Gaussian parameters used by metrics and Viser."""

    centers: np.ndarray
    scales: np.ndarray
    quats_wxyz: np.ndarray
    opacities: np.ndarray
    colors: np.ndarray
    # Optional SH coefficients above DC. Shape (N, K, 3) where
    # K = (sh_degree+1)**2 - 1 (15 for degree 3), in the 3DGS / gsplat
    # band-then-channel order. ``None`` preserves the legacy SH0-only path.
    sh_rest: Optional[np.ndarray] = None
    sh_degree: int = 0
    # Raw DC band, shape (N, 1, 3). ``colors`` is the same band converted to
    # displayable RGB and clamped to [0, 1] for viewers; that clamp is lossy on
    # out-of-gamut gaussians, so exporters that need to reproduce the model
    # (rather than just show it) persist this instead. ``None`` on backends
    # that only expose colours.
    sh0: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        count = len(self.centers)
        expected = {
            "centers": (count, 3),
            "scales": (count, 3),
            "quats_wxyz": (count, 4),
            "opacities": (count,),
            "colors": (count, 3),
        }
        for name, shape in expected.items():
            if np.asarray(getattr(self, name)).shape != shape:
                raise ValueError("%s must have shape %s" % (name, shape))
        if self.sh_degree < 0:
            raise ValueError("sh_degree must be non-negative")
        if self.sh_rest is not None:
            coeffs = (self.sh_degree + 1) ** 2 - 1
            if np.asarray(self.sh_rest).shape != (count, coeffs, 3):
                raise ValueError(
                    "sh_rest must have shape (%d, %d, 3) for sh_degree=%d"
                    % (count, coeffs, self.sh_degree)
                )
        elif self.sh_degree > 0:
            raise ValueError("sh_degree > 0 requires sh_rest")


class Reconstruction(Protocol):
    """A fitted scene representation returned by a backend."""

    @property
    def metadata(self) -> Dict[str, Any]: ...

    def render(
        self,
        c2w_cv: np.ndarray,
        camera: Optional[ReconstructionCamera] = None,
    ) -> ReconstructionRender: ...

    def gaussian_splats(self) -> GaussianSplats: ...


class ReconstructionBackend(Protocol):
    """Train or infer a reconstruction from the standardized dataset."""

    name: str

    def reconstruct(
        self,
        dataset: ReconstructionDataset,
        *,
        iterations: int,
        seed: int,
        options: Dict[str, Any],
    ) -> Reconstruction: ...


def create_reconstruction_backend(name: str, **_unused) -> ReconstructionBackend:
    """Phase 1 uses the method-independent gsplat backend."""
    if name.strip().lower() != "gsplat":
        raise ValueError("unknown reconstruction backend %r; available: gsplat" % name)
    from activebench.eval.gsplat_backend import GsplatVanilla3DGSBackend
    return GsplatVanilla3DGSBackend()
