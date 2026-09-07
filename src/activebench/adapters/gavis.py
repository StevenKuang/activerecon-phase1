"""ViewSelector adapter for GAVIS (CVPR 2026) — uncertainty-driven 3DGS NBV.

Wraps the `gavis` package from https://github.com/gatech-rl2/GAVIS (checked
out at ``repo_root``) behind the benchmark's tier-1 ``ViewSelector``
interface, lifted into an ``ActiveAgent`` by ``PoolNBVAgent``:

1. every selection round, retrain a 3DGS model on all frames captured so far
   (their own ``train_gaussian_model``, RGB + optional GT depth supervision);
2. build the anisotropic visibility field (``GAVIS.load_model``);
3. score every remaining candidate pose by the mean of
   ``render_uncertainty_map(pose)`` and pick the argmax.

GAVIS consumes c2w poses in COLMAP convention; conversion happens at this
boundary only. The visibility-field AABB comes from the benchmark's scene
bounds (Habitat stage AABB), matching the role of the depth-derived AABB in
their example pipeline.

Runtime requirements (the ``gavis`` conda env): torch + CUDA,
``diff_gaussian_rasterization`` (vanilla 3DGS), ``gavis_rasterizer``,
``simple-knn``, ``nerfacc``, ``omegaconf``, ``plyfile``.
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from activebench.api import CaptureRecord, ViewSelector
from activebench.convention import pose_to_c2w_cv
from activebench.common.camera import CameraIntrinsics, CameraPose

from activebench.runtime import external_repo

DEFAULT_REPO_ROOT = external_repo("gavis")


@dataclass
class GavisSelector:
    """Next-best-view scoring with GAVIS uncertainty maps."""

    intrinsics: CameraIntrinsics
    # Scene bounds in benchmark world coordinates, shape (2, 3) (min, max).
    scene_bbox: Any
    repo_root: str = DEFAULT_REPO_ROOT
    train_iterations: int = 1500
    # Points sampled per view when initializing gaussians from depth. This is
    # the dominant driver of the scene gaussian count (pts/view x n_views), and
    # hence of the O(n_gaussians x n_views) visibility-field memory in
    # ``load_model``. Lowering it caps the gaussian budget so large open scenes
    # (which otherwise densify past the GPU's visibility-tensor ceiling) fit,
    # at the cost of reconstruction density. Matches the gavis default (10k).
    depth_init_pts_per_view: int = 10_000
    use_depth: bool = True
    # Downscale factor applied to the uncertainty render resolution; scoring
    # a candidate needs relative ranking, not full-res maps.
    score_downscale: int = 2
    # Relative padding applied to the scene bounds, mirroring the AABB_MARGIN
    # headroom their example gives the visibility field.
    aabb_margin: float = 0.1
    seed: int = 0
    gavis_overrides: Dict[str, Any] = field(default_factory=dict)
    verbose: bool = False

    def __post_init__(self) -> None:
        repo = str(Path(self.repo_root).expanduser())
        if repo not in sys.path:
            sys.path.insert(0, repo)
        bbox = np.asarray(self.scene_bbox, dtype=np.float64)
        if bbox.shape != (2, 3):
            raise ValueError("scene_bbox must have shape (2, 3)")
        pad = self.aabb_margin * (bbox[1] - bbox[0])
        bbox = np.stack([bbox[0] - pad, bbox[1] + pad])
        # gavis wants aabb as [[xmin, xmax], [ymin, ymax], [zmin, zmax]].
        self._aabb = bbox.T.copy()
        intr = self.intrinsics
        self._fovx = 2.0 * float(np.arctan(intr.width / (2.0 * intr.fx)))
        self._fovy = 2.0 * float(np.arctan(intr.height / (2.0 * intr.fy)))

    # -- data marshalling ----------------------------------------------------

    def _load_history(self, history: List[CaptureRecord]):
        from PIL import Image

        images, poses, depths = [], [], []
        for record in history:
            if record.rgb_path is None:
                raise ValueError("GavisSelector needs rgb_path on capture records")
            rgb = np.asarray(Image.open(record.rgb_path))[..., :3]
            images.append(rgb.astype(np.float32) / 255.0)
            poses.append(pose_to_c2w_cv(record.pose))
            if self.use_depth and record.depth_path:
                depths.append(np.load(record.depth_path))
        if depths and len(depths) != len(images):
            depths = []
        return images, poses, depths if depths else None

    # -- ViewSelector protocol -------------------------------------------------

    def select(self, history: List[CaptureRecord], candidates: List[CameraPose]) -> int:
        import torch
        from omegaconf import OmegaConf

        from gavis import GAVIS
        from gavis.train_utils import train_gaussian_model

        torch.manual_seed(self.seed + len(history))
        images, poses, depths = self._load_history(history)
        h, w = images[0].shape[:2]

        model = train_gaussian_model(
            images=images,
            poses=poses,
            fovx=self._fovx,
            fovy=self._fovy,
            H=h,
            W=w,
            aabb=self._aabb,
            depths=depths,
            use_depth_init=depths is not None,
            use_depth_supervision=depths is not None,
            depth_init_pts_per_view=self.depth_init_pts_per_view,
            iterations=self.train_iterations,
            verbose=self.verbose,
        )

        down = max(1, int(self.score_downscale))
        cfg = OmegaConf.create(
            {
                "H": h // down,
                "W": w // down,
                "fovx": self._fovx,
                "fovy": self._fovy,
                "aabb": self._aabb.tolist(),
                **self.gavis_overrides,
            }
        )
        uq = GAVIS(cfg)
        uq.load_model(model, [np.asarray(p) for p in poses])

        scores = np.empty(len(candidates), dtype=np.float64)
        for index, pose in enumerate(candidates):
            umap = uq.render_uncertainty_map(pose_to_c2w_cv(pose))
            scores[index] = float(umap.mean())
        best = int(np.argmax(scores))
        print(
            "[gavis] round %d: %d train views, %d candidates, best #%d "
            "(score %.5f, spread %.5f)"
            % (
                len(history),
                len(images),
                len(candidates),
                best,
                scores[best],
                scores.max() - scores.min(),
            )
        )
        return best
