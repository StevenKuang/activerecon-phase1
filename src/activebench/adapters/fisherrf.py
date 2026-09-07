"""ViewSelector adapter for FisherRF (ECCV 2024) — Fisher-information NBV.

Wraps the official implementation from https://github.com/JiangWenPL/FisherRF
(checked out at ``repo_root``) behind the benchmark's tier-1 ``ViewSelector``
interface, lifted into an ``ActiveAgent`` by ``PoolNBVAgent``:

1. every selection round, train their 3DGS (their ``GaussianModel`` + loss +
   densification schedule, vanilla gaussian-splatting) on all frames captured
   so far;
2. score candidates with their ``HRegSelector`` — the acquisition is computed
   by their modified rasterizer, whose backward pass returns per-parameter
   Hessian diagonals (that is the method; it is taken verbatim);
3. return the argmax candidate.

Faithfulness notes:
- Selector hyperparameters follow ``active_train.py`` defaults:
  ``reg_lambda=1e-6``, ``filter_out_grad=["rotation"]``, ``I_test=False``,
  ``I_acq_reg=False``.
- FisherRF is an RGB-only method (their pipeline never consumes depth), so
  the manifest declares ``needs_depth=False`` and the model initializes from
  random points in the scene bounds — their protocol for synthetic scenes.
- The per-round training loop mirrors their ``active_train.py`` inner loop
  (L1+D-SSIM, absolute densification schedule) with the iteration budget as
  the only benchmark-controlled knob, matching how the GAVIS adapter treats
  its trainer.
- FisherRF consumes c2w poses in COLMAP convention; conversion happens at
  this boundary only.

Runtime requirements (the ``fisherrf`` conda env): torch + CUDA and the two
submodules from their repo. Their rasterizer fork installs under the same
module name as vanilla 3DGS (``diff_gaussian_rasterization``), which is why
this adapter must never share an env with the GAVIS adapter.
"""

import sys
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from activebench.api import CaptureRecord, ViewSelector
from activebench.convention import pose_to_c2w_cv
from activebench.common.camera import CameraIntrinsics, CameraPose

from activebench.runtime import external_repo

DEFAULT_REPO_ROOT = external_repo("FisherRF")


def _c2w_to_colmap_rt(c2w_cv: np.ndarray):
    """3DGS Camera convention: R is w2c rotation transposed, T is w2c translation."""

    w2c = np.linalg.inv(c2w_cv)
    return np.transpose(w2c[:3, :3]), w2c[:3, 3]


@dataclass
class FisherRFSelector:
    """Next-best-view scoring with FisherRF's Fisher-information acquisition."""

    intrinsics: CameraIntrinsics
    # Scene bounds in benchmark world coordinates, shape (2, 3) (min, max);
    # used only for the random point-cloud initialization.
    scene_bbox: Any
    repo_root: str = DEFAULT_REPO_ROOT
    train_iterations: int = 1500
    num_init_points: int = 100_000
    sh_degree: int = 3
    reg_lambda: float = 1e-6
    filter_out_grad: List[str] = field(default_factory=lambda: ["rotation"])
    seed: int = 0
    verbose: bool = False

    def __post_init__(self) -> None:
        repo = str(Path(self.repo_root).expanduser())
        if repo not in sys.path:
            sys.path.insert(0, repo)
        bbox = np.asarray(self.scene_bbox, dtype=np.float64)
        if bbox.shape != (2, 3):
            raise ValueError("scene_bbox must have shape (2, 3)")
        self._bbox = bbox
        intr = self.intrinsics
        self._fovx = 2.0 * float(np.arctan(intr.width / (2.0 * intr.fx)))
        self._fovy = 2.0 * float(np.arctan(intr.height / (2.0 * intr.fy)))

    # -- their config objects -------------------------------------------------

    def _opt_pipe(self):
        from arguments import OptimizationParams, PipelineParams

        parser = ArgumentParser()
        opt_group = OptimizationParams(parser)
        pipe_group = PipelineParams(parser)
        args = parser.parse_args([])
        opt = opt_group.extract(args)
        pipe = pipe_group.extract(args)
        opt.iterations = self.train_iterations
        opt.position_lr_max_steps = self.train_iterations
        opt.densify_until_iter = min(opt.densify_until_iter, self.train_iterations)
        return opt, pipe

    # -- data marshalling ------------------------------------------------------

    def _camera(self, pose: CameraPose, image, uid: int):
        from scene.cameras import Camera

        r, t = _c2w_to_colmap_rt(pose_to_c2w_cv(pose))
        return Camera(
            colmap_id=uid,
            R=r,
            T=t,
            FoVx=self._fovx,
            FoVy=self._fovy,
            image=image,
            gt_alpha_mask=None,
            image_name="view_%04d" % uid,
            uid=uid,
        )

    def _train_cameras(self, history: List[CaptureRecord]):
        import torch
        from PIL import Image

        cameras = []
        for uid, record in enumerate(history):
            if record.rgb_path is None:
                raise ValueError("FisherRFSelector needs rgb_path on capture records")
            rgb = np.asarray(Image.open(record.rgb_path))[..., :3]
            image = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1)
            cameras.append(self._camera(record.pose, image, uid))
        return cameras

    def _candidate_cameras(self, candidates: List[CameraPose]):
        # Candidate views have no ground-truth image; the Hessian render only
        # needs the camera geometry, so a zero image placeholder is fine (the
        # backward is driven by torch.ones_like on the prediction).
        import torch

        blank = torch.zeros(3, self.intrinsics.height, self.intrinsics.width)
        return [
            self._camera(pose, blank, 10_000 + index)
            for index, pose in enumerate(candidates)
        ]

    @staticmethod
    def _cameras_extent(cameras) -> float:
        centers = np.stack([np.asarray(c.camera_center.cpu()) for c in cameras])
        center = centers.mean(axis=0)
        radius = float(np.linalg.norm(centers - center, axis=1).max())
        return max(1.1 * radius, 1e-3)

    # -- their training loop, condensed ---------------------------------------

    def _train_model(self, cameras, opt, pipe, background):
        import torch
        from gaussian_renderer import render
        from scene.gaussian_model import GaussianModel
        from utils.graphics_utils import BasicPointCloud
        from utils.loss_utils import l1_loss, ssim

        rng = np.random.default_rng(self.seed + len(cameras))
        points = rng.uniform(self._bbox[0], self._bbox[1], size=(self.num_init_points, 3))
        colors = rng.uniform(0.0, 1.0, size=points.shape)
        pcd = BasicPointCloud(
            points=points, colors=colors, normals=np.zeros_like(points)
        )

        gaussians = GaussianModel(self.sh_degree)
        extent = self._cameras_extent(cameras)
        gaussians.create_from_pcd(pcd, extent)
        gaussians.training_setup(opt)

        order = []
        for iteration in range(1, opt.iterations + 1):
            gaussians.update_learning_rate(iteration)
            if iteration % 1000 == 0:
                gaussians.oneupSHdegree()
            if not order:
                order = list(rng.permutation(len(cameras)))
            cam = cameras[order.pop()]

            render_pkg = render(cam, gaussians, pipe, background)
            image = render_pkg["render"]
            gt = cam.original_image.cuda()
            loss_l1 = l1_loss(image, gt)
            loss = (1.0 - opt.lambda_dssim) * loss_l1 + opt.lambda_dssim * (
                1.0 - ssim(image, gt)
            )
            loss.backward()

            with torch.no_grad():
                if iteration < opt.densify_until_iter:
                    visibility = render_pkg["visibility_filter"]
                    gaussians.max_radii2D[visibility] = torch.max(
                        gaussians.max_radii2D[visibility],
                        render_pkg["radii"][visibility],
                    )
                    gaussians.add_densification_stats(
                        render_pkg["viewspace_points"], visibility
                    )
                    if (
                        iteration > opt.densify_from_iter
                        and iteration % opt.densification_interval == 0
                    ):
                        size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                        gaussians.densify_and_prune(
                            opt.densify_grad_threshold, 0.005, extent, size_threshold
                        )
                    if iteration % opt.opacity_reset_interval == 0:
                        gaussians.reset_opacity()
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)
            if self.verbose and iteration % 500 == 0:
                print(
                    "[fisherrf] iter %d/%d loss %.4f gaussians %d"
                    % (iteration, opt.iterations, loss.item(), gaussians.get_xyz.shape[0])
                )
        return gaussians

    # -- ViewSelector protocol -------------------------------------------------

    def select(self, history: List[CaptureRecord], candidates: List[CameraPose]) -> int:
        import torch

        from active.H_reg import HRegSelector

        torch.manual_seed(self.seed + len(history))
        opt, pipe = self._opt_pipe()
        background = torch.zeros(3, device="cuda")

        train_cameras = self._train_cameras(history)
        gaussians = self._train_model(train_cameras, opt, pipe, background)

        candidate_cameras = self._candidate_cameras(candidates)

        class _SceneShim:
            def get_candidate_set(self_inner):
                return list(range(len(candidate_cameras)))

            def getTrainCameras(self_inner):
                return train_cameras

            def getTestCameras(self_inner):
                return []

            def getCandidateCameras(self_inner):
                return candidate_cameras

        selector_args = Namespace(
            seed=self.seed,
            reg_lambda=self.reg_lambda,
            I_test=False,
            I_acq_reg=False,
            filter_out_grad=list(self.filter_out_grad),
        )
        selector = HRegSelector(selector_args)
        selected = selector.nbvs(
            gaussians, _SceneShim(), 1, pipe, background, exit_func=lambda: False
        )
        best = int(selected[0])
        print(
            "[fisherrf] round %d: %d train views, %d candidates, %d gaussians, best #%d"
            % (
                len(history),
                len(train_cameras),
                len(candidates),
                gaussians.get_xyz.shape[0],
                best,
            )
        )
        return best
