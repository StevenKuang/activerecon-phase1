"""Vanilla 3DGS training on gsplat — no method-repo dependencies.

This is the benchmark's standard Tier-2 backend: the graphdeco training
recipe (L1 + DSSIM loss, exponential means-LR decay, SH warmup, the
500-15000/100 densification window with 3000-step opacity resets) executed
with gsplat's rasterizer and ``DefaultStrategy``, whose defaults mirror that
schedule. Initialization uses the deterministic benchmark RGB-D point cloud
(not COLMAP), like the historical gavis-vendored vanilla backend it replaces.
Runs in the ``bencheval`` env.

Numbers differ slightly from the gavis-vendored dr_aa rasterizer (kernel and
2D-pruning details); rounds record their backend, and cross-method
comparisons always share one backend.
"""

import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from activebench.eval.reconstruction import (
    GaussianSplats,
    ReconstructionCamera,
    ReconstructionDataset,
    ReconstructionRender,
)

_SH_C0 = 0.28209479177387814


def ensure_gsplat_build_env() -> None:
    """Point torch's extension loader at this env's ninja/nvcc toolchain.

    torch re-verifies ninja availability on every gsplat import even when the
    compiled kernel cache is warm, so launchers with a bare PATH (campaign
    subprocesses) would fail. Everything derives from ``sys.executable``, so
    any caller running inside the bencheval env works unchanged.
    """

    import os
    import sys

    prefix = Path(sys.executable).resolve().parent.parent
    bin_dir = str(prefix / "bin")
    if bin_dir not in os.environ.get("PATH", "").split(os.pathsep):
        os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
    os.environ.setdefault("CUDA_HOME", str(prefix))
    targets = prefix / "targets" / "x86_64-linux"
    if targets.exists():
        os.environ.setdefault("CPATH", str(targets / "include"))
        os.environ.setdefault(
            "LIBRARY_PATH",
            os.pathsep.join([str(targets / "lib"), str(prefix / "lib")]),
        )


def _focal(fov: float, pixels: int) -> float:
    return pixels / (2.0 * math.tan(fov / 2.0))


def _camera_extent(dataset: ReconstructionDataset) -> float:
    centers = np.stack([np.asarray(p)[:3, 3] for p in dataset.poses_c2w_cv])
    center = centers.mean(axis=0)
    return max(1.1 * float(np.linalg.norm(centers - center, axis=1).max()), 1e-3)


def _initial_point_cloud(dataset: ReconstructionDataset, per_view: int, seed: int):
    """Subsampled RGB-D unprojection, matching the historical backend."""

    rng = np.random.default_rng(seed)
    fx = _focal(dataset.fovx, dataset.width)
    fy = _focal(dataset.fovy, dataset.height)
    cx, cy = dataset.width / 2.0, dataset.height / 2.0
    xx, yy = np.meshgrid(
        np.arange(dataset.width), np.arange(dataset.height), indexing="xy"
    )
    xx = ((xx - cx) / fx).reshape(-1)
    yy = ((yy - cy) / fy).reshape(-1)
    points: List[np.ndarray] = []
    colors: List[np.ndarray] = []
    for index, (image, pose, depth) in enumerate(
        zip(dataset.images, dataset.poses_c2w_cv, dataset.depths)
    ):
        d = np.asarray(depth, dtype=np.float64).reshape(-1)
        valid = d > 0.0
        if dataset.alpha_masks is not None:
            valid &= np.asarray(dataset.alpha_masks[index]).reshape(-1) > 0.5
        valid_indices = np.flatnonzero(valid)
        if len(valid_indices) == 0:
            continue
        if len(valid_indices) > per_view:
            valid_indices = rng.choice(valid_indices, size=per_view, replace=False)
        dd = d[valid_indices]
        camera_points = np.stack(
            [xx[valid_indices] * dd, yy[valid_indices] * dd, dd], axis=-1
        )
        homogeneous = np.column_stack([camera_points, np.ones(len(camera_points))])
        world = (np.asarray(pose, dtype=np.float64) @ homogeneous.T).T[:, :3]
        points.append(world)
        colors.append(
            np.asarray(image, dtype=np.float64)[..., :3].reshape(-1, 3)[valid_indices]
        )
    if not points:
        raise ValueError("all training depth maps are empty")
    return np.concatenate(points, axis=0), np.concatenate(colors, axis=0)


def _knn_mean_sq_dist(points: np.ndarray) -> np.ndarray:
    """Mean squared distance to the 3 nearest neighbors (vanilla scale init)."""

    from scipy.spatial import cKDTree

    distances, _ = cKDTree(points).query(points, k=4)
    return np.clip((distances[:, 1:] ** 2).mean(axis=1), 1e-7, None)


def _ssim(pred, gt):
    """Differentiable channel-averaged SSIM (11x11 Gaussian, standard C1/C2)."""

    import torch
    import torch.nn.functional as functional

    coords = torch.arange(11, dtype=pred.dtype, device=pred.device) - 5
    gaussian = torch.exp(-(coords**2) / (2 * 1.5**2))
    gaussian = gaussian / gaussian.sum()
    window = (gaussian[:, None] * gaussian[None, :]).expand(3, 1, 11, 11)
    x, y = pred[None], gt[None]
    mu_x = functional.conv2d(x, window, padding=5, groups=3)
    mu_y = functional.conv2d(y, window, padding=5, groups=3)
    sigma_x = functional.conv2d(x * x, window, padding=5, groups=3) - mu_x**2
    sigma_y = functional.conv2d(y * y, window, padding=5, groups=3) - mu_y**2
    sigma_xy = functional.conv2d(x * y, window, padding=5, groups=3) - mu_x * mu_y
    score = ((2 * mu_x * mu_y + 0.01**2) * (2 * sigma_xy + 0.03**2)) / (
        (mu_x**2 + mu_y**2 + 0.01**2) * (sigma_x + sigma_y + 0.03**2)
    )
    return score.mean()


class _GsplatReconstruction:
    """Fitted splats rendered with gsplat at arbitrary held-out poses."""

    # The exported npz holds every parameter this renderer uses, and reloading
    # it renders through the same flat gsplat rasterizer, so scoring the export
    # reproduces scoring this object. Backends with their own renderer (i3dgs's
    # hierarchical LOD cut, the feed-forward pair) must NOT set this: a flat
    # re-render of their export measures a different model.
    scores_from_export = True

    def __init__(self, params, dataset: ReconstructionDataset, metadata: Dict[str, Any]):
        self._params = params
        self._dataset = dataset
        self._metadata = metadata

    @property
    def metadata(self) -> Dict[str, Any]:
        return dict(self._metadata)

    def _intrinsics_tensor(self, camera: ReconstructionCamera):
        import torch

        return torch.tensor([
            [camera.fx, 0.0, camera.cx],
            [0.0, camera.fy, camera.cy],
            [0.0, 0.0, 1.0],
        ], dtype=torch.float32, device="cuda")[None]

    def render(
        self,
        c2w_cv: np.ndarray,
        camera: Optional[ReconstructionCamera] = None,
    ) -> ReconstructionRender:
        ensure_gsplat_build_env()
        import torch
        from gsplat import rasterization

        p = self._params
        camera = camera or ReconstructionCamera.from_dataset(self._dataset)
        viewmat = torch.from_numpy(
            np.linalg.inv(np.asarray(c2w_cv, dtype=np.float64))
        ).float().cuda()[None]
        with torch.no_grad():
            rendered, _alphas, _meta = rasterization(
                means=p["means"],
                quats=p["quats"],
                scales=torch.exp(p["scales"]),
                opacities=torch.sigmoid(p["opacities"]),
                colors=torch.cat([p["sh0"], p["shN"]], dim=1),
                sh_degree=int(self._metadata["sh_degree"]),
                viewmats=viewmat,
                Ks=self._intrinsics_tensor(camera),
                width=camera.width,
                height=camera.height,
                render_mode="RGB+ED",
            )
        image = rendered[0].cpu().numpy()
        return ReconstructionRender(
            rgb=np.clip(image[..., :3], 0.0, 1.0), depth=image[..., 3]
        )

    def gaussian_splats(self) -> GaussianSplats:
        import torch

        p = self._params
        sh_degree = int(self._metadata.get("sh_degree", 0))
        with torch.no_grad():
            quats = torch.nn.functional.normalize(p["quats"], dim=-1)
            # colors is the viewer-facing clamped RGB; sh0 keeps the raw band
            # so an export can round-trip the model exactly.
            colors = torch.clamp(p["sh0"][:, 0, :] * _SH_C0 + 0.5, 0.0, 1.0)
            kwargs: Dict[str, Any] = dict(
                sh0=p["sh0"].cpu().numpy(),
                centers=p["means"].cpu().numpy(),
                scales=torch.exp(p["scales"]).cpu().numpy(),
                quats_wxyz=quats.cpu().numpy(),
                opacities=torch.sigmoid(p["opacities"]).cpu().numpy(),
                colors=colors.cpu().numpy(),
            )
            if sh_degree > 0 and "shN" in p:
                kwargs["sh_rest"] = p["shN"].cpu().numpy()
                kwargs["sh_degree"] = sh_degree
            return GaussianSplats(**kwargs)


@dataclass
class GsplatVanilla3DGSBackend:
    """Vanilla 3DGS optimization with gsplat's rasterizer and DefaultStrategy."""

    name: str = "gsplat"

    def reconstruct(
        self,
        dataset: ReconstructionDataset,
        *,
        iterations: int,
        seed: int,
        options: Dict[str, Any],
    ) -> _GsplatReconstruction:
        if iterations <= 0:
            raise ValueError("iterations must be positive")
        defaults = {
            "depth_init_pts_per_view": 10_000,
            "sh_degree": 3,
            "lambda_dssim": 0.2,
            "position_lr_init": 0.00016,
            "position_lr_final": 0.0000016,
            "position_lr_max_steps": 30_000,
            "feature_lr": 0.0025,
            "opacity_lr": 0.05,
            "scaling_lr": 0.005,
            "rotation_lr": 0.001,
            "init_opacity": 0.1,
            "densify_from": 500,
            "densify_until": 15_000,
            "densification_interval": 100,
            "opacity_reset_interval": 3_000,
            "densify_grad_threshold": 0.0002,
            "prune_opacity": 0.005,
            # Hard ceiling on the gaussian count: when reached, refinement
            # (grow + prune + opacity resets — all gated on refine_stop_iter
            # in gsplat 1.5.3) stops and optimization continues on the frozen
            # set. None keeps the unbounded vanilla schedule. High-resolution
            # streams (1600x1200) densify ~6x harder than 640x480 and can
            # exceed 31 GiB VRAM without a cap.
            "max_gaussians": None,
        }
        unknown = sorted(set(options) - set(defaults))
        if unknown:
            raise ValueError("unknown backend options: %s" % ", ".join(unknown))
        params_cfg = {**defaults, **options}

        ensure_gsplat_build_env()
        import torch
        from gsplat import DefaultStrategy, rasterization

        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        points, colors = _initial_point_cloud(
            dataset, int(params_cfg["depth_init_pts_per_view"]), seed
        )
        extent = _camera_extent(dataset)
        count = len(points)
        sh_degree = int(params_cfg["sh_degree"])
        sh_coeffs = (sh_degree + 1) ** 2
        scales0 = np.log(np.sqrt(_knn_mean_sq_dist(points)))[:, None].repeat(3, axis=1)
        sh0 = ((colors - 0.5) / _SH_C0)[:, None, :]

        def parameter(array, dtype=np.float32):
            return torch.nn.Parameter(
                torch.from_numpy(np.asarray(array, dtype=dtype)).cuda()
            )

        quats0 = np.zeros((count, 4), dtype=np.float32)
        quats0[:, 0] = 1.0
        opacity0 = math.log(
            params_cfg["init_opacity"] / (1.0 - params_cfg["init_opacity"])
        )
        params = torch.nn.ParameterDict({
            "means": parameter(points),
            "scales": parameter(scales0),
            "quats": parameter(quats0),
            "opacities": parameter(np.full(count, opacity0, dtype=np.float32)),
            "sh0": parameter(sh0),
            "shN": parameter(np.zeros((count, sh_coeffs - 1, 3), dtype=np.float32)),
        }).cuda()
        optimizers = {
            "means": torch.optim.Adam(
                [params["means"]],
                lr=float(params_cfg["position_lr_init"]) * extent, eps=1e-15,
            ),
            "scales": torch.optim.Adam(
                [params["scales"]], lr=float(params_cfg["scaling_lr"]), eps=1e-15),
            "quats": torch.optim.Adam(
                [params["quats"]], lr=float(params_cfg["rotation_lr"]), eps=1e-15),
            "opacities": torch.optim.Adam(
                [params["opacities"]], lr=float(params_cfg["opacity_lr"]), eps=1e-15),
            "sh0": torch.optim.Adam(
                [params["sh0"]], lr=float(params_cfg["feature_lr"]), eps=1e-15),
            "shN": torch.optim.Adam(
                [params["shN"]], lr=float(params_cfg["feature_lr"]) / 20.0, eps=1e-15),
        }
        gamma = (
            float(params_cfg["position_lr_final"]) / float(params_cfg["position_lr_init"])
        ) ** (1.0 / float(params_cfg["position_lr_max_steps"]))
        means_schedule = torch.optim.lr_scheduler.ExponentialLR(
            optimizers["means"], gamma=gamma
        )
        strategy = DefaultStrategy(
            prune_opa=float(params_cfg["prune_opacity"]),
            grow_grad2d=float(params_cfg["densify_grad_threshold"]),
            refine_start_iter=int(params_cfg["densify_from"]),
            refine_stop_iter=int(params_cfg["densify_until"]),
            refine_every=int(params_cfg["densification_interval"]),
            reset_every=int(params_cfg["opacity_reset_interval"]),
        )
        strategy.check_sanity(params, optimizers)
        state = strategy.initialize_state(scene_scale=extent)

        intrinsics = torch.tensor([
            [_focal(dataset.fovx, dataset.width), 0.0, dataset.width / 2.0],
            [0.0, _focal(dataset.fovy, dataset.height), dataset.height / 2.0],
            [0.0, 0.0, 1.0],
        ], dtype=torch.float32, device="cuda")[None]
        viewmats = [
            torch.from_numpy(
                np.linalg.inv(np.asarray(pose, dtype=np.float64))
            ).float().cuda()[None]
            for pose in dataset.poses_c2w_cv
        ]
        images = [
            torch.from_numpy(np.asarray(image, dtype=np.float32)).cuda()
            for image in dataset.images
        ]
        alphas = None
        if dataset.alpha_masks is not None:
            alphas = [
                torch.from_numpy(np.asarray(mask, dtype=np.float32)).cuda()[..., None]
                for mask in dataset.alpha_masks
            ]
        lambda_dssim = float(params_cfg["lambda_dssim"])
        viewpoint_stack: List[int] = []
        optimization_started = time.perf_counter()
        print(
            "[gsplat] initialized %d Gaussians from %d views at %dx%d"
            % (count, len(images), dataset.width, dataset.height),
            flush=True,
        )
        for step in range(1, iterations + 1):
            if not viewpoint_stack:
                viewpoint_stack = list(range(len(images)))
            view = viewpoint_stack.pop(random.randrange(len(viewpoint_stack)))
            active_sh = min(step // 1000, sh_degree)
            rendered, _alphas, info = rasterization(
                means=params["means"],
                quats=params["quats"],
                scales=torch.exp(params["scales"]),
                opacities=torch.sigmoid(params["opacities"]),
                colors=torch.cat([params["sh0"], params["shN"]], dim=1),
                sh_degree=active_sh,
                viewmats=viewmats[view],
                Ks=intrinsics,
                width=dataset.width,
                height=dataset.height,
                render_mode="RGB",
                packed=False,  # DefaultStrategy reads dense-mode info tensors
            )
            pred = rendered[0].clamp(0.0, 1.0)
            gt = images[view]
            if alphas is not None:
                pred = pred * alphas[view]
                gt = gt * alphas[view]
            strategy.step_pre_backward(params, optimizers, state, step, info)
            loss_l1 = (pred - gt).abs().mean()
            loss = (1.0 - lambda_dssim) * loss_l1 + lambda_dssim * (
                1.0 - _ssim(pred.permute(2, 0, 1), gt.permute(2, 0, 1))
            )
            loss.backward()
            strategy.step_post_backward(params, optimizers, state, step, info)
            max_gaussians = params_cfg["max_gaussians"]
            if (
                max_gaussians is not None
                and step < strategy.refine_stop_iter
                and len(params["means"]) >= int(max_gaussians)
            ):
                strategy.refine_stop_iter = step
                print(
                    "[gsplat] gaussian cap %d reached at iter %d "
                    "(%d Gaussians) — densification frozen"
                    % (int(max_gaussians), step, len(params["means"])),
                    flush=True,
                )
            for optimizer in optimizers.values():
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            means_schedule.step()
            if step == 1 or step % 1_000 == 0 or step == iterations:
                elapsed = time.perf_counter() - optimization_started
                rate = step / max(elapsed, 1e-9)
                eta = (iterations - step) / max(rate, 1e-9)
                print(
                    "[gsplat] iter %d/%d, loss %.5f, Gaussians %d, "
                    "elapsed %.1fs, ETA %.1fs"
                    % (
                        step,
                        iterations,
                        float(loss.detach()),
                        len(params["means"]),
                        elapsed,
                        eta,
                    ),
                    flush=True,
                )

        metadata = {
            "backend": self.name,
            "implementation": "gsplat rasterization + DefaultStrategy",
            "initialization": "benchmark RGB-D point cloud (not COLMAP SfM)",
            "gaussian_scale_initialization": "nearest-neighbor distance (vanilla 3DGS)",
            "supervision": "RGB L1 + DSSIM",
            "isotropic_loss": False,
            "depth_supervision": False,
            "iterations": iterations,
            "initial_num_gaussians": count,
            **params_cfg,
        }
        return _GsplatReconstruction(params, dataset, metadata)
