"""Tier-2 metric: standardized reconstruction + quality evaluation.

Protocol (identical for every method, that is the point):

1. Pass exactly the episode's standardized reconstruction frames to a selected
   backend. The default is vanilla 3DGS optimization with RGB-D point-cloud
   initialization and RGB-only supervision. Distractor masks are NOT applied
   by default: contaminated captures bake ghosts into the model and cost score
   against the clean ground truth (``use_masks=True`` gives the masked upper
   bound).
2. Render the held-out clean eval views and report appearance
   (PSNR / SSIM / optional LPIPS) and depth MAE **per eval stratum**
   (level / lookup / lookdown, Tier-3).
3. Compare opacity-filtered Gaussian means against the GT surface samples
   and report completeness@τ **per orientation bin** (up / side / down,
   Tier-1 bins) plus accuracy (recon→GT distances, floaters penalized).

The geometry helpers and evaluation schema are backend-independent. Individual
backends declare their own runtime dependencies.
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from activebench.eval.reconstruction import (
    ReconstructionBackend,
    ReconstructionCamera,
    ReconstructionDataset,
    create_reconstruction_backend,
    reconstruction_artifacts,
)

class _LazyFrames:
    """Decode-on-access frame sequence so a streaming backend never holds the
    whole set in RAM. ``len()`` and integer/slice indexing behave like a list;
    iteration re-decodes. Only used for backends that read frames from disk
    themselves (i3dgs) -- materializing backends still get eager lists.
    """

    def __init__(self, items, loader):
        self._items = list(items)
        self._loader = loader

    def __len__(self):
        return len(self._items)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self._loader(x) for x in self._items[index]]
        return self._loader(self._items[index])


DEFAULT_TAUS = (0.05, 0.10)
DEFAULT_BIN_COS = 0.7
DEFAULT_OPACITY_THRESHOLD = 0.5
# Gaussians below this opacity are dropped from the exported model: they are
# individually invisible and keeping them roughly doubles the artifact. Scoring
# runs on the export, so this defines the evaluated model rather than creating a
# gap between the measured and the shipped one.
_VISIBILITY_CUTOFF = 0.05
ACCURACY_CAP = 1.0  # meters; keeps far floaters from dominating the mean
_SH_C0 = 0.28209479177387814


def psnr(pred: np.ndarray, gt: np.ndarray) -> float:
    """PSNR in dB between images scaled to [0, 1]."""

    mse = float(np.mean((pred.astype(np.float64) - gt.astype(np.float64)) ** 2))
    if mse <= 1e-12:
        return 99.0
    return float(10.0 * np.log10(1.0 / mse))


def _torch_ssim(pred: np.ndarray, gt: np.ndarray) -> float:
    """Channel-averaged SSIM matching the standard 3DGS metric."""

    import torch
    import torch.nn.functional as functional

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.from_numpy(pred).permute(2, 0, 1).unsqueeze(0).to(device)
    y = torch.from_numpy(gt).permute(2, 0, 1).unsqueeze(0).to(device)
    coords = torch.arange(11, dtype=x.dtype, device=device) - 5
    gaussian = torch.exp(-(coords**2) / (2 * 1.5**2))
    gaussian /= gaussian.sum()
    window = (gaussian[:, None] * gaussian[None, :]).expand(3, 1, 11, 11)
    mu_x = functional.conv2d(x, window, padding=5, groups=3)
    mu_y = functional.conv2d(y, window, padding=5, groups=3)
    mu_x_sq = mu_x.square()
    mu_y_sq = mu_y.square()
    mu_xy = mu_x * mu_y
    sigma_x = functional.conv2d(x * x, window, padding=5, groups=3) - mu_x_sq
    sigma_y = functional.conv2d(y * y, window, padding=5, groups=3) - mu_y_sq
    sigma_xy = functional.conv2d(x * y, window, padding=5, groups=3) - mu_xy
    score = ((2 * mu_xy + 0.01**2) * (2 * sigma_xy + 0.03**2)) / (
        (mu_x_sq + mu_y_sq + 0.01**2) * (sigma_x + sigma_y + 0.03**2)
    )
    return float(score.mean())


def geometry_metrics(
    recon_points: np.ndarray,
    gt_points: np.ndarray,
    gt_normals: np.ndarray,
    taus=DEFAULT_TAUS,
    bin_cos: float = DEFAULT_BIN_COS,
) -> Dict[str, Any]:
    """Completeness@τ per orientation bin and recon→GT accuracy."""

    from scipy.spatial import cKDTree

    result: Dict[str, Any] = {"num_recon_points": int(len(recon_points))}
    n_y = gt_normals[:, 1]
    bins = {
        "up": n_y > bin_cos,
        "down": n_y < -bin_cos,
        "side": np.abs(n_y) <= bin_cos,
    }
    if len(recon_points) == 0:
        result["bins"] = {
            name: {"num_points": int(mask.sum()), **{"completeness@%.2f" % t: 0.0 for t in taus}}
            for name, mask in bins.items()
        }
        result["accuracy"] = {"mean": ACCURACY_CAP, "median": ACCURACY_CAP}
        return result

    recon_tree = cKDTree(recon_points)
    gt_to_recon, _ = recon_tree.query(gt_points, k=1)
    result["bins"] = {}
    for name, mask in bins.items():
        entry = {"num_points": int(mask.sum())}
        for tau in taus:
            entry["completeness@%.2f" % tau] = (
                float((gt_to_recon[mask] < tau).mean()) if mask.any() else 0.0
            )
        result["bins"][name] = entry
    for tau in taus:
        result["completeness@%.2f" % tau] = float((gt_to_recon < tau).mean())

    gt_tree = cKDTree(gt_points)
    recon_to_gt, _ = gt_tree.query(recon_points, k=1)
    recon_to_gt = np.minimum(recon_to_gt, ACCURACY_CAP)
    result["accuracy"] = {
        "mean": float(recon_to_gt.mean()),
        "median": float(np.median(recon_to_gt)),
        "frac_within@%.2f" % taus[0]: float((recon_to_gt < taus[0]).mean()),
    }
    return result


@dataclass
class RetrainConfig:
    backend: str = "vanilla-3dgs"
    run_name: Optional[str] = None
    backend_options: Dict[str, Any] = field(default_factory=dict)
    train_iterations: int = 30_000
    use_masks: bool = False
    opacity_threshold: float = DEFAULT_OPACITY_THRESHOLD
    aabb_margin: float = 0.1
    seed: int = 0
    max_depth: float = 8.0
    # Additional common output sizes scored from the canonical eval render.
    # Targets must not exceed the eval-set resolution.
    evaluation_resolutions: Tuple[Tuple[int, int], ...] = ()
    # Protocol runs use LPIPS for model selection and must not silently emit
    # null values when its package or pretrained weights are unavailable.
    require_lpips: bool = False
    # Persist renders and generic Gaussian artifacts under the named backend run.
    save_renders: bool = False
    # Re-score an existing gaussians.npz without retraining (e.g. against a
    # rerendered higher-resolution shared eval set). Training-time fields of
    # a previous eval.json are preserved; only scores are recomputed.
    eval_only: bool = False
    # Ignore a raw sh0 band on reload so that a matrix mixing pre- and
    # post-sh0 artifacts is scored in one regime. See NpzGaussianReconstruction.
    force_legacy_dc: bool = False
    # Inherit completeness/accuracy from the eval being replaced instead of
    # recomputing an identical value. Only valid when the model and the GT
    # surface samples are unchanged, i.e. for an eval-set-only re-score.
    reuse_geometry: bool = False


class NpzGaussianReconstruction:
    """Renderable reconstruction reloaded from a saved ``gaussians.npz``.

    The artifact stores post-activation parameters (linear scales, sigmoid
    opacities, normalized wxyz quaternions) above the visibility cutoff, plus
    ``sh_rest``/``sh_degree`` when the backend trained view-dependent color.

    Colour comes from the raw ``sh0`` band when present, which reproduces the
    exported model exactly. Artifacts written before ``sh0`` was stored only
    carry the clamped [0, 1] ``colors``; the DC is then recovered from those and
    every gaussian that had clamped out of gamut comes back altered, so such a
    file cannot reproduce the numbers it was scored with (``raw_dc`` is False).
    """

    def __init__(self, npz_path: Path, force_legacy_dc: bool = False):
        from activebench.eval.gsplat_backend import ensure_gsplat_build_env

        ensure_gsplat_build_env()
        import torch

        self._npz_path = Path(npz_path)
        data = np.load(self._npz_path)
        self.num_loaded = int(data["opacities"].shape[0])
        self.sh_degree = int(data["sh_degree"]) if "sh_degree" in data else 0
        # ``force_legacy_dc`` exists for cross-cell homogeneity, not fidelity:
        # when part of a matrix predates the ``sh0`` export, scoring the newer
        # cells through their raw band and the older ones through clamped RGB
        # mixes two regimes and biases the comparison between them. Forcing the
        # clamped path everywhere costs ~0.15 dB uniformly and keeps the matrix
        # internally comparable, which is what a ranking needs.
        if "sh0" in data and not force_legacy_dc:
            sh0 = data["sh0"].astype(np.float32)
            self.raw_dc = True
        else:
            # Legacy artifact: recover the DC band from the clamped RGB. Any
            # gaussian whose DC mapped outside [0, 1] comes back changed, so a
            # re-score of such a file cannot reproduce its original numbers.
            sh0 = ((data["colors"].astype(np.float32) - 0.5) / _SH_C0)[:, None, :]
            self.raw_dc = False
        if self.sh_degree > 0 and "sh_rest" in data:
            sh = np.concatenate([sh0, data["sh_rest"].astype(np.float32)], axis=1)
        else:
            self.sh_degree = 0
            sh = sh0
        to_cuda = lambda a: torch.from_numpy(np.ascontiguousarray(a)).float().cuda()
        self._means = to_cuda(data["centers"])
        self._scales = to_cuda(data["scales"])
        self._quats = to_cuda(data["quats_wxyz"])
        self._opacities = to_cuda(data["opacities"])
        self._sh = to_cuda(sh)
        self._arrays = {
            key: np.asarray(data[key])
            for key in ("centers", "scales", "quats_wxyz", "opacities", "colors")
        }
        self._sh_rest = np.asarray(data["sh_rest"]) if "sh_rest" in data else None

    @property
    def metadata(self) -> Dict[str, Any]:
        return {
            "renderer": "npz-reload (gsplat rasterization)",
            "source_npz": str(self._npz_path),
            "num_loaded_gaussians": self.num_loaded,
            "sh_degree": self.sh_degree,
        }

    def render(self, c2w_cv: np.ndarray, camera) -> "ReconstructionRender":
        import torch
        from gsplat import rasterization

        from activebench.eval.reconstruction import ReconstructionRender

        viewmat = torch.from_numpy(
            np.linalg.inv(np.asarray(c2w_cv, dtype=np.float64))
        ).float().cuda()[None]
        intrinsics = torch.tensor([
            [camera.fx, 0.0, camera.cx],
            [0.0, camera.fy, camera.cy],
            [0.0, 0.0, 1.0],
        ], dtype=torch.float32, device="cuda")[None]
        with torch.no_grad():
            rendered, _alphas, _meta = rasterization(
                means=self._means,
                quats=self._quats,
                scales=self._scales,
                opacities=self._opacities,
                colors=self._sh,
                sh_degree=self.sh_degree,
                viewmats=viewmat,
                Ks=intrinsics,
                width=camera.width,
                height=camera.height,
                render_mode="RGB+ED",
            )
        image = rendered[0].cpu().numpy()
        return ReconstructionRender(
            rgb=np.clip(image[..., :3], 0.0, 1.0), depth=image[..., 3]
        )

    def gaussian_splats(self):
        from activebench.eval.reconstruction import GaussianSplats

        kwargs: Dict[str, Any] = dict(
            centers=self._arrays["centers"],
            scales=self._arrays["scales"],
            quats_wxyz=self._arrays["quats_wxyz"],
            opacities=self._arrays["opacities"],
            colors=self._arrays["colors"],
        )
        if self._sh_rest is not None and self.sh_degree > 0:
            kwargs["sh_rest"] = self._sh_rest
            kwargs["sh_degree"] = self.sh_degree
        return GaussianSplats(**kwargs)


def _camera_from_payload(
    payload: Dict[str, Any],
    fallback: ReconstructionCamera,
) -> ReconstructionCamera:
    """Read exact eval intrinsics, falling back for legacy local eval sets."""

    return ReconstructionCamera(
        width=int(payload.get("w", fallback.width)),
        height=int(payload.get("h", fallback.height)),
        fx=float(payload.get("fl_x", fallback.fx)),
        fy=float(payload.get("fl_y", fallback.fy)),
        cx=float(payload.get("cx", fallback.cx)),
        cy=float(payload.get("cy", fallback.cy)),
    )


def export_gaussian_artifacts(reconstruction, artifacts, config) -> Dict[str, Any]:
    """Write ``points.npz`` + ``gaussians.npz`` and describe what was kept.

    ``gaussians.npz`` is both the viewer's model and the unit a re-score loads,
    so it stores everything rendering depends on:

    - the raw DC band as ``sh0``. ``colors`` is the clamped [0, 1] RGB the
      viewer wants, and on real scenes ~22% of its channels sit on a clamp
      boundary; reconstructing the DC from it would silently darken or brighten
      those gaussians on reload. ``colors`` stays for viewer compatibility.
    - ``sh_rest``/``sh_degree`` when the backend trained view-dependent colour.

    The ``opacity > 0.05`` visibility cutoff is kept: it roughly halves the file
    and those gaussians are individually invisible. Because scoring now happens
    on this artifact, the cutoff is part of the evaluated model rather than a
    discrepancy between what is measured and what is shipped.
    """

    from activebench.common.io import save_npz_compressed

    splats = reconstruction.gaussian_splats()
    opacity = splats.opacities
    keep = opacity > config.opacity_threshold
    save_npz_compressed(
        artifacts.points_path,
        points=splats.centers[keep].astype(np.float32),
        colors=(splats.colors[keep] * 255).astype(np.uint8),
    )

    vis = opacity > _VISIBILITY_CUTOFF
    npz_payload = dict(
        centers=splats.centers[vis].astype(np.float32),
        scales=splats.scales[vis].astype(np.float32),
        quats_wxyz=splats.quats_wxyz[vis].astype(np.float32),
        opacities=opacity[vis].astype(np.float32),
        colors=splats.colors[vis].astype(np.float32),
    )
    if splats.sh0 is not None:
        npz_payload["sh0"] = splats.sh0[vis].astype(np.float32)
    if splats.sh_rest is not None:
        npz_payload["sh_rest"] = splats.sh_rest[vis].astype(np.float32)
        npz_payload["sh_degree"] = int(splats.sh_degree)
    save_npz_compressed(artifacts.gaussians_path, **npz_payload)
    return {
        "trained_gaussians": int(len(splats.centers)),
        "exported_gaussians": int(vis.sum()),
        "visibility_cutoff": _VISIBILITY_CUTOFF,
        "raw_dc_exported": splats.sh0 is not None,
        "scored_from_export": True,
    }


def _aggregate_appearance(per_stratum: Dict[str, Dict[str, list]]) -> Dict[str, Any]:
    appearance = {
        stratum: {
            metric: float(np.nanmean(values)) if values else None
            for metric, values in metrics.items()
        }
        for stratum, metrics in per_stratum.items()
    }
    all_metrics: Dict[str, Any] = {}
    for metric in (
        "psnr", "ssim", "lpips", "depth_mae",
        "render_coverage", "depth_mae_fixed", "depth_p50_fixed",
        "depth_mae_zerofill",
    ):
        values = [
            value
            for metrics in per_stratum.values()
            for value in metrics.get(metric, [])
        ]
        all_metrics[metric] = float(np.nanmean(values)) if values else None
    appearance["all"] = all_metrics
    return appearance


def _resize_rgb(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """Area-resample an RGB float image without an 8-bit round trip."""

    import torch
    import torch.nn.functional as functional

    if image.shape[:2] == (height, width):
        return image
    tensor = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1)[None]
    resized = functional.interpolate(tensor, size=(height, width), mode="area")
    return resized[0].permute(1, 2, 0).numpy()


# LPIPS's AlexNet backbone pools the input five times, so a side shorter than
# this collapses an intermediate feature map to zero width and raises inside
# torch. Measured on the pinned net: 31 px per side still runs, 24 px does not.
# Benchmark evaluations are 640x480 and up, so this only guards degenerate
# resolutions (unit fixtures, smoke episodes), where lpips is reported as null
# via the existing empty-metric path rather than crashing the whole eval.
_LPIPS_MIN_SIDE = 32


def _load_lpips_model(device: str):
    """Load the pinned AlexNet LPIPS model on ``device``."""

    import lpips

    return lpips.LPIPS(net="alex").eval().to(device)


def run_retrain_eval(
    episode_dir: Path,
    surface_samples_path: Path,
    config: Optional[RetrainConfig] = None,
    gavis_repo: str = "/home/steven/Projects/gavis",
    output_name: Optional[str] = None,
    backend_impl: Optional[ReconstructionBackend] = None,
    eval_set_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run one named reconstruction backend and score its output.

    Held-out appearance is scored on the SHARED eval set when
    ``eval_set_dir`` is given (fixed poses reused across methods and
    difficulties; also writes the two-class ``eval_shared.json`` summary at
    full model fidelity). Without it, legacy episode-local
    ``transforms_eval.json`` views are used — kept only so archived pre-shared
    episodes stay evaluable.
    """

    import torch
    from PIL import Image

    from activebench.eval.dataset import training_transforms_path

    episode_dir = Path(episode_dir)
    config = config or RetrainConfig()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # Validate the package and pretrained weights before an expensive protocol
    # fit. Keep the probe on CPU so it does not consume reconstruction VRAM.
    lpips_model = None
    lpips_error = None
    if config.require_lpips:
        try:
            lpips_model = _load_lpips_model("cpu")
        except Exception as exc:
            lpips_error = "%s: %s" % (type(exc).__name__, exc)
            raise RuntimeError(
                "LPIPS is required for this evaluation but could not be initialized: %s"
                % lpips_error
            ) from exc

    def load_split(name):
        payload = json.loads((episode_dir / name).read_text())
        return payload

    train_transforms = training_transforms_path(episode_dir)
    train_payload = json.loads(train_transforms.read_text())
    if eval_set_dir is not None:
        eval_root = Path(eval_set_dir)
        eval_payload = json.loads(
            (eval_root / "transforms_eval_shared.json").read_text()
        )
    else:
        eval_root = episode_dir
        eval_payload = load_split("transforms_eval.json")
    if len(train_payload["frames"]) < 3:
        raise ValueError(
            "episode has only %d captured frames; the standardized retrain "
            "needs at least 3 (degenerate/aborted episode)" % len(train_payload["frames"])
        )
    w, h = int(train_payload["w"]), int(train_payload["h"])
    fovx = 2.0 * float(np.arctan(w / (2.0 * train_payload["fl_x"])))
    fovy = 2.0 * float(np.arctan(h / (2.0 * train_payload["fl_y"])))
    train_camera = ReconstructionCamera(
        width=w,
        height=h,
        fx=float(train_payload["fl_x"]),
        fy=float(train_payload["fl_y"]),
        cx=float(train_payload.get("cx", w / 2.0)),
        cy=float(train_payload.get("cy", h / 2.0)),
    )
    eval_camera = _camera_from_payload(eval_payload, train_camera)
    evaluation_resolutions = []
    for width, height in ((eval_camera.width, eval_camera.height),) + tuple(
        config.evaluation_resolutions
    ):
        width, height = int(width), int(height)
        if width > eval_camera.width or height > eval_camera.height:
            raise ValueError(
                "evaluation resolution %dx%d exceeds canonical eval set %dx%d"
                % (width, height, eval_camera.width, eval_camera.height)
            )
        if (width, height) not in evaluation_resolutions:
            evaluation_resolutions.append((width, height))

    gl_cv = np.diag([1.0, -1.0, -1.0, 1.0])
    samples = np.load(surface_samples_path)
    gt_points = samples["points"].astype(np.float64)
    gt_normals = samples["normals"].astype(np.float64)
    run_name = config.run_name or config.backend
    artifacts = reconstruction_artifacts(episode_dir, run_name)

    if config.eval_only:
        # Reload the saved model; the (possibly large) training images are
        # never touched and training-time provenance is preserved below.
        if not artifacts.gaussians_path.exists():
            raise FileNotFoundError(
                "eval_only needs an existing %s" % artifacts.gaussians_path
            )
        reconstruction = NpzGaussianReconstruction(
                artifacts.gaussians_path, force_legacy_dc=config.force_legacy_dc
            )
        backend = None
        resource_usage = None
    else:
        # Build the backend first so we know whether it streams frames from
        # disk (i3dgs) and can skip materializing the whole frame set in RAM.
        backend = backend_impl or create_reconstruction_backend(
            config.backend, gavis_repo=gavis_repo
        )
        stream_from_disk = getattr(backend, "consumes_frame_paths", False)

        max_depth = config.max_depth

        def _load_rgb(path):
            return np.asarray(Image.open(path))[..., :3].astype(np.float32) / 255.0

        def _load_depth(path):
            depth = np.load(path).astype(np.float32)
            depth[depth > max_depth] = 0.0
            return depth

        def _load_mask(frame):
            if frame.get("mask_path"):
                mask = np.asarray(Image.open(episode_dir / frame["mask_path"]))
                if mask.ndim == 3:
                    mask = mask[..., 0]
                return (mask == 0).astype(np.float32)
            return np.ones((h, w), dtype=np.float32)

        poses, image_paths, depth_paths = [], [], []
        for frame in train_payload["frames"]:
            poses.append(np.asarray(frame["transform_matrix"], dtype=np.float64) @ gl_cv)
            image_paths.append(episode_dir / frame["file_path"])
            depth_paths.append(episode_dir / frame["depth_path"])

        if stream_from_disk:
            images = _LazyFrames(image_paths, _load_rgb)
            depths = _LazyFrames(depth_paths, _load_depth)
            alpha_masks = (
                _LazyFrames(train_payload["frames"], _load_mask)
                if config.use_masks else None
            )
        else:
            images = [_load_rgb(p) for p in image_paths]
            depths = [_load_depth(p) for p in depth_paths]
            alpha_masks = (
                [_load_mask(f) for f in train_payload["frames"]]
                if config.use_masks else None
            )
        lo, hi = gt_points.min(axis=0), gt_points.max(axis=0)
        pad = config.aabb_margin * (hi - lo)
        aabb = np.stack([lo - pad, hi + pad]).T  # (3, 2) as gavis expects

        dataset = ReconstructionDataset(
            images=images,
            poses_c2w_cv=poses,
            depths=depths,
            alpha_masks=alpha_masks,
            width=w,
            height=h,
            fovx=fovx,
            fovy=fovy,
            aabb=aabb,
            image_paths=image_paths,
            depth_paths=depth_paths,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        train_started = time.perf_counter()
        reconstruction = backend.reconstruct(
            dataset,
            iterations=config.train_iterations,
            seed=config.seed,
            options=dict(config.backend_options),
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        train_wall_time = time.perf_counter() - train_started
        resource_usage = {
            "train_wall_time_s": train_wall_time,
            "peak_cuda_allocated_bytes": (
                int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None
            ),
            "peak_cuda_reserved_bytes": (
                int(torch.cuda.max_memory_reserved()) if torch.cuda.is_available() else None
            ),
        }
    artifacts.root.mkdir(parents=True, exist_ok=True)

    # Move the preflighted model to CUDA, or try an optional late load for
    # legacy ad-hoc evaluations.
    try:
        if lpips_model is None:
            lpips_model = _load_lpips_model("cuda")
        else:
            lpips_model = lpips_model.cuda()
    except Exception as exc:
        lpips_error = "%s: %s" % (type(exc).__name__, exc)
        lpips_model = None
        if config.require_lpips:
            raise RuntimeError(
                "LPIPS is required for this evaluation but could not be initialized: %s"
                % lpips_error
            ) from exc

    # Export the model BEFORE scoring, then score the exported artifact rather
    # than the in-memory one. The artifact is what ships (viewer, PLY, RAD) and
    # what anyone re-scoring the run will load, so making it the thing measured
    # keeps a single number per cell instead of a train-time score that a reload
    # cannot reproduce. See ``export_gaussian_artifacts``.
    exported = None
    # The backend's own provenance must survive the swap below: the reloaded
    # artifact only knows it is an npz, not which backend or settings made it.
    backend_metadata = reconstruction.metadata
    if config.save_renders and not config.eval_only:
        exported = export_gaussian_artifacts(reconstruction, artifacts, config)
        # Only re-score through the export when the backend says a flat gsplat
        # re-render reproduces its own renderer. i3dgs renders a hierarchical
        # LOD cut and the feed-forward backends bring their own rasterizers, so
        # for them the export is a viewer artifact, not the measured model --
        # and their envs need not even have gsplat installed.
        if getattr(reconstruction, "scores_from_export", False):
            reconstruction = NpzGaussianReconstruction(
                artifacts.gaussians_path, force_legacy_dc=config.force_legacy_dc
            )
        else:
            exported["scored_from_export"] = False

    renders_dir = artifacts.renders_dir
    if config.save_renders:
        renders_dir.mkdir(parents=True, exist_ok=True)

    metric_stores: Dict[str, Dict[str, Dict[str, list]]] = {
        "%dx%d" % resolution: {} for resolution in evaluation_resolutions
    }
    canonical_key = "%dx%d" % (eval_camera.width, eval_camera.height)
    view_records: list = []
    for index, frame in enumerate(eval_payload["frames"]):
        stratum = frame.get("stratum", "unstratified")
        gt_rgb = np.asarray(Image.open(eval_root / frame["file_path"]))[..., :3].astype(np.float32) / 255.0
        gt_depth = np.load(eval_root / frame["depth_path"]).astype(np.float32)
        c2w_cv = np.asarray(frame["transform_matrix"], dtype=np.float64) @ gl_cv
        rendered = reconstruction.render(c2w_cv, camera=eval_camera)
        pred = np.clip(rendered.rgb, 0.0, 1.0)
        pred_depth = rendered.depth
        if pred.shape != gt_rgb.shape:
            raise ValueError(
                "render shape %s does not match eval GT %s"
                % (pred.shape, gt_rgb.shape)
            )

        view_psnr = None
        for width, height in evaluation_resolutions:
            key = "%dx%d" % (width, height)
            pred_scaled = _resize_rgb(pred, width, height)
            gt_scaled = _resize_rgb(gt_rgb, width, height)
            entry = metric_stores[key].setdefault(
                stratum,
                {
                    "psnr": [], "ssim": [], "lpips": [], "depth_mae": [],
                    # Geometry measured on a FIXED pixel set. ``depth_mae``
                    # above only averages pixels the reconstruction rendered,
                    # which silently drops its worst failures -- see
                    # docs/results/2026-07-31-geometry-label-audit.md.
                    "render_coverage": [], "depth_mae_fixed": [],
                    "depth_p50_fixed": [], "depth_mae_zerofill": [],
                },
            )
            score = psnr(pred_scaled, gt_scaled)
            entry["psnr"].append(score)
            with torch.no_grad():
                entry["ssim"].append(_torch_ssim(pred_scaled, gt_scaled))
                if lpips_model is not None and min(width, height) >= _LPIPS_MIN_SIDE:
                    entry["lpips"].append(
                        float(
                            lpips_model(
                                torch.from_numpy(pred_scaled).permute(2, 0, 1)[None].cuda() * 2 - 1,
                                torch.from_numpy(gt_scaled).permute(2, 0, 1)[None].cuda() * 2 - 1,
                            )
                        )
                    )
            if key == canonical_key:
                view_psnr = score
        assert view_psnr is not None
        record = {
            "index": index,
            "stratum": stratum,
            "psnr": round(view_psnr, 3),
            "occlusion": frame.get("occlusion", {}),
        }
        view_records.append(record)
        if pred_depth is not None:
            valid = (gt_depth > 0.0) & (gt_depth < config.max_depth) & (pred_depth > 0.0)
            view_depth_mae = (
                float(np.abs(pred_depth[valid] - gt_depth[valid]).mean())
                if valid.any()
                else float("nan")
            )
            metric_stores[canonical_key][stratum]["depth_mae"].append(view_depth_mae)

            # Geometry on a FIXED pixel set: every pixel with valid GT depth,
            # whether or not the reconstruction rendered there. ``depth_mae``
            # above averages only rendered pixels, so a model that produces no
            # geometry is not penalised -- those pixels simply leave the average,
            # and the number of views scored then varies with quality (5 to 10
            # of 10 across one root). See
            # docs/results/2026-07-31-geometry-label-audit.md.
            gt_valid = (gt_depth > 0.0) & (gt_depth < config.max_depth)
            if gt_valid.any():
                rendered = pred_depth[gt_valid] > 0.0
                gt_sel = gt_depth[gt_valid]
                err = np.abs(pred_depth[gt_valid] - gt_sel)
                # un-rendered pixels charged the depth range...
                err_capped = np.where(rendered, err, config.max_depth)
                # ...and, parameter-free, charged as if predicted at depth 0,
                # which keeps the penalty on the same scale as a wrong depth
                # instead of dominated by max_depth.
                err_zero = np.where(rendered, err, gt_sel)
                store = metric_stores[canonical_key][stratum]
                store["render_coverage"].append(float(rendered.mean()))
                store["depth_mae_fixed"].append(float(err_capped.mean()))
                store["depth_p50_fixed"].append(float(np.median(err_capped)))
                store["depth_mae_zerofill"].append(float(err_zero.mean()))
                record["render_coverage"] = round(float(rendered.mean()), 4)
                record["depth_mae_fixed"] = round(float(err_capped.mean()), 4)
                record["depth_p50_fixed"] = round(float(np.median(err_capped)), 4)
                record["depth_mae_zerofill"] = round(float(err_zero.mean()), 4)
            # Record depth per view, not only in the aggregate. Whether a metric
            # is outlier-driven cannot be judged from a mean, and the aggregate
            # already hid one defect this way: an escaping view contributes the
            # psnr() 99.0 sentinel but is *silently dropped* from depth_mae
            # (no valid pixels -> NaN -> nanmean), so the two metrics disagree
            # about which views they even measure.
            record["depth_mae"] = (
                None if np.isnan(view_depth_mae) else round(view_depth_mae, 4)
            )
            record["depth_valid_fraction"] = round(float(valid.mean()), 4)
        if config.save_renders:
            side = np.concatenate([pred, gt_rgb], axis=1)
            Image.fromarray((side * 255).astype(np.uint8)).save(
                renders_dir / ("%s_%03d_pred_vs_gt.png" % (stratum, index))
            )

    appearance_by_resolution = {
        key: _aggregate_appearance(store) for key, store in metric_stores.items()
    }
    appearance = appearance_by_resolution[canonical_key]

    # Geometry is a function of (model, GT samples) alone -- no camera enters
    # it -- so re-scoring one model against a different eval set recomputes an
    # identical number. That recompute is two KD-trees over millions of points
    # and dominates an eval-only run (measured: 180 s against 6 s of rendering),
    # so a re-score may inherit it from the eval it replaces.
    splats = reconstruction.gaussian_splats()
    geometry = None
    if config.eval_only and config.reuse_geometry:
        prior_path = episode_dir / output_name if output_name else artifacts.eval_path
        if prior_path.exists():
            prior_geometry = json.loads(prior_path.read_text()).get("geometry")
            if prior_geometry:
                geometry = dict(prior_geometry)
                geometry["reused_from"] = str(prior_path)
    if geometry is None:
        keep = splats.opacities > config.opacity_threshold
        recon_points = splats.centers[keep]
        geometry = geometry_metrics(recon_points, gt_points, gt_normals)

    result = {
        "episode_dir": str(episode_dir),
        "train_transforms": train_transforms.name,
        "config": {
            "backend": config.backend,
            "run_name": run_name,
            "train_iterations": config.train_iterations,
            "use_masks": config.use_masks,
            "opacity_threshold": config.opacity_threshold,
            "seed": config.seed,
            "require_lpips": config.require_lpips,
            "lpips_available": lpips_model is not None,
            "lpips_error": lpips_error,
            "evaluation_resolutions": [list(value) for value in evaluation_resolutions],
            "backend_metadata": backend_metadata,
        },
        "num_train_views": len(train_payload["frames"]),
        "num_gaussians": int(len(splats.centers)),
        # What was actually scored: with save_renders the exported artifact is
        # reloaded and measured, so these numbers are reproducible from the npz.
        "scored_artifact": exported,
        "train_resolution": [w, h],
        "eval_resolution": [eval_camera.width, eval_camera.height],
        "resource_usage": resource_usage,
        "appearance_per_stratum": appearance,
        "appearance_by_resolution": appearance_by_resolution,
        "geometry": geometry,
    }
    if config.eval_only:
        # Preserve training-time provenance from the eval this one replaces;
        # only the scoring context (eval set, resolutions, metrics) is new.
        result_path_existing = episode_dir / output_name if output_name else artifacts.eval_path
        prior = (
            json.loads(result_path_existing.read_text())
            if result_path_existing.exists() else {}
        )
        for key in ("resource_usage", "num_gaussians", "train_resolution"):
            if key in prior:
                result[key] = prior[key]
        prior_metadata = (prior.get("config") or {}).get("backend_metadata")
        if prior_metadata:
            result["config"]["backend_metadata"] = prior_metadata
        result["eval_mode"] = {
            "reloaded_npz": True,
            "loaded_gaussians": reconstruction.num_loaded,
            "renderer": reconstruction.metadata["renderer"],
        }
        if result_path_existing.exists():
            backup = result_path_existing.with_suffix(".json.bak")
            if not backup.exists():
                backup.write_text(result_path_existing.read_text())
    if eval_set_dir is not None:
        label_keys = sorted({key for view in view_records for key in view["occlusion"]})
        class_psnr: Dict[str, Dict[str, Any]] = {}
        for key in label_keys:
            groups: Dict[str, list] = {}
            for view in view_records:
                label = view["occlusion"].get(key, {}).get("class")
                if label:
                    groups.setdefault(label, []).append(view["psnr"])
            class_psnr[key] = {
                label: {"psnr": round(float(np.mean(values)), 3), "n": len(values)}
                for label, values in sorted(groups.items())
            }
        parts = episode_dir.parent.name.rsplit("__", 2)
        (artifacts.root / "eval_shared.json").write_text(json.dumps({
            "episode_dir": str(episode_dir),
            "difficulty": parts[1] if len(parts) == 3 else "-",
            "shared_set": str(eval_set_dir),
            "renderer": (
                "%s full model" % backend.name if backend is not None
                else reconstruction.metadata["renderer"]
            ),
            "color_model": "full",
            "views": view_records,
            "class_psnr": class_psnr,
        }, indent=2))
        result["eval_set"] = str(eval_set_dir)

    result_path = episode_dir / output_name if output_name else artifacts.eval_path
    result_path.write_text(json.dumps(result, indent=2))
    return result
