"""Verify SH3 PLY export: round-trip coefficients through gsplat.

Renders 6 viewpoints (2 level / 2 lookup / 2 lookdown from the shared eval
set, or the first training frames as fallback) with gsplat using:

- (ref)  the NPZ's full SH3 coefficients (the trained-model reference);
- (ply)  the exported SH3 PLY's coefficients (validates the export path);
- (sh0)  DC-only colors from the NPZ (shows the view-dependence signal).

A lossless PLY export should match ``ref`` within float precision (PSNR > 50
dB); ``sh0`` diverges from ``ref`` at oblique viewpoints where SH3 carries
view-dependent color. Reports PSNR / SSIM / LPIPS for each comparison and
writes the rendered images alongside the metrics JSON.

The browser-side Spark rendering of the PLY/RAD is a separate manual step;
this script validates everything up to that boundary (the PLY is a valid,
lossless encoding of the trained SH3 model that gsplat can consume).

Run in the ``bencheval`` env (same as gsplat training):
    python scripts/verify_sh3_export.py \
        --episode runs_campaign_v4/skokloster_castle__d0__s0/r3con-pano \
        --reconstruction gsplat \
        --shared-root eval_assets/shared_eval_v4
"""

import argparse
import json
import struct
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from activebench.web_export import ply_from_gaussians, read_sh_degree  # noqa: E402

SH_C0 = 0.28209479177387814
_GL_CV_FLIP = np.diag([1.0, -1.0, -1.0, 1.0])


def _parse_3dgs_ply(path: Path) -> Dict[str, np.ndarray]:
    """Minimal standard 3DGS binary PLY reader (verification only)."""

    payload = path.read_bytes()
    end = payload.index(b"end_header\n") + len(b"end_header\n")
    header = payload[:end].decode("ascii").splitlines()
    count = int(next(l.split()[-1] for l in header if l.startswith("element vertex")))
    names = [l.split()[-1] for l in header if l.startswith("property")]
    data = np.frombuffer(payload[end:], dtype=np.float32).reshape(count, len(names))
    return {name: data[:, i] for i, name in enumerate(names)}


def _sh_from_npz(npz_path: Path) -> Tuple[np.ndarray, np.ndarray, int]:
    """Load (means, full SH tensor (N, K, 3), sh_degree) from a gaussians NPZ.

    The raw DC band is loaded from ``sh0`` when present, otherwise reconstructed
    from the legacy activated ``colors`` field; higher bands come from
    ``sh_rest``. Returns the available SH coefficients in
    the gsplat band-then-channel layout.
    """

    with np.load(npz_path) as data:
        means = np.asarray(data["centers"], dtype=np.float32)
        colors = np.asarray(data["colors"], dtype=np.float32)
        sh_degree = int(data["sh_degree"]) if "sh_degree" in data.files else 0
        sh_rest = (np.asarray(data["sh_rest"], dtype=np.float32)
                   if "sh_rest" in data.files else None)
        raw_dc = np.asarray(data["sh0"], dtype=np.float32) if "sh0" in data.files else None
    sh0 = (raw_dc.reshape(-1, 1, 3) if raw_dc is not None else
           ((colors.astype(np.float64) - 0.5) / SH_C0).astype(np.float32)[:, None, :])
    if sh_rest is not None and sh_degree > 0:
        full = np.concatenate([sh0, sh_rest], axis=1)
    else:
        full = sh0
        sh_degree = 0
    return means, full, sh_degree


def _sh_from_ply(ply_path: Path) -> Tuple[np.ndarray, np.ndarray, int]:
    """Load (means, full SH tensor (N, K, 3), sh_degree) from an exported PLY."""

    cols = _parse_3dgs_ply(ply_path)
    means = np.stack([cols["x"], cols["y"], cols["z"]], axis=1).astype(np.float32)
    dc = np.stack([cols["f_dc_%d" % i] for i in range(3)], axis=1).astype(np.float32)
    sh0 = dc[:, None, :]
    rest_names = [n for n in cols if n.startswith("f_rest_")]
    if rest_names:
        coeffs = len(rest_names) // 3
        ordered = np.stack(
            [cols["f_rest_%d" % k] for k in range(coeffs * 3)], axis=1
        ).reshape(-1, 3, coeffs).transpose(0, 2, 1).astype(np.float32)
        full = np.concatenate([sh0, ordered], axis=1)
        sh_degree = int(np.sqrt(coeffs + 1)) - 1
    else:
        full = sh0
        sh_degree = 0
    return means, full, sh_degree


def _select_views(transforms: Dict[str, Any], n_per_stratum: int = 2) -> List[int]:
    """Pick 2 frames each from level / lookup / lookdown strata when present."""

    strata = ("level", "lookup", "lookdown")
    frames = transforms["frames"]
    picks: List[int] = []
    for stratum in strata:
        indices = [i for i, f in enumerate(frames) if f.get("stratum") == stratum]
        picks.extend(indices[:n_per_stratum])
    if not picks:
        picks = list(range(min(6, len(frames))))
    return picks


def _render(
    means: np.ndarray,
    sh: np.ndarray,
    quats: np.ndarray,
    scales: np.ndarray,
    opacities: np.ndarray,
    sh_degree: int,
    c2w_cv: np.ndarray,
    intrinsics,
    width: int,
    height: int,
):
    import torch
    from gsplat import rasterization

    viewmat = torch.from_numpy(np.linalg.inv(c2w_cv)).float().cuda()[None]
    with torch.no_grad():
        rendered, _alphas, _meta = rasterization(
            means=torch.from_numpy(means).float().cuda(),
            quats=torch.from_numpy(quats).float().cuda(),
            scales=torch.from_numpy(scales).float().cuda(),
            opacities=torch.from_numpy(opacities).float().cuda(),
            colors=torch.from_numpy(sh).float().cuda(),
            sh_degree=sh_degree,
            viewmats=viewmat,
            Ks=intrinsics,
            width=width,
            height=height,
            render_mode="RGB",
        )
    return rendered[0].clamp(0, 1).cpu().numpy()


def _metrics(pred: np.ndarray, gt: np.ndarray, lpips_model) -> Dict[str, float]:
    import torch

    from activebench.eval.retrain import _torch_ssim, psnr

    result = {"psnr": round(psnr(pred, gt), 3), "ssim": round(_torch_ssim(pred, gt), 4)}
    if lpips_model is not None:
        with torch.no_grad():
            score = float(lpips_model(
                torch.from_numpy(pred).permute(2, 0, 1)[None].cuda() * 2 - 1,
                torch.from_numpy(gt).permute(2, 0, 1)[None].cuda() * 2 - 1,
            ))
        result["lpips"] = round(score, 4)
    return result


def _load_lpips():
    try:
        import lpips
        return lpips.LPIPS(net="alex").eval().cuda()
    except Exception as exc:
        print("[verify] LPIPS unavailable: %s" % exc, file=sys.stderr)
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", required=True, type=Path)
    parser.add_argument("--reconstruction", default="gsplat")
    parser.add_argument(
        "--shared-root", type=Path, default=_REPO_ROOT / "eval_assets" / "shared_eval_v4",
    )
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="default: <episode>/sh3_verify")
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--height", type=int, default=600)
    args = parser.parse_args()

    from activebench.eval.reconstruction import reconstruction_artifacts
    from activebench.replay import shared_eval_set_dir

    episode_dir = args.episode.resolve()
    artifacts = reconstruction_artifacts(episode_dir, args.reconstruction)
    npz_path = artifacts.gaussians_path
    if not npz_path.exists():
        sys.exit("gaussians.npz not found for %s in %s" % (args.reconstruction, episode_dir))

    sh_degree = read_sh_degree(npz_path)
    if sh_degree == 0:
        sys.exit("NPZ %s carries no SH3 coefficients; retrain with the gsplat SH3 backend first"
                 % npz_path)
    print("[verify] NPZ sh_degree=%d, gaussians=%d" % (sh_degree, len(np.load(npz_path)["centers"])))

    # Pick viewpoints from the shared eval set (clean held-out poses).
    shared_dir = shared_eval_set_dir(episode_dir, args.shared_root)
    if shared_dir is None:
        sys.exit("no shared eval set for %s under %s" % (episode_dir, args.shared_root))
    transforms = json.loads((shared_dir / "transforms_eval_shared.json").read_text())
    view_indices = _select_views(transforms)
    print("[verify] %d viewpoints: %s" % (len(view_indices), view_indices))

    # Source arrays.
    means_npz, sh_npz, deg_npz = _sh_from_npz(npz_path)
    with np.load(npz_path) as data:
        quats = np.asarray(data["quats_wxyz"], dtype=np.float32)
        scales = np.asarray(data["scales"], dtype=np.float32)
        opacities = np.asarray(data["opacities"], dtype=np.float32)
    # Re-export the PLY through the production path, then read it back.
    out_dir = args.out_dir or (episode_dir / "sh3_verify")
    out_dir.mkdir(parents=True, exist_ok=True)
    ply_path = out_dir / ("%s.sh%d.ply" % (args.reconstruction, sh_degree))
    from activebench.web_export import splats_npz_to_ply
    count, used_sh = splats_npz_to_ply(npz_path, ply_path)
    print("[verify] exported PLY %s: %d gaussians, sh%d, %.1f MB"
          % (ply_path.name, count, used_sh, ply_path.stat().st_size / 1e6))
    means_ply, sh_ply, deg_ply = _sh_from_ply(ply_path)
    assert deg_ply == sh_degree, "PLY degree %d != NPZ degree %d" % (deg_ply, sh_degree)

    # Camera intrinsics from the shared eval set, rescaled to the target.
    sw, sh_h = int(transforms["w"]), int(transforms["h"])
    sx, sy = args.width / sw, args.height / sh_h
    import torch
    intrinsics = torch.tensor([
        [transforms["fl_x"] * sx, 0.0, transforms["cx"] * sx],
        [0.0, transforms["fl_y"] * sy, transforms["cy"] * sy],
        [0.0, 0.0, 1.0],
    ], dtype=torch.float32, device="cuda")[None]

    lpips_model = _load_lpips()
    records = []
    for pick in view_indices:
        frame = transforms["frames"][pick]
        c2w_cv = np.asarray(frame["transform_matrix"], dtype=np.float64) @ _GL_CV_FLIP
        t0 = time.perf_counter()
        ref = _render(means_npz, sh_npz, quats, scales, opacities, deg_npz,
                      c2w_cv, intrinsics, args.width, args.height)
        ply = _render(means_ply, sh_ply, quats, scales, opacities, deg_ply,
                      c2w_cv, intrinsics, args.width, args.height)
        sh0 = _render(means_npz, sh_npz[:, :1, :], quats, scales, opacities, 0,
                      c2w_cv, intrinsics, args.width, args.height)
        elapsed = time.perf_counter() - t0
        record = {
            "index": pick,
            "stratum": frame.get("stratum"),
            "ply_vs_ref": _metrics(ply, ref, lpips_model),
            "sh0_vs_ref": _metrics(sh0, ref, lpips_model),
            "render_seconds": round(elapsed, 3),
        }
        records.append(record)
        print("[verify] view %d (%s): ply_vs_ref PSNR %s dB, sh0_vs_ref PSNR %s dB (%.2fs)"
              % (pick, record["stratum"],
                 record["ply_vs_ref"]["psnr"], record["sh0_vs_ref"]["psnr"], elapsed))
        try:
            from PIL import Image
            for label, img in (("ref", ref), ("ply", ply), ("sh0", sh0)):
                Image.fromarray((img * 255).astype(np.uint8)).save(
                    out_dir / ("view_%02d_%s.png" % (pick, label)))
        except ImportError:
            pass

    summary = {
        "episode": str(episode_dir),
        "reconstruction": args.reconstruction,
        "npz": str(npz_path),
        "ply": str(ply_path),
        "sh_degree": sh_degree,
        "width": args.width,
        "height": args.height,
        "views": records,
    }
    out_path = out_dir / "verify_sh3.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print("[verify] wrote %s" % out_path)


if __name__ == "__main__":
    main()
