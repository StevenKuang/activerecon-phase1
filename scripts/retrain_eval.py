"""Standard Phase 1 gsplat training and full-metric evaluation.

Run in the method-independent bencheval environment.
"""

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from activebench.eval.retrain import RetrainConfig, run_retrain_eval


def parse_resolution(value: str):
    try:
        width, height = (int(part) for part in value.lower().split("x", 1))
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError("resolution must be WIDTHxHEIGHT")
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("resolution dimensions must be positive")
    return width, height


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episodes", nargs="+")
    parser.add_argument("--samples", required=True, help="GT surface samples npz")
    parser.add_argument(
        "--backend",
        choices=["gsplat"],
        default="gsplat",
    )
    parser.add_argument(
        "--run-name", default=None,
        help="artifact name under <episode>/reconstructions (default: backend name)",
    )
    parser.add_argument("--iterations", type=int, default=30_000)
    parser.add_argument(
        "--train-seed", type=int, default=0,
        help="gsplat initialization/view-order seed (independent of episode seed)",
    )
    parser.add_argument(
        "--backend-options", default="{}",
        help="backend-specific JSON overrides",
    )
    parser.add_argument("--use-masks", action="store_true",
                        help="drop distractor pixels during retrain (upper bound)")
    parser.add_argument("--save-renders", action="store_true",
                        help="write pred|GT side-by-sides and generic Gaussian artifacts")
    parser.add_argument(
        "--shared-dir", default=str(_REPO_ROOT / "eval_assets/shared_eval_v3"),
        help="shared eval sets root; per-episode group resolved automatically",
    )
    parser.add_argument(
        "--eval-set", default=None,
        help="explicit shared eval set dir (overrides --shared-dir resolution)",
    )
    parser.add_argument(
        "--eval-resolution", action="append", type=parse_resolution, default=[],
        metavar="WIDTHxHEIGHT",
        help="additional downsampled appearance scale; repeat for multiple sizes",
    )
    parser.add_argument(
        "--require-lpips", action="store_true",
        help="fail the evaluation if LPIPS or its pretrained weights are unavailable",
    )
    parser.add_argument(
        "--force-legacy-dc", action="store_true",
        help="ignore a raw sh0 band when reloading, so a matrix that mixes "
             "pre- and post-sh0 artifacts is scored in one regime",
    )
    parser.add_argument(
        "--reuse-geometry", action="store_true",
        help="with --eval-only, inherit completeness/accuracy from the eval "
             "being replaced instead of recomputing an identical value; valid "
             "only when the model and GT surface samples are unchanged",
    )
    parser.add_argument(
        "--eval-only", action="store_true",
        help="re-score the existing gaussians.npz (no retraining); training "
        "provenance in eval.json is preserved and the old file backed up",
    )
    args = parser.parse_args()

    config = RetrainConfig(
        backend=args.backend,
        run_name=args.run_name,
        backend_options=json.loads(args.backend_options),
        train_iterations=args.iterations,
        seed=args.train_seed,
        evaluation_resolutions=tuple(args.eval_resolution),
        require_lpips=args.require_lpips,
        use_masks=args.use_masks,
        save_renders=args.save_renders,
        eval_only=args.eval_only,
        force_legacy_dc=args.force_legacy_dc,
        reuse_geometry=args.reuse_geometry,
    )
    header = "%-24s %7s %7s %8s | %8s %8s %8s | %7s" % (
        "episode", "psnr", "ssim", "dep.mae", "cmp:up", "cmp:side", "cmp:down", "acc.med")
    print(header)
    print("-" * len(header))
    from activebench.replay import shared_eval_set_dir

    failures = []
    for episode in args.episodes:
        eval_set = (
            Path(args.eval_set) if args.eval_set
            else shared_eval_set_dir(Path(episode), Path(args.shared_dir))
        )
        if eval_set is None:
            print("[retrain] %s: no shared eval set — legacy episode-local views"
                  % episode)
        try:
            result = run_retrain_eval(
                Path(episode), Path(args.samples), config, eval_set_dir=eval_set
            )
        except Exception as exc:
            # One broken episode (e.g. an aborted run without transforms.json)
            # must not take down the rest of a multi-episode batch.
            failures.append(str(episode))
            print("[retrain] FAILED %-17s %s: %s" % (Path(episode).name, type(exc).__name__, exc))
            continue
        strata = result["appearance_per_stratum"]
        keys = [k for k in strata if k != "all"]
        mean = lambda metric: sum(strata[k][metric] for k in keys if strata[k][metric] is not None) / max(
            1, sum(1 for k in keys if strata[k][metric] is not None))
        bins = result["geometry"]["bins"]
        print("%-24s %7.2f %7.3f %8.3f | %7.1f%% %7.1f%% %7.1f%% | %7.3f" % (
            Path(episode).name,
            strata["all"]["psnr"], mean("ssim"), mean("depth_mae"),
            100 * bins["up"]["completeness@0.05"],
            100 * bins["side"]["completeness@0.05"],
            100 * bins["down"]["completeness@0.05"],
            result["geometry"]["accuracy"]["median"],
        ))
        for stratum in sorted(keys):
            s = strata[stratum]
            extras = "" if s.get("lpips") is None else ", lpips %.3f" % s["lpips"]
            print("    %-12s psnr %6.2f, ssim %.3f, depth mae %.3f%s" % (
                stratum, s["psnr"], s["ssim"], s["depth_mae"], extras))
    print("(completeness@5cm per orientation bin; details in each named reconstruction eval.json)")
    if failures:
        sys.exit("%d episode(s) failed: %s" % (len(failures), ", ".join(failures)))


if __name__ == "__main__":
    main()
