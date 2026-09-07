"""Evaluate orientation-binned observation coverage for recorded episodes.

Builds (and caches) GT surface samples for the scene with Habitat, then scores
each episode directory offline from its stored poses + depth maps.

Example (habitat conda env):
    python scripts/eval_coverage.py \
        --scene-config configs/bench/apartment_patrol.yaml \
        --samples eval_assets/apartment_1_surface.npz \
        runs_bench/random_apartment_smoke runs_bench/r3con_smoke runs_bench/gavis_smoke
"""

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from activebench.eval.coverage import CoverageParams, evaluate_episode_dir
from activebench.eval.surface_samples import (
    SurfaceSampleConfig,
    SurfaceSamples,
    build_surface_samples,
)


def get_samples(args) -> SurfaceSamples:
    samples_path = Path(args.samples)
    if samples_path.exists() and not args.rebuild_samples:
        return SurfaceSamples.load(samples_path)
    if args.scene_config is None:
        raise SystemExit("--scene-config is required to build missing surface samples")
    from activebench.episode import EpisodeSpec
    from activebench.sim import DynamicSceneSim

    spec = EpisodeSpec.from_yaml(Path(args.scene_config))
    sim = DynamicSceneSim(spec.scene)
    try:
        samples = build_surface_samples(sim, SurfaceSampleConfig())
    finally:
        sim.close()
    samples_path.parent.mkdir(parents=True, exist_ok=True)
    samples.save(samples_path)
    print("built %d surface samples -> %s" % (len(samples.points), samples_path))
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episodes", nargs="+", help="episode directories")
    parser.add_argument("--samples", required=True, help="surface samples npz (cache)")
    parser.add_argument("--scene-config", default=None, help="episode YAML for building samples")
    parser.add_argument("--rebuild-samples", action="store_true")
    parser.add_argument("--max-range", type=float, default=6.0)
    args = parser.parse_args()

    samples = get_samples(args)
    params = CoverageParams(max_range=args.max_range)
    header = "%-28s %8s %8s %8s %8s" % ("episode", "up", "side", "down", "overall")
    print(header)
    print("-" * len(header))
    for episode in args.episodes:
        result = evaluate_episode_dir(Path(episode), samples.points, samples.normals, params)
        bins = result["bins"]
        print(
            "%-28s %7.1f%% %7.1f%% %7.1f%% %7.1f%%"
            % (
                Path(episode).name,
                100 * bins["up"]["observed_frac"],
                100 * bins["side"]["observed_frac"],
                100 * bins["down"]["observed_frac"],
                100 * result["overall"]["observed_frac"],
            )
        )
    print("(observed fraction per surface-orientation bin; details in each episode's coverage.json)")


if __name__ == "__main__":
    main()
