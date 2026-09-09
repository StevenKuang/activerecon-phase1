"""Build clean surface samples and shared cube cameras before running any method.

Run in the matching simulator environment. Use a new output directory whenever
the scene, camera sampling or surface recipe changes. Defaults are a full
reference recipe; small settings are intended for pipeline smoke tests only.
"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from activebench.configuration import cell_identity, digest, file_digest, input_file_hashes, load_yaml, reference_scene, write_json
from activebench.episode import EpisodeSpec
from activebench.eval.surface_samples import SurfaceSampleConfig, build_surface_samples
from activebench.sim import DynamicSceneSim
from build_uniform_eval_set import process_group


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--configs-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--scenes", nargs="+")
    ap.add_argument("--points", type=int, default=24)
    ap.add_argument("--resolution", type=int, default=1200, help="square cube-face width/height")
    ap.add_argument("--surface-spacing", type=float, default=2.0)
    ap.add_argument("--surface-points-per-frame", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=20260727, help="shared evaluation seed; independent of policy")
    ap.add_argument("--eye-height", type=float, default=1.4)
    ap.add_argument("--eye-placement", choices=["fixed", "mid-height"], default="mid-height")
    ap.add_argument("--min-valid-depth-frac", type=float, default=0.5)
    args = ap.parse_args()
    if min(args.points, args.resolution, args.surface_spacing, args.surface_points_per_frame) <= 0:
        ap.error("points, resolution and surface sampling settings must be positive")
    if not 0 < args.min_valid_depth_frac <= 1:
        ap.error("min-valid-depth-frac must be in (0, 1]")
    groups = {}
    for path in sorted(args.configs_dir.glob("*.yaml")):
        payload = load_yaml(path)
        scene, condition, seed = cell_identity(payload)
        if condition == "d0" and (not args.scenes or scene in args.scenes):
            groups[scene + "__s%d" % seed] = (path, payload)
    if not groups:
        ap.error("no matching d0 configs; prepare the clean condition even for a dynamic-only run")
    if args.out_dir.exists():
        ap.error("choose a new --out-dir; reference assets are immutable after preparation")
    args.out_dir = args.out_dir.resolve()
    surface_dir = args.out_dir / "surface"
    surface_dir.mkdir(parents=True)
    surface_cfg = SurfaceSampleConfig(grid_spacing=args.surface_spacing,
                                     points_per_frame=args.surface_points_per_frame, seed=args.seed)
    recipe = {"surface": asdict(surface_cfg), "cube": {"points": args.points, "resolution": args.resolution,
              "seed": args.seed, "eye_height": args.eye_height, "eye_placement": args.eye_placement,
              "min_valid_depth_frac": args.min_valid_depth_frac, "min_clearance": 0.75,
              "clearance_floor": 0.4, "clearance_band": 0.5, "candidate_factor": 8, "hfov": 90.0}}
    record = {"schema_version": 1, "recipe": recipe, "groups": {}, "files": {}}
    surfaces = {}
    for group, (config, payload) in groups.items():
        scene = payload["campaign"]["scene"]
        identity = digest(reference_scene(payload))
        if scene in surfaces and surfaces[scene] != identity:
            raise ValueError("one scene name refers to different scene/camera/start configs")
        if scene not in surfaces:
            clean = dict(payload, distractors=[])
            sim = DynamicSceneSim(EpisodeSpec.from_dict(clean).scene)
            try:
                surface = build_surface_samples(sim, surface_cfg)
            finally:
                sim.close()
            if not len(surface.points):
                raise RuntimeError("reference surface is empty: %s" % scene)
            surface.save(surface_dir / (scene + ".npz"))
            surfaces[scene] = identity
            print("[references] %s: %d surface samples" % (scene, len(surface.points)), flush=True)
        # Write expanded inputs so downstream renderers never depend on shell
        # interpolation; keep the resolved config next to the references.
        import yaml
        resolved = args.out_dir / (group + ".yaml")
        resolved.write_text(yaml.safe_dump(payload, sort_keys=False))
        cube = recipe["cube"]
        options = SimpleNamespace(out_dir=args.out_dir / "eval", surface_dir=surface_dir,
                    overwrite=False, width=args.resolution, height=args.resolution,
                    **{k: v for k, v in cube.items() if k != "resolution"})
        process_group(group, resolved, options)
        record["groups"][group] = {"scene_identity": identity, "source_config": str(config.resolve()),
                                    "scene_files": input_file_hashes(dict(payload, distractors=[]))}
    for path in sorted(args.out_dir.rglob("*")):
        if path.is_file():
            record["files"][str(path.relative_to(args.out_dir))] = file_digest(path)
    write_json(args.out_dir / "assets.json", record)
    print("Reference assets ready:", args.out_dir)


if __name__ == "__main__":
    main()
