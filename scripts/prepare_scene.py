"""Prepare static/dynamic episode YAMLs for an installed Habitat mesh or GS scene.

Run in habitat for mesh scenes and habitat-gs for Gaussian-splat stages.
The generated configuration records the actual start, camera and route geometry.
"""

import argparse
from pathlib import Path
import sys

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from activebench.common.habitat_env import HabitatEnvConfig
from activebench.common.camera import CameraPose
from activebench.configuration import safe_name, write_json
from activebench.sim import DynamicSceneConfig, DynamicSceneSim


def patrol_routes(sim, points, count, seed, clearance):
    """Use navmesh polylines; reject insufficient clearance and short routes."""
    rng = np.random.default_rng(seed)
    routes = []
    for _ in range(2000):
        a, b = points[rng.integers(len(points), size=2)]
        route = sim.navmesh_route(a, b)
        if route is None or not 1.0 <= route[1] <= 10.0:
            continue
        waypoints = route[0]
        dense = []
        for start, end in zip(waypoints[:-1], waypoints[1:]):
            steps = max(1, int(np.ceil(np.linalg.norm(end - start) / 0.2)))
            dense.extend(start + (end - start) * t for t in np.linspace(0, 1, steps + 1))
        pf = sim.env.sim.pathfinder
        if not dense or min(float(pf.distance_to_closest_obstacle(p.astype(np.float32), 3.0))
                            for p in dense) < clearance:
            continue
        routes.append(waypoints)
        if len(routes) == count:
            return routes
    raise RuntimeError("only %d/%d routes satisfy clearance; use fewer/smaller distractors or another scene"
                       % (len(routes), count))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scene-path", required=True, help="installed stage file or scene-dataset handle")
    ap.add_argument("--dataset-config", default="default")
    ap.add_argument("--name", required=True, help="unique output scene name")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--conditions", nargs="+", choices=["d0", "dyn"], default=["d0"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--protocol", default="activebench-navmesh-cube-v1")
    ap.add_argument("--seconds", type=float, default=300)
    ap.add_argument("--max-captures", type=int, default=640)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--hfov", type=float, default=75.1781789379499)
    ap.add_argument("--eye-height", type=float, default=1.4)
    ap.add_argument("--start-pose", type=float, nargs=5, metavar=("X", "Y", "Z", "YAW_RAD", "PITCH_RAD"))
    ap.add_argument("--collision", choices=["navmesh", "none"], default="navmesh")
    ap.add_argument("--object-template", type=Path, help="required for dyn; installed .object_config.json")
    ap.add_argument("--object-diag-m", type=float, default=0.5, help="conservative scaled object diagonal")
    ap.add_argument("--object-center-height", type=float, default=0.5, help="template origin above floor (m)")
    ap.add_argument("--object-scale", type=float, default=1.0)
    ap.add_argument("--distractors", type=int, default=3)
    args = ap.parse_args()
    safe_name(args.name)
    if min(args.seconds, args.max_captures, args.width, args.height, args.eye_height) <= 0:
        ap.error("time, capture budget, image dimensions and eye height must be positive")
    if not 0 < args.hfov < 180 or any(s < 0 for s in args.seeds):
        ap.error("hfov must be in (0, 180); seeds must be nonnegative")
    if "dyn" in args.conditions:
        if not args.object_template or not args.object_template.is_file():
            ap.error("dyn requires an installed --object-template")
        if min(args.distractors, args.object_diag_m, args.object_scale) <= 0:
            ap.error("distractor count, diagonal and scale must be positive")
    targets = [args.out_dir / ("%s__%s__s%d.yaml" % (args.name, d, s))
               for d in set(args.conditions) for s in set(args.seeds)]
    if any(p.exists() for p in targets):
        ap.error("a target config already exists; use a new name/output directory")
    stage = Path(args.scene_path).expanduser()
    scene_path = str(stage.resolve()) if stage.is_file() else args.scene_path
    dataset_config = (str(Path(args.dataset_config).expanduser().resolve())
                      if args.dataset_config != "default" else "default")
    if stage.suffix and not stage.is_file():
        ap.error("scene file does not exist: %s" % stage)
    if dataset_config != "default" and not Path(dataset_config).is_file():
        ap.error("dataset config does not exist")
    hab = HabitatEnvConfig(scene_path=scene_path, scene_dataset_config_file=dataset_config,
                           width=args.width, height=args.height, hfov=args.hfov, enable_physics=True)
    sim = DynamicSceneSim(DynamicSceneConfig(habitat=hab, distractors=[]))
    generated, diagnostics = [], []
    try:
        points = sim.sample_navigable_points(500, seed=0)
        if points is None or not len(points):
            raise RuntimeError("scene preparation needs a loadable navmesh; install/build it first")
        # Prefer a central point; then render all candidate starts to avoid an
        # eye placed inside geometry. The fixed start is shared across seeds.
        ranked = points[np.argsort(np.linalg.norm(points - np.median(points, axis=0), axis=1))]
        if args.start_pose:
            candidates = [args.start_pose]
        else:
            candidates = [[*list(p + [0, args.eye_height, 0]), yaw, 0.0]
                          for p in ranked[:32] for yaw in (0.0, np.pi / 2, np.pi, -np.pi / 2)]
        best = None
        for values in candidates:
            pose = CameraPose.from_xyz_yaw_pitch(values[:3], yaw=values[3], pitch=values[4])
            depth = sim.render_clean(pose)["depth"]
            quality = float((depth > 0.2).mean())
            if best is None or quality > best[0]:
                best = quality, [float(v) for v in values]
            if quality >= 0.95:
                break
        if best[0] < 0.6:
            raise RuntimeError("no start view with sufficient valid depth; provide --start-pose")
        start = best[1]
        sim.set_navigation_anchor(start[:3])
        points = sim.sample_navigable_points(500, seed=1, anchor_xyz=start[:3])
        for seed in sorted(set(args.seeds)):
            moving = []
            if "dyn" in args.conditions:
                routes = patrol_routes(sim, points, args.distractors, seed, args.object_diag_m / 2 + 0.1)
                for index, route in enumerate(routes):
                    moving.append({"name": "distractor_%d" % index,
                                   "object_template": str(args.object_template.resolve()),
                                   "object_diag_m": args.object_diag_m, "scale": args.object_scale,
                                   "trajectory": {"type": "waypoint_patrol", "speed": 0.35,
                                                  "mode": "pingpong", "waypoints":
                                                  (route + [0, args.object_center_height, 0]).tolist()}})
            for condition in sorted(set(args.conditions)):
                payload = {"habitat": {"scene_path": scene_path, "scene_dataset_config_file": dataset_config,
                                      "width": args.width, "height": args.height, "hfov": args.hfov,
                                      "enable_physics": True},
                           "start_pose": start, "seed": seed, "trajectory_seed": seed,
                           "motion": {"speed": 0.5, "yaw_rate_deg": 60.0, "pitch_rate_deg": 60.0},
                           "episode": {"max_captures": args.max_captures, "max_sim_time": args.seconds,
                                       "capture_cost": 0.0, "reconstruction_interval": 1.0,
                                       "stream_observations": True, "provide_depth": True,
                                       "collision": args.collision},
                           "distractors": moving if condition == "dyn" else [],
                           "campaign": {"protocol": args.protocol, "scene": args.name, "difficulty": condition}}
                generated.append((args.out_dir / ("%s__%s__s%d.yaml" % (args.name, condition, seed)), payload))
        diagnostics.append({"start_pose": start, "start_valid_depth_fraction": best[0],
                            "navmesh_floor_samples": len(points), "collision": args.collision})
    finally:
        sim.close()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for path, payload in generated:
        path.write_text(yaml.safe_dump(payload, sort_keys=False))
        print(path)
    write_json(args.out_dir / (args.name + ".preparation.json"), diagnostics)


if __name__ == "__main__":
    main()
