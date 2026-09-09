"""Build shared held-out eval sets for a campaign config directory.

One eval set per (scene, seed) group, reused by every method AND difficulty
(clean GT is difficulty-invariant, so d0 -> dyn deltas are paired on
pixel-identical ground truth). Per group, in the habitat env:

1. candidate poses spread over the navmesh (farthest-point, strata cycled),
   validity-checked against the clean render;
2. per dynamic difficulty: occlusion count maps over the 1 Hz training times
   (distractor silhouettes in front of the clean surface — contamination can
   only enter reconstruction streams at those times, so this is exact) plus a
   geometric clean certification (bounding-sphere sweep at 10 Hz against the
   pose's visible 3D region);
3. two-pole selection (default 20 severe + 20 clean = 40 views, in line
   with or above per-scene NVS test-set sizes), written as clean GT
   renders, per-difficulty occlusion count maps, and
   transforms_eval_shared.json with per-frame class labels.

Run in the habitat env (one group per invocation isolates GL context reuse):
    for g in $(python scripts/build_shared_eval_set.py --configs-dir \
            configs/bench/campaign_v2 --list-groups); do
        python scripts/build_shared_eval_set.py --configs-dir \
            configs/bench/campaign_v2 --group "$g"
    done
"""

import argparse
import sys
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import numpy as np
import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from activebench.episode import EpisodeSpec
from activebench.eval.geometry import unproject_depth_cv
from activebench.eval.shared_eval import (
    DEFAULT_CLEAN_MARGIN_M,
    DEFAULT_SEVERE_THRESHOLD,
    aim_at,
    candidate_orientations,
    certify_clean,
    farthest_point_indices,
    route_biased_positions,
    select_two_pole,
    sweep_centers,
)
from activebench.sim import MASK_DEPTH_EPS, DynamicSceneSim
from activebench.common.transforms_export import export_transforms_json, frame_from_pose
from activebench.common.camera import CameraPose
from activebench.common.image import save_rgb_png
from activebench.common.io import ensure_dir, save_json, save_npz_compressed

_GL_CV_FLIP = np.diag([1.0, -1.0, -1.0, 1.0])
MAX_REGION_DEPTH_M = 8.0
VISIBLE_POINT_STRIDE = 4
EYE_HEIGHT_RANGE_M = (1.2, 1.7)
MIN_CLEAR_FRAC = 0.5


@dataclass
class Candidate:
    pose: CameraPose
    stratum: str
    rgb: np.ndarray
    depth: np.ndarray
    visible_points: np.ndarray = field(default=None, repr=False)


def discover_groups(configs_dir: Path) -> Dict[str, Dict[str, Path]]:
    """(scene__seed) -> {difficulty: config path}, from <scene>__<diff>__<seed>.yaml."""

    groups: Dict[str, Dict[str, Path]] = {}
    for path in sorted(configs_dir.glob("*.yaml")):
        try:
            scene, difficulty, seed = path.stem.rsplit("__", 2)
        except ValueError:
            continue
        groups.setdefault("%s__%s" % (scene, seed), {})[difficulty] = path
    return groups


def visible_region_points(candidate: Candidate, intrinsics) -> np.ndarray:
    """Subsampled world-space points seen from a candidate (clean cert region)."""

    from activebench.common.camera import CameraIntrinsics

    frame = frame_from_pose(candidate.pose, "unused")
    c2w_cv = np.asarray(frame["transform_matrix"], dtype=np.float64) @ _GL_CV_FLIP
    depth = candidate.depth[::VISIBLE_POINT_STRIDE, ::VISIBLE_POINT_STRIDE]
    strided = CameraIntrinsics(
        width=depth.shape[1], height=depth.shape[0],
        fx=intrinsics.fx / VISIBLE_POINT_STRIDE, fy=intrinsics.fy / VISIBLE_POINT_STRIDE,
        cx=intrinsics.cx / VISIBLE_POINT_STRIDE, cy=intrinsics.cy / VISIBLE_POINT_STRIDE,
    )
    cam = unproject_depth_cv(depth.astype(np.float64), strided)
    valid = (depth > 0.0) & (depth < MAX_REGION_DEPTH_M)
    return cam[valid] @ c2w_cv[:3, :3].T + c2w_cv[:3, 3]


def build_candidates(
    sim: DynamicSceneSim, rng: np.random.Generator, args, route_points=None
) -> List[Candidate]:
    pathfinder = sim.env.sim.pathfinder
    if not pathfinder.is_loaded:
        return []
    pathfinder.seed(int(rng.integers(2**31 - 1)))
    nav = np.asarray(
        [pathfinder.get_random_navigable_point() for _ in range(args.nav_samples)],
        dtype=np.float64,
    )
    nav = nav[np.isfinite(nav).all(axis=1)]
    if len(nav) == 0:
        return []

    # A third of the budget stands near patrol routes, aimed at them, so the
    # severe pool survives in large scenes; the rest spreads scene-wide.
    route_points = (
        np.zeros((0, 3)) if route_points is None else np.asarray(route_points)
    )
    route_quota = args.candidates // 3 if len(route_points) else 0
    route_indices = route_biased_positions(nav, route_points, route_quota)
    spread = farthest_point_indices(nav, args.candidates - len(route_indices))
    orientations = candidate_orientations(len(spread), rng)

    plans = []
    for nav_index, (yaw, pitch, stratum) in zip(spread, orientations):
        plans.append((nav_index, yaw, pitch, stratum))
    for nav_index in route_indices:
        nearest = route_points[
            np.linalg.norm(
                route_points[:, [0, 2]] - nav[nav_index][[0, 2]], axis=1
            ).argmin()
        ]
        eye = nav[nav_index].copy()
        eye[1] += np.mean(EYE_HEIGHT_RANGE_M)
        yaw, pitch = aim_at(eye, nearest)
        yaw += float(rng.uniform(-0.35, 0.35))
        pitch = float(np.clip(pitch + rng.uniform(-0.15, 0.15), -0.6, 0.6))
        plans.append((nav_index, yaw, pitch, "route"))

    candidates: List[Candidate] = []
    for nav_index, yaw, pitch, stratum in plans:
        position = nav[nav_index].copy()
        position[1] += rng.uniform(*EYE_HEIGHT_RANGE_M)
        pose = CameraPose.from_xyz_yaw_pitch(position, yaw=yaw, pitch=pitch)
        render = sim.render_clean(pose)
        if float((render["depth"] > 0.25).mean()) <= MIN_CLEAR_FRAC:
            continue
        candidate = Candidate(
            pose=pose, stratum=stratum,
            rgb=render["rgb"][..., :3].copy(), depth=render["depth"].copy(),
        )
        candidate.visible_points = visible_region_points(candidate, sim.intrinsics)
        candidates.append(candidate)
    return candidates


def occlusion_counts(
    sim: DynamicSceneSim, candidates: List[Candidate], train_times: np.ndarray
) -> List[np.ndarray]:
    """Per candidate: count of training times a distractor sat in front."""

    # uint16: 300 s missions have 301 training times, overflowing uint8.
    counts = [np.zeros(c.depth.shape, dtype=np.uint16) for c in candidates]
    for t in train_times:
        for candidate, count in zip(candidates, counts):
            observed = sim.observe(candidate.pose, float(t))
            in_front = (candidate.depth - observed["depth"]) > MASK_DEPTH_EPS
            count += in_front.astype(np.uint16)
    return counts


def clean_certifications(
    spec: EpisodeSpec, payload: dict, candidates: List[Candidate], t_end: float, margin: float
) -> np.ndarray:
    trajectories, radii = [], []
    for index, distractor in enumerate(spec.scene.distractors):
        trajectories.append(
            distractor.build_trajectory(seed=spec.scene.trajectory_seed + index)
        )
        raw = payload["distractors"][index]
        diag = float(raw.get("object_diag_m", 1.0)) * float(raw.get("scale") or 1.0)
        radii.append(diag / 2.0)
    centers, sphere_radii = sweep_centers(trajectories, radii, t_end=t_end)
    return np.asarray(
        [certify_clean(c.visible_points, centers, sphere_radii, margin) for c in candidates]
    )


def process_group(group: str, config_paths: Dict[str, Path], args) -> None:
    payloads = {d: yaml.safe_load(p.read_text()) for d, p in sorted(config_paths.items())}
    dynamic = {d: p for d, p in payloads.items() if p.get("distractors")}
    if not dynamic:
        print("[%s] dynamic condition unavailable; skipped" % group)
        return
    out_dir = Path(args.out_dir) / group
    transforms_path = out_dir / "transforms_eval_shared.json"
    if transforms_path.exists() and not args.overwrite:
        print("[%s] exists; skipped (use --overwrite)" % group)
        return

    rng = np.random.default_rng((zlib.crc32(group.encode()) + args.eval_seed) % 2**32)
    specs = {d: EpisodeSpec.from_yaml(config_paths[d]) for d in dynamic}
    first = specs[sorted(dynamic)[0]]
    intrinsics = first.scene.habitat.intrinsics
    interval = first.reconstruction_interval or 1.0
    t_end = first.max_sim_time
    train_times = np.arange(0.0, t_end + 1e-9, interval)

    route_points: List[List[float]] = []
    for difficulty in sorted(dynamic):
        for distractor in payloads[difficulty].get("distractors", []):
            trajectory = distractor.get("trajectory", {})
            if "waypoints" in trajectory:
                route_points.extend(trajectory["waypoints"])
            elif "center" in trajectory:
                route_points.append(trajectory["center"])

    candidates: List[Candidate] = []
    occ: Dict[str, List[np.ndarray]] = {}
    clean: Dict[str, np.ndarray] = {}
    for index, difficulty in enumerate(sorted(dynamic)):
        spec = specs[difficulty]
        sim = DynamicSceneSim(spec.scene)
        try:
            if index == 0:
                candidates = build_candidates(sim, rng, args, route_points=route_points)
                print("[%s] %d valid candidates" % (group, len(candidates)))
                if not candidates:
                    return
            occ[difficulty] = occlusion_counts(sim, candidates, train_times)
            clean[difficulty] = clean_certifications(
                spec, payloads[difficulty], candidates, t_end, args.clean_margin
            )
        finally:
            sim.close()

    positions = np.stack([c.pose.position for c in candidates])
    fracs = {
        d: np.asarray([float((count > 0).mean()) for count in counts])
        for d, counts in occ.items()
    }
    selection = select_two_pole(
        positions, fracs, clean,
        severe_quota=args.severe, clean_quota=args.clean,
        severe_threshold=args.severe_threshold,
    )
    for shortfall in selection.shortfalls:
        print("[%s] SHORTFALL %s" % (group, shortfall))

    ensure_dir(out_dir / "gt")
    frames = []
    for out_index, cand_index in enumerate(selection.indices):
        candidate = candidates[cand_index]
        rgb_rel = "gt/eval_%04d.png" % out_index
        depth_rel = "gt/eval_depth_%04d.npy" % out_index
        save_rgb_png(out_dir / rgb_rel, candidate.rgb)
        np.save(out_dir / depth_rel, candidate.depth)
        frame = frame_from_pose(candidate.pose, rgb_rel, depth_path=depth_rel)
        frame["stratum"] = candidate.stratum
        frame["occlusion"] = {
            d: {
                "frac": round(float(fracs[d][cand_index]), 5),
                # Brief close passes flash-cover whole frames once; persistent
                # occlusion (>= 2 of the 61 training times) separates them.
                "frac_persistent": round(
                    float((occ[d][cand_index] >= 2).mean()), 5
                ),
                "class": selection.labels[cand_index][d],
            }
            for d in sorted(dynamic)
        }
        frames.append(frame)
    for difficulty in sorted(dynamic):
        save_npz_compressed(
            out_dir / ("masks_%s.npz" % difficulty),
            occ_count=np.stack([occ[difficulty][i] for i in selection.indices]),
        )
    export_transforms_json(
        out_dir, intrinsics, frames, filename=transforms_path.name
    )
    save_json(out_dir / "meta.json", {
        "group": group,
        "configs": {d: str(p) for d, p in config_paths.items()},
        "train_times": [float(t) for t in train_times],
        "severe_threshold": args.severe_threshold,
        "clean_margin_m": args.clean_margin,
        "quotas": {"severe": args.severe, "clean": args.clean},
        "candidates_total": len(candidates),
        "selected": len(selection.indices),
        "shortfalls": selection.shortfalls,
        "class_counts": {
            d: {
                label: sum(
                    1 for i in selection.indices
                    if selection.labels[i][d] == label
                )
                for label in ("severe", "clean", "mixed")
            }
            for d in sorted(dynamic)
        },
    })
    print("[%s] wrote %d shared eval views -> %s" % (group, len(frames), out_dir))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs-dir", default=str(_REPO_ROOT / "configs/bench/campaign_v2"))
    parser.add_argument("--out-dir", default=str(_REPO_ROOT / "eval_assets/shared_eval"))
    parser.add_argument("--group", default=None, help="process one <scene>__s<seed> group")
    parser.add_argument("--list-groups", action="store_true")
    parser.add_argument("--candidates", type=int, default=160)
    parser.add_argument("--nav-samples", type=int, default=2000)
    parser.add_argument("--severe", type=int, default=20)
    parser.add_argument("--clean", type=int, default=20)
    parser.add_argument("--severe-threshold", type=float, default=DEFAULT_SEVERE_THRESHOLD)
    parser.add_argument("--clean-margin", type=float, default=DEFAULT_CLEAN_MARGIN_M)
    parser.add_argument("--eval-seed", type=int, default=20260715)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    groups = discover_groups(Path(args.configs_dir))
    if args.list_groups:
        print("\n".join(sorted(groups)))
        return
    if args.group:
        if args.group not in groups:
            sys.exit("unknown group %r; --list-groups shows options" % args.group)
        groups = {args.group: groups[args.group]}
    for group, config_paths in sorted(groups.items()):
        process_group(group, config_paths, args)


if __name__ == "__main__":
    main()
