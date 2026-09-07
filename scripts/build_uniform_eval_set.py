"""Build a uniform cube eval set: N standing points x the 6 cube-face views.

A second, independent appearance metric alongside the severe/clean shared-eval
catalog, which it does NOT replace. The two answer different questions:

- ``build_shared_eval_set.py`` aims a third of its cameras AT the distractor
  patrol routes, because unbiased views almost never see a distractor (measured
  on campaign_v6: only 7% of the non-route views land in the severe class). It
  is a *damage probe* and its severe class is, by construction, nearly the same
  partition as its "route" stratum.
- This set is the *map-quality* metric: positions are spread over the navigable
  area with no knowledge of the distractors, and every point contributes a full
  spherical view, so a method cannot hide a badly reconstructed direction.

Each point contributes the six cube-face directions -- four horizontal at
90 deg spacing plus straight up and straight down -- rendered through a SQUARE
90 deg frustum, which is the one configuration that tiles the whole sphere with
neither gaps nor double-counted borders. The eval camera is deliberately not the
planner camera: the planner runs at 75.18 deg HFOV (forced by a roster method's
intrinsics) while ``retrain_eval`` reads ``fl_x`` from this set's own transforms,
so the two are independent.

A level-only ring was the earlier design. It was replaced because it cannot see
ceilings or floors, which flatters any policy that surveys rooms from their
thresholds instead of entering them.

Standing points keep a clearance from geometry (``--min-clearance``), measured
against the scene's GT surface samples, so cameras never start inside a wall or
hard against furniture.

Clearance alone is not enough, and pushing points away from surfaces actively
makes the remaining failure *more* likely: a point can be perfectly clear and
still have faces that look out of the scene into empty space. Such a view
renders all-black GT with no valid depth, a reconstruction that also renders
nothing matches it exactly, and ``psnr()`` returns its ``99.0``
identical-images sentinel -- a constant that silently inflates every score
computed against the set (two such views in a 12-view set added 16.5 dB; see
``docs/results/2026-07-31-calibration-label-audit.md``). So every candidate
point is *rendered* during selection and accepted only if **all** of its
faces see geometry over at least ``--min-valid-depth-frac`` of the frame.
Rejection is at point level, not view level, because uniformity is what gives
this set its resolving power -- dropping individual views would leave points
contributing different numbers of faces.

Output is drop-in compatible with the shared-eval consumers: the directory
holds ``transforms_eval_shared.json`` plus ``gt/`` RGB-D, so it can be passed
straight to ``retrain_eval.py --shared-dir``.

Run in the sim env (habitat-gs for .gs.ply scenes):

    python scripts/build_uniform_eval_set.py \
        --configs-dir configs/bench/campaign_v6 \
        --out-dir eval_assets/uniform_eval_gs_v6 \
        --points 24 --width 1200 --height 1200
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from activebench.episode import EpisodeSpec  # noqa: E402
from activebench.sim import DynamicSceneSim  # noqa: E402
from activebench.common.transforms_export import (  # noqa: E402
    export_transforms_json,
    frame_from_pose,
)
from activebench.common.camera import CameraIntrinsics, CameraPose  # noqa: E402
from activebench.common.image import save_rgb_png  # noqa: E402
from activebench.common.io import ensure_dir, save_json  # noqa: E402

# Sampling the navmesh densely and thinning by farthest-point gives an even
# spread; a plain random draw clumps.
NAV_SAMPLES = 4000


def _repo_relative(path: Path) -> str:
    """Repo-relative path for provenance, tolerating a cwd-relative argument."""

    try:
        return str(path.resolve().relative_to(_REPO_ROOT))
    except ValueError:
        return str(path)


def discover_groups(configs_dir: Path) -> Dict[str, Path]:
    """(scene__seed) -> the d0 config, which defines the distractor-free scene."""

    groups: Dict[str, Path] = {}
    for path in sorted(configs_dir.glob("*.yaml")):
        parts = path.stem.split("__")
        if len(parts) != 3:
            continue
        scene, difficulty, seed = parts
        if difficulty == "d0":
            groups["%s__%s" % (scene, seed)] = path
    return groups


def farthest_points(points: np.ndarray, count: int) -> np.ndarray:
    """Indices of up to ``count`` points in greedy farthest-point order.

    Returning fewer than requested is fine; returning them *unordered* is not.
    An earlier version short-circuited to ``arange`` whenever ``count`` reached
    the input size, which silently handed back navmesh sampling order -- so a
    caller that over-requests a ranked pool and then keeps a prefix got an
    arbitrary subset with no spreading at all. The prefix-stability this
    function promises has to hold in that branch too, so the greedy loop now
    always runs.
    """

    count = min(int(count), len(points))
    if count <= 0:
        return np.asarray([], dtype=int)
    picked = [int(np.argmax(np.linalg.norm(points - points.mean(0), axis=1)))]
    distances = np.linalg.norm(points - points[picked[0]], axis=1)
    distances[picked[0]] = -1.0
    for _ in range(count - 1):
        nxt = int(np.argmax(distances))
        picked.append(nxt)
        distances = np.minimum(distances, np.linalg.norm(points - points[nxt], axis=1))
        # Never re-pick: duplicate coordinates would otherwise tie at distance 0.
        distances[picked] = -1.0
    return np.asarray(picked)


def view_is_valid(depth: np.ndarray, min_valid_frac: float) -> bool:
    """True when the camera actually saw geometry over enough of the frame.

    A ray that escapes the scene returns depth 0. A view that is mostly escape
    is not a reconstruction test: both the GT and any reconstruction render
    nothing there, so the matching region is free score. At the limit -- no
    valid depth at all -- ``psnr()`` hits its ``99.0`` identical-images sentinel
    and contributes a per-view constant to every model ever scored.
    """

    return float((np.asarray(depth) > 0.0).mean()) >= min_valid_frac


def standing_points(
    sim: DynamicSceneSim,
    surface: np.ndarray,
    count: int,
    min_clearance: float,
    seed: int,
    eye_height: float,
    band: float,
    pool: int = 0,
    clearance_floor: float = 0.4,
) -> np.ndarray:
    """Evenly spread navigable positions that keep clear of walls and furniture.

    Clearance is horizontal distance to surface samples lying within ``band``
    metres of eye height. Restricting to that slab is what makes the measure
    mean anything: projecting the whole cloud to the ground plane would include
    the floor and ceiling, which blanket the entire footprint and drive the
    nearest-surface distance to zero everywhere -- the filter would then keep
    only positions over gaps in the surface sampling.

    ``pool`` returns that many *ranked* candidates instead of exactly ``count``,
    so the caller can walk the ranking and skip points whose panorama escapes
    the scene. The greedy farthest-point order is built incrementally, so its
    first ``count`` entries do not depend on how many were requested: with no
    rejections the selection is identical to asking for ``count`` directly.
    """

    nav = sim.sample_navigable_points(NAV_SAMPLES, seed=seed)
    if nav is None or len(nav) == 0:
        raise RuntimeError("navmesh sampling returned no points")

    want = max(int(count), int(pool))

    if len(surface):
        from scipy.spatial import cKDTree

        # 4.9M GT surface samples: a strided subset resolves clearance to well
        # under the tolerance we care about and builds the tree in a blink.
        thinned = surface[:: max(1, len(surface) // 400_000)]
        eye_y = float(np.median(nav[:, 1])) + eye_height
        slab = thinned[np.abs(thinned[:, 1] - eye_y) <= band]
        if len(slab) < 100:
            print(
                "[uniform-eval] WARN only %d surface samples within %.2f m of eye "
                "height %.2f m; clearance filter disabled" % (len(slab), band, eye_y),
                flush=True,
            )
            return nav[farthest_points(nav, want)]
        tree = cKDTree(slab[:, [0, 2]])
        clearance, _ = tree.query(nav[:, [0, 2]])
        # Adaptive threshold. A fixed clearance is scene-dependent: in a
        # furnished room almost nothing clears 0.75 m, and the survivors are
        # the middles of the few open areas -- a candidate pool too small and
        # too clustered to spread over, whatever the sampler does afterwards.
        # So take the LARGEST threshold that still supplies a full pool, and
        # clamp it to the requested clearance above and to a physical floor
        # below (the same 0.4 m the route audit calls "clear of obstacles").
        # Open scenes are unaffected: the quantile lands above min_clearance
        # and the clamp restores the original behaviour exactly.
        quantile = float(np.quantile(clearance, max(0.0, 1.0 - want / len(nav))))
        threshold = float(np.clip(quantile, clearance_floor, min_clearance))
        keep = clearance >= threshold
        if int(keep.sum()) < count:
            order = np.argsort(-clearance)[: max(count, 1)]
            print(
                "[uniform-eval] WARN only %d/%d navigable points clear the %.2f m "
                "floor; falling back to the %d most-clear points (min %.2f m)"
                % (int(keep.sum()), len(nav), threshold, len(order),
                   float(clearance[order].min())),
                flush=True,
            )
            nav = nav[order]
        else:
            if threshold < min_clearance - 1e-9:
                print(
                    "[uniform-eval] clearance relaxed %.2f -> %.2f m to supply "
                    "%d candidates (%d/%d navigable points pass)"
                    % (min_clearance, threshold, want, int(keep.sum()), len(nav)),
                    flush=True,
                )
            nav = nav[keep]
    return nav[farthest_points(nav, want)]


# The six cube-face directions as (yaw, pitch) in degrees. Four horizontal
# faces at 90 deg spacing plus straight up and straight down; at a 90 deg
# square FOV these tile the whole sphere with no gaps and no overlap. The
# benchmark's poses are OpenGL-style (-Z view), so +pitch looks up, matching
# the shared catalog's "lookup" stratum convention.
CUBE_DIRECTIONS = (
    ("front", 0.0, 0.0),
    ("right", 90.0, 0.0),
    ("back", 180.0, 0.0),
    ("left", 270.0, 0.0),
    ("up", 0.0, 90.0),
    ("down", 0.0, -90.0),
)


def cube_poses(centers: np.ndarray, eye_height) -> List[CameraPose]:
    """The six cube-face views at every standing point, in point-major order.

    ``eye_height`` is metres above the navigable floor, either one value for
    every centre or one per centre.
    """

    heights = np.broadcast_to(np.asarray(eye_height, dtype=float), (len(centers),))
    poses: List[CameraPose] = []
    for center, height in zip(centers, heights):
        eye = [float(center[0]), float(center[1]) + float(height), float(center[2])]
        for _, yaw_deg, pitch_deg in CUBE_DIRECTIONS:
            poses.append(
                CameraPose.from_xyz_yaw_pitch(
                    eye, yaw=np.radians(yaw_deg), pitch=np.radians(pitch_deg)
                )
            )
    return poses


def _central_depth(depth: np.ndarray, frac: float = 0.1) -> float:
    """Median valid depth in a central box: the perpendicular range of the view.

    The median over a whole 90 deg frame is dominated by oblique rays and
    overstates the distance to a flat surface; the centre of an up- or
    down-facing view looks straight at the ceiling or the floor.
    """

    h, w = depth.shape[:2]
    dy, dx = max(1, int(h * frac / 2)), max(1, int(w * frac / 2))
    patch = depth[h // 2 - dy: h // 2 + dy, w // 2 - dx: w // 2 + dx]
    valid = patch[patch > 0.0]
    return float(np.median(valid)) if valid.size else 0.0


def mid_height_correction(
    sim,
    center: np.ndarray,
    height: float,
    limits=(0.4, 3.0),
) -> float:
    """Height above the floor that puts the camera midway to the ceiling.

    A cube view samples up and down symmetrically only if the camera sits
    halfway between them. At a fixed 1.0 m eye height the ``down`` face images
    the floor from 1.0 m while ``up`` images the ceiling from roughly 1.7 m, so
    the two faces sample at ranges differing by most of a metre and their
    scores are not comparable.

    The ceiling is measured by *rendering* rather than from the GT surface
    samples: on GS stages those samples carry floaters metres above the room
    (interior_0007 spans 18 m vertically), so the highest sample in a column is
    a floater, not a ceiling. The renderer's own depth has no such problem.
    Raising the camera by delta moves it delta closer to the ceiling and delta
    further from the floor, so one correction is exact for a flat pair.
    """

    probe = cube_poses(np.asarray([center]), height)
    up_depth = _central_depth(sim.observe(probe[4], 0.0)["depth"])
    down_depth = _central_depth(sim.observe(probe[5], 0.0)["depth"])
    if up_depth <= 0.0 or down_depth <= 0.0:
        return height
    corrected = height + 0.5 * (up_depth - down_depth)
    if not (limits[0] <= corrected <= limits[1]):
        return height
    return corrected

def process_group(group: str, config_path: Path, args) -> None:
    out_dir = Path(args.out_dir) / group
    if (out_dir / "transforms_eval_shared.json").exists() and not args.overwrite:
        print("[uniform-eval] %s exists, skipping" % group, flush=True)
        return

    payload = yaml.safe_load(config_path.read_text())
    payload["distractors"] = []  # the map metric is measured on the clean scene
    if args.width and args.height:
        # Render at the canonical reconstruction resolution rather than the
        # planner's, matching what the shared-eval catalog is re-rendered to.
        payload.setdefault("habitat", {})["width"] = int(args.width)
        payload["habitat"]["height"] = int(args.height)
    # The eval camera is independent of the planner camera: retrain_eval reads
    # fl_x from this set's own transforms, so the FOV here does not have to be
    # the episode's. A cube needs a square 90 deg frustum to tile the sphere.
    payload.setdefault("habitat", {})["hfov"] = float(args.hfov)
    if payload["habitat"]["width"] != payload["habitat"]["height"]:
        raise ValueError(
            "cube views need a square frustum: got %dx%d. Pass --width/--height "
            "equal (e.g. 1200x1200)."
            % (payload["habitat"]["width"], payload["habitat"]["height"])
        )
    spec = EpisodeSpec.from_dict(payload)
    sim = DynamicSceneSim(spec.scene)
    try:
        surface_path = _REPO_ROOT / "eval_assets/surface" / ("%s.npz" % group.split("__")[0])
        surface = (
            np.load(surface_path)["points"] if surface_path.exists() else np.zeros((0, 3))
        )
        if not len(surface):
            print(
                "[uniform-eval] WARN no surface samples at %s; clearance unchecked"
                % surface_path,
                flush=True,
            )

        pool = max(args.points * max(1, args.candidate_factor), args.points)
        candidates = standing_points(
            sim, surface, args.points, args.min_clearance, args.seed,
            args.eye_height, args.clearance_band, pool=pool,
            clearance_floor=args.clearance_floor,
        )

        # Walk the farthest-point ranking and keep a point only if its whole
        # panorama sees geometry. Renders of accepted points are reused, so a
        # scene with no escaping views costs exactly what it did before.
        gt_dir = ensure_dir(out_dir / "gt")
        eye_heights: List[float] = []
        centers: List[np.ndarray] = []
        frames = []
        valid_fractions: List[List[float]] = []
        rejected: List[Dict] = []
        for center in candidates:
            if len(centers) >= args.points:
                break
            height = args.eye_height
            if args.eye_placement == "mid-height":
                height = mid_height_correction(sim, center, height)
            panorama = cube_poses(np.asarray([center]), height)
            # The scene carries no distractors here, so the plain render at any
            # time is already the clean GT.
            observations = [sim.observe(pose, 0.0) for pose in panorama]
            fractions = [
                float((obs["depth"] > 0.0).mean()) for obs in observations
            ]
            bad = [
                i for i, obs in enumerate(observations)
                if not view_is_valid(obs["depth"], args.min_valid_depth_frac)
            ]
            if bad:
                rejected.append({
                    "center": np.asarray(center).tolist(),
                    "failing_faces": [CUBE_DIRECTIONS[i][0] for i in bad],
                    "valid_depth_fraction": [round(v, 4) for v in fractions],
                })
                print(
                    "[uniform-eval] %s: rejected standing point %s -- faces %s "
                    "escape the scene (valid-depth fraction %s < %.2f)"
                    % (group, np.round(center, 2).tolist(), bad,
                       [round(fractions[i], 3) for i in bad],
                       args.min_valid_depth_frac),
                    flush=True,
                )
                continue
            for (face, _, _), pose, obs in zip(CUBE_DIRECTIONS, panorama, observations):
                index = len(frames)
                rgb_rel = "gt/eval_%04d.png" % index
                depth_rel = "gt/eval_depth_%04d.npy" % index
                save_rgb_png(out_dir / rgb_rel, obs["rgb"])
                np.save(out_dir / depth_rel, obs["depth"].astype(np.float32))
                frames.append(frame_from_pose(pose, rgb_rel, depth_rel, stratum=face))
            centers.append(np.asarray(center))
            eye_heights.append(round(float(height), 4))
            valid_fractions.append([round(v, 4) for v in fractions])

        if len(centers) < args.points:
            raise RuntimeError(
                "%s: only %d/%d standing points have a fully valid panorama out of "
                "%d ranked candidates. Raise --candidate-factor, lower "
                "--min-valid-depth-frac, or accept a smaller --points; do not ship "
                "a set with escaping views, they add a psnr() sentinel constant to "
                "every score." % (group, len(centers), args.points, len(candidates))
            )
        centers = np.asarray(centers)

        intrinsics = CameraIntrinsics.from_hfov(
            spec.scene.habitat.width, spec.scene.habitat.height, spec.scene.habitat.hfov
        )
        export_transforms_json(
            out_dir, intrinsics, frames, filename="transforms_eval_shared.json"
        )
        save_json(
            out_dir / "meta.json",
            {
                "group": group,
                "config": _repo_relative(config_path),
                "protocol": "uniform-cube",
                "points": int(len(centers)),
                "views_per_point": len(CUBE_DIRECTIONS),
                "views_total": int(len(frames)),
                "cube_directions": [d[0] for d in CUBE_DIRECTIONS],
                "hfov_deg": float(args.hfov),
                "eye_placement": args.eye_placement,
                "eye_height_m": float(args.eye_height),
                "eye_heights_m": eye_heights,
                "min_clearance_m": float(args.min_clearance),
                "clearance_band_m": float(args.clearance_band),
                "min_valid_depth_frac": float(args.min_valid_depth_frac),
                "candidate_factor": int(args.candidate_factor),
                "candidates_ranked": int(len(candidates)),
                "points_rejected": len(rejected),
                "rejected_points": rejected,
                "valid_depth_fraction_per_point": valid_fractions,
                "nav_samples": NAV_SAMPLES,
                "seed": int(args.seed),
                "point_centers": centers.tolist(),
                "note": (
                    "Map-quality metric on the distractor-free scene; the "
                    "severe/clean damage probe lives in the shared-eval "
                    "catalog and is unaffected by this set. Every view is "
                    "render-verified to see geometry, so no view can contribute "
                    "the psnr() identical-images sentinel."
                ),
            },
        )
        print(
            "[uniform-eval] %s: %d points x %d views = %d frames at %dx%d "
            "(%d candidate points rejected for escaping views)"
            % (group, len(centers), len(CUBE_DIRECTIONS), len(frames),
               spec.scene.habitat.width, spec.scene.habitat.height, len(rejected)),
            flush=True,
        )
    finally:
        sim.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--group", default=None, help="process one <scene>__s<seed>")
    parser.add_argument(
        "--points", type=int, default=24,
        help="standing points per scene (drives the effective sample size: the "
             "views at one point are spatially correlated)",
    )
    parser.add_argument(
        "--hfov", type=float, default=90.0,
        help="eval camera horizontal FOV, degrees. With a square frame, 90 is "
             "the only value at which the six cube faces tile the sphere "
             "exactly; anything smaller leaves blind wedges between faces and "
             "anything larger double-counts their borders",
    )
    parser.add_argument("--width", type=int, default=None,
                        help="render width override (default: the config's)")
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--eye-height", type=float, default=1.0,
                        help="metres above the floor; with --eye-placement "
                             "mid-height this is only the fallback")
    parser.add_argument(
        "--eye-placement", choices=("mid-height", "fixed"), default="mid-height",
        help="mid-height puts the camera halfway between floor and ceiling so "
             "the up and down cube faces sample symmetrically; fixed keeps the "
             "camera at --eye-height above the floor",
    )
    parser.add_argument(
        "--min-clearance", type=float, default=0.75,
        help="minimum horizontal distance from GT surface samples, metres",
    )
    parser.add_argument(
        "--clearance-floor", type=float, default=0.4,
        help="clearance is relaxed no further than this when a scene cannot "
             "supply a full candidate pool at --min-clearance; 0.4 m matches "
             "the route audit's obstacle-clearance rule",
    )
    parser.add_argument(
        "--clearance-band", type=float, default=0.5,
        help="half-thickness of the eye-height slab the clearance is measured "
             "against, metres; excludes floor and ceiling",
    )
    parser.add_argument(
        "--min-valid-depth-frac", type=float, default=0.5,
        help="a view must have at least this fraction of pixels with valid "
             "depth, i.e. must actually see geometry rather than escape the "
             "scene. A point is rejected unless ALL its faces pass. Measured "
             "separation on the existing sets is wide: legitimate views sit at "
             "0.57-1.00 while escaping ones sit at 0.00-0.20",
    )
    parser.add_argument(
        "--candidate-factor", type=int, default=8,
        help="rank this many times --points candidates so rejected points can "
             "be replaced; the farthest-point order is prefix-stable, so this "
             "does not change the selection when nothing is rejected",
    )
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    groups = discover_groups(args.configs_dir)
    if args.group:
        groups = {k: v for k, v in groups.items() if k == args.group}
        if not groups:
            parser.error("group %s not found in %s" % (args.group, args.configs_dir))
    for group, config_path in groups.items():
        process_group(group, config_path, args)


if __name__ == "__main__":
    main()
