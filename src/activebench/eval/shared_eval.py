"""Shared held-out eval sets: fixed poses reused across methods and difficulties.

Design (2026-07-15): the runner's per-method eval sets are jittered around each
agent's own trajectory, which biases PSNR toward wherever that method happened
to operate and makes cross-method numbers incomparable. A shared eval set fixes
one pose list per (scene, seed) campaign group, reused by every method AND
every difficulty. Clean GT renders are difficulty-invariant, so d0 -> dyn
comparisons are paired on pixel-identical ground truth.

Poses are sampled for whole-scene coverage (farthest-point over navmesh
samples) and selected two-pole:

- **severe**: at least ``severe_threshold`` (default 40%) of pixels saw a distractor in
  front of the clean surface at one of the 1 Hz training times. Contamination
  can enter any method's reconstruction stream only at those times, so the
  render-based union mask over them is exact, not an approximation.
- **clean**: certified never-disturbed. Certification is geometric and
  region-based, not silhouette-based: no distractor bounding sphere (plus a
  margin) ever comes near the 3D surface region visible from the pose, at a
  fine time sampling. This also excludes poses whose visible region a
  distractor may have occluded from *other* methods' camera angles.

Mean PSNR over the severe class (d0 vs dyn) isolates damage where distractors
lived; the clean class isolates collateral damage everywhere else.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np

DEFAULT_SEVERE_THRESHOLD = 0.4
DEFAULT_CLEAN_MARGIN_M = 0.25

# Pitch ranges (radians) per stratum, matching the runner's eval conventions:
# positive pitch looks up in the benchmark convention.
_STRATUM_PITCH = {
    "level": (-0.35, 0.35),
    "lookup": (np.deg2rad(40.0), np.deg2rad(70.0)),
    "lookdown": (np.deg2rad(-70.0), np.deg2rad(-40.0)),
}
# level : lookup : lookdown cycle; half the views stay level.
_STRATUM_CYCLE = ("level", "lookup", "level", "lookdown")


def farthest_point_indices(points: np.ndarray, count: int, start: int = 0) -> List[int]:
    """Deterministic farthest-point subset of ``points`` (N, D), seeded at ``start``."""

    points = np.asarray(points, dtype=np.float64)
    if len(points) == 0 or count <= 0:
        return []
    count = min(count, len(points))
    chosen = [int(start) % len(points)]
    distances = np.linalg.norm(points - points[chosen[0]], axis=1)
    while len(chosen) < count:
        nxt = int(np.argmax(distances))
        chosen.append(nxt)
        distances = np.minimum(distances, np.linalg.norm(points - points[nxt], axis=1))
    return chosen


def candidate_orientations(
    count: int, rng: np.random.Generator
) -> List[Tuple[float, float, str]]:
    """(yaw, pitch, stratum) per candidate, cycling strata for coverage."""

    orientations = []
    for index in range(count):
        stratum = _STRATUM_CYCLE[index % len(_STRATUM_CYCLE)]
        lo, hi = _STRATUM_PITCH[stratum]
        orientations.append(
            (float(rng.uniform(-np.pi, np.pi)), float(rng.uniform(lo, hi)), stratum)
        )
    return orientations


def aim_at(from_pos: np.ndarray, to_pos: np.ndarray) -> Tuple[float, float]:
    """(yaw, pitch) that points the benchmark camera from one point at another.

    Benchmark convention: yaw about +Y with yaw=0 facing -Z, positive pitch
    looks up (see activebench.common.transforms).
    """

    d = np.asarray(to_pos, dtype=np.float64) - np.asarray(from_pos, dtype=np.float64)
    norm = float(np.linalg.norm(d))
    if norm == 0.0:
        return 0.0, 0.0
    yaw = float(np.arctan2(-d[0], -d[2]))
    pitch = float(np.arcsin(np.clip(d[1] / norm, -1.0, 1.0)))
    return yaw, pitch


def route_biased_positions(
    nav_points: np.ndarray,
    route_points: np.ndarray,
    count: int,
    near: float = 3.0,
    far_enough: float = 0.8,
) -> List[int]:
    """Indices of navmesh points that stand near (not inside) patrol routes.

    Whole-scene farthest-point sampling under-covers distractor routes in
    large scenes (MP3D: <10% of spread candidates reach severe occlusion), so
    a share of candidates is placed within ``near`` meters of a route and
    aimed at it by the caller. Distances are horizontal: route waypoints float
    at object hover height.
    """

    if len(nav_points) == 0 or len(route_points) == 0 or count <= 0:
        return []
    nav_xz = np.asarray(nav_points, dtype=np.float64)[:, [0, 2]]
    route_xz = np.asarray(route_points, dtype=np.float64)[:, [0, 2]]
    distances = np.linalg.norm(nav_xz[:, None, :] - route_xz[None, :, :], axis=2).min(axis=1)
    eligible = np.flatnonzero((distances >= far_enough) & (distances <= near))
    if len(eligible) == 0:
        return []
    chosen = farthest_point_indices(np.asarray(nav_points)[eligible], count)
    return [int(eligible[i]) for i in chosen]


def sweep_centers(
    trajectories: Sequence, radii: Sequence[float], t_end: float, dt: float = 0.1
) -> Tuple[np.ndarray, np.ndarray]:
    """Distractor center positions and radii swept over [0, t_end] at ``dt``."""

    times = np.arange(0.0, t_end + 1e-9, dt)
    centers, sphere_radii = [], []
    for trajectory, radius in zip(trajectories, radii):
        for t in times:
            position, _ = trajectory.pose_at(float(t))
            centers.append(np.asarray(position, dtype=np.float64))
            sphere_radii.append(float(radius))
    if not centers:
        return np.zeros((0, 3)), np.zeros(0)
    return np.stack(centers), np.asarray(sphere_radii)


def certify_clean(
    visible_points: np.ndarray,
    centers: np.ndarray,
    radii: np.ndarray,
    margin: float = DEFAULT_CLEAN_MARGIN_M,
) -> bool:
    """True when no swept distractor sphere comes near the visible 3D region.

    ``visible_points``: world-space points seen from the eval pose (subsampled
    unprojection of its clean depth). Region-based on purpose: a distractor
    close to the visible surface is disqualifying even if this particular view
    angle never had it between camera and surface — it may have occluded the
    region from other angles that reconstruction methods observed.
    """

    if len(centers) == 0:
        return True
    if len(visible_points) == 0:
        return False
    from scipy.spatial import cKDTree

    distances, _ = cKDTree(np.asarray(visible_points, dtype=np.float64)).query(centers)
    return bool(np.all(distances > radii + margin))


@dataclass
class SelectionResult:
    """Selected candidate indices with per-difficulty class labels."""

    indices: List[int] = field(default_factory=list)
    labels: Dict[int, Dict[str, str]] = field(default_factory=dict)  # index -> {difficulty: class}
    shortfalls: List[str] = field(default_factory=list)

    def classes_for(self, index: int) -> Dict[str, str]:
        return self.labels.get(index, {})


def select_two_pole(
    positions: np.ndarray,
    fracs: Dict[str, np.ndarray],
    clean: Dict[str, np.ndarray],
    severe_quota: int,
    clean_quota: int,
    severe_threshold: float = DEFAULT_SEVERE_THRESHOLD,
) -> SelectionResult:
    """Pick spatially-diverse severe/clean views per dynamic difficulty.

    ``fracs`` / ``clean`` map difficulty name -> per-candidate arrays. A pose
    joins the clean class only when certified clean under EVERY difficulty, so
    the clean baseline is valid for all of them. Severe poses are picked per
    difficulty (hardest first, reusing overlaps) since swept volumes differ.
    """

    positions = np.asarray(positions, dtype=np.float64)
    count = len(positions)
    difficulties = sorted(fracs)
    selected: List[int] = []
    shortfalls: List[str] = []

    def pick(pool_mask: np.ndarray, quota: int, tag: str) -> None:
        reused = sum(1 for i in selected if pool_mask[i])
        available = [int(i) for i in np.flatnonzero(pool_mask) if i not in selected]
        needed = quota - reused
        if needed > len(available):
            shortfalls.append("%s: %d/%d available" % (tag, int(pool_mask.sum()), quota))
            needed = len(available)
        if needed > 0:
            chosen = farthest_point_indices(positions[available], needed)
            selected.extend(available[i] for i in chosen)

    # Clean picks need BOTH the geometric certification and a zero rendered
    # occlusion fraction: a distractor crossing right in front of the lens
    # occludes pixels while staying far from every visible surface.
    all_clean = np.ones(count, dtype=bool)
    for difficulty in difficulties:
        all_clean &= np.asarray(clean[difficulty], dtype=bool)
        all_clean &= np.asarray(fracs[difficulty]) <= 0.0
    pick(all_clean, clean_quota, "clean")
    # Hardest difficulty first: its severe pool usually covers the others'.
    for difficulty in reversed(difficulties):
        severe_pool = np.asarray(fracs[difficulty]) >= severe_threshold
        pick(severe_pool, severe_quota, "severe[%s]" % difficulty)

    # Labels are pure functions of the stats, independent of pick order.
    labels: Dict[int, Dict[str, str]] = {}
    for index in sorted(selected):
        labels[index] = {}
        for difficulty in difficulties:
            if fracs[difficulty][index] >= severe_threshold:
                labels[index][difficulty] = "severe"
            elif clean[difficulty][index]:
                labels[index][difficulty] = "clean"
            else:
                labels[index][difficulty] = "mixed"
    return SelectionResult(indices=sorted(selected), labels=labels, shortfalls=shortfalls)
