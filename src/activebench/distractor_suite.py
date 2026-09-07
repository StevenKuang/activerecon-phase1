"""Seeded distractor generation from audited assets and navmesh-valid routes.

The scene suite supplies patrol polylines built from Habitat shortest paths.
This module chooses objects that fit each route's audited clearance, lifts the
route above its floor, and assigns speeds relative to the benchmark vehicle.
It never treats a stage AABB as free space.
"""

import json
import zlib
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

AGENT_SPEED_MPS = 0.5
DIFFICULTIES: Dict[str, Dict[str, float]] = {
    "d0": {
        "per_m2": 0.0,
        "min_count": 0,
        "max_count": 0,
        "speed_min": 0.0,
        "speed_max": 0.0,
    },
    "d1": {
        "per_m2": 1 / 25.0,
        "min_count": 3,
        "max_count": 6,
        "speed_min": 0.15,
        "speed_max": 0.40,
    },
    "d2": {
        "per_m2": 1 / 8.0,
        "min_count": 8,
        "max_count": 16,
        "speed_min": 0.25,
        "speed_max": 0.80,
    },
    # Single dynamic level for v3 rounds (replaces the d1/d2 axis): few but
    # unmissably large objects, per-object speeds stratified across the full
    # slower-to-faster-than-the-agent range.
    "dyn": {
        "per_m2": 1 / 40.0,
        "min_count": 4,
        "max_count": 6,
        "speed_min": 0.15,
        "speed_max": 0.80,
        "diag_range": (0.9, 2.0),
    },
}

POOL_DIAG_RANGE = (0.6, 1.8)
ROUTE_MARGIN = 0.1
MIN_HOVER_MARGIN = 0.15
MAX_HOVER_HEIGHT = 1.5


def load_object_pool(audit_path: Path, diag_range=POOL_DIAG_RANGE) -> List[Dict[str, Any]]:
    """Return audited objects large enough to be visible in campaign renders."""

    audit = json.loads(Path(audit_path).read_text())
    lo, hi = diag_range
    pool = [
        {"handle": entry["handle"], "diag": float(entry["diag"])}
        for entry in audit
        if lo <= float(entry["diag"]) < hi
    ]
    pool.sort(key=lambda entry: entry["handle"])
    if not pool:
        raise ValueError("object audit yielded an empty distractor pool")
    return pool


def _rng_for(scene_name: str, difficulty: str, seed: int) -> np.random.Generator:
    return np.random.default_rng(
        [seed, zlib.crc32(scene_name.encode()), zlib.crc32(difficulty.encode())]
    )


def _stratified_speeds(
    count: int, speed_min: float, speed_max: float, rng: np.random.Generator
) -> np.ndarray:
    edges = np.linspace(speed_min, speed_max, count + 1)
    speeds = np.array([rng.uniform(edges[i], edges[i + 1]) for i in range(count)])
    return speeds[rng.permutation(count)]


def _fits_route(route: Dict[str, Any], obj: Dict[str, Any], ceiling_y: float) -> bool:
    diagonal = float(obj["diag"])
    horizontal_fit = 0.5 * diagonal + ROUTE_MARGIN <= float(route["clearance_m"])
    floor_y = max(float(point[1]) for point in route["waypoints"])
    vertical_fit = floor_y + diagonal + MIN_HOVER_MARGIN <= ceiling_y
    return horizontal_fit and vertical_fit


def _compatible_pairs(
    routes: List[Dict[str, Any]],
    pool: List[Dict[str, Any]],
    ceiling_y: float,
) -> List[tuple]:
    return [
        (route, obj)
        for route in routes
        for obj in pool
        if _fits_route(route, obj, ceiling_y)
    ]


def generate_distractors(
    scene_entry: Dict[str, Any],
    difficulty: str,
    pool: List[Dict[str, Any]],
    seed: int = 0,
) -> List[Dict[str, Any]]:
    """Generate deterministic distractors on pre-audited scene-valid routes."""

    if difficulty not in DIFFICULTIES:
        raise KeyError("difficulty must be one of %s" % sorted(DIFFICULTIES))
    spec = DIFFICULTIES[difficulty]
    if spec["max_count"] == 0:
        return []

    routes = scene_entry.get("patrol_routes") or []
    if not routes:
        raise ValueError(
            "scene %s has no patrol_routes; rebuild the scene suite in Habitat"
            % scene_entry["name"]
        )
    diag_lo, diag_hi = spec.get("diag_range", POOL_DIAG_RANGE)
    pool = [obj for obj in pool if diag_lo <= float(obj["diag"]) < diag_hi]
    if not pool:
        raise ValueError(
            "no audited object in diag range [%.2f, %.2f) for difficulty %s"
            % (diag_lo, diag_hi, difficulty)
        )
    lo = np.asarray(scene_entry["aabb_min"], dtype=np.float64)
    hi = np.asarray(scene_entry["aabb_max"], dtype=np.float64)
    pairs = _compatible_pairs(routes, pool, float(hi[1]))
    if not pairs:
        raise ValueError("no audited distractor fits routes for %s" % scene_entry["name"])

    area = float(scene_entry.get("navigable_area_m2", (hi[0] - lo[0]) * (hi[2] - lo[2])))
    count = int(np.clip(round(area * spec["per_m2"]), spec["min_count"], spec["max_count"]))
    rng = _rng_for(scene_entry["name"], difficulty, seed)
    speeds = _stratified_speeds(count, spec["speed_min"], spec["speed_max"], rng)

    # Spread selections over the audited route list before reusing a route.
    route_order = list(rng.permutation(len(routes)))
    distractors = []
    used_handles = set()
    for i in range(count):
        route = routes[route_order[i % len(route_order)]]
        compatible = [
            obj
            for obj in pool
            if _fits_route(route, obj, float(hi[1]))
            and obj["handle"] not in used_handles
        ]
        if not compatible:
            compatible = [
                obj
                for obj in pool
                if _fits_route(route, obj, float(hi[1]))
            ]
        if not compatible:
            # Another audited route can still accommodate this object slot.
            route, obj = pairs[int(rng.integers(len(pairs)))]
        else:
            obj = compatible[int(rng.integers(len(compatible)))]
        used_handles.add(obj["handle"])

        waypoints = np.asarray(route["waypoints"], dtype=np.float64)
        radius = 0.5 * float(obj["diag"])
        max_lift = min(MAX_HOVER_HEIGHT, float(hi[1] - waypoints[:, 1].max() - radius))
        min_lift = radius + MIN_HOVER_MARGIN
        if max_lift < min_lift:
            raise RuntimeError("audited route/object pair no longer fits scene height")
        lift = float(rng.uniform(min_lift, max_lift))
        waypoints = waypoints + np.array([0.0, lift, 0.0])

        distractors.append(
            {
                "name": "%s_%s_obj%d" % (scene_entry["name"], difficulty, i),
                "object_template": obj["handle"],
                "object_diag_m": round(float(obj["diag"]), 3),
                "trajectory": {
                    "type": "waypoint_patrol",
                    "waypoints": [
                        [round(float(value), 3) for value in point] for point in waypoints
                    ],
                    "speed": round(float(speeds[i]), 3),
                    "mode": "pingpong",
                },
            }
        )
    return distractors


def generate_episode_config(
    scene_entry: Dict[str, Any],
    intrinsics: Dict[str, Any],
    difficulty: str,
    pool: List[Dict[str, Any]],
    seed: int = 0,
    max_captures: int = 128,
    max_sim_time: float = 60.0,
    reconstruction_interval: float = 1.0,
    stream_observations: bool = False,
) -> Dict[str, Any]:
    """Build a campaign episode configuration (v2, or v3 when streaming)."""

    return {
        "habitat": {
            "scene_path": scene_entry["scene_path"],
            "scene_dataset_config_file": scene_entry.get("scene_dataset_config_file", "default"),
            "width": int(intrinsics["width"]),
            "height": int(intrinsics["height"]),
            "hfov": float(intrinsics["hfov"]),
            "enable_physics": True,
        },
        "seed": seed,
        "trajectory_seed": seed,
        "start_pose": [float(v) for v in scene_entry["start_pose"]],
        "distractors": generate_distractors(scene_entry, difficulty, pool, seed),
        "motion": {
            "speed": AGENT_SPEED_MPS,
            "yaw_rate_deg": 60.0,
            "pitch_rate_deg": 60.0,
        },
        "episode": {
            "max_captures": max_captures,
            "max_sim_time": max_sim_time,
            "capture_cost": 0.0,
            "capture_interval": None,
            "reconstruction_interval": reconstruction_interval,
            "stream_observations": bool(stream_observations),
            "provide_depth": True,
        },
        "campaign": {
            "protocol": "video-stream-v3" if stream_observations else "mission-time-v2",
            "scene": scene_entry["name"],
            "difficulty": difficulty,
            "has_ceiling": bool(scene_entry.get("has_ceiling", False)),
        },
    }
