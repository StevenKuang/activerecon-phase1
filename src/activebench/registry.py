"""Agent registry: build benchmark agents by name, in-process or in a worker.

The same factory runs on both sides of the RPC boundary, so ``options`` must
stay JSON-serializable. ``default_env`` names the conda env a method expects;
``None`` means the current interpreter is fine.
"""

from typing import Any, Dict, List, Optional

import numpy as np

from activebench.runtime import external_repo


def _build_random(options: Dict[str, Any]):
    from activebench.baselines import RandomAgent

    return RandomAgent()


def _build_pool_random(options: Dict[str, Any]):
    """Pool baseline on the same shared candidate pool as fisherrf/gavis.

    (Until 2026-07-17 it drew its own ±1.5 m box around the start pose;
    unified so every pool agent sees one benchmark-owned pool policy —
    navmesh-sampled under protocol v5.)
    """

    from activebench.baselines import PoolNBVAgent, RandomSelector

    seed = int(options.get("seed", 0))
    rng = np.random.default_rng(seed)
    if "candidate_positions" not in options and "scene_bbox" not in options:
        options = {**options, "scene_bbox": [[-2.0, 0.0, -2.0], [2.0, 3.0, 2.0]]}
    pool = _scene_candidate_pool(options, rng)
    return PoolNBVAgent(selector=RandomSelector(seed=seed), pool=pool, name="pool-random")


def _scene_candidate_pool(options: Dict[str, Any], rng: np.random.Generator):
    """Seeded NBV candidate pool spanning the scene bounds.

    Positions are drawn inside the horizontal scene extent (0.5 m wall
    margin) around the start height. Some samples may fall inside geometry;
    navmesh-aware sampling is a planned upgrade (docs/ARCHITECTURE.md).
    """

    from activebench.common.camera import CameraPose

    provided = options.get("candidate_positions")
    if provided is not None:
        # v5: the launcher sampled these on the navmesh (floor + camera
        # height) — physically valid by construction. Orientation stays a
        # pool policy, drawn here with the shared per-episode rng.
        return [
            CameraPose.from_xyz_yaw_pitch(
                np.asarray(xyz, dtype=np.float64),
                yaw=float(rng.uniform(-np.pi, np.pi)),
                pitch=float(rng.uniform(-0.3, 0.3)),
            )
            for xyz in provided
        ]

    bbox = np.asarray(options["scene_bbox"], dtype=np.float64)
    start = options.get("start_pose") or [0.0, 1.5, 0.0, 0.0, 0.0]
    pool_size = int(options.get("pool_size", 3 * int(options.get("max_captures", 20))))
    lo, hi = bbox[0].copy(), bbox[1].copy()
    margin = np.minimum(0.5, 0.25 * (hi - lo))
    lo += margin
    hi -= margin
    pool = []
    for _ in range(pool_size):
        xyz = rng.uniform(lo, hi)
        xyz[1] = float(np.clip(start[1] + rng.uniform(-0.4, 0.4), lo[1], hi[1]))
        pool.append(
            CameraPose.from_xyz_yaw_pitch(
                xyz,
                yaw=float(rng.uniform(-np.pi, np.pi)),
                pitch=float(rng.uniform(-0.3, 0.3)),
            )
        )
    return pool


def _pool_agent(selector, name: str, options: Dict[str, Any], needs_depth: bool):
    """Wrap a tier-1 selector with the shared candidate-pool policy.

    Default is the R3CON-style ``local`` pool (observed free space within a
    radius of the current pose, isotropic view directions, regenerated each
    round — see activebench/free_space.py); ``pool_mode: "fixed"`` restores
    the legacy global pool.
    """

    from activebench.baselines import PoolNBVAgent

    seed = int(options.get("seed", 0))
    mode = str(options.get("pool_mode", "local"))
    if mode == "fixed":
        pool = _scene_candidate_pool(options, np.random.default_rng(seed))
        return PoolNBVAgent(selector=selector, pool=pool, name=name, needs_depth=needs_depth)
    return PoolNBVAgent(
        selector=selector,
        name=name,
        needs_depth=needs_depth,
        pool_mode="local",
        radius=float(options.get("pool_radius", 2.0)),
        sample_num=int(options.get("pool_sample_num", 50)),
        scene_bbox=np.asarray(options["scene_bbox"], dtype=np.float64),
        seed=seed,
    )


def _build_gavis(options: Dict[str, Any]):
    from activebench.adapters.gavis import GavisSelector
    from activebench.common.camera import CameraIntrinsics

    seed = int(options.get("seed", 0))
    use_depth = bool(options.get("use_depth", True))
    selector = GavisSelector(
        intrinsics=CameraIntrinsics.from_hfov(
            int(options["width"]), int(options["height"]), float(options["hfov"])
        ),
        scene_bbox=np.asarray(options["scene_bbox"], dtype=np.float64),
        train_iterations=int(options.get("train_iterations", 1500)),
        depth_init_pts_per_view=int(options.get("depth_init_pts_per_view", 10_000)),
        use_depth=use_depth,
        seed=seed,
        gavis_overrides=options.get("gavis_overrides", {}),
    )
    return _pool_agent(selector, "gavis", options, needs_depth=use_depth)


def _build_fisherrf(options: Dict[str, Any]):
    from activebench.adapters.fisherrf import FisherRFSelector
    from activebench.common.camera import CameraIntrinsics

    seed = int(options.get("seed", 0))
    selector = FisherRFSelector(
        intrinsics=CameraIntrinsics.from_hfov(
            int(options["width"]), int(options["height"]), float(options["hfov"])
        ),
        scene_bbox=np.asarray(options["scene_bbox"], dtype=np.float64),
        train_iterations=int(options.get("train_iterations", 1500)),
        seed=seed,
    )
    # The selector itself is RGB-only; depth in the agent process feeds only
    # the harness free-space tracker of the local pool.
    return _pool_agent(selector, "fisherrf", options, needs_depth=False)


def _build_r3con(options: Dict[str, Any]):
    from activebench.adapters.r3con import R3ConAgent

    return R3ConAgent(
        scene_bbox=np.asarray(options["scene_bbox"], dtype=np.float64),
        planner_type=options.get("planner_type", "confidence_pano"),
        planner_overrides=options.get("planner_overrides", {}),
        path_stride=int(options.get("path_stride", 1)),
    )


def _build_magician(options: Dict[str, Any]):
    from activebench.adapters.magician import MagicianAgent

    samples = options.get("surface_samples_path")
    if not samples:
        raise ValueError(
            "magician needs surface_samples_path in --agent-options (GT surface "
            "npz from scripts/eval_coverage.py; used for gt coverage logging and "
            "step-1 collision checks)"
        )
    return MagicianAgent(
        scene_bbox=np.asarray(options["scene_bbox"], dtype=np.float64),
        surface_samples_path=str(samples),
        world_scale=options.get("world_scale"),
        beam_width=int(options.get("beam_width", 10)),
        beam_steps=int(options.get("beam_steps", 10)),
        max_captures=int(options.get("max_captures", 20)),
        seed=int(options.get("seed", 0)),
    )


def _build_gleam(options: Dict[str, Any]):
    from activebench.adapters.gleam import GleamAgent

    return GleamAgent(
        scene_bbox=np.asarray(options["scene_bbox"], dtype=np.float64),
        checkpoint=options.get("checkpoint"),
        repo_root=options.get("repo_root", external_repo("GLEAM")),
        spin_legs=int(options.get("spin_legs", 4)),
    )


def _build_wander(options: Dict[str, Any]):
    from activebench.baselines import WanderAgent

    return WanderAgent()


















_REGISTRY: Dict[str, Dict[str, Any]] = {
    'random': {'factory': _build_random, 'default_env': None},
    'pool-random': {'factory': _build_pool_random, 'default_env': None},
    'wander': {'factory': _build_wander, 'default_env': None},
    'r3con-pano': {'factory': _build_r3con, 'default_env': 'r3con'},
    'magician': {'factory': _build_magician, 'default_env': 'magician'},
    'fisherrf': {'factory': _build_fisherrf, 'default_env': 'fisherrf'},
    'gavis': {'factory': _build_gavis, 'default_env': 'gavis'},
    'gleam': {'factory': _build_gleam, 'default_env': 'gleam'},
}


def available_agents() -> List[str]:
    return sorted(_REGISTRY)


def default_env(name: str) -> Optional[str]:
    return _entry(name)["default_env"]


def build_agent(name: str, options: Dict[str, Any]):
    return _entry(name)["factory"](options)


def _entry(name: str) -> Dict[str, Any]:
    if name not in _REGISTRY:
        raise KeyError(
            "unknown agent %r (available: %s)" % (name, ", ".join(available_agents()))
        )
    return _REGISTRY[name]
