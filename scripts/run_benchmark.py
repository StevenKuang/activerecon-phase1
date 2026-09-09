"""Run one benchmark episode: an agent exploring a dynamic Habitat scene.

Examples (habitat conda env):
    python scripts/run_benchmark.py \
        --config configs/bench/apartment_patrol.yaml \
        --agent random --out runs_bench/random_apartment

    # Methods with their own conda env run in a subprocess automatically:
    python scripts/run_benchmark.py \
        --config configs/bench/apartment_patrol.yaml \
        --agent r3con-pano --out runs_bench/r3con_apartment
"""

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from activebench import registry
from activebench.episode import EpisodeSpec
from activebench.rpc import AgentProcessProxy
from activebench.runner import run_episode
from activebench.sim import DynamicSceneSim

from activebench.runtime import conda_envs_dir

CONDA_ENVS_DIR = conda_envs_dir()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="episode spec YAML")
    parser.add_argument("--agent", default="random",
                        help="registered name or module:factory or /path/agent.py:factory")
    parser.add_argument("--out", required=True, help="episode output directory")
    parser.add_argument(
        "--agent-python", help="explicit Python executable for an isolated external agent",
    )
    parser.add_argument(
        "--agent-env",
        default=None,
        help="conda env for the agent subprocess (default: the agent's "
        "registered env; 'inline' forces in-process execution)",
    )
    parser.add_argument(
        "--agent-options",
        default=None,
        help="JSON dict merged into the agent factory options "
        '(e.g. \'{"train_iterations": 400}\')',
    )
    args = parser.parse_args()

    if args.agent_python and args.agent_env:
        parser.error("choose --agent-python or --agent-env, not both")
    if ":" in args.agent:
        from activebench.plugins import normalize_factory
        args.agent = normalize_factory(args.agent)
    if (Path(args.out) / "manifest.json").exists():
        parser.error("completed output already exists; choose a new --out (campaign CLI supports resume)")

    spec = EpisodeSpec.from_yaml(Path(args.config))
    out_dir = Path(args.out)

    env_name = args.agent_env or registry.default_env(args.agent)
    inline = not args.agent_python and env_name in (None, "inline")
    if inline and registry.default_env(args.agent) is not None:
        # In-process CUDA agents need the CUDA context created before
        # Habitat's GL context, or custom CUDA kernels can crash.
        import torch

        if torch.cuda.is_available():
            torch.zeros(1, device="cuda")

    sim = DynamicSceneSim(spec.scene)
    proxy = None
    try:
        options = {
            "seed": spec.seed,
            "start_pose": spec.start_pose,
            "max_captures": spec.max_captures,
            "scene_bbox": sim.scene_aabb().tolist(),
            "width": spec.scene.habitat.width,
            "height": spec.scene.habitat.height,
            "hfov": spec.scene.habitat.hfov,
            "max_sim_time": spec.max_sim_time,
        }
        if args.agent_options:
            import json

            options.update(json.loads(args.agent_options))
        if spec.collision == "navmesh":
            # Benchmark-owned candidate pools must be physically valid: the
            # launcher (which owns the navmesh) samples seeded floor points
            # and ships them in options; RPC workers have no habitat, so
            # they only add yaw/pitch on top.
            pool_size = int(options.get("pool_size", 3 * spec.max_captures))
            start = spec.resolved_start_pose(sim.start_pose())
            floor = sim.sample_navigable_points(
                pool_size, seed=spec.seed, anchor_xyz=start.position)
            if floor is not None:
                anchor = sim.snap_navigable(start.position)
                height = float(start.position[1] - anchor[1]) if anchor is not None else 1.5
                options["candidate_positions"] = [
                    [float(p[0]), float(p[1] + height), float(p[2])] for p in floor
                ]
        if inline:
            agent = registry.build_agent(args.agent, options)
        else:
            python_exe = (Path(args.agent_python).expanduser().resolve() if args.agent_python
                          else CONDA_ENVS_DIR / env_name / "bin" / "python")
            if not python_exe.exists():
                raise SystemExit("conda env %r not found at %s" % (env_name, python_exe))
            proxy = AgentProcessProxy(
                agent_name=args.agent,
                options=options,
                python_exe=str(python_exe),
                stderr_log=out_dir / "agent_worker.log",
            )
            agent = proxy
        manifest = run_episode(spec, agent, out_dir, sim=sim)
    finally:
        if proxy is not None:
            proxy.close()
        sim.close()

    reconstruction = manifest.get("reconstruction") or {}
    frames = reconstruction.get("frames", manifest["captures"])
    print(
        "episode complete: %d reconstruction frames, %d agent observations, "
        "%.1fs sim time, %.1fm path, mean distractor fraction %.4f"
        % (
            reconstruction.get("num_frames", manifest["num_captures"]),
            manifest["num_captures"],
            manifest["clock"]["final_sim_time"],
            manifest["clock"]["path_length_m"],
            sum(frame["distractor_pixel_fraction"] for frame in frames)
            / max(1, len(frames)),
        )
    )
    print("output: %s" % args.out)


if __name__ == "__main__":
    main()
