"""General campaign planning, immutable run receipts and sequential execution.

No historical cell roster is consulted. A campaign is selected episode YAMLs
crossed with method configurations, using independently prepared references.
"""

import argparse
import json
from pathlib import Path
import sys
import time

import yaml

from activebench import registry
from activebench.configuration import (cell_identity, digest, file_digest, input_file_hashes, load_yaml,
                                       reference_scene, safe_name, write_json)
from activebench.plugins import normalize_factory
from activebench.runtime import conda_python


ROOT = Path(__file__).resolve().parents[2]


def resolve_methods(names, path, overrides=None):
    catalog = load_yaml(path) if path else {"schema_version": 1, "methods": {}}
    if catalog.get("schema_version") != 1 or not isinstance(catalog.get("methods"), dict):
        raise ValueError("method config needs schema_version: 1 and methods: mapping")
    result = {}
    for name in names:
        safe_name(name)
        entry = dict(catalog["methods"].get(name, {"agent": name}))
        unknown = set(entry) - {"agent", "environment", "python", "options", "version"}
        if unknown:
            raise ValueError("unknown method settings for %s: %s" % (name, sorted(unknown)))
        agent = entry.get("agent", name)
        if ":" in agent:
            module = agent.rsplit(":", 1)[0]
            if module.endswith(".py") and not Path(module).is_absolute() and path:
                agent = str(Path(path).resolve().parent / module) + ":" + agent.rsplit(":", 1)[1]
            agent = normalize_factory(agent)
        env = entry.get("environment", registry.default_env(agent))
        python = entry.get("python")
        if python and "environment" in entry:
            raise ValueError("method %s: choose python or environment" % name)
        options = entry.get("options", {})
        if not isinstance(options, dict):
            raise ValueError("method options must be a mapping")
        options = {**options, **(overrides or {}).get(name, {})}
        # These are benchmark inputs, not tunable method hyperparameters.
        reserved = {"seed", "scene_bbox", "start_pose", "width", "height", "hfov",
                    "max_captures", "max_sim_time", "candidate_positions", "surface_samples_path"}
        if reserved & set(options):
            raise ValueError("benchmark-owned method options cannot be overridden: %s" % sorted(reserved & set(options)))
        result[name] = {"agent": agent, "environment": None if python else env,
                        "python": str(Path(python).expanduser().resolve()) if python else None,
                        "options": options, "version": entry.get("version")}
        if ":" in agent and agent.rsplit(":", 1)[0].endswith(".py"):
            result[name]["source_sha256"] = file_digest(agent.rsplit(":", 1)[0])
    extra = set(overrides or {}) - set(names)
    if extra:
        raise ValueError("method overrides do not match selection: %s" % sorted(extra))
    return result


def select_configs(directory, scenes=None, conditions=None, seeds=None):
    selected, seen = [], set()
    for path in sorted(Path(directory).glob("*.yaml")):
        payload = load_yaml(path)
        scene, condition, seed = cell_identity(payload)
        if scenes and scene not in scenes or conditions and condition not in conditions or seeds is not None and seed not in seeds:
            continue
        cell = "%s__%s__s%d" % (scene, condition, seed)
        if cell in seen:
            raise ValueError("duplicate scene/condition/seed: %s" % cell)
        seen.add(cell)
        selected.append((cell, path.resolve(), payload))
    if not selected:
        raise ValueError("no episode configs match the selection")
    identities = [cell_identity(item[2]) for item in selected]
    for requested, index, label in ((scenes, 0, "scenes"), (conditions, 1, "conditions"), (seeds, 2, "seeds")):
        missing = set(requested or []) - {i[index] for i in identities}
        if missing:
            raise ValueError("requested %s absent from selected configs: %s" % (label, sorted(missing)))
    return selected


def validate_assets(directory, selected):
    record = json.loads((directory / "assets.json").read_text())
    if record.get("schema_version") != 1:
        raise ValueError("unsupported reference assets schema")
    for rel, expected in record["files"].items():
        path = (directory / rel).resolve()
        if not path.is_relative_to(directory.resolve()) or file_digest(path) != expected:
            raise ValueError("reference asset changed or escaped its directory: %s" % rel)
    file_cache = {}
    for _, _, payload in selected:
        scene, _, seed = cell_identity(payload)
        group = "%s__s%d" % (scene, seed)
        entry = record["groups"].get(group)
        if not entry or entry["scene_identity"] != digest(reference_scene(payload)):
            raise ValueError("references do not match scene/camera/start for %s" % group)
        if "scene_files" in entry and entry["scene_files"] != input_file_hashes(dict(payload, distractors=[]), file_cache):
            raise ValueError("scene asset bytes changed since reference preparation: %s" % group)
        for rel in ("surface/%s.npz" % scene, "eval/%s/transforms_eval_shared.json" % group):
            if rel not in record["files"]:
                raise ValueError("reference manifest omits required file: %s" % rel)
    return record


def check_resume(directory, identity):
    receipt = directory / "benchmark-run.json"
    if receipt.exists():
        old = json.loads(receipt.read_text())
        if old.get("identity") != identity:
            raise ValueError("run configuration/code/assets changed at %s; use a new --out-dir" % directory)
    elif directory.exists() and any(directory.iterdir()):
        raise ValueError("untracked existing output at %s; use a new --out-dir" % directory)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--configs-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--methods", nargs="+", default=["random"])
    ap.add_argument("--method-config", type=Path, default=ROOT / "configs/methods.yaml")
    ap.add_argument("--method-options", default="{}", help='JSON mapping alias to option overrides')
    ap.add_argument("--scenes", nargs="+")
    ap.add_argument("--conditions", "--difficulties", dest="conditions", nargs="+")
    ap.add_argument("--seeds", nargs="+", type=int)
    ap.add_argument("--assets-dir", type=Path, help="immutable output of prepare_evaluation.py")
    ap.add_argument("--acquisition-only", "--skip-retrain", dest="acquisition_only", action="store_true")
    ap.add_argument("--reconstruction-resolution", default="1600x1200")
    ap.add_argument("--retrain-iterations", type=int, default=30000)
    ap.add_argument("--reconstruction-options", default="{}")
    ap.add_argument("--reconstruction-run-name", default="gsplat")
    ap.add_argument("--episode-retries", type=int, default=1)
    ap.add_argument("--export-spark", action="store_true", help="also convert final model to browser PLY/RAD")
    ap.add_argument("--execute", action="store_true", help="launch the printed plan")
    args = ap.parse_args()
    sys.path.insert(0, str(ROOT / "scripts"))
    import run_campaign as stages
    try:
        selected = select_configs(args.configs_dir, args.scenes, args.conditions, args.seeds)
        methods = resolve_methods(list(dict.fromkeys(args.methods)), args.method_config, json.loads(args.method_options))
        width, height = map(int, args.reconstruction_resolution.lower().split("x"))
        if min(width, height, args.retrain_iterations, args.episode_retries) <= 0:
            raise ValueError("resolution, iterations and retries must be positive")
        safe_name(args.reconstruction_run_name)
        reconstruction_options = json.loads(args.reconstruction_options)
        if not isinstance(reconstruction_options, dict):
            raise ValueError("reconstruction options must be a JSON object")
        if not args.acquisition_only and not args.assets_dir:
            raise ValueError("full pipeline needs --assets-dir from prepare_evaluation.py")
        if any(m["agent"] == "magician" for m in methods.values()) and not args.assets_dir:
            raise ValueError("MAGICIAN needs --assets-dir for its declared reference-surface input")
        for cell, _, payload in selected:
            episode = payload.get("episode", {})
            if not args.acquisition_only and episode.get("reconstruction_interval") != 1.0:
                raise ValueError("the common pipeline requires reconstruction_interval: 1.0 (%s)" % cell)
        assets_dir = args.assets_dir.resolve() if args.assets_dir else None
        assets = validate_assets(assets_dir, selected) if assets_dir else None
        recipe = {"acquisition_only": args.acquisition_only, "reconstruction_resolution": [width, height],
                  "iterations": args.retrain_iterations, "backend": "gsplat", "options": reconstruction_options,
                  "run_name": args.reconstruction_run_name,
                  "reference_recipe": assets["recipe"] if assets else None}
        code_files = sorted((ROOT / "src/activebench").rglob("*.py"))
        code_files += [ROOT / "scripts" / (name + ".py") for name in
                       ("run_benchmark", "run_campaign", "retrain_eval", "resample_stream")]
        code = digest({str(p.relative_to(ROOT)): file_digest(p) for p in code_files})
        file_cache = {}
        plan = []
        for cell, source, payload in selected:
            scene, condition, seed = cell_identity(payload)
            for name, method in methods.items():
                directory = args.out_dir.resolve() / cell / name
                material = {"config": payload, "method": method, "recipe": recipe,
                            "reference_assets": digest(assets) if assets else None, "platform_code": code,
                            "simulation_files": input_file_hashes(payload, file_cache)}
                identity = digest(material)
                check_resume(directory, identity)
                plan.append({"cell": cell, "scene": scene, "condition": condition, "seed": seed,
                             "alias": name, "source": str(source), "out": str(directory),
                             "identity": identity, **material})
    except (ValueError, KeyError, OSError, TypeError) as exc:
        ap.error(str(exc))
    print("%d configs x %d methods = %d cells" % (len(selected), len(methods), len(plan)), flush=True)
    for item in plan:
        print("  %s / %s -> %s" % (item["cell"], item["alias"], item["out"]), flush=True)
    if not args.execute:
        print("Plan only. Add --execute to run. Use --acquisition-only to collect without reconstruction.")
        return
    # Fail before the first cell when an interpreter is missing. Import/CUDA
    # checks are available separately through scripts/doctor.py.
    interpreters = {stages._sim_python(source) for _, source, _ in selected}
    if not args.acquisition_only:
        interpreters.add(conda_python("bencheval"))
    for method in methods.values():
        if method["python"]:
            interpreters.add(method["python"])
        elif method["environment"] not in (None, "inline"):
            interpreters.add(conda_python(method["environment"]))
    for executable in interpreters:
        if not Path(executable).is_file():
            ap.error("missing interpreter: %s (see docs/SETUP.md)" % executable)
    failures, completed = [], []
    # Persist the entire planned denominator before the first expensive cell.
    # An interrupted process must not make unstarted cells disappear.
    for item in plan:
        directory = Path(item["out"])
        directory.mkdir(parents=True, exist_ok=True)
        write_json(directory / "benchmark-run.json", item)
        config = directory / "episode.yaml"
        config.write_text(yaml.safe_dump(item["config"], sort_keys=False))
    write_json(args.out_dir / "campaign-plan.json", {"cells": [i["cell"] + "/" + i["alias"] for i in plan]})
    for item in plan:
        directory = Path(item["out"])
        config = directory / "episode.yaml"
        samples = assets_dir / "surface" / (item["scene"] + ".npz") if assets_dir else None
        started = time.time()
        try:
            method = item["method"]
            ok, attempts = stages.run_episode(config, item["alias"], directory, samples,
                            retries=args.episode_retries, agent_name=method["agent"],
                            agent_options=method["options"], agent_env=method["environment"],
                            agent_python=method["python"])
            if not ok:
                raise RuntimeError("acquisition failed; inspect %s.launcher.log and agent_worker.log" % item["alias"])
            if samples is not None:
                stages.run_coverage(directory, samples)
            if not args.acquisition_only:
                stages.run_resample_stream(config, directory, width=width, height=height)
                stages.run_retrain(directory, samples, backend="gsplat", run_name=args.reconstruction_run_name,
                        iterations=args.retrain_iterations, shared_dir=str(assets_dir / "eval"),
                        backend_options=json.dumps(reconstruction_options), expected_resolution=(width, height))
                if args.export_spark:
                    stages.run_spark_export(directory, args.reconstruction_run_name, build_lod=stages._find_build_lod())
            status = {"ok": True, "wall_seconds": time.time() - started, "attempts": attempts}
            completed.append(item["cell"] + "/" + item["alias"])
        except Exception as exc:
            status = {"ok": False, "wall_seconds": time.time() - started, "error": str(exc)}
            failures.append({"cell": item["cell"], "method": item["alias"], "error": str(exc)})
            print("FAILED %s/%s: %s" % (item["cell"], item["alias"], exc), flush=True)
        write_json(directory / "benchmark-status.json", status)
    write_json(args.out_dir / "campaign-status.json", {"planned": len(plan), "completed": completed, "failures": failures})
    print("Finished: %d completed, %d failed. Summarize with scripts/summarize_benchmark.py." % (len(completed), len(failures)))
    if failures:
        raise SystemExit(1)
