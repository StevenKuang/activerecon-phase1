"""Run the frozen Phase 1 campaign, retaining separate mesh and GS groups."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from string import Template

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from activebench.runtime import conda_python, external_repo


def resolve(value):
    env = dict(os.environ)
    env.setdefault("HABITAT_SIM_ROOT", str(Path.home() / "Projects/habitat-sim"))
    env.setdefault("HABITAT_GS_ROOT", str(Path.home() / "Projects/habitat-gs"))
    for name in ("R3CON", "MAGICIAN", "FISHERRF", "GAVIS", "GLEAM"):
        dirname = {"FISHERRF": "FisherRF", "GAVIS": "gavis"}.get(name, name)
        env.setdefault(name + "_REPO", external_repo(dirname))
    if isinstance(value, str):
        return Template(value).substitute(env)
    if isinstance(value, dict):
        return {k: resolve(v) for k, v in value.items()}
    if isinstance(value, list):
        return [resolve(v) for v in value]
    return value


def cell_config(cell, campaign_dir, smoke_seconds=None):
    import yaml
    cfg = resolve(yaml.safe_load((campaign_dir / cell["config"]).read_text()))
    cfg["episode"]["max_sim_time"] = smoke_seconds or cell["configured_budget_s"]
    return cfg


def worker(cell, args):
    import yaml
    spec = importlib.util.spec_from_file_location("phase1_campaign", ROOT / "scripts/run_campaign.py")
    campaign = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(campaign)
    run = args.out / cell["id"]
    run.mkdir(parents=True, exist_ok=True)
    config = cell_config(cell, args.campaign.parent, args.smoke_seconds)
    config_path = run / "episode.yaml"
    encoded = yaml.safe_dump(config, sort_keys=False)
    if config_path.exists() and config_path.read_text() != encoded:
        raise RuntimeError(f"Configuration changed for {run}; use a new output root")
    config_path.write_text(encoded)
    samples = args.data / "surface" / (cell["scene"] + ".npz")
    if not samples.exists():
        raise FileNotFoundError(f"Frozen surface reference missing: {samples}; restore the evaluation archive")
    options = resolve(cell["method_options"])
    receipt = dict(cell=cell["id"], config=config, method_options=options,
                   iterations=args.iterations, reconstruction_name=args.reconstruction_name,
                   data=str(args.data), smoke=bool(args.smoke_seconds))
    receipt_path = run / "phase1-run.json"
    if receipt_path.exists() and json.loads(receipt_path.read_text()) != receipt:
        raise RuntimeError(f"Recipe changed for {run}; use a new output root")
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    campaign.METHOD_OPTIONS[cell["method"]] = options
    if args.stage in ("all", "acquire"):
        ok, _ = campaign.run_episode(config_path, cell["method"], run, samples, retries=1)
        if not ok:
            raise RuntimeError(f"Acquisition failed: {run}")
    if args.stage in ("all", "reconstruct"):
        if not (run / "manifest.json").exists():
            raise FileNotFoundError(f"No acquisition at {run}")
        campaign.run_resample_stream(config_path, run, width=1600, height=1200, rate=1.0)
        campaign.run_coverage(run, samples)
        campaign.run_retrain(run, samples, backend="gsplat", run_name=args.reconstruction_name,
            iterations=args.iterations, shared_dir=str(args.data / "eval" / cell["group"]),
            expected_resolution=(1600, 1200), secondary_eval_resolution=None)
    if args.stage in ("all", "export"):
        campaign.run_spark_export(run, args.reconstruction_name, build_lod=campaign._find_build_lod())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--campaign", type=Path, default=ROOT / "phase1/campaign.json")
    ap.add_argument("--data", type=Path, default=ROOT / "data/phase1")
    ap.add_argument("--out", type=Path, default=ROOT / "runs_phase1")
    ap.add_argument("--group", choices=["mesh", "gs"])
    ap.add_argument("--cell", action="append")
    ap.add_argument("--stage", choices=["all", "acquire", "reconstruct", "export"], default="all")
    ap.add_argument("--execute", action="store_true", help="Execute the displayed plan")
    ap.add_argument("--include-missing", action="store_true", help="Attempt the four historically missing/excluded cells too")
    ap.add_argument("--smoke-seconds", type=float, help="Small validation only, requires a separate output root")
    ap.add_argument("--iterations", type=int, default=30000)
    ap.add_argument("--reconstruction-name", default="gsplat")
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()
    args.campaign, args.data, args.out = args.campaign.resolve(), args.data.resolve(), args.out.resolve()
    if args.smoke_seconds is not None and args.smoke_seconds < 3:
        ap.error("smoke duration must be at least 3 seconds")
    if args.iterations != 30000 and args.smoke_seconds is None:
        ap.error("non-standard training iterations require --smoke-seconds")
    if args.smoke_seconds and args.out == (ROOT / "runs_phase1").resolve():
        ap.error("smoke runs require an explicit separate --out")
    cells = [c for c in json.loads(args.campaign.read_text())["cells"]
             if (not args.group or c["group"] == args.group)
             and (not args.cell or c["id"] in args.cell)
             and (args.include_missing or c["status"] == "retained")]
    if not cells:
        ap.error("selection contains no eligible cells")
    if args.cell and set(args.cell) - {c["id"] for c in cells}:
        ap.error("unknown or excluded cell requested")
    if args.worker:
        if len(cells) != 1:
            ap.error("worker requires one cell")
        worker(cells[0], args)
        return
    failures = []
    for cell in cells:
        command = [conda_python("habitat-gs" if cell["group"] == "gs" else "habitat"),
            str(Path(__file__).resolve()), "--campaign", str(args.campaign), "--data", str(args.data),
            "--out", str(args.out), "--cell", cell["id"], "--stage", args.stage,
            "--iterations", str(args.iterations), "--reconstruction-name", args.reconstruction_name, "--worker"]
        if args.include_missing:
            command += ["--include-missing"]
        if args.smoke_seconds:
            command += ["--smoke-seconds", str(args.smoke_seconds)]
        print(json.dumps(dict(cell=cell["id"], historical_status=cell["status"],
            budget_s=args.smoke_seconds or cell["configured_budget_s"],
            method_options=resolve(cell["method_options"]), command=command)), flush=True)
        if args.execute:
            code = subprocess.run(command, cwd=ROOT).returncode
            if code:
                failures.append(dict(cell=cell["id"], exit_code=code))
    if args.execute:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "failures.json").write_text(json.dumps(failures, indent=2) + "\n")
    print(f"{len(cells)} cells; {'executed' if args.execute else 'plan only'}; {len(failures)} failures")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
