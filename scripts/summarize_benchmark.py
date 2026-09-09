"""Summarize fresh generic campaigns; never write the frozen Phase 1 tables."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from activebench.configuration import digest, write_json
from activebench.audit import world_fingerprint


def load_rows(runs_dir, scenes=None):
    rows, worlds, regimes, methods = [], {}, set(), {}
    recorded_worlds = {}
    for path in sorted(Path(runs_dir).glob("*/*/benchmark-run.json")):
        record = json.loads(path.read_text())
        if scenes and record["scene"] not in scenes:
            continue
        config = record["config"]
        world = (record["scene"], record["condition"], record["seed"])
        identity = digest({"config": config, "references": record.get("reference_assets")})
        if world in worlds and worlds[world] != identity:
            raise ValueError("methods saw different episode configs for %s" % (world,))
        worlds[world] = identity
        method_identity = digest(record.get("method"))
        if record["alias"] in methods and methods[record["alias"]] != method_identity:
            raise ValueError("method alias %s has different recipes/source versions; use distinct aliases" % record["alias"])
        methods[record["alias"]] = method_identity
        regimes.add(digest({"protocol": config["campaign"]["protocol"], "motion": config.get("motion"),
            "episode": config["episode"], "camera": {k: config["habitat"].get(k) for k in ("width", "height", "hfov")},
            "recipe": record["recipe"], "platform_code": record.get("platform_code")}))
        directory = path.parent
        status_path = directory / "benchmark-status.json"
        status = json.loads(status_path.read_text()) if status_path.exists() else {}
        row = {"scene": record["scene"], "condition": record["condition"], "seed": record["seed"],
               "method": record["alias"], "status": "complete" if status.get("ok") else "failed/incomplete",
               "error": status.get("error", ""), "protocol": config["campaign"]["protocol"]}
        manifest = directory / "manifest.json"
        if manifest.exists():
            data = json.loads(manifest.read_text())
            fingerprint = world_fingerprint(data)
            if world in recorded_worlds and recorded_worlds[world] != fingerprint:
                raise ValueError("recorded worlds differ across methods for %s" % (world,))
            recorded_worlds[world] = fingerprint
            row.update(agent_observations=data["num_captures"], sim_seconds=data["clock"]["final_sim_time"],
                       planning_wall_seconds=data["clock"].get("planning_wall_time_s"),
                       path_m=data["clock"].get("path_length_m"), pose_access=data["method"].get("pose_access", "gt"))
            frames = (data.get("reconstruction") or {}).get("frames", data["captures"])
            row["reconstruction_frames"] = len(frames)
            row["distractor_pixel_fraction"] = statistics.mean(f["distractor_pixel_fraction"] for f in frames) if frames else None
        coverage = directory / "coverage.json"
        if coverage.exists():
            row["coverage"] = json.loads(coverage.read_text())["overall"]["observed_frac"]
        evaluation = directory / "reconstructions" / record["recipe"]["run_name"] / "eval.json"
        if evaluation.exists() and status.get("ok"):
            data = json.loads(evaluation.read_text())
            for stratum, scores in data.get("appearance_per_stratum", {}).items():
                for metric in ("psnr", "ssim", "lpips"):
                    if metric in scores:
                        row[metric if stratum == "all" else metric + "_" + stratum] = scores[metric]
            geometry = data.get("geometry", {})
            row["completeness_at_5cm"] = geometry.get("completeness@0.05")
            row["accuracy_median_m"] = geometry.get("accuracy", {}).get("median")
        rows.append(row)
    if not rows:
        raise ValueError("no generic campaign receipts found (historical tables use scripts/phase1/report.py)")
    if len(regimes) > 1:
        raise ValueError("mixed motion/camera/budget/reconstruction/evaluation protocols; select one group with --scenes or separate run roots")
    return rows


def summarize(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["scene"], row["condition"], row["method"])].append(row)
    summaries = []
    for (scene, condition, method), cells in sorted(grouped.items()):
        values = [r["psnr"] for r in cells if r.get("psnr") is not None and r["status"] == "complete"]
        summaries.append({"scene": scene, "condition": condition, "method": method,
                          "complete": sum(r["status"] == "complete" for r in cells), "planned": len(cells),
                          "psnr_n": len(values), "psnr_mean": statistics.mean(values) if values else None,
                          "psnr_std": statistics.stdev(values) if len(values) > 1 else None})
    pairs = []
    indexed = {(r["scene"], r["seed"], r["method"], r["condition"]): r for r in rows}
    for key, clean in sorted(indexed.items()):
        if key[-1] != "d0":
            continue
        dynamic = indexed.get((*key[:3], "dyn"))
        if dynamic and all(r["status"] == "complete" and r.get("psnr") is not None for r in (clean, dynamic)):
            pairs.append({"scene": key[0], "seed": key[1], "method": key[2], "d0_psnr": clean["psnr"],
                          "dyn_psnr": dynamic["psnr"], "dyn_minus_d0_psnr": dynamic["psnr"] - clean["psnr"]})
    return summaries, pairs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs-dir", type=Path, required=True)
    ap.add_argument("--scenes", nargs="+")
    ap.add_argument("--out-dir", type=Path, help="default: <runs-dir>/summary")
    args = ap.parse_args()
    try:
        rows = load_rows(args.runs_dir, args.scenes)
    except ValueError as exc:
        ap.error(str(exc))
    summaries, pairs = summarize(rows)
    out = args.out_dir or args.runs_dir / "summary"
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "summary.json", {"cells": rows, "by_scene_method_condition": summaries, "paired_static_dynamic": pairs})
    with (out / "results.csv").open("w", newline="") as stream:
        fields = sorted({k for r in rows for k in r})
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# Fresh benchmark results", "", "Each row summarizes seeds within one scene. Std is sample standard deviation; n=1 has no std.",
             "Failed or incomplete cells remain in the denominator. Acquisition-only runs have no reconstruction PSNR.", "",
             "| Scene | Condition | Method | Complete / planned | PSNR mean ± std (n) |",
             "|---|---|---|---:|---:|"]
    for row in summaries:
        score = "—" if row["psnr_mean"] is None else "%.3f%s (%d)" % (
                row["psnr_mean"], " ± %.3f" % row["psnr_std"] if row["psnr_std"] is not None else "", row["psnr_n"])
        lines.append("| %s | %s | %s | %d / %d | %s |" % (row["scene"], row["condition"], row["method"], row["complete"], row["planned"], score))
    lines += ["", "Per-cell metrics and matched d0/dyn deltas are in summary.json. These are new runs, not frozen report evidence.", ""]
    (out / "RESULTS.md").write_text("\n".join(lines))
    print(out / "RESULTS.md")


if __name__ == "__main__":
    main()
