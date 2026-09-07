"""Check the companion report's numerical result tables against the release CSVs."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def tables(markdown):
    result, current = [], []
    for line in markdown.splitlines() + [""]:
        if line.startswith("|"):
            values = [v.strip() for v in line.strip().strip("|").split("|")]
            if not all(set(v) <= set(":- ") for v in values):
                current.append(values)
        elif current:
            result.append(current)
            current = []
    return result


def audit(report, root):
    rows = list(csv.DictReader((root / "results.csv").open()))
    pairs = list(csv.DictReader((root / "pairs.csv").open()))
    tabs = tables(report.read_text())
    checks = []
    def find(header):
        found = [t for t in tabs if t[0] == header]
        if len(found) != 1:
            raise ValueError(f"Expected one result table with header {header}")
        return found[0][1:]
    def method(label):
        return label.lower().split(",")[0].strip()
    def mean(rs, key):
        return sum(float(r[key]) for r in rs) / len(rs)
    def check(label, actual, value, spec):
        normalized = actual.replace("−", "-").strip()
        expected = format(value, spec)
        ok = value < float(normalized[1:]) if normalized.startswith("<") else normalized == expected
        checks.append(dict(label=label, actual=actual, expected=expected, unrounded=value, ok=ok))
    static = lambda m: [r for r in rows if r["group"] == "gs" and r["condition"] == "d0" and r["method"] == m]
    for c in find(["Method", "n", "Mission (s)", "PSNR shared (dB)", "PSNR cube, provisional (dB)", "Completeness@5cm (%)"]):
        m = method(c[0]); rs = static(m)
        if c[1] != f"{len(rs)}/2":
            raise ValueError(f"Wrong completion count: {m}")
        for j,key,spec,factor in [(2,"configured_budget_s",".0f",1),(3,"psnr_shared",".2f",1),(4,"psnr_cube_provisional",".2f",1),(5,"historical_completeness_5cm",".1f",100)]:
            check(m+" "+key,c[j],factor*mean(rs,key),spec)
    for c in find(["Method", "0007 shared", "0044 shared", "0007 cube, provisional", "0044 cube, provisional"]):
        m = method(c[0]); rs = static(m)
        for j,(scene,key) in enumerate([("interior_0007","psnr_shared"),("interior_0044","psnr_shared"),("interior_0007","psnr_cube_provisional"),("interior_0044","psnr_cube_provisional")],1):
            check(m+" "+scene+" "+key,c[j],float(next(r for r in rs if r["scene"]==scene)[key]),".2f")
    for c in find(["Method", "Mean path (m)", "Mean decision captures", "Reconstruction frames", "Mean planning wall time (s)"]):
        m = method(c[0]); rs = static(m)
        for j,key,spec in [(1,"path_length_m",".1f"),(2,"decision_captures",".1f"),(3,"reconstruction_frames",".0f"),(4,"planning_wall_time_s",".1f")]:
            check(m+" "+key,c[j],mean(rs,key),spec)
    for c in find(["Scene", "r3con-pano", "MAGICIAN", "FisherRF", "GAVIS", "Random"]):
        for j,m in enumerate(["r3con-pano","magician","fisherrf","gavis","random"],1):
            rs=[r for r in rows if r["scene"]==c[0] and r["method"]==m and r["condition"]=="d0"]
            if rs:
                check(c[0]+" "+m,c[j],float(rs[0]["psnr_shared"]),".2f")
            elif c[j] not in ("—", "–", "N/A", "Missing"):
                raise ValueError(f"Missing cell filled in report: {c[0]} {m}")
    for c in find(["Method", "Δsevere (dB)", "Δclean (dB)", "Regional contrast (dB)", "Mean path change (%)"]):
        m = method(c[0]); rs=[r for r in pairs if r["group"]=="gs" and r["method"]==m]
        for j,key in [(1,"delta_severe"),(2,"delta_clean"),(3,"regional_contrast"),(4,"path_change_percent")]:
            check(m+" "+key,c[j],mean(rs,key),"+.1f" if j==4 else "+.2f")
    if len(checks) != 115:
        raise ValueError(f"Incomplete report tables: {len(checks)} numeric cells; expected 115")
    return checks


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--report",type=Path,required=True,help="Companion report Markdown file")
    ap.add_argument("--root",type=Path,default=ROOT / "phase1")
    ap.add_argument("--out",type=Path,default=ROOT / "outputs/report-table-check.json")
    args=ap.parse_args()
    checks=audit(args.report,args.root)
    failures=[c for c in checks if not c["ok"]]
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(dict(numeric_cells=len(checks),checks=checks,failures=failures),indent=2)+"\n")
    print(f"{len(checks)} numerical report cells; {len(failures)} mismatches")
    if failures:
        for row in failures: print(row)
        raise SystemExit(1)


if __name__=="__main__":
    main()
