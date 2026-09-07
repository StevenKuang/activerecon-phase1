"""Verify the selected runtime paths and required imports without changing them."""
import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from activebench.runtime import conda_python

IMPORTS = {
    "habitat": ["habitat_sim", "numpy", "PIL", "yaml"],
    "habitat-gs": ["habitat_sim", "torch", "numpy", "PIL", "yaml"],
    "bencheval": ["torch", "gsplat", "PIL", "scipy", "lpips"],
    "r3con": ["torch", "diff_gauss", "diff_gaussian_rasterization_2d", "open3d"],
    "magician": ["torch", "pytorch3d", "diff_gaussian_rasterization"],
    "fisherrf": ["torch", "modified_diff_gaussian_rasterization", "simple_knn"],
    "gavis": ["torch", "diff_gaussian_rasterization", "gavis_rasterizer", "nerfacc"],
    "gleam": ["torch", "numpy", "PIL"],
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--environment", choices=sorted(IMPORTS), action="append",
                    help="Check only this conda environment; repeat as needed")
    args = ap.parse_args()
    results = {}
    for name, modules in IMPORTS.items():
        if args.environment and name not in args.environment:
            continue
        code = "import importlib,json,sys; mods=" + repr(modules) + "; [importlib.import_module(x) for x in mods]; print(json.dumps({'python':sys.version,'imports':mods}))"
        try:
            proc = subprocess.run([conda_python(name), "-c", code], capture_output=True, text=True, timeout=90)
            results[name] = dict(ok=proc.returncode == 0, stdout=proc.stdout.strip(), stderr=proc.stderr[-1500:])
        except Exception as exc:
            results[name] = dict(ok=False, error=str(exc))
        print(name, "OK" if results[name]["ok"] else "FAILED", flush=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2) + "\n")
    if not all(r["ok"] for r in results.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
