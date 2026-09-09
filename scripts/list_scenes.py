"""Discover installed dataset scenes without loading a simulator or a campaign."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from activebench.scene_datasets import load_scene_catalog, select_scene_candidates, inventory


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--catalog", type=Path, default=ROOT / "configs/datasets.yaml")
    ap.add_argument("--data-root", type=Path, required=True, help="Habitat checkout's data directory")
    ap.add_argument("--datasets", nargs="+")
    ap.add_argument("--split", default="all")
    args = ap.parse_args()
    catalog = load_scene_catalog(args.catalog)
    if args.datasets:
        print(json.dumps(select_scene_candidates(catalog, args.data_root.resolve(),
                         datasets=args.datasets, split=args.split), indent=2))
    else:
        for item in inventory(catalog, args.data_root.resolve()):
            print("%-16s %4d scenes  %s" % (item.name, item.scene_count, item.root))


if __name__ == "__main__":
    main()
