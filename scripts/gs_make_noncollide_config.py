#!/usr/bin/env python3
"""Generate an activebench-friendly non-collidable habitat-gs dataset config.

A Gaussian-splat (``.gs.ply``) stage carries no collision mesh, so enabling
physics — which activebench's :class:`DynamicSceneConfig` forces on for its
kinematic distractors — makes Habitat try to build a stage collision mesh and
hard-aborts (``getCollisionMesh`` assertion). Marking the stages non-collidable
via a dataset-config default lets Bullet skip stage collision. Distractors are
kinematic and the agent navigates via the navmesh, so no stage collision
geometry is needed.

Usage::

    python scripts/gs_make_noncollide_config.py \
        --gs-dir /path/to/habitat-gs/data/scene_datasets/gs_scenes

Writes ``{split}_activebench_noncollide.scene_dataset_config.json`` next to each
original ``{split}.scene_dataset_config.json``. Point an episode config's
``habitat.scene_dataset_config_file`` at the generated file.
"""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gs-dir",
        required=True,
        type=Path,
        help="habitat-gs gs_scenes directory holding {split}.scene_dataset_config.json",
    )
    parser.add_argument("--splits", nargs="*", default=["train", "val"])
    args = parser.parse_args()

    wrote = 0
    for split in args.splits:
        src = args.gs_dir / f"{split}.scene_dataset_config.json"
        if not src.exists():
            print(f"skip {split}: {src} not found")
            continue
        cfg = json.loads(src.read_text())
        stages = cfg.setdefault("stages", {})
        stages["default_attributes"] = {"is_collidable": False}
        out = args.gs_dir / f"{split}_activebench_noncollide.scene_dataset_config.json"
        out.write_text(json.dumps(cfg, indent=2))
        print(f"wrote {out}")
        wrote += 1
    if wrote == 0:
        raise SystemExit("no dataset configs found under %s" % args.gs_dir)


if __name__ == "__main__":
    main()
