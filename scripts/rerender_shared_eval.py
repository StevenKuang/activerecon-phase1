"""Re-render an existing shared-eval catalog at one canonical resolution.

Camera poses, strata, and occlusion labels are copied verbatim. Only clean
Habitat RGB-D observations and their pinhole intrinsics change, so models
trained at different input resolutions can be compared against the same
high-resolution target.

Run in the habitat environment:
    python scripts/rerender_shared_eval.py \
        --source-dir eval_assets/shared_eval_v4 \
        --configs-dir configs/bench/campaign_v4 \
        --out-dir eval_assets/shared_eval_resolution_v1/1600x1200 \
        --width 1600 --height 1200
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))


def config_for_group(configs_dir: Path, group_name: str) -> Path:
    scene, seed = group_name.rsplit("__", 1)
    path = configs_dir / ("%s__d0__%s.yaml" % (scene, seed))
    if not path.exists():
        raise FileNotFoundError("clean config not found for %s: %s" % (group_name, path))
    return path


def target_hfov_degrees(payload, width: int, height: int) -> float:
    source_width, source_height = int(payload["w"]), int(payload["h"])
    if abs(source_width / source_height - width / height) > 1e-9:
        raise ValueError(
            "target %dx%d must preserve source aspect ratio %dx%d"
            % (width, height, source_width, source_height)
        )
    return float(np.rad2deg(
        2.0 * np.arctan(source_width / (2.0 * float(payload["fl_x"])))
    ))


def rerender_group(
    source_group: Path,
    config_path: Path,
    out_group: Path,
    width: int,
    height: int,
    overwrite: bool,
) -> int:
    from activebench.episode import EpisodeSpec
    from activebench.sim import DynamicSceneSim
    from activebench.common.camera import CameraPose
    from activebench.common.image import save_rgb_png
    from activebench.common.io import ensure_dir, save_json
    from activebench.common.transforms import rotation_to_yaw_pitch

    output_transforms = out_group / "transforms_eval_shared.json"
    if output_transforms.exists() and not overwrite:
        payload = json.loads(output_transforms.read_text())
        print("cached %s: %d views" % (out_group.name, len(payload["frames"])))
        return len(payload["frames"])

    source_transforms = source_group / "transforms_eval_shared.json"
    payload = json.loads(source_transforms.read_text())
    spec = EpisodeSpec.from_yaml(config_path)
    spec.scene.distractors = []
    spec.scene.habitat.width = width
    spec.scene.habitat.height = height
    spec.scene.habitat.hfov = target_hfov_degrees(payload, width, height)

    ensure_dir(out_group / "gt")
    frames = []
    sim = DynamicSceneSim(spec.scene)
    intrinsics = sim.intrinsics
    try:
        for index, source_frame in enumerate(payload["frames"]):
            matrix = np.asarray(source_frame["transform_matrix"], dtype=np.float64)
            yaw, pitch = rotation_to_yaw_pitch(matrix[:3, :3])
            pose = CameraPose.from_xyz_yaw_pitch(matrix[:3, 3], yaw=yaw, pitch=pitch)
            if not np.allclose(pose.as_matrix(), matrix, atol=1e-6):
                raise ValueError("shared eval pose %d is not a roll-free benchmark pose" % index)
            observation = sim.render_clean(pose)
            rgb_rel = "gt/eval_%04d.png" % index
            depth_rel = "gt/eval_depth_%04d.npy" % index
            save_rgb_png(out_group / rgb_rel, observation["rgb"])
            np.save(out_group / depth_rel, observation["depth"].astype(np.float32))
            frame = dict(source_frame)
            frame["file_path"] = rgb_rel
            frame["depth_path"] = depth_rel
            frames.append(frame)
    finally:
        sim.close()

    output = dict(payload)
    output.update({
        "w": intrinsics.width,
        "h": intrinsics.height,
        "fl_x": float(intrinsics.fx),
        "fl_y": float(intrinsics.fy),
        "cx": float(intrinsics.cx),
        "cy": float(intrinsics.cy),
        "frames": frames,
        "resolution_provenance": {
            "source_set": str(source_group.resolve()),
            "source_resolution": [int(payload["w"]), int(payload["h"])],
            "rendered_resolution": [intrinsics.width, intrinsics.height],
            "poses_and_labels": "copied verbatim",
        },
    })
    save_json(output_transforms, output)
    print("rendered %s: %d views at %dx%d" % (
        out_group.name, len(frames), intrinsics.width, intrinsics.height
    ))
    return len(frames)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--configs-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--scenes", nargs="*", default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.width <= 0 or args.height <= 0:
        parser.error("width and height must be positive")
    if args.source_dir.resolve() == args.out_dir.resolve():
        parser.error("--out-dir must differ from --source-dir")

    groups = sorted(
        path.parent
        for path in args.source_dir.glob("*/transforms_eval_shared.json")
    )
    if args.scenes:
        wanted = set(args.scenes)
        groups = [group for group in groups if group.name.rsplit("__", 1)[0] in wanted]
    if not groups:
        parser.error("no shared-eval groups selected")

    total = 0
    for source_group in groups:
        config = config_for_group(args.configs_dir, source_group.name)
        total += rerender_group(
            source_group,
            config,
            args.out_dir / source_group.name,
            args.width,
            args.height,
            args.overwrite,
        )
    print("complete: %d groups, %d views -> %s" % (
        len(groups), total, args.out_dir
    ))


if __name__ == "__main__":
    main()
