"""Re-render a recorded episode at a new temporal rate and/or resolution.

The planned trajectory is NOT re-run: poses come from the episode's recorded
1 Hz stream (verbatim at recorded times, EpisodeReplay's interpolation in
between — the same trajectory definition the viewer replays), and the scene
is re-simulated from the episode config, whose distractor scripts are
deterministic. Resolution changes preserve aspect ratio and horizontal FOV.

Output is an episode-mirror directory (patched manifest with provenance,
frames/ symlink, copied transforms) so retrain_eval / aggregate_campaign /
the replay viewer work on it unchanged. The agent never saw the extra
frames: manifest reconstruction.streamed_to_agent is set false.

Correctness gates: at every recorded time the resampled pose must match the
recording, and re-rendered anchor frames must match the stored pixels.

Run in the habitat env:
    conda run -n habitat python scripts/resample_stream.py \
        --episode runs_campaign_v4/mp3d_17DRP5sb8fy__d0__s0/r3con-pano \
        --config configs/bench/campaign_v4/mp3d_17DRP5sb8fy__d0__s0.yaml \
        --rate 1 --width 1280 --height 960 \
        --out runs_ablation_resolution_v1/1280x960/mp3d_17DRP5sb8fy__d0__s0/r3con-pano
"""

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Optional

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))


def resample_grid(t_end: float, rate: float):
    """Uniform sample times 0..t_end inclusive at ``rate`` Hz."""

    count = int(round(t_end * rate))
    return [k / rate for k in range(count + 1)]


def _verify_frame_decodes(path: Path, expected_rgb: np.ndarray, index: int, t: float,
                          attempts: int = 3) -> None:
    """Read back a saved RGB PNG and confirm it decodes to the source array.

    Catches silent disk corruption that survives the byte-level fsync check in
    save_rgb_png (e.g. SSD wear-leveling returning bad blocks days later, or
    mid-flight page-cache issues that produce decodable-but-wrong bytes). The
    decoder is PIL — the same loader retrain_eval uses — so any OSError raised
    here would otherwise have crashed gsplat 30k training minutes later.
    """

    from PIL import Image

    # save_rgb_png accepts HxWx3 or HxWx4 but always writes the first 3 channels.
    expected = np.asarray(expected_rgb)[..., :3]
    if expected.dtype != np.uint8:
        expected = np.clip(expected, 0, 255).astype(np.uint8)

    last_error: Optional[Exception] = None
    for attempt in range(attempts):
        try:
            decoded = np.asarray(Image.open(path))[..., :3]
        except (OSError, ValueError) as exc:
            last_error = exc
        else:
            if decoded.shape == expected.shape and np.array_equal(decoded, expected):
                return
            last_error = IOError(
                "decoded RGB for frame %d (t=%.1f) does not match what we wrote "
                "(shape %s vs %s)" % (
                    index, t, decoded.shape, expected.shape,
                )
            )
        # Re-write the file with a fresh block allocation and try again.
        from activebench.common.image import save_rgb_png

        save_rgb_png(path, expected)
    raise IOError(
        "frame %d (t=%.1f) at %s failed decode-verification after %d attempts: %s"
        % (index, t, path, attempts, last_error)
    )


def patched_manifest(
    source_manifest,
    reconstruction_log,
    rate,
    transforms_name,
    source_episode,
    resolution=None,
    source_resolution=None,
):
    """Episode manifest for the resampled mirror, with provenance."""

    manifest = dict(source_manifest)
    reconstruction = {
        "policy": "uniform-time-resampled",
        "interval_s": 1.0 / rate,
        "streamed_to_agent": False,
        "agent_interval_s": (source_manifest.get("reconstruction") or {}).get("interval_s"),
        "num_frames": len(reconstruction_log),
        "frames": reconstruction_log,
        "transforms": transforms_name,
        "resampled_from": str(source_episode),
        "source_interval_s": (source_manifest.get("reconstruction") or {}).get("interval_s"),
    }
    if resolution is not None:
        reconstruction["resolution"] = {
            "width": int(resolution[0]), "height": int(resolution[1])
        }
    if source_resolution is not None:
        reconstruction["source_resolution"] = {
            "width": int(source_resolution[0]), "height": int(source_resolution[1])
        }
    manifest["reconstruction"] = reconstruction
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", required=True, help="source episode dir (recorded stream)")
    parser.add_argument("--config", required=True, help="episode config yaml (scene + distractor scripts)")
    parser.add_argument("--rate", type=float, required=True, help="target sampling rate in Hz")
    parser.add_argument("--width", type=int, default=None, help="target RGB-D width")
    parser.add_argument("--height", type=int, default=None, help="target RGB-D height")
    parser.add_argument("--out", required=True, help="output episode-mirror dir")
    args = parser.parse_args()

    from activebench.episode import EpisodeSpec
    from activebench.replay import EpisodeReplay
    from activebench.runner import _save_mask_png
    from activebench.sim import DynamicSceneSim
    from activebench.common.transforms_export import export_transforms_json, frame_from_pose
    from activebench.common.image import save_rgb_png
    from activebench.common.io import ensure_dir, save_json

    source = Path(args.episode).resolve()
    out_dir = ensure_dir(Path(args.out)).resolve()
    if out_dir == source:
        sys.exit("--out must differ from --episode")

    replay = EpisodeReplay(source)
    if (args.width is None) != (args.height is None):
        parser.error("--width and --height must be provided together")
    target_width = args.width or replay.intrinsics.width
    target_height = args.height or replay.intrinsics.height
    if target_width <= 0 or target_height <= 0:
        parser.error("target width and height must be positive")
    source_aspect = replay.intrinsics.width / replay.intrinsics.height
    target_aspect = target_width / target_height
    if abs(source_aspect - target_aspect) > 1e-9:
        parser.error(
            "target resolution must preserve source aspect ratio %.6f" % source_aspect
        )
    recorded = {round(f.time / (1.0 / args.rate)): f for f in replay.frames}
    spec = EpisodeSpec.from_yaml(Path(args.config))
    source_hfov = float(np.rad2deg(
        2.0 * np.arctan(replay.intrinsics.width / (2.0 * replay.intrinsics.fx))
    ))
    spec.scene.habitat.width = target_width
    spec.scene.habitat.height = target_height
    spec.scene.habitat.hfov = source_hfov
    sim = DynamicSceneSim(spec.scene)
    try:
        ensure_dir(out_dir / "stream")
        times = resample_grid(replay.t_end, args.rate)
        frames, log = [], []
        for index, t in enumerate(times):
            grid_key = round(t * args.rate)
            anchor = recorded.get(grid_key)
            if anchor is not None and abs(anchor.time - t) > 1e-6:
                anchor = None
            pose = anchor.pose if anchor is not None else replay.camera_pose_at(t)
            if anchor is not None:
                # The interpolator must be pinned at recorded times.
                drift = np.linalg.norm(
                    np.asarray(replay.camera_pose_at(t).position) - np.asarray(pose.position))
                if drift > 1e-6:
                    sys.exit("interpolation drifts %.2e m off the recorded pose at t=%.1f" % (drift, t))
            result = sim.observe(pose, t, include_mask=True)
            if (
                anchor is not None
                and target_width == replay.intrinsics.width
                and target_height == replay.intrinsics.height
            ):
                from PIL import Image

                stored = np.asarray(Image.open(anchor.rgb_path))[..., :3]
                diff = np.abs(stored.astype(np.int16) - result["rgb"][..., :3].astype(np.int16)).mean()
                if diff > 1.0:
                    sys.exit(
                        "re-rendered frame at t=%.1f deviates from the recording "
                        "(mean |diff| %.2f gray levels); sim/world mismatch" % (t, diff))
            rgb_rel = "stream/frame_%05d.png" % index
            depth_rel = "stream/depth_%05d.npy" % index
            mask_rel = "stream/mask_%05d.png" % index
            save_rgb_png(out_dir / rgb_rel, result["rgb"])
            np.save(out_dir / depth_rel, result["depth"])
            _save_mask_png(out_dir / mask_rel, result["distractor_mask"])
            _verify_frame_decodes(out_dir / rgb_rel, result["rgb"], index, t)
            frame = frame_from_pose(pose, rgb_rel, depth_path=depth_rel)
            frame["time"] = float(t)
            frame["mask_path"] = mask_rel
            frames.append(frame)
            log.append({
                "index": index,
                "time": float(t),
                "pose": pose.as_list(),
                "distractor_pixel_fraction": float(result["distractor_mask"].mean()),
            })
    finally:
        sim.close()

    transforms_path = export_transforms_json(
        out_dir, sim.intrinsics, frames, filename="transforms_stream.json")
    source_manifest = json.loads((source / "manifest.json").read_text())
    manifest = patched_manifest(
        source_manifest,
        log,
        args.rate,
        transforms_path.name,
        source,
        resolution=(target_width, target_height),
        source_resolution=(replay.intrinsics.width, replay.intrinsics.height),
    )
    save_json(out_dir / "manifest.json", manifest)

    frames_link = out_dir / "frames"
    if not frames_link.exists():
        frames_link.symlink_to(source / "frames")
    for name in ("transforms.json", "coverage.json"):
        if (source / name).exists() and not (out_dir / name).exists():
            shutil.copy2(source / name, out_dir / name)

    print(
        "resampled %d frames at %.1f Hz, %dx%d -> %s"
        % (len(frames), args.rate, target_width, target_height, out_dir)
    )


if __name__ == "__main__":
    main()
