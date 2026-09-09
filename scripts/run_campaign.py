"""Run a configurable ActiveBench campaign and its shared evaluation pipeline.

Use --configs-dir, --methods and --out-dir; inspect the plan, then add --execute.
See docs/RUNNING.md. Frozen report reproduction remains scripts/phase1/run.py.
The stage helpers are shared by the generic and historical campaign launchers.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional, Tuple

import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from activebench.runtime import conda_python, external_repo

HABITAT_PY = conda_python("habitat")
# Gaussian-splat (.gs.ply) stages render in the habitat-gs env; mesh scenes use
# habitat. Run this launcher itself in habitat-gs for a GS campaign so the
# in-process surface-sample build (ensure_surface_samples) can render GS too.
HABITAT_GS_PY = conda_python("habitat-gs")
GAVIS_PY = conda_python("gavis")


def _sim_python(config_path: Path) -> str:
    """Pick the sim env for one episode: habitat-gs for GS stages, else habitat."""
    scene = str(
        yaml.safe_load(Path(config_path).read_text()).get("habitat", {}).get("scene_path", "")
    )
    return HABITAT_GS_PY if scene.endswith((".gs.ply", ".3dgs.ply")) else HABITAT_PY

METHODS = ["random", "r3con-pano", "gavis", "fisherrf", "magician"]
# Campaign budgets: deliberately not the smoke settings. gavis/fisherrf
# retrain per round (800 its balances fidelity and a ~day-scale campaign);
# magician uses its published beam 10x10.
METHOD_OPTIONS = {
    "gavis": {"train_iterations": 800},
    "fisherrf": {"train_iterations": 800},
    "magician": {"beam_width": 10, "beam_steps": 10},
    # Upstream ships three 40k-step checkpoints and recommends the stage-2 run
    # that excludes the 96 Gibson scenes as "more robust and stable overall".
    # On interior_0007/d0 that is not a nuance: it explores for the whole 300 s
    # mission (47 observations, 90.6 m), while plain stage 2 stalls at 143 s
    # and stage 1 at 89 s — a deterministic argmax goal that keeps landing on
    # an occupied cell leaves the agent re-planning the same move forever.
    "gleam": {
        "checkpoint": str(
            Path(external_repo("GLEAM")) / "ckpt/train_gleam_stage2_wo_gibson_40000000_steps.zip"
        )
    },
}
EPISODE_TIMEOUT_S = 3 * 3600
RECONSTRUCTION_BACKEND = "gsplat"
RETRAIN_ITERATIONS = 30_000
BENCHEVAL_PY = conda_python("bencheval")
# The standard backend runs in the method-independent bencheval env; the
# gavis-repo-backed backends remain for reproducing historical rounds.
BACKEND_PYTHON = {
    "gsplat": BENCHEVAL_PY,
}


def _is_transient_episode_crash(
    episode_dir: Path,
    wall_s: float,
    max_wall_s: float,
    max_frames: int,
) -> bool:
    """Classify whether a failed episode is a transient early crash.

    Returns False (do NOT retry) when:
      - manifest.json exists (the episode actually succeeded);
      - wall_s >= max_wall_s (ran long enough to be a real method failure).
    Returns True (retry) when the episode produced fewer than max_frames —
    including when frames/ was never created: crashes before the first frame
    (agent-worker startup OOM while the previous cell's retrain VRAM is still
    being reclaimed) are the MOST transient class, not the least.
    """
    if (episode_dir / "manifest.json").exists():
        return False
    if wall_s >= max_wall_s:
        return False
    frames_dir = episode_dir / "frames"
    if not frames_dir.exists():
        return True
    num_frames = len(list(frames_dir.glob("frame_*.png")))
    return num_frames < max_frames


def _wait_for_gpu_settle(min_free_gib: float = 15.0, timeout_s: float = 60.0) -> None:
    """Best-effort poll until a GPU has enough free VRAM, or timeout."""
    print(
        "[campaign] GPU settle: waiting for >= %.0f GiB free (timeout %ds)"
        % (min_free_gib, timeout_s),
        flush=True,
    )
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            raw = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                text=True,
                timeout=5,
            )
            free_mib = int(raw.strip().split("\n")[0])
            if free_mib >= min_free_gib * 1024:
                print(
                    "[campaign] GPU settle: %.0f GiB free; proceeding" % (free_mib / 1024),
                    flush=True,
                )
                return
        except (subprocess.SubprocessError, ValueError, IndexError, FileNotFoundError):
            print("[campaign] WARN nvidia-smi unavailable, sleeping 5s", flush=True)
            time.sleep(5)
            return
        time.sleep(2)
    print("[campaign] WARN GPU settle timed out after %.0fs" % timeout_s, flush=True)


def parse_resolution(value: str) -> Tuple[int, int]:
    """Parse a 'WIDTHxHEIGHT' string like retrain_eval.py does."""
    try:
        width, height = (int(part) for part in value.lower().split("x", 1))
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError("resolution must be WIDTHxHEIGHT")
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("resolution dimensions must be positive")
    return width, height


def _find_build_lod(explicit: Optional[Path] = None) -> Optional[Path]:
    """Locate the Spark ``build-lod`` binary.

    Order: explicit arg → PATH → sibling of this Python → sibling of
    bencheval Python → any miniconda env.
    """
    if explicit is not None:
        if explicit.exists():
            return explicit
        print(f"[campaign] WARN: --build-lod {explicit} not found, PLY-only export (no RAD)")
        return None
    from shutil import which

    found = which("build-lod")
    if found:
        return Path(found)
    candidates = [
        Path(sys.executable).parent / "build-lod",
        Path(BENCHEVAL_PY).parent / "build-lod",
    ]
    candidates += list(Path.home().glob("miniconda3/envs/*/bin/build-lod"))
    for cand in candidates:
        if cand.exists():
            return cand
    return None


def ensure_surface_samples(config_path: Path, scene_name: str) -> Path:
    out = _REPO_ROOT / "eval_assets" / "surface" / ("%s.npz" % scene_name)
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    print("[campaign] building GT surface samples for %s" % scene_name, flush=True)
    from activebench.episode import EpisodeSpec
    from activebench.eval.surface_samples import SurfaceSampleConfig, build_surface_samples
    from activebench.sim import DynamicSceneSim

    payload = yaml.safe_load(config_path.read_text())
    payload["distractors"] = []  # clean scene for GT
    spec = EpisodeSpec.from_dict(payload)
    sim = DynamicSceneSim(spec.scene)
    try:
        samples = build_surface_samples(sim, SurfaceSampleConfig())
    finally:
        sim.close()
    samples.save(out)
    print("[campaign] %s: %d surface samples" % (scene_name, len(samples.points)), flush=True)
    return out


def run_episode(
    config_path: Path,
    method: str,
    episode_dir: Path,
    samples_path: Path,
    *,
    retries: int = 1,
    retry_cooldown_s: float = 30.0,
    retry_max_wall_s: float = 120.0,
    retry_max_frames: int = 5,
    agent_name: Optional[str] = None,
    agent_options: Optional[dict] = None,
    agent_env: Optional[str] = None,
    agent_python: Optional[str] = None,
) -> Tuple[bool, int]:
    """Run one (config, method) episode, with retry for transient early crashes.

    Returns (success, total_attempts).
    """
    if (episode_dir / "manifest.json").exists():
        return True, 1

    episode_dir.parent.mkdir(parents=True, exist_ok=True)
    options = dict(METHOD_OPTIONS.get(method, {}) if agent_options is None else agent_options)
    if (agent_name or method) == "magician":
        options["surface_samples_path"] = str(samples_path)
        options["max_captures"] = int(
            yaml.safe_load(config_path.read_text())["episode"]["max_captures"]
        )
    cmd = [
        _sim_python(config_path), str(_REPO_ROOT / "scripts/run_benchmark.py"),
        "--config", str(config_path),
        "--agent", agent_name or method,
        "--out", str(episode_dir),
    ]
    if options:
        cmd += ["--agent-options", json.dumps(options)]
    if agent_env:
        cmd += ["--agent-env", agent_env]
    if agent_python:
        cmd += ["--agent-python", agent_python]

    log_path = episode_dir.parent / ("%s.launcher.log" % method)
    wall_s = 0.0
    last_returncode = -1
    # A 30k-iteration 1600x1200 retrain in the previous cell peaks near the
    # whole GPU; give its memory a moment to be reclaimed before the next
    # method worker starts, or its startup allocation can OOM in seconds.
    _wait_for_gpu_settle()
    for attempt in range(1, retries + 1):
        attempt_start = time.monotonic()
        # Always append: a rerun must never overwrite the previous run's
        # crash evidence. The header separates invocations.
        with open(log_path, "ab") as log:
            log.write(
                ("\n===== campaign attempt %d @ %s =====\n"
                 % (attempt, time.strftime("%Y-%m-%d %H:%M:%S"))).encode()
            )
            log.flush()
            proc = subprocess.run(
                cmd, stdout=log, stderr=subprocess.STDOUT, timeout=EPISODE_TIMEOUT_S,
            )
        wall_s = time.monotonic() - attempt_start
        ok = proc.returncode == 0 and (episode_dir / "manifest.json").exists()
        if ok:
            print(
                "[campaign] %-32s %-10s OK (%.0fs, %d attempt%s)"
                % (episode_dir.parent.name, method, wall_s, attempt,
                   "s" if attempt > 1 else ""),
                flush=True,
            )
            return True, attempt

        last_returncode = proc.returncode
        is_transient = _is_transient_episode_crash(
            episode_dir, wall_s, retry_max_wall_s, retry_max_frames,
        )
        if not is_transient or attempt >= retries:
            break

        print(
            "[campaign] retry %d/%d %s after transient crash (wall %.0fs, frames %d)"
            % (attempt, retries, method, wall_s,
               len(list((episode_dir / "frames").glob("frame_*.png")))),
            flush=True,
        )
        _wait_for_gpu_settle()
        time.sleep(retry_cooldown_s)

    print(
        "[campaign] %-32s %-10s FAILED(rc=%d) (%.0fs, %d attempt%s)"
        % (episode_dir.parent.name, method, last_returncode,
           wall_s, attempt, "s" if attempt > 1 else ""),
        flush=True,
    )
    return False, attempt


def run_coverage(episode_dir: Path, samples_path: Path) -> None:
    if (episode_dir / "coverage.json").exists():
        return
    import numpy as np

    from activebench.eval.coverage import evaluate_episode_dir

    samples = np.load(samples_path)
    evaluate_episode_dir(episode_dir, samples["points"], samples["normals"])


def _assert_stream_resolution(episode_dir: Path, width: int, height: int) -> None:
    """Refuse to train on a stream that is not at the requested resolution.

    This is the hard gate behind the resample step: a silently skipped or
    failed resample must never produce a model trained at the planner
    resolution and passed off as comparable (it costs ~6x fewer pixels).
    """
    ts_path = episode_dir / "transforms_stream.json"
    if not ts_path.exists():
        raise RuntimeError("no transforms_stream.json in %s" % episode_dir)
    payload = json.loads(ts_path.read_text())
    cur = (payload.get("w"), payload.get("h"))
    if cur != (width, height):
        raise RuntimeError(
            "stream in %s is %sx%s but the campaign trains at %dx%d; "
            "resample it first" % (episode_dir, cur[0], cur[1], width, height)
        )


def run_retrain(
    episode_dir: Path,
    samples_path: Path,
    *,
    backend: str,
    run_name: Optional[str],
    iterations: int,
    shared_dir: str,
    backend_options: str = "{}",
    expected_resolution: Optional[Tuple[int, int]] = None,
    secondary_eval_resolution: Optional[str] = None,
) -> None:
    run_name = run_name or backend
    output_dir = episode_dir / "reconstructions" / run_name
    if (output_dir / "eval.json").exists():
        return
    if expected_resolution is not None:
        _assert_stream_resolution(episode_dir, *expected_resolution)
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        BACKEND_PYTHON[backend], str(_REPO_ROOT / "scripts/retrain_eval.py"),
        "--samples", str(samples_path),
        "--backend", backend,
        "--run-name", run_name,
        "--iterations", str(iterations),
        "--backend-options", backend_options,
        "--shared-dir", shared_dir,
        "--save-renders",
    ]
    if secondary_eval_resolution:
        cmd += ["--eval-resolution", secondary_eval_resolution]
    cmd.append(str(episode_dir))
    log_path = output_dir / "train.log"
    env = dict(os.environ)
    # Headroom against fragmentation at 1600x1200 gaussian counts; measured
    # to cut reserved-unallocated from GiBs to hundreds of MiBs.
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    with open(log_path, "wb") as log:
        proc = subprocess.run(
            cmd, stdout=log, stderr=subprocess.STDOUT, timeout=3 * 3600, env=env,
        )
    if proc.returncode != 0 or not (output_dir / "eval.json").exists():
        raise RuntimeError(
            "retrain failed for %s (exit %d); see %s"
            % (episode_dir, proc.returncode, log_path)
        )


def run_resample_stream(
    config_path: Path,
    episode_dir: Path,
    *,
    width: int,
    height: int,
    rate: float = 1.0,
) -> None:
    """Re-render the recorded stream at reconstruction resolution.

    The episode's ``frames/`` (planner input at 640x480) is never touched.
    """
    ts_path = episode_dir / "transforms_stream.json"
    if ts_path.exists():
        ts_data = json.loads(ts_path.read_text())
        cur_w = ts_data.get("w", ts_data.get("width"))
        cur_h = ts_data.get("h", ts_data.get("height"))
        if cur_w == width and cur_h == height:
            print(
                "[campaign] stream resample skip %s (already %dx%d)"
                % (episode_dir.name, width, height),
                flush=True,
            )
            return

    # Remember the old planner resolution for provenance.
    cfg_payload = yaml.safe_load(config_path.read_text())
    old_w = cfg_payload.get("habitat", {}).get("width", 640)
    old_h = cfg_payload.get("habitat", {}).get("height", 480)

    tmp_mirror = None
    try:
        tmp_mirror = Path(
            tempfile.mkdtemp(prefix=".resample_", dir=episode_dir.parent)
        )
        log_path = episode_dir.parent / ("%s.resample.log" % episode_dir.name)
        cmd = [
            # Re-rendering is sim work, so it needs the scene's own sim env:
            # a GS (.gs.ply) stage cannot be rasterized by plain habitat.
            _sim_python(config_path),
            str(_REPO_ROOT / "scripts/resample_stream.py"),
            "--episode", str(episode_dir),
            "--config", str(config_path),
            "--rate", str(rate),
            "--width", str(width),
            "--height", str(height),
            "--out", str(tmp_mirror),
        ]
        with open(log_path, "wb") as log:
            proc = subprocess.run(
                cmd, stdout=log, stderr=subprocess.STDOUT, timeout=3600
            )
        if proc.returncode != 0:
            raise RuntimeError(
                "resample_stream exited %d; see %s" % (proc.returncode, log_path)
            )
        if not (tmp_mirror / "stream").exists():
            raise RuntimeError(
                "resample_stream produced no stream/; see %s" % log_path
            )

        # Replace the stream directory in-place.
        old_stream = episode_dir / "stream"
        if old_stream.exists():
            shutil.move(str(old_stream), str(episode_dir / "stream.bak"))
        shutil.move(str(tmp_mirror / "stream"), str(episode_dir / "stream"))
        shutil.move(
            str(tmp_mirror / "transforms_stream.json"),
            str(episode_dir / "transforms_stream.json"),
        )
        # Clean up backup only after everything is confirmed in place
        old_bak = episode_dir / "stream.bak"
        if old_bak.exists():
            shutil.rmtree(old_bak)

        # Patch provenance into manifest.
        manifest_path = episode_dir / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            reconstruction = manifest.setdefault("reconstruction", {})
            reconstruction["resolution"] = {"width": width, "height": height}
            reconstruction["planner_resolution"] = {"width": old_w, "height": old_h}
            _tmp = manifest_path.with_suffix(".json.tmp")
            _tmp.write_text(json.dumps(manifest, indent=2))
            os.replace(str(_tmp), str(manifest_path))

        print(
            "[campaign] stream resample %s %dx%d -> %dx%d"
            % (episode_dir.name, old_w, old_h, width, height),
            flush=True,
        )
    finally:
        if tmp_mirror is not None and tmp_mirror.exists():
            shutil.rmtree(tmp_mirror, ignore_errors=True)


def run_extra_eval(
    episode_dir: Path,
    samples_path: Path,
    *,
    backend: str,
    run_name: str,
    eval_set: Path,
    suffix: str,
) -> None:
    """Score an already-trained cell against an additional pose set.

    Scoring runs on the exported gaussians.npz, so re-scoring it here yields
    exactly what the training pass recorded (verified to full float precision)
    -- the extra metric needs no retraining and no second training session.
    The result lands in its own run dir, ``<run_name><suffix>``, hard-linked to
    the same npz so the primary run's artifacts are never touched.
    """

    src = episode_dir / "reconstructions" / run_name / "gaussians.npz"
    if not src.exists():
        print("[campaign] extra eval skipped, no %s" % src, flush=True)
        return
    out_run = "%s%s" % (run_name, suffix)
    out_dir = episode_dir / "reconstructions" / out_run
    if (out_dir / "eval.json").exists():
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / "gaussians.npz"
    if not dst.exists():
        try:
            os.link(src, dst)
        except OSError:
            shutil.copyfile(src, dst)
    cmd = [
        BACKEND_PYTHON[backend], str(_REPO_ROOT / "scripts/retrain_eval.py"),
        str(episode_dir), "--samples", str(samples_path), "--backend", backend,
        "--run-name", out_run, "--eval-only", "--eval-set", str(eval_set),
    ]
    log_path = episode_dir.parent / ("%s.extraeval.log" % episode_dir.name)
    with open(log_path, "ab") as log:
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, timeout=3600)
    if proc.returncode != 0:
        print("[campaign] WARN extra eval failed for %s - see %s"
              % (episode_dir, log_path), flush=True)


def run_spark_export(
    episode_dir: Path,
    run_name: str,
    build_lod: Optional[Path] = None,
) -> None:
    """Export versioned standard PLY and optional RAD without discarding SH.

    Reuse only matching source/export provenance. Failure markers also include
    the builder identity, so an older converter's failure cannot suppress a
    fixed export on a resumed campaign.
    """
    from activebench.web_export import (
        build_rad_from_ply,
        current_campaign_rad,
        rad_provenance,
        read_sh_degree,
        splats_npz_to_ply,
        write_rad_provenance,
    )

    npz = episode_dir / "reconstructions" / run_name / "gaussians.npz"
    if not npz.exists():
        return
    degree = read_sh_degree(npz)
    ply_path = npz.with_suffix(".ply")
    rad_path = ply_path.with_name(ply_path.stem + "-lod.rad")
    marker = rad_path.with_name(rad_path.name + ".failed")
    signature = {"model": rad_provenance(npz, degree), "builder": None}
    if build_lod is not None:
        builder = Path(build_lod).resolve()
        stat = builder.stat()
        signature["builder"] = {
            "path": str(builder), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
        }

    if current_campaign_rad(npz, rad_path, degree):
        print("[campaign] spark export skip %s (current RAD exists)" % episode_dir.name,
              flush=True)
        return
    try:
        previous_failure = json.loads(marker.read_text())
    except (OSError, ValueError):
        previous_failure = {}
    if build_lod is not None and previous_failure.get("signature") == signature:
        print("[campaign] spark export skip %s (same converter failed; remove %s to retry)"
              % (episode_dir.name, marker.name), flush=True)
        return

    # The provenance schema applies to both encodings. Pre-v3 PLYs can have
    # incorrect SH ordering even when newer than their NPZ.
    count = "cached"
    if not current_campaign_rad(npz, ply_path, degree):
        count, degree = splats_npz_to_ply(npz, ply_path)
        write_rad_provenance(npz, ply_path, degree)
    if build_lod is None:
        print("[campaign] spark export %s: %s gaussians sh%d PLY only"
              % (episode_dir.name, count, degree), flush=True)
        return

    try:
        build_rad_from_ply(ply_path, build_lod, sh_degree=degree)
        write_rad_provenance(npz, rad_path, degree)
    except Exception as exc:
        marker.write_text(json.dumps({"signature": signature, "error": str(exc)}))
        raise
    marker.unlink(missing_ok=True)
    print("[campaign] spark export %s: %s gaussians sh%d PLY %s RAD %s"
          % (episode_dir.name, count, degree, ply_path.name, rad_path.name), flush=True)


def main() -> None:
    from activebench.campaign import main as campaign_main
    campaign_main()


if __name__ == "__main__":
    main()
