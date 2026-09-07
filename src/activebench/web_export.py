"""Static web-demo export: replay bundle + 3DGS PLY for browser splat renderers.

Two decoupled asset groups, exported side by side (docs/results/
2026-07-17-spark2-viewer-survey.md):

- the *replay bundle* — poses/timeline, progressive point-cloud chunks,
  capture thumbnails, distractor tracks — is per-(episode, method) process
  data;
- the *reconstruction assets* — one standard 3DGS PLY per backend — are the
  results a browser renderer (Spark et al.) loads independently, so the demo
  page can switch reconstructions without touching the replay clock.

Full SH coefficients are exported in Spark/INRIA's channel-major PLY layout.
Pure array/dict helpers stay import-light (numpy only) for
headless tests; I/O lives in :func:`export_episode_bundle` and the CLI.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import warnings
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

SH_C0 = 0.28209479177387814

# v6's canonical reconstruction precedes historical recipes and score aliases.
RECON_PREFERENCE = ("gsplat1600", "gsplat", "vanilla-3dgs-gsplat", "vanilla-3dgs", "legacy")
RECONSTRUCTION_EXPORT_VERSION = 3

_PLY_PROPS = (
    "x", "y", "z", "nx", "ny", "nz",
    "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
    "scale_0", "scale_1", "scale_2",
    "rot_0", "rot_1", "rot_2", "rot_3",
)


def _sh_rest_props(sh_degree: int) -> Tuple[str, ...]:
    """Standard 3DGS PLY property names for SH bands above DC.

    ``f_rest_{channel * K + coefficient}``: all R coefficients, then G, then B.
    Spark v2.1.0's rust/spark-lib/src/ply.rs ``f_rest_name`` uses this order;
    the in-memory gsplat array instead has shape (N, K, 3).
    """

    coeffs = (sh_degree + 1) ** 2 - 1
    return tuple("f_rest_%d" % k for k in range(coeffs * 3))


def ply_from_gaussians(
    centers: np.ndarray,
    scales: np.ndarray,
    quats_wxyz: np.ndarray,
    opacities: np.ndarray,
    colors: np.ndarray,
    sh_rest: Optional[np.ndarray] = None,
    sh_degree: int = 0,
    sh0: Optional[np.ndarray] = None,
) -> bytes:
    """Standard 3DGS binary PLY from raw gaussian arrays.

    Inverts the activations 3DGS readers apply: colors -> DC coefficients,
    linear scales -> log, opacities -> logits; clamps keep degenerate
    gaussians (scale 0, opacity 1) finite instead of propagating inf.

    When ``sh_rest`` is provided with ``sh_degree > 0``, appends the
    ``f_rest_*`` properties matching the standard 3DGS layout so Spark's
    LoD builder and other SH-aware renderers can evaluate view-dependent
    color. ``sh_rest`` must have shape ``(N, K, 3)`` where
    ``K = (sh_degree+1)**2 - 1``.
    """

    n = len(centers)
    opacities = np.clip(np.asarray(opacities, np.float64).reshape(-1), 1e-6, 1 - 1e-6)
    quats = np.asarray(quats_wxyz, np.float64)
    norms = np.linalg.norm(quats, axis=1, keepdims=True)
    if not np.all(np.isfinite(norms)) or np.any(norms <= 0):
        raise ValueError("Gaussian rotations must be finite nonzero quaternions")
    quats = quats / norms

    sh_rest_props: Tuple[str, ...] = ()
    sh_rest_flat: Optional[np.ndarray] = None
    if sh_degree > 0 and sh_rest is not None:
        coeffs = (sh_degree + 1) ** 2 - 1
        arr = np.asarray(sh_rest, np.float32)
        if arr.shape != (n, coeffs, 3):
            raise ValueError(
                "sh_rest must have shape (%d, %d, 3) for sh_degree=%d, got %s"
                % (n, coeffs, sh_degree, arr.shape)
            )
        sh_rest_flat = arr.transpose(0, 2, 1).reshape(n, coeffs * 3)
        sh_rest_props = _sh_rest_props(sh_degree)

    props = _PLY_PROPS + sh_rest_props
    out = np.empty((n, len(props)), dtype=np.float32)
    out[:, 0:3] = centers
    out[:, 3:6] = 0.0  # normals: unused by 3DGS, present for reader compat
    if sh0 is not None:
        dc = np.asarray(sh0, np.float32)
        if dc.shape not in ((n, 3), (n, 1, 3)):
            raise ValueError("sh0 must have shape (N, 3) or (N, 1, 3)")
        out[:, 6:9] = dc.reshape(n, 3)
    else:
        out[:, 6:9] = (np.asarray(colors, np.float64) - 0.5) / SH_C0
    out[:, 9] = np.log(opacities / (1.0 - opacities))
    out[:, 10:13] = np.log(np.clip(np.asarray(scales, np.float64), 1e-9, None))
    out[:, 13:17] = quats
    if sh_rest_flat is not None:
        out[:, 17:] = sh_rest_flat

    header = "\n".join(
        ["ply", "format binary_little_endian 1.0", "element vertex %d" % n]
        + ["property float %s" % p for p in props]
        + ["end_header", ""]
    )
    return header.encode("ascii") + out.tobytes()


def splats_npz_to_ply(
    npz_path: Path,
    ply_path: Path,
    sh_degree: Optional[int] = None,
) -> Tuple[int, int]:
    """Convert a saved ``gaussians.npz`` to a 3DGS PLY.

    Returns ``(gaussian_count, sh_degree_used)``. When the NPZ stores
    ``sh_rest`` (gsplat SH3 backends do), the PLY preserves the full SH3
    coefficients unless ``sh_degree`` is explicitly overridden. A SH0-only
    NPZ falls back to the legacy DC-only PLY so older artifacts stay
    readable without retraining.
    """

    with np.load(npz_path) as data:
        centers = data["centers"]
        scales = data["scales"]
        quats_wxyz = data["quats_wxyz"]
        opacities = data["opacities"]
        colors = data["colors"]
        sh_rest = data["sh_rest"] if "sh_rest" in data.files else None
        sh0 = data["sh0"] if "sh0" in data.files else None
        recorded_degree = int(data["sh_degree"]) if "sh_degree" in data.files else 0
        count = len(centers)
    if sh_degree is None:
        sh_degree = recorded_degree
        if sh_degree == 0 and sh_rest is not None:
            # NPZ predates the explicit sh_degree field but carries
            # coefficients — infer the degree from their count.
            sh_degree = int(np.sqrt(sh_rest.shape[1] + 1)) - 1
    payload = ply_from_gaussians(
        centers, scales, quats_wxyz, opacities, colors,
        sh_rest=sh_rest, sh_degree=sh_degree, sh0=sh0,
    )
    Path(ply_path).write_bytes(payload)
    return count, sh_degree


def build_rad_from_ply(
    ply_path: Path,
    build_lod: Path,
    sh_degree: int = 0,
) -> Path:
    """Run Spark's standalone ``build-lod`` and return its fixed output path."""

    ply_path = Path(ply_path).resolve()
    output = ply_path.with_name(ply_path.stem + "-lod.rad")
    output.unlink(missing_ok=True)
    # Tiny-LoD keeps the base model and SH, and avoids the quality builder's
    # large-tree crashes seen on v6. Prefer it for reliable paged delivery.
    for method in ("--quick", "--quality"):
        try:
            subprocess.run(
                [
                    str(Path(build_lod).resolve()), str(ply_path),
                    method, "--max-sh=%d" % sh_degree,
                ],
                cwd=ply_path.parent, check=True,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            )
            break
        except subprocess.CalledProcessError as exc:
            if exc.returncode == -2:
                raise KeyboardInterrupt from exc
            if method == "--quick":
                # Both builders preserve the requested SH. Never silently
                # replace the model's SH3 appearance with SH0.
                warnings.warn(
                    "Spark quick LoD failed; retrying quality LoD with SH%d preserved"
                    % sh_degree, RuntimeWarning)
                output.unlink(missing_ok=True)
                continue
            tail = "\n".join((exc.stdout or "").strip().splitlines()[-20:])
            raise RuntimeError("build-lod failed with exit code %d\n%s" % (
                exc.returncode, tail)) from exc
    if not output.exists():
        raise RuntimeError("build-lod did not create %s" % output)
    return output


def pack_point_chunks(
    chunks: Iterable[Tuple[np.ndarray, np.ndarray]],
) -> Tuple[bytes, List[Dict[str, int]]]:
    """Concatenate (points float32 Nx3, colors uint8 Nx3) chunks into one blob.

    Per chunk: positions then colors, padded so every chunk offset stays
    4-byte aligned for zero-copy Float32Array views in the browser.
    """

    parts: List[bytes] = []
    index: List[Dict[str, int]] = []
    offset = 0
    for points, colors in chunks:
        n = len(points)
        blob = (
            np.ascontiguousarray(points, dtype=np.float32).tobytes()
            + np.ascontiguousarray(colors, dtype=np.uint8).tobytes()
        )
        pad = (-len(blob)) % 4
        parts.append(blob + b"\0" * pad)
        index.append({"offset": offset, "count": n})
        offset += len(blob) + pad
    return b"".join(parts), index


def sample_tracks(replay, dt: float = 0.2) -> List[Dict]:
    """Distractor pose tracks sampled on the sim clock, [x, y, z, yaw] rows."""

    times = np.arange(0.0, replay.t_end + dt / 2, dt)
    tracks = []
    for name, trajectory, asset in replay.distractors:
        rows = []
        for t in times:
            position, yaw = trajectory.pose_at(float(t))
            rows.append([float(position[0]), float(position[1]), float(position[2]), float(yaw)])
        tracks.append({"name": name, "dt": dt, "track": rows, "glb": None})
    return tracks


def run_manifest(
    replay,
    method: str,
    points_bin: str,
    contam_bin: str,
    reconstructions: List[Dict],
    point_index: List[Dict[str, int]],
    contam_index: List[Dict[str, int]],
    thumbs: List[str],
    tracks: List[Dict],
    eval_frusta: List[Dict],
    shared_frusta: List[Dict],
    summary: Optional[str] = None,
    eval_intrinsics: Optional[Dict] = None,
    shared_intrinsics: Optional[Dict] = None,
) -> Dict:
    """One run (method) as a self-contained entry in a bundle's ``runs`` list."""

    frames = replay.frames
    capture_by_index = {
        int(capture.get("index", position)): capture
        for position, capture in enumerate(getattr(replay, "capture_entries", ()))
    }
    if not (len(point_index) == len(contam_index) == len(thumbs) == len(frames)):
        raise ValueError(
            "chunk/thumb indexes must match frame count %d" % len(frames))
    intr = replay.intrinsics
    return {
        "method": method,
        "episode": "%s/%s" % (Path(replay.episode_dir).parent.name,
                              Path(replay.episode_dir).name),
        # A reusable replay must not auto-select among multiple score catalogs.
        "summary": summary if summary is not None else replay.summary(""),
        "t_end": float(replay.t_end),
        "intrinsics": {
            "width": intr.width, "height": intr.height,
            "fx": float(intr.fx), "fy": float(intr.fy),
            "cx": float(intr.cx), "cy": float(intr.cy),
        },
        "frames": [
            {
                "index": int(f.index),
                "time": float(f.time),
                "c2w": [float(v) for v in np.asarray(f.c2w_gl).reshape(-1)],
                "thumb": thumbs[i],
                "distractor_pixel_fraction": float(
                    capture_by_index.get(int(f.index), {}).get(
                        "distractor_pixel_fraction", 0.0
                    )
                ),
            }
            for i, f in enumerate(frames)
        ],
        "points": {"bin": points_bin, "chunks": point_index},
        "contam": {"bin": contam_bin, "chunks": contam_index},
        "distractors": tracks,
        "eval_frusta": eval_frusta,
        "shared_frusta": shared_frusta,
        # Absent for eval sets that share the run's camera; the viewer then
        # falls back to the run intrinsics, matching older bundles.
        "eval_intrinsics": eval_intrinsics,
        "shared_intrinsics": shared_intrinsics,
        "reconstructions": reconstructions,
    }


def _frusta_from_transforms(path: Path, label_key: str, label_fn):
    """Frusta plus the intrinsics they were rendered with.

    The intrinsics travel with the frusta because an eval set does not have to
    share the run's camera: the cube set is a square 90 deg frustum while the
    episode is 75.18 deg at 4:3. Drawing eval cameras with the run's shape would
    misstate what they actually see, which is the whole point of showing them.
    Returns ``(frusta, intrinsics or None)``; ``None`` means "fall back to the
    run's camera", which is what pre-existing bundles carry.
    """

    if not path.exists():
        return [], None
    payload = json.loads(path.read_text())
    out = []
    for frame in payload["frames"]:
        m = [float(v) for v in np.asarray(frame["transform_matrix"]).reshape(-1)]
        entry = {"c2w": m, label_key: label_fn(frame)}
        if frame.get("stratum") is not None and label_key != "stratum":
            entry["stratum"] = frame["stratum"]
        out.append(entry)
    intrinsics = None
    if all(k in payload for k in ("w", "h", "fl_x", "fl_y")):
        intrinsics = {
            "width": int(payload["w"]), "height": int(payload["h"]),
            "fx": float(payload["fl_x"]), "fy": float(payload["fl_y"]),
            "cx": float(payload.get("cx", payload["w"] / 2.0)),
            "cy": float(payload.get("cy", payload["h"] / 2.0)),
        }
    return out, intrinsics


def campaign_rad_for(npz_path: Path) -> Path:
    """Return the expected path of a campaign-built sibling RAD.

    Campaign exports place ``gaussians-lod.rad`` next to ``gaussians.npz``
    with the same stem prefix (``build_rad_from_ply`` naming convention).
    """

    return npz_path.with_name(npz_path.stem + "-lod.rad")


def rad_provenance(npz_path: Path, sh_degree: int) -> Dict:
    """The source and exporter contract required to reuse a generated RAD."""

    stat = Path(npz_path).stat()
    return {
        "export_version": RECONSTRUCTION_EXPORT_VERSION,
        "source_size": stat.st_size, "source_mtime_ns": stat.st_mtime_ns,
        "sh_degree": sh_degree, "dc_mode": "raw-sh0-if-present",
        "lod_policy": "quick-then-quality",
    }


def write_rad_provenance(npz_path: Path, rad_path: Path, sh_degree: int) -> None:
    Path(str(rad_path) + ".json").write_text(
        json.dumps(rad_provenance(npz_path, sh_degree), indent=2) + "\n")


def current_campaign_rad(npz_path: Path, rad_path: Path, sh_degree: int) -> bool:
    """Old RADs may have shuffled SH or silent SH0 fallback; mtime is insufficient."""

    try:
        return (
            rad_path.is_file() and rad_path.stat().st_size > 0
            and rad_path.stat().st_mtime_ns >= npz_path.stat().st_mtime_ns
            and json.loads(Path(str(rad_path) + ".json").read_text())
            == rad_provenance(npz_path, sh_degree)
        )
    except (OSError, ValueError, TypeError):
        return False


def export_reconstruction(
    episode_dir: Path,
    out_dir: Path,
    data_prefix: str,
    name: str,
    rad_builder: Optional[Path] = None,
    sh_degree: Optional[int] = None,
    progress: Optional[Callable[[float, str], None]] = None,
) -> Dict:
    """Export one reconstruction without duplicating its episode replay.

    ``sh_degree`` overrides the NPZ's recorded SH degree; ``None`` reads it
    from the artifact (gsplat SH3 NPZs emit a full-SH3 PLY, legacy SH0
    NPZs fall back to the DC-only PLY). The chosen degree is recorded on
    the manifest entry and the cache fingerprint so SH3 and SH0 exports of
    the same source NPZ never alias.
    """

    import os

    from activebench.replay import EpisodeReplay

    episode_dir = Path(episode_dir)
    out_dir = Path(out_dir)
    models = discover_reconstructions(episode_dir)
    if name not in models:
        raise ValueError("reconstruction not found in %s: %s" % (episode_dir, name))
    (out_dir / data_prefix / "splats").mkdir(parents=True, exist_ok=True)

    npz_path = models[name]
    used_sh = sh_degree if sh_degree is not None else read_sh_degree(npz_path)

    # Fast path: reuse a campaign-built RAD sibling sitting next to the npz.
    sibling_rad = campaign_rad_for(npz_path)
    is_fresh = (
        rad_builder is not None
        and current_campaign_rad(npz_path, sibling_rad, used_sh)
    )
    if is_fresh:
        if progress:
            progress(0.10, "Reusing campaign-built RAD")
        recon_dir = out_dir / data_prefix / "splats"
        rad_rel = "%s/splats/%s-lod.rad" % (data_prefix, name)
        rad_dest = out_dir / rad_rel
        try:
            os.link(sibling_rad, rad_dest)
        except OSError:
            shutil.copyfile(sibling_rad, rad_dest)
        with np.load(npz_path) as data:
            count = int(data["opacities"].shape[0])
        reconstruction = {
            "name": name, "rad": rad_rel, "count": count,
            "sh_degree": used_sh,
        }
        reconstruction["summary"] = EpisodeReplay(episode_dir).summary(name)
        if progress:
            progress(1.0, "Reconstruction ready (reused RAD)")
        return reconstruction

    # Slow path: convert npz → PLY → RAD from scratch.
    if progress:
        progress(0.02, "Converting reconstruction to PLY")
    rel = "%s/splats/%s.ply" % (data_prefix, name)
    ply_path = out_dir / rel
    count, used_sh = splats_npz_to_ply(npz_path, ply_path, sh_degree=sh_degree)
    if progress:
        progress(0.28, "PLY ready (sh%d)" % used_sh)
    reconstruction = {
        "name": name, "ply": rel, "count": count, "sh_degree": used_sh,
    }
    reconstruction["summary"] = EpisodeReplay(episode_dir).summary(name)
    if rad_builder is not None:
        try:
            if progress:
                progress(0.32, "Building streamable RAD (sh%d)" % used_sh)
            rad_path = build_rad_from_ply(ply_path, rad_builder, sh_degree=used_sh)
            write_rad_provenance(npz_path, rad_path, used_sh)
            reconstruction["rad"] = rad_path.relative_to(out_dir).as_posix()
            ply_path.unlink()
            del reconstruction["ply"]
        except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
            warnings.warn("RAD conversion failed for %s; keeping PLY only: %s" % (
                name, exc), RuntimeWarning)
    if progress:
        progress(1.0, "Reconstruction ready")
    return reconstruction


def export_replay(
    episode_dir: Path,
    out_dir: Path,
    data_prefix: str,
    method: str,
    shared_eval_root: Optional[Path] = None,
    thumb_width: int = 240,
    track_dt: float = 0.2,
    progress: Optional[Callable[[float, str], None]] = None,
    summary_run: Optional[str] = None,
) -> Dict:
    """Export one episode replay without any reconstruction assets.

    ``summary_run`` optionally selects the headline reconstruction metrics.
    A reusable replay leaves it unset and exports a model-neutral summary;
    scoring aliases must never make replay-only export ambiguous.
    """

    from PIL import Image

    from activebench.replay import (
        EpisodeReplay,
        shared_eval_set_dir,
        shared_view_class,
    )

    episode_dir = Path(episode_dir)
    out_dir = Path(out_dir)
    for sub in ("thumbs", "distractors"):
        (out_dir / data_prefix / sub).mkdir(parents=True, exist_ok=True)

    if progress:
        progress(0.02, "Reading episode replay")
    replay = EpisodeReplay(episode_dir)

    # Replay bundle: point-cloud chunks, contamination, thumbnails.
    frame_count = max(1, len(replay.frames))

    def replay_chunks():
        for index in range(len(replay.frames)):
            chunk = replay.chunk(index)
            if progress:
                progress(
                    0.08 + 0.52 * (index + 1) / frame_count,
                    "Packing replay frame %d/%d" % (index + 1, len(replay.frames)),
                )
            yield chunk.points, chunk.colors

    point_blob, point_index = pack_point_chunks(
        replay_chunks())
    if progress:
        progress(0.64, "Packing contamination data")
    contam_blob, contam_index = pack_point_chunks(
        (replay.chunk(i).distractor_points,
         np.zeros((len(replay.chunk(i).distractor_points), 3), np.uint8))
        for i in range(len(replay.frames)))
    points_bin = "%s/points.bin" % data_prefix
    contam_bin = "%s/contam.bin" % data_prefix
    (out_dir / points_bin).write_bytes(point_blob)
    (out_dir / contam_bin).write_bytes(contam_blob)

    thumbs = []
    for position, frame in enumerate(replay.frames):
        rel = "%s/thumbs/%03d.jpg" % (data_prefix, frame.index)
        img = Image.open(frame.rgb_path).convert("RGB")
        height = max(1, round(img.height * thumb_width / img.width))
        img.resize((thumb_width, height)).save(out_dir / rel, quality=80)
        thumbs.append(rel)
        if progress:
            progress(
                0.70 + 0.20 * (position + 1) / frame_count,
                "Writing thumbnail %d/%d" % (position + 1, len(replay.frames)),
            )

    # Distractor tracks + meshes (falls back to a marker client-side).
    if progress:
        progress(0.93, "Preparing replay overlays")
    tracks = sample_tracks(replay, dt=track_dt)
    for track, (_name, _traj, asset) in zip(tracks, replay.distractors):
        if asset is not None and Path(asset).suffix.lower() in (".glb", ".gltf"):
            rel = "%s/distractors/%s%s" % (data_prefix, track["name"], Path(asset).suffix.lower())
            shutil.copyfile(asset, out_dir / rel)
            track["glb"] = rel

    eval_frusta, eval_intrinsics = _frusta_from_transforms(
        episode_dir / "transforms_eval.json", "stratum",
        lambda frame: frame.get("stratum", "level"))
    shared_frusta, shared_intrinsics = [], None
    if shared_eval_root is not None:
        shared_dir = shared_eval_set_dir(episode_dir, shared_eval_root)
        if shared_dir is not None:
            parts = episode_dir.parent.name.rsplit("__", 2)
            difficulty = parts[1] if len(parts) == 3 else ""
            shared_frusta, shared_intrinsics = _frusta_from_transforms(
                shared_dir / "transforms_eval_shared.json", "cls",
                lambda frame: shared_view_class(frame, difficulty))

    result = run_manifest(
        replay, method, points_bin, contam_bin, [],
        point_index, contam_index, thumbs, tracks, eval_frusta, shared_frusta,
        summary=replay.summary(summary_run or ""),
        eval_intrinsics=eval_intrinsics, shared_intrinsics=shared_intrinsics,
    )
    if progress:
        progress(1.0, "Replay ready")
    return result


def export_run(
    episode_dir: Path,
    out_dir: Path,
    data_prefix: str,
    method: str,
    reconstruction_runs: Optional[Sequence[str]] = None,
    shared_eval_root: Optional[Path] = None,
    thumb_width: int = 240,
    track_dt: float = 0.2,
    rad_builder: Optional[Path] = None,
) -> Dict:
    """Write one self-contained replay plus its selected reconstructions."""

    models = discover_reconstructions(episode_dir)
    if reconstruction_runs is not None:
        missing = [name for name in reconstruction_runs if name not in models]
        if missing:
            raise ValueError("reconstructions not found in %s: %s" % (episode_dir, missing))
        names = list(reconstruction_runs)
    else:
        names = list(models)
    run = export_replay(
        episode_dir, out_dir, data_prefix, method,
        shared_eval_root=shared_eval_root,
        thumb_width=thumb_width,
        track_dt=track_dt,
        summary_run=names[0] if names else None,
    )
    run["reconstructions"] = [
        export_reconstruction(
            episode_dir, out_dir, data_prefix, name,
            rad_builder=rad_builder,
        )
        for name in names
    ]
    return run


def export_bundle(
    episode_dirs: Sequence[Path],
    out_dir: Path,
    reconstruction_runs: Optional[Sequence[str]] = None,
    per_run_reconstructions: Optional[Sequence[Sequence[str]]] = None,
    shared_eval_root: Optional[Path] = None,
    thumb_width: int = 240,
    track_dt: float = 0.2,
    template_dir: Optional[Path] = None,
    rad_builder: Optional[Path] = None,
) -> Dict:
    """Write a static demo bundle for one or more runs (compare = >1).

    Runs share a synced camera and clock in the viewer; each keeps its own
    replay (path, point cloud, distractors) and reconstruction set.
    """

    if not episode_dirs:
        raise ValueError("need at least one episode dir")
    if (per_run_reconstructions is not None
            and len(per_run_reconstructions) != len(episode_dirs)):
        raise ValueError("per-run reconstructions must match episode dirs")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = []
    used = {}
    for run_index, episode_dir in enumerate(episode_dirs):
        episode_dir = Path(episode_dir)
        method = episode_dir.name
        # namespace assets by method; disambiguate a repeated method name
        used[method] = used.get(method, 0) + 1
        prefix_name = method if used[method] == 1 else "%s-%d" % (method, used[method])
        runs.append(export_run(
            episode_dir, out_dir, "data/%s" % prefix_name, method,
            reconstruction_runs=(per_run_reconstructions[run_index]
                                  if per_run_reconstructions is not None
                                  else reconstruction_runs),
            shared_eval_root=shared_eval_root,
            thumb_width=thumb_width, track_dt=track_dt,
            rad_builder=rad_builder,
        ))

    manifest = {
        "mode": "compare" if len(runs) > 1 else "single",
        "up": "+y",
        "runs": runs,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest))
    if template_dir is not None:
        for name in ("index.html", "app.js", "style.css"):
            shutil.copyfile(Path(template_dir) / name, out_dir / name)
    return manifest


def export_episode_bundle(
    episode_dir: Path,
    out_dir: Path,
    **kwargs,
) -> Dict:
    """Single-run bundle (thin wrapper over :func:`export_bundle`)."""

    return export_bundle([episode_dir], out_dir, **kwargs)


def discover_reconstructions(episode_dir: Path) -> Dict[str, Path]:
    """Named gaussians.npz models of an episode (same layout the viewer uses)."""

    episode_dir = Path(episode_dir)
    models: Dict[str, Path] = {}
    legacy = episode_dir / "retrain_gaussians.npz"
    if legacy.exists():
        models["legacy"] = legacy
    root = episode_dir / "reconstructions"
    if root.exists():
        for path in sorted(root.glob("*/gaussians.npz")):
            models[path.parent.name] = path
    ordered = sorted(
        models, key=lambda name: (RECON_PREFERENCE.index(name)
                                  if name in RECON_PREFERENCE else len(RECON_PREFERENCE), name))
    return {name: models[name] for name in ordered}


def read_sh_degree(npz_path: Path) -> int:
    """Read the recorded SH degree from a saved gaussians.npz.

    ``np.load`` is lazy on the central directory, so this does not decode
    the gaussian arrays — it only peeks at the file list. Returns 0 for
    legacy SH0-only NPZs and unreadable files so the SH0 export path
    stays the safe fallback.
    """

    try:
        with np.load(npz_path) as data:
            if "sh_degree" in data.files:
                return int(data["sh_degree"])
    except (OSError, ValueError, KeyError, EOFError):
        pass
    return 0
