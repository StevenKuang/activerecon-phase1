"""Lazy, cached bundle selection for the local Spark web-demo server."""

from __future__ import annotations

import concurrent.futures
import copy
import hashlib
import http.server
import json
import os
import re
import shutil
import tempfile
import threading
import time
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Sequence

from activebench.replay import (
    EpisodeReplay,
    ExperimentRound,
    RunIndex,
    shared_eval_set_dir,
)
from activebench.web_export import (
    RECONSTRUCTION_EXPORT_VERSION,
    discover_reconstructions,
    export_reconstruction,
    export_replay,
    read_sh_degree,
)


class RangeRequestHandler(http.server.SimpleHTTPRequestHandler):
    """Static-file handler with the single byte ranges used by paged RAD."""

    _byte_range = None

    def end_headers(self) -> None:
        self.send_header("Accept-Ranges", "bytes")
        if self.path.startswith(("/artifacts/", "/views/")):
            self.send_header("Cache-Control", "public, max-age=31536000, immutable")
        super().end_headers()

    def send_head(self):
        self._byte_range = None
        header = self.headers.get("Range")
        path = self.translate_path(self.path)
        if not header or not os.path.isfile(path):
            return super().send_head()
        match = re.fullmatch(r"bytes=(\d*)-(\d*)", header.strip())
        if match is None or not any(match.groups()):
            self.send_error(400, "invalid byte range")
            return None
        size = os.path.getsize(path)
        if match.group(1):
            start = int(match.group(1))
            end = int(match.group(2)) if match.group(2) else size - 1
        else:
            length = int(match.group(2))
            start, end = max(0, size - length), size - 1
        end = min(end, size - 1)
        if start >= size or end < start:
            self.send_response(416)
            self.send_header("Content-Range", "bytes */%d" % size)
            self.end_headers()
            return None
        source = open(path, "rb")
        self.send_response(206)
        self.send_header("Content-Type", self.guess_type(path))
        self.send_header("Content-Range", "bytes %d-%d/%d" % (start, end, size))
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Last-Modified", self.date_time_string(os.path.getmtime(path)))
        self.end_headers()
        self._byte_range = (start, end)
        return source

    def copyfile(self, source, outputfile) -> None:
        try:
            if self._byte_range is None:
                super().copyfile(source, outputfile)
                return
            start, end = self._byte_range
            source.seek(start)
            remaining = end - start + 1
            while remaining:
                chunk = source.read(min(64 * 1024, remaining))
                if not chunk:
                    break
                outputfile.write(chunk)
                remaining -= len(chunk)
        except (BrokenPipeError, ConnectionResetError):
            # Browsers routinely cancel stale range/download requests during
            # model switches; this is not a server failure.
            return


@dataclass(frozen=True)
class LiveRound:
    label: str
    index: RunIndex


def discover_live_rounds(
    runs_dirs: Sequence[Path] = (),
    rounds_dir: Optional[Path] = None,
) -> Dict[str, LiveRound]:
    """Build viewer indexes from explicit run roots or experiment rounds."""

    if runs_dirs:
        index = RunIndex.scan([Path(path) for path in runs_dirs])
        return {"adhoc": LiveRound("Ad-hoc", index)} if index.scenes() else {}
    if rounds_dir is None:
        return {}
    found = {}
    for round_ in ExperimentRound.scan(Path(rounds_dir)):
        if not round_.runs_dir.exists():
            continue
        index = RunIndex.scan([round_.runs_dir])
        if index.scenes():
            found[round_.round_id] = LiveRound(round_.label, index)
    return found


PANE_DIMENSIONS = ("difficulty", "method", "reconstruction", "seed")
ProgressCallback = Callable[[Sequence[int], str, float, str], None]


def live_catalog(
    rounds: Mapping[str, LiveRound],
    reconstruction_runs: Optional[Sequence[str]] = None,
) -> Dict:
    """Flat valid-view catalog used to keep A/B dimensions independent."""

    views = []
    allowed = set(reconstruction_runs) if reconstruction_runs is not None else None
    for round_id, live_round in rounds.items():
        index = live_round.index
        for scene in index.scenes():
            for difficulty in index.difficulties(scene):
                for seed in index.seeds(scene, difficulty):
                    for method in index.methods(scene, difficulty, seed):
                        episode_dir = index.episode_dir(scene, difficulty, seed, method)
                        reconstructions = list(discover_reconstructions(episode_dir))
                        if allowed is not None:
                            reconstructions = [name for name in reconstructions if name in allowed]
                        elif "gsplat1600" in reconstructions:
                            # v6: cube/uni are scoring aliases of the same model;
                            # isolated older ablations are not deliverable choices.
                            # Explicit --reconstruction still opens an archive.
                            reconstructions = ["gsplat1600"]
                        for reconstruction in reconstructions or [""]:
                            views.append({
                                "round": round_id,
                                "scene": scene,
                                "difficulty": difficulty,
                                "method": method,
                                "reconstruction": reconstruction,
                                "seed": seed,
                            })
    return {
        "rounds": [
            {"id": round_id, "label": live_round.label}
            for round_id, live_round in rounds.items()
        ],
        "views": views,
    }


class LiveBundleCache:
    """Compose views from reusable replay and reconstruction artifacts."""

    def __init__(
        self,
        rounds: Mapping[str, LiveRound],
        cache_dir: Path,
        *,
        reconstruction_runs: Optional[Sequence[str]] = None,
        shared_eval_root: Optional[Path] = None,
        shared_eval_roots: Sequence[Path] = (),
        thumb_width: int = 240,
        track_dt: float = 0.2,
        template_dir: Optional[Path] = None,
        rad_builder: Optional[Path] = None,
    ) -> None:
        self.rounds = dict(rounds)
        self.reconstruction_runs = (tuple(reconstruction_runs)
                                    if reconstruction_runs is not None else None)
        self.catalog = live_catalog(self.rounds, self.reconstruction_runs)
        self.views = self.catalog["views"]
        self._catalog_digest = self._digest(self.catalog)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.shared_eval_root = Path(shared_eval_root) if shared_eval_root else None
        self.shared_eval_roots = tuple(Path(path) for path in shared_eval_roots)
        self.thumb_width = thumb_width
        self.track_dt = track_dt
        self.template_dir = Path(template_dir) if template_dir else None
        self.rad_builder = Path(rad_builder) if rad_builder else None
        self._locks_guard = threading.Lock()
        self._locks: Dict[str, threading.Lock] = {}
        self._rad_lock = threading.Lock()
        self._index_lock = threading.Lock()
        self._index_dirty = False
        self._template_digest = self._digest({
            name: hashlib.sha256((self.template_dir / name).read_bytes()).hexdigest()
            for name in ("index.html", "app.js", "style.css")
        }) if self.template_dir is not None else "none"
        self._episodes = self._index_episodes()

    @staticmethod
    def _digest(payload: Mapping) -> str:
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(raw).hexdigest()[:20]

    @staticmethod
    def _stat(path: Path) -> Dict[str, int]:
        stat = Path(path).stat()
        return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}

    @staticmethod
    def _view_key(view: Mapping[str, str]):
        return tuple(view[key] for key in ("round", "scene") + PANE_DIMENSIONS)

    def _index_episodes(self) -> Dict[tuple, Path]:
        episodes = {}
        for view in self.views:
            key = self._view_key(view)
            index = self.rounds[view["round"]].index
            episodes[key] = index.episode_dir(
                view["scene"], view["difficulty"], view["seed"], view["method"])
        return episodes

    def _matching_views(self, round_id: str, scene: str) -> List[Dict[str, str]]:
        return [view for view in self.views
                if view["round"] == round_id and view["scene"] == scene]

    def _pane_from_query(
        self,
        query: Mapping[str, str],
        prefix: str,
        candidates: Sequence[Dict[str, str]],
    ) -> Dict[str, str]:
        legacy = prefix == "a" and not any(
            ("a_" + key) in query for key in PANE_DIMENSIONS)
        values = {}
        for key in PANE_DIMENSIONS:
            query_key = key if legacy else "%s_%s" % (prefix, key)
            if query_key in query:
                values[key] = query[query_key]
        if not values:
            return {key: candidates[0][key] for key in PANE_DIMENSIONS}
        matches = [view for view in candidates
                   if all(view[key] == value for key, value in values.items())]
        if not matches:
            raise ValueError("%s pane selection not found" % prefix.upper())
        chosen = matches[0]
        return {key: chosen[key] for key in PANE_DIMENSIONS}

    def selection(self, query: Mapping[str, str]) -> Dict:
        if not self.views:
            raise ValueError("no runnable views found")
        round_id = query.get("round", self.views[0]["round"])
        round_views = [view for view in self.views if view["round"] == round_id]
        if not round_views:
            raise ValueError("unknown round: %s" % round_id)
        scene = query.get("scene", round_views[0]["scene"])
        candidates = self._matching_views(round_id, scene)
        if not candidates:
            raise ValueError("scene not found: %s" % scene)
        pane_a = self._pane_from_query(query, "a", candidates)

        compare_value = query.get("compare", "")
        legacy_compare = compare_value not in ("", "0", "1", "false", "true")
        compare = compare_value.lower() in ("1", "true") or legacy_compare
        pane_b = None
        if compare:
            if legacy_compare:
                legacy_query = dict(query)
                legacy_query.update({
                    "b_difficulty": pane_a["difficulty"],
                    "b_method": compare_value,
                    "b_reconstruction": query.get(
                        "compare_reconstruction", pane_a["reconstruction"]),
                    "b_seed": pane_a["seed"],
                })
                pane_b = self._pane_from_query(legacy_query, "b", candidates)
            elif not any(("b_" + key) in query for key in PANE_DIMENSIONS):
                pane_b = dict(pane_a)
            else:
                inherited = dict(query)
                for key in PANE_DIMENSIONS:
                    inherited.setdefault("b_" + key, pane_a[key])
                pane_b = self._pane_from_query(inherited, "b", candidates)
        return {"round": round_id, "scene": scene, "a": pane_a, "b": pane_b}

    def resolve(self, query: Mapping[str, str]) -> List[Path]:
        selection = self.selection(query)
        panes = [selection["a"]] + ([selection["b"]] if selection["b"] else [])
        return [self._episode(selection, pane) for pane in panes]

    def default_query(self) -> Dict[str, str]:
        selection = self.selection({})
        return self.selection_query(selection)

    @staticmethod
    def selection_query(selection: Mapping) -> Dict[str, str]:
        query = {"round": selection["round"], "scene": selection["scene"]}
        for key in PANE_DIMENSIONS:
            query["a_" + key] = selection["a"][key]
        if selection.get("b") is not None:
            query["compare"] = "1"
            for key in PANE_DIMENSIONS:
                query["b_" + key] = selection["b"][key]
        return query

    def _episode(self, selection: Mapping, pane: Mapping[str, str]) -> Path:
        view = {"round": selection["round"], "scene": selection["scene"], **pane}
        return self._episodes[self._view_key(view)]

    def _shared_eval_for(self, episode_dir: Path) -> Optional[Path]:
        if self.shared_eval_root is not None:
            return self.shared_eval_root
        for root in self.shared_eval_roots:
            if root.exists() and shared_eval_set_dir(episode_dir, root) is not None:
                return root
        return None

    def _lock_for(self, key: str) -> threading.Lock:
        with self._locks_guard:
            return self._locks.setdefault(key, threading.Lock())

    @staticmethod
    def _asset_url(base: str, path: str) -> str:
        return base + urllib.parse.quote(path, safe="/")

    def _prefix_run(self, run: Mapping, base: str) -> Dict:
        run = copy.deepcopy(run)
        run["points"]["bin"] = self._asset_url(base, run["points"]["bin"])
        run["contam"]["bin"] = self._asset_url(base, run["contam"]["bin"])
        for frame in run["frames"]:
            frame["thumb"] = self._asset_url(base, frame["thumb"])
        for track in run["distractors"]:
            if track.get("glb"):
                track["glb"] = self._asset_url(base, track["glb"])
        return run

    def _prefix_reconstruction(self, reconstruction: Mapping, base: str) -> Dict:
        reconstruction = dict(reconstruction)
        for key in ("rad", "ply"):
            if key in reconstruction:
                reconstruction[key] = self._asset_url(base, reconstruction[key])
        return reconstruction

    @staticmethod
    def _view_metadata(episode_dir: Path, reconstruction: str) -> Dict:
        """Small selection-specific UI data kept outside reusable artifacts."""

        try:
            replay = EpisodeReplay(episode_dir)
            capture_by_index = {
                int(capture.get("index", position)): capture
                for position, capture in enumerate(replay.capture_entries)
            }
            return {
                "summary": replay.summary(reconstruction or ""),
                "distractor_pixel_fractions": [
                    float(capture_by_index.get(int(frame.index), {}).get(
                        "distractor_pixel_fraction", 0.0))
                    for frame in replay.frames
                ],
            }
        except (KeyError, OSError, TypeError, ValueError, IndexError):
            # Discovery tests and partially-written episodes may only have the
            # minimal manifest needed for indexing. Missing display metadata
            # must not prevent the reusable artifacts themselves from loading.
            return {}

    def _run_key(self, episode_dir: Path, shared_eval_root: Optional[Path]) -> str:
        payload = {
            "run_format": 3,
            "episode": str(episode_dir.resolve()),
            "manifest": self._stat(episode_dir / "manifest.json"),
            "shared_eval": str(shared_eval_root.resolve()) if shared_eval_root else None,
            "shared_transforms": self._stat(
                shared_eval_set_dir(episode_dir, shared_eval_root) / "transforms_eval_shared.json"
            ) if shared_eval_root and shared_eval_set_dir(episode_dir, shared_eval_root) else None,
            "eval_transforms": self._stat(episode_dir / "transforms_eval.json")
            if (episode_dir / "transforms_eval.json").is_file() else None,
            "thumb_width": self.thumb_width,
            "track_dt": self.track_dt,
        }
        return self._digest(payload)

    def _reconstruction_key(self, episode_dir: Path, name: str) -> str:
        model = discover_reconstructions(episode_dir)[name]
        builder = self.rad_builder.resolve() if self.rad_builder else None
        payload = {
            "reconstruction_format": RECONSTRUCTION_EXPORT_VERSION,
            "name": name,
            "model": str(model.resolve()),
            "source": self._stat(model),
            # Include the NPZ's recorded SH degree so an SH3 export and an
            # SH0 fallback of the same source never alias in the cache.
            "sh_degree": read_sh_degree(model),
            "lod_policy": "quick-then-quality",
            "rad_builder": ({"path": str(builder), "source": self._stat(builder)}
                            if builder else None),
        }
        return self._digest(payload)

    def _ensure_run(
        self,
        episode_dir: Path,
        progress: Optional[Callable[[float, str], None]] = None,
    ) -> str:
        shared_eval_root = self._shared_eval_for(episode_dir)
        key = self._run_key(episode_dir, shared_eval_root)
        parent = self.cache_dir / "artifacts" / "runs"
        target = parent / key
        manifest_path = target / "run.json"
        was_cached = manifest_path.exists()
        if progress:
            progress(0.01, "Checking replay cache")
        with self._lock_for("run:" + key):
            if not manifest_path.exists():
                parent.mkdir(parents=True, exist_ok=True)
                stage = Path(tempfile.mkdtemp(prefix=".run-", dir=parent))
                try:
                    run = export_replay(
                        episode_dir, stage, "data", episode_dir.name,
                        shared_eval_root=shared_eval_root,
                        thumb_width=self.thumb_width,
                        track_dt=self.track_dt,
                        progress=progress,
                    )
                    run = self._prefix_run(
                        run, "/artifacts/runs/%s/" % key)
                    (stage / "run.json").write_text(json.dumps(run))
                    stage.rename(target)
                    self._index_dirty = True
                except Exception:
                    shutil.rmtree(stage, ignore_errors=True)
                    raise
        if progress:
            progress(1.0, "Replay cached" if was_cached else "Replay ready")
        return "/artifacts/runs/%s/run.json" % key

    def _ensure_reconstruction(
        self,
        episode_dir: Path,
        name: str,
        progress: Optional[Callable[[float, str], None]] = None,
    ) -> Optional[str]:
        if not name:
            return None
        key = self._reconstruction_key(episode_dir, name)
        parent = self.cache_dir / "artifacts" / "reconstructions"
        target = parent / key
        manifest_path = target / "reconstruction.json"
        was_cached = manifest_path.exists()
        if progress:
            progress(0.01, "Checking reconstruction cache")
        with self._lock_for("reconstruction:" + key):
            if not manifest_path.exists():
                parent.mkdir(parents=True, exist_ok=True)
                stage = Path(tempfile.mkdtemp(prefix=".reconstruction-", dir=parent))
                try:
                    if progress:
                        progress(0.03, "Waiting for RAD worker")
                    with self._rad_lock:
                        reconstruction = export_reconstruction(
                            episode_dir, stage, "data", name,
                            rad_builder=self.rad_builder,
                            progress=progress,
                        )
                    reconstruction = self._prefix_reconstruction(
                        reconstruction,
                        "/artifacts/reconstructions/%s/" % key,
                    )
                    (stage / "reconstruction.json").write_text(
                        json.dumps(reconstruction))
                    stage.rename(target)
                    self._index_dirty = True
                except Exception:
                    shutil.rmtree(stage, ignore_errors=True)
                    raise
        if progress:
            progress(1.0, "Reconstruction cached" if was_cached else "Reconstruction ready")
        return "/artifacts/reconstructions/%s/reconstruction.json" % key

    @staticmethod
    def _json_atomic(path: Path, payload: Mapping) -> None:
        """Write small cache metadata without exposing a partial JSON file."""

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, stage_name = tempfile.mkstemp(prefix=".%s-" % path.name, dir=path.parent)
        os.close(fd)
        stage = Path(stage_name)
        try:
            stage.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            os.replace(stage, path)
        finally:
            if stage.exists():
                stage.unlink()

    def artifact_plan(
        self,
        reconstruction_names: Optional[Sequence[str]] = None,
    ) -> List[Dict]:
        """Unique persistent artifacts needed by the current live catalog.

        reconstruction_names limits model preparation only. Replays are always
        included because every selectable model needs its episode data. The
        returned target paths are fingerprinted; selections is the readable
        logical index for those immutable targets.
        """

        wanted = (set(reconstruction_names)
                  if reconstruction_names is not None else None)
        records: Dict[tuple, Dict] = {}
        for view in self.views:
            episode_dir = self._episodes[self._view_key(view)]
            base_selection = {
                key: view[key]
                for key in ("round", "scene", "difficulty", "method", "seed")
            }
            shared_eval_root = self._shared_eval_for(episode_dir)
            run_key = self._run_key(episode_dir, shared_eval_root)
            run_id = ("replay", run_key)
            run = records.setdefault(run_id, {
                "kind": "replay",
                "fingerprint": run_key,
                "episode": str(episode_dir.resolve()),
                "source": str((episode_dir / "manifest.json").resolve()),
                "source_size": self._stat(episode_dir / "manifest.json")["size"],
                "target": "artifacts/runs/%s/run.json" % run_key,
                "label": "%s / %s / %s / %s / %s / replay" % (
                    view["round"], view["scene"], view["difficulty"],
                    view["method"], view["seed"],
                ),
                "selections": [],
            })
            if base_selection not in run["selections"]:
                run["selections"].append(base_selection)

            name = view["reconstruction"]
            if not name or (wanted is not None and name not in wanted):
                continue
            model = discover_reconstructions(episode_dir)[name]
            reconstruction_key = self._reconstruction_key(episode_dir, name)
            reconstruction_id = ("reconstruction", reconstruction_key)
            logical_selection = {**base_selection, "reconstruction": name}
            reconstruction = records.setdefault(reconstruction_id, {
                "kind": "reconstruction",
                "fingerprint": reconstruction_key,
                "episode": str(episode_dir.resolve()),
                "name": name,
                "source": str(model.resolve()),
                "source_size": self._stat(model)["size"],
                "target": (
                    "artifacts/reconstructions/%s/reconstruction.json"
                    % reconstruction_key
                ),
                "label": "%s / %s / %s / %s / %s / %s" % (
                    view["round"], view["scene"], view["difficulty"],
                    view["method"], view["seed"], name,
                ),
                "selections": [],
            })
            if logical_selection not in reconstruction["selections"]:
                reconstruction["selections"].append(logical_selection)
        return sorted(
            records.values(),
            key=lambda item: (item["kind"] != "replay", item["label"]),
        )

    def write_cache_index(self) -> Path:
        """Refresh the readable logical-to-fingerprinted artifact index."""

        with self._index_lock:
            artifacts = self.artifact_plan()
            for artifact in artifacts:
                artifact["cached"] = (self.cache_dir / artifact["target"]).is_file()
            path = self.cache_dir / "cache-index.json"
            self._json_atomic(path, {
                "format": 1,
                "catalog_fingerprint": self._catalog_digest,
                "artifacts": artifacts,
            })
            self._index_dirty = False
            return path

    def _load_prepare_stats(self) -> Dict:
        path = self.cache_dir / "prepare-stats.json"
        try:
            payload = json.loads(path.read_text())
        except (OSError, ValueError, TypeError):
            payload = {}
        kinds = payload.get("kinds", {}) if isinstance(payload, dict) else {}
        normalized = {"format": 1, "kinds": {}}
        for kind in ("replay", "reconstruction"):
            source = kinds.get(kind, {}) if isinstance(kinds, dict) else {}
            normalized["kinds"][kind] = {
                "samples": max(0, int(source.get("samples", 0))),
                "seconds": max(0.0, float(source.get("seconds", 0.0))),
                "units": max(0.0, float(source.get("units", 0.0))),
            }
        return normalized

    def prepare(
        self,
        reconstruction_names: Optional[Sequence[str]] = None,
        *,
        replay_workers: int = 2,
        progress: Optional[Callable[[Mapping], None]] = None,
    ) -> Dict:
        """Prebuild each unique artifact once, reporting aggregate progress/ETA."""

        artifacts = self.artifact_plan(reconstruction_names)
        replay_workers = max(1, int(replay_workers))
        state_lock = threading.Lock()
        started_at = time.monotonic()
        stats = self._load_prepare_stats()
        states = {}
        for artifact in artifacts:
            cached = (self.cache_dir / artifact["target"]).is_file()
            states[(artifact["kind"], artifact["fingerprint"])] = {
                "artifact": artifact,
                "status": "cached" if cached else "queued",
                "progress": 1.0 if cached else 0.0,
                "message": "Already cached" if cached else "Queued",
                "started_at": None,
                "duration": None,
                "error": None,
            }

        def units(state: Mapping) -> float:
            if state["artifact"]["kind"] == "replay":
                return 1.0
            mib = state["artifact"]["source_size"] / float(1024 * 1024)
            return max(1.0, mib)

        def rate(kind: str) -> float:
            entry = stats["kinds"][kind]
            if entry["units"] > 0.0:
                return entry["seconds"] / entry["units"]
            if kind == "replay":
                return 4.0
            # A first-run estimate, replaced by measured MiB throughput as soon
            # as one model completes. RAD construction is the expensive path.
            return 0.25 if self.rad_builder is not None else 0.06

        def lane_eta(kind: str, workers: int, now: float) -> float:
            pending = [state for state in states.values()
                       if state["artifact"]["kind"] == kind
                       and state["status"] in ("queued", "running")]
            if not pending:
                return 0.0
            seconds_per_unit = rate(kind)
            running = [state for state in pending if state["status"] == "running"]
            queued = [state for state in pending if state["status"] == "queued"]
            loads = [0.0] * workers
            for index, state in enumerate(running[:workers]):
                estimate = seconds_per_unit * units(state)
                elapsed = max(0.0, now - state["started_at"])
                loads[index] = max(1.0, 0.1 * elapsed, estimate - elapsed)
            for state in sorted(queued, key=units, reverse=True):
                index = min(range(workers), key=loads.__getitem__)
                loads[index] += seconds_per_unit * units(state)
            return max(loads)

        def snapshot_locked() -> Dict:
            now = time.monotonic()
            cached = sum(state["status"] == "cached" for state in states.values())
            built = sum(state["status"] == "built" for state in states.values())
            failed = sum(state["status"] == "error" for state in states.values())
            total = len(states)
            processed = cached + built + failed
            fraction = (sum(state["progress"] for state in states.values()) / total
                        if total else 1.0)
            current = []
            for state in states.values():
                if state["status"] != "running":
                    continue
                artifact = state["artifact"]
                current.append({
                    "kind": artifact["kind"],
                    "label": artifact["label"],
                    "message": state["message"],
                    "progress": round(100.0 * state["progress"], 1),
                })
            errors = [
                "%s: %s" % (state["artifact"]["label"], state["error"])
                for state in states.values() if state["status"] == "error"
            ]
            eta = max(
                lane_eta("replay", replay_workers, now),
                lane_eta("reconstruction", 1, now),
            )
            return {
                "status": ("running" if processed < total else
                           "error" if failed else "complete"),
                "total": total,
                "ready": cached + built,
                "cached": cached,
                "built": built,
                "failed": failed,
                "progress": round(100.0 * fraction, 1),
                "elapsed_seconds": now - started_at,
                "eta_seconds": 0.0 if processed == total else eta,
                "current": current,
                "errors": errors,
            }

        def emit() -> None:
            if progress is None:
                return
            with state_lock:
                current_snapshot = snapshot_locked()
            progress(current_snapshot)

        def update(task_id: tuple, value: float, message: str) -> None:
            with state_lock:
                state = states[task_id]
                state["progress"] = min(0.99, max(state["progress"], float(value)))
                state["message"] = message
            emit()

        def build(task_id: tuple) -> None:
            with state_lock:
                state = states[task_id]
                state["status"] = "running"
                state["started_at"] = time.monotonic()
                state["message"] = "Checking persistent cache"
                artifact = state["artifact"]
            emit()
            task_started = time.monotonic()
            try:
                callback = lambda value, message: update(task_id, value, message)
                episode_dir = Path(artifact["episode"])
                if artifact["kind"] == "replay":
                    self._ensure_run(episode_dir, callback)
                else:
                    self._ensure_reconstruction(
                        episode_dir, artifact["name"], callback)
                duration = max(0.001, time.monotonic() - task_started)
                with state_lock:
                    state = states[task_id]
                    state["status"] = "built"
                    state["progress"] = 1.0
                    state["message"] = "Ready"
                    state["duration"] = duration
                    entry = stats["kinds"][artifact["kind"]]
                    entry["samples"] += 1
                    entry["seconds"] += duration
                    entry["units"] += units(state)
            except Exception as exc:
                with state_lock:
                    state = states[task_id]
                    state["status"] = "error"
                    state["progress"] = 1.0
                    state["message"] = "Failed"
                    state["error"] = str(exc)
            emit()

        emit()
        pending_replays = [task_id for task_id, state in states.items()
                           if state["status"] == "queued"
                           and state["artifact"]["kind"] == "replay"]
        pending_reconstructions = [task_id for task_id, state in states.items()
                                   if state["status"] == "queued"
                                   and state["artifact"]["kind"] == "reconstruction"]
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=replay_workers, thread_name_prefix="spark-prewarm-replay",
        ) as replay_pool, concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="spark-prewarm-model",
        ) as reconstruction_pool:
            futures = [replay_pool.submit(build, task_id) for task_id in pending_replays]
            futures.extend(
                reconstruction_pool.submit(build, task_id)
                for task_id in pending_reconstructions
            )
            pending = set(futures)
            while pending:
                done, pending = concurrent.futures.wait(
                    pending, timeout=0.5,
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )
                for future in done:
                    future.result()
                if pending:
                    # build-lod has one long phase without granular callbacks.
                    # Keep elapsed time and ETA alive while it is running.
                    emit()

        self._json_atomic(self.cache_dir / "prepare-stats.json", stats)
        self.write_cache_index()
        with state_lock:
            summary = snapshot_locked()
        if summary["failed"]:
            raise RuntimeError(
                "cache preparation failed for %d artifact(s): %s"
                % (summary["failed"], "; ".join(summary["errors"]))
            )
        return summary

    @staticmethod
    def _tree_size(path: Path) -> int:
        return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())

    def prune_stale(self) -> Dict[str, int]:
        """Delete only generated cache entries outside the current full catalog."""

        valid = {
            (self.cache_dir / artifact["target"]).parent.resolve()
            for artifact in self.artifact_plan()
        }
        removed = 0
        removed_bytes = 0
        for relative in ("artifacts/runs", "artifacts/reconstructions", "views"):
            parent = self.cache_dir / relative
            if not parent.is_dir():
                continue
            for child in parent.iterdir():
                if not child.is_dir():
                    continue
                if relative != "views" and child.resolve() in valid:
                    continue
                removed_bytes += self._tree_size(child)
                shutil.rmtree(child)
                removed += 1
        self._index_dirty = True
        self.write_cache_index()
        return {"removed": removed, "bytes": removed_bytes}

    def manifest(
        self,
        selection: Mapping,
        progress: Optional[ProgressCallback] = None,
    ) -> Dict:
        panes = [selection["a"]] + ([selection["b"]] if selection.get("b") else [])
        run_specs = {}
        reconstruction_specs = {}
        for pane_index, pane in enumerate(panes):
            episode_dir = self._episode(selection, pane)
            run_key = str(episode_dir.resolve())
            run_spec = run_specs.setdefault(run_key, {
                "episode": episode_dir, "panes": [],
            })
            run_spec["panes"].append(pane_index)
            reconstruction = pane["reconstruction"]
            if reconstruction:
                reconstruction_key = (run_key, reconstruction)
                reconstruction_spec = reconstruction_specs.setdefault(
                    reconstruction_key,
                    {"episode": episode_dir, "name": reconstruction, "panes": []},
                )
                reconstruction_spec["panes"].append(pane_index)

        jobs = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(4, 2 * len(panes))) as pool:
            for run_key, spec in run_specs.items():
                callback = None
                if progress:
                    callback = lambda value, message, panes=tuple(spec["panes"]): progress(
                        panes, "replay", value, message)
                jobs[("run", run_key)] = pool.submit(
                    self._ensure_run, spec["episode"], callback)
            for (run_key, reconstruction), spec in reconstruction_specs.items():
                callback = None
                if progress:
                    callback = lambda value, message, panes=tuple(spec["panes"]): progress(
                        panes, "reconstruction", value, message)
                jobs[("reconstruction", run_key, reconstruction)] = pool.submit(
                    self._ensure_reconstruction,
                    spec["episode"], spec["name"], callback,
                )
            runs = []
            for pane in panes:
                episode_dir = self._episode(selection, pane)
                run_key = str(episode_dir.resolve())
                run = {"replay": jobs[("run", run_key)].result()}
                reconstruction = pane["reconstruction"]
                if reconstruction:
                    run["reconstruction"] = jobs[
                        ("reconstruction", run_key, reconstruction)].result()
                run.update(self._view_metadata(episode_dir, reconstruction))
                runs.append(run)
        return {
            "mode": "compare" if selection.get("b") else "single",
            "up": "+y",
            "runs": runs,
            "live": {"catalog": self.catalog, "selection": selection},
        }

    def request_key(self, selection: Mapping) -> str:
        artifacts = []
        panes = [selection["a"]] + ([selection["b"]] if selection.get("b") else [])
        for pane in panes:
            episode_dir = self._episode(selection, pane)
            shared_eval_root = self._shared_eval_for(episode_dir)
            metadata = self._view_metadata(
                episode_dir, pane["reconstruction"])
            artifacts.append({
                "run": self._run_key(episode_dir, shared_eval_root),
                "reconstruction": (
                    self._reconstruction_key(episode_dir, pane["reconstruction"])
                    if pane["reconstruction"] else None
                ),
                # Evaluation files are not part of the replay/model artifact
                # fingerprints. Including their rendered summary here makes a
                # changed metric produce a fresh tiny view manifest without
                # rebuilding either large artifact.
                "summary": metadata.get("summary"),
            })
        return self._digest({
            "selection": selection,
            "catalog": self._catalog_digest,
            "template": self._template_digest,
            "artifacts": artifacts,
        })

    def export(
        self,
        query: Mapping[str, str],
        progress: Optional[ProgressCallback] = None,
    ) -> Path:
        selection = self.selection(query)
        manifest = self.manifest(selection, progress=progress)
        if self._index_dirty:
            self.write_cache_index()
        key = self._digest({
            "view_format": 5,
            "selection": selection,
            "catalog": self._catalog_digest,
            "template": self._template_digest,
            "runs": manifest["runs"],
        })
        parent = self.cache_dir / "views"
        target = parent / key
        with self._lock_for("view:" + key):
            if (target / "manifest.json").exists():
                return target
            parent.mkdir(parents=True, exist_ok=True)
            stage = Path(tempfile.mkdtemp(prefix=".view-", dir=parent))
            try:
                (stage / "manifest.json").write_text(json.dumps(manifest))
                if self.template_dir is not None:
                    for name in ("index.html", "app.js", "style.css"):
                        shutil.copyfile(self.template_dir / name, stage / name)
                stage.rename(target)
            except Exception:
                shutil.rmtree(stage, ignore_errors=True)
                raise
        return target


class LiveExportJobs:
    """Run live exports off the HTTP thread and expose per-pane progress."""

    def __init__(self, cache: LiveBundleCache, concurrent_jobs: int = 2):
        self.cache = cache
        self._lock = threading.Lock()
        self._slots = threading.Semaphore(max(1, concurrent_jobs))
        self._jobs: Dict[str, Dict] = {}
        self._events: Dict[str, threading.Event] = {}

    @staticmethod
    def _pane(tag: str, selection: Mapping[str, str]) -> Dict:
        has_reconstruction = bool(selection["reconstruction"])
        return {
            "tag": tag,
            "selection": dict(selection),
            "progress": 0,
            "message": "Queued",
            "components": {
                "replay": 0.0,
                "reconstruction": 0.0 if has_reconstruction else 1.0,
            },
        }

    def start(self, query: Mapping[str, str]) -> str:
        selection = self.cache.selection(query)
        job_id = self.cache.request_key(selection)
        with self._lock:
            existing = self._jobs.get(job_id)
            if existing is not None and existing["status"] != "error":
                return job_id
            now = time.time()
            panes = [self._pane("A", selection["a"])]
            if selection.get("b") is not None:
                panes.append(self._pane("B", selection["b"]))
            self._jobs[job_id] = {
                "id": job_id,
                "status": "queued",
                "created_at": now,
                "updated_at": now,
                "target": None,
                "error": None,
                "panes": panes,
            }
            self._events[job_id] = threading.Event()
        done_event = self._events[job_id]
        thread = threading.Thread(
            target=self._run,
            args=(job_id, self.cache.selection_query(selection), done_event),
            daemon=True,
            name="spark-export-" + job_id[:8],
        )
        thread.start()
        return job_id

    def _run(
        self,
        job_id: str,
        query: Mapping[str, str],
        done_event: threading.Event,
    ) -> None:
        with self._slots:
            with self._lock:
                job = self._jobs[job_id]
                job["status"] = "running"
                job["updated_at"] = time.time()
            try:
                target = self.cache.export(
                    query,
                    progress=lambda panes, component, value, message: self._progress(
                        job_id, panes, component, value, message),
                )
                relative = target.relative_to(self.cache.cache_dir).as_posix()
                with self._lock:
                    job = self._jobs[job_id]
                    job["status"] = "complete"
                    job["target"] = "/%s/" % relative
                    job["updated_at"] = time.time()
                    for pane in job["panes"]:
                        pane["progress"] = 100
                        pane["message"] = "Ready"
            except Exception as exc:
                with self._lock:
                    job = self._jobs[job_id]
                    job["status"] = "error"
                    job["error"] = str(exc)
                    job["updated_at"] = time.time()
            finally:
                done_event.set()

    def _progress(
        self,
        job_id: str,
        pane_indices: Sequence[int],
        component: str,
        value: float,
        message: str,
    ) -> None:
        value = min(1.0, max(0.0, float(value)))
        with self._lock:
            job = self._jobs[job_id]
            for index in pane_indices:
                pane = job["panes"][index]
                pane["components"][component] = value
                replay = pane["components"]["replay"]
                reconstruction = pane["components"]["reconstruction"]
                if pane["selection"]["reconstruction"]:
                    overall = 0.25 * replay + 0.75 * reconstruction
                else:
                    overall = replay
                pane["progress"] = round(100 * overall)
                pane["message"] = message
            job["updated_at"] = time.time()

    def snapshot(self, job_id: str) -> Dict:
        with self._lock:
            if job_id not in self._jobs:
                raise KeyError(job_id)
            return copy.deepcopy(self._jobs[job_id])

    def wait(self, job_id: str, timeout: float) -> Dict:
        with self._lock:
            if job_id not in self._events:
                raise KeyError(job_id)
            event = self._events[job_id]
        event.wait(timeout)
        return self.snapshot(job_id)
