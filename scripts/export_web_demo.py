"""Export episode(s) as a static, shareable Spark web demo.

Bundles a run's replay (poses, progressive point cloud, thumbnails,
distractor tracks) plus its 3DGS reconstructions (standard PLY) with the
webdemo/ viewer page into a directory any static host can serve:

    conda run -n habitat python scripts/export_web_demo.py \
        --episode data/phase1/runs/mesh/skokloster_castle__dyn__s0/r3con-pano \
        --out exports/skokloster_dyn_r3con --serve 8090

Pass several --episode dirs (same scene/difficulty/seed, different methods)
for a single-window split compare with synced camera; each pane picks its
own method + reconstruction:

    ... --episode data/phase1/runs/mesh/skokloster_castle__dyn__s0/r3con-pano \
        --episode data/phase1/runs/mesh/skokloster_castle__dyn__s0/random \
        --out exports/skokloster_dyn_compare --serve 8090

The output has no backend or CUDA dependency; --serve just runs a local
http.server for a quick look. Deploy = copy the directory.
"""

import argparse
import functools
import hashlib
import http.server
import json
import os
import shutil
import sys
import threading
import time
import urllib.parse
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from activebench.web_export import discover_reconstructions, export_bundle  # noqa: E402
from activebench.web_live import (  # noqa: E402
    LiveBundleCache,
    LiveExportJobs,
    RangeRequestHandler,
    discover_live_rounds,
)

_PHASE1_DATA = Path(os.environ.get("ACTIVEBENCH_PHASE1_DATA", str(_REPO_ROOT / "data/phase1")))
SHARED_EVAL_ROOTS = (str(_PHASE1_DATA / "eval/gs"), str(_PHASE1_DATA / "eval/mesh"))
DEFAULT_LIVE_PORT = 8090
SHARE_DOMAIN = "share.viser.studio"


def default_shared_eval_root(episode_dir: Path):
    """Newest shared-eval set that actually covers this episode's scene__seed."""

    from activebench.replay import shared_eval_set_dir

    for rel in SHARED_EVAL_ROOTS:
        root = _REPO_ROOT / rel
        if root.exists() and shared_eval_set_dir(episode_dir, root) is not None:
            return root
    return None


def apply_mode_defaults(args) -> None:
    """Make indexed browsing concise while keeping static exports non-blocking."""

    if args.runs_dir or args.rounds_dir:
        args.live = True
    if args.live and args.serve is None:
        args.serve = DEFAULT_LIVE_PORT
    if getattr(args, "share", False) and args.serve is None:
        args.serve = DEFAULT_LIVE_PORT
    if args.build_lod:
        args.rad = True


def find_build_lod():
    """Find build-lod on PATH or beside the active Python executable."""

    found = shutil.which("build-lod")
    if found:
        return Path(found).resolve()
    candidate = Path(sys.executable).resolve().parent / "build-lod"
    if candidate.is_file() and os.access(candidate, os.X_OK):
        return candidate
    return None


def default_live_cache_dir(runs_dirs, rounds_dir) -> Path:
    """Stable cache location tied to the selected local campaign."""

    roots = [Path(path).expanduser().resolve() for path in runs_dirs]
    if len(roots) == 1:
        return roots[0] / ".spark-web-cache"
    if rounds_dir:
        return Path(rounds_dir).expanduser().resolve() / ".spark-web-cache"
    digest = hashlib.sha256(
        "\n".join(str(path) for path in roots).encode()
    ).hexdigest()[:12]
    return _REPO_ROOT / ".spark-web-cache" / digest


def _duration(seconds) -> str:
    seconds = max(0, int(round(seconds or 0)))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return "%d:%02d:%02d" % (hours, minutes, seconds)
    return "%02d:%02d" % (minutes, seconds)


def _byte_size(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0 or unit == "TiB":
            return "%.1f %s" % (value, unit)
        value /= 1024.0
    return "%.1f TiB" % value


class TerminalPrepareProgress:
    """Dependency-free, throttled terminal progress bar for cache preparation."""

    def __init__(self, stream=None) -> None:
        self.stream = stream or sys.stdout
        self.is_tty = bool(getattr(self.stream, "isatty", lambda: False)())
        self._lock = threading.Lock()
        self._last_render = 0.0
        self._last_ready = -1
        self._last_width = 0

    def __call__(self, snapshot) -> None:
        with self._lock:
            now = time.monotonic()
            final = snapshot["status"] in ("complete", "error")
            ready_changed = snapshot["ready"] != self._last_ready
            interval = 0.12 if self.is_tty else 5.0
            if not final and not ready_changed and now - self._last_render < interval:
                return
            self._last_render = now
            self._last_ready = snapshot["ready"]

            width = 24
            filled = min(width, max(0, round(width * snapshot["progress"] / 100.0)))
            bar = "#" * filled + "-" * (width - filled)
            line = (
                "[%s] %d/%d ready (%d cached, %d built) %5.1f%%"
                " | elapsed %s | ETA ~%s"
                % (
                    bar, snapshot["ready"], snapshot["total"],
                    snapshot["cached"], snapshot["built"], snapshot["progress"],
                    _duration(snapshot["elapsed_seconds"]),
                    _duration(snapshot["eta_seconds"]),
                )
            )
            if snapshot["current"]:
                current = sorted(
                    snapshot["current"],
                    key=lambda item: item["kind"] != "reconstruction",
                )[0]
                line += " | %s — %s (%.0f%%)" % (
                    current["label"], current["message"], current["progress"])
            if snapshot["failed"]:
                line += " | %d failed" % snapshot["failed"]

            columns = shutil.get_terminal_size((140, 24)).columns
            if len(line) > columns:
                line = line[:max(1, columns - 1)] + "…"
            if self.is_tty:
                padded = line.ljust(self._last_width)
                self.stream.write("\r" + padded)
                self._last_width = len(line)
                if final:
                    self.stream.write("\n")
                    self._last_width = 0
            else:
                self.stream.write(line + "\n")
            self.stream.flush()


def start_share_tunnel(port: int, stream=None, tunnel_factory=None):
    """Expose an existing local HTTP server through Viser's raw TCP tunnel."""

    stream = stream or sys.stdout
    if tunnel_factory is None:
        try:
            from viser._tunnel import ViserTunnel
        except ImportError as exc:
            raise RuntimeError(
                "--share requires viser>=1.0.26; install the share extra"
            ) from exc
        tunnel_factory = ViserTunnel

    tunnel = tunnel_factory(SHARE_DOMAIN, int(port))

    def emit(message: str) -> None:
        print(message, file=stream, flush=True)

    def connected(max_clients: int) -> None:
        url = tunnel.get_url()
        if url:
            emit("share URL (expires in 24 hours, max %d clients): %s" % (
                max_clients, url))
        else:
            emit("share URL unavailable")

    def disconnected() -> None:
        emit("share tunnel disconnected")

    tunnel.on_disconnect(disconnected)
    emit("requesting share URL from %s..." % SHARE_DOMAIN)
    tunnel.on_connect(connected)

    def report_failure() -> None:
        while tunnel.get_status() in ("ready", "connecting"):
            time.sleep(0.05)
        if tunnel.get_status() == "failed":
            emit("share URL unavailable; local viewer is still running")

    threading.Thread(
        target=report_failure, name="spark-share-status", daemon=True,
    ).start()
    return tunnel


def serve_http_server(server, share: bool = False) -> None:
    """Run one local server and close its optional share tunnel cleanly."""

    tunnel = None
    try:
        if share:
            tunnel = start_share_tunnel(server.server_port)
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nviewer stopped")
    finally:
        if tunnel is not None and tunnel.get_status() not in ("failed", "closed"):
            tunnel.close()
        server.server_close()


def serve_live(args) -> None:
    rounds = discover_live_rounds(
        [Path(path) for path in args.runs_dir],
        Path(args.rounds_dir) if args.rounds_dir else None,
    )
    if not rounds:
        sys.exit("no completed runs found")
    cache_dir = (
        Path(args.cache_dir).expanduser().resolve()
        if args.cache_dir
        else default_live_cache_dir(args.runs_dir, args.rounds_dir)
    )
    shared_root = Path(args.shared_eval_dir) if args.shared_eval_dir else None
    cache = LiveBundleCache(
        rounds, cache_dir,
        reconstruction_runs=args.reconstruction,
        shared_eval_root=shared_root,
        shared_eval_roots=[_REPO_ROOT / rel for rel in SHARED_EVAL_ROOTS],
        thumb_width=args.thumb_width,
        track_dt=args.track_dt,
        template_dir=_REPO_ROOT / "webdemo",
        rad_builder=args.rad_builder,
    )
    jobs = LiveExportJobs(cache)
    loading_page = (_REPO_ROOT / "webdemo" / "loading.html").read_bytes()
    # The service worker must be served from the cache root to hold "/" scope.
    shutil.copyfile(_REPO_ROOT / "webdemo" / "sw.js", cache_dir / "sw.js")

    print("live cache: %s" % cache_dir)
    if args.prune_cache:
        result = cache.prune_stale()
        print("pruned %d stale cache entries (%s)" % (
            result["removed"], _byte_size(result["bytes"])))
    index_path = cache.write_cache_index()
    print("cache index: %s" % index_path)
    if args.prepare_cache is not None:
        if args.prepare_cache:
            prepare_names = (
                None if args.prepare_cache == ["all"] else args.prepare_cache)
        else:
            # The visible v6 catalog already contains only gsplat1600. Do not
            # ask for the absent historical name "gsplat" on a bare flag.
            prepare_names = args.reconstruction or sorted({
                view["reconstruction"] for view in cache.views if view["reconstruction"]
            })
        if prepare_names is not None:
            available = {view["reconstruction"] for view in cache.views
                         if view["reconstruction"]}
            missing = sorted(set(prepare_names) - available)
            if missing:
                sys.exit("prewarm reconstruction(s) not found: %s" % ", ".join(missing))
        plan = cache.artifact_plan(prepare_names)
        replay_count = sum(item["kind"] == "replay" for item in plan)
        model_count = len(plan) - replay_count
        model_label = "all reconstructions" if prepare_names is None else ", ".join(prepare_names)
        print("prewarming %d reusable artifacts (%d replays + %d %s models)" % (
            len(plan), replay_count, model_count, model_label))
        try:
            summary = cache.prepare(
                prepare_names, progress=TerminalPrepareProgress())
        except RuntimeError as exc:
            sys.exit(str(exc))
        print("prewarm complete: %d reused, %d built in %s" % (
            summary["cached"], summary["built"],
            _duration(summary["elapsed_seconds"])))

    class Handler(RangeRequestHandler):
        def __init__(self, *handler_args, **kwargs):
            super().__init__(*handler_args, directory=str(cache_dir), **kwargs)

        def log_request(self, code="-", size="-") -> None:
            if urllib.parse.urlparse(self.path).path.startswith("/api/jobs/"):
                return
            super().log_request(code, size)

        def _send_payload(self, payload: bytes, content_type: str) -> None:
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

        def _redirect(self, target: str) -> None:
            self.send_response(303)
            self.send_header("Location", target)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()

        def do_GET(self):
            parsed = urllib.parse.urlparse(self.path)
            if parsed.path.startswith("/api/jobs/"):
                job_id = parsed.path.rsplit("/", 1)[-1]
                try:
                    snapshot = jobs.snapshot(job_id)
                except KeyError:
                    self.send_error(404, "export job not found")
                    return
                self._send_payload(
                    json.dumps(snapshot).encode(), "application/json; charset=utf-8")
                return
            if parsed.path.startswith("/loading/"):
                job_id = parsed.path.rstrip("/").rsplit("/", 1)[-1]
                try:
                    jobs.snapshot(job_id)
                except KeyError:
                    self.send_error(404, "export job not found")
                    return
                self._send_payload(loading_page, "text/html; charset=utf-8")
                return
            if parsed.path in ("/", "/view", "/api/view"):
                try:
                    if parsed.path == "/":
                        query = cache.default_query()
                    else:
                        values = urllib.parse.parse_qs(parsed.query, keep_blank_values=True)
                        query = {key: items[-1] for key, items in values.items()}
                    job_id = jobs.start(query)
                    snapshot = jobs.wait(job_id, 0.05)
                except (KeyError, ValueError) as exc:
                    self.send_error(400, str(exc))
                    return
                except Exception as exc:
                    self.send_error(500, "bundle export failed: %s" % exc)
                    return
                if parsed.path == "/api/view":
                    if snapshot["status"] == "complete":
                        payload = {"status": "complete", "target": snapshot["target"]}
                    elif snapshot["status"] == "error":
                        payload = {"status": "error", "error": snapshot["error"]}
                    else:
                        payload = {"status": "running", "job": job_id}
                    self._send_payload(
                        json.dumps(payload).encode(), "application/json; charset=utf-8")
                    return
                if snapshot["status"] == "complete":
                    self._redirect(snapshot["target"])
                elif snapshot["status"] == "error":
                    self.send_error(500, "bundle export failed: %s" % snapshot["error"])
                else:
                    self._redirect("/loading/%s" % job_id)
                return
            super().do_GET()

    print("serving live viewer at http://127.0.0.1:%d" % args.serve)
    server = http.server.ThreadingHTTPServer(("127.0.0.1", args.serve), Handler)
    serve_http_server(server, share=args.share)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--episode", action="append", default=[],
                        help="episode dir (the method dir); repeat for a compare bundle")
    parser.add_argument("--out", help="output bundle directory")
    parser.add_argument("--live", action="store_true",
                        help="serve the viewer and lazily cache selected A/B views")
    parser.add_argument("--runs-dir", action="append", default=[],
                        help="runs root to scan in --live mode; repeatable")
    parser.add_argument("--rounds-dir",
                        help="experiment-round manifest directory to scan in --live mode")
    parser.add_argument(
        "--cache-dir",
        help="live artifact cache (default: <runs-dir>/.spark-web-cache)",
    )
    parser.add_argument(
        "--prepare-cache", nargs="*", metavar="NAME", default=None,
        help=(
            "prewarm reusable artifacts before serving; defaults to visible models, "
            "or pass names / 'all'"
        ),
    )
    parser.add_argument(
        "--prune-cache", action="store_true",
        help="remove stale generated artifacts and views from the live cache",
    )
    parser.add_argument("--reconstruction", nargs="*", default=None,
                        help="explicit reconstruction names; v6 live defaults to canonical gsplat1600")
    parser.add_argument("--rad", action=argparse.BooleanOptionalAction, default=True,
                        help="convert splats to streamable Spark RAD (default: enabled)")
    parser.add_argument("--build-lod", metavar="PATH",
                        help="standalone Spark build-lod binary (implies --rad)")
    parser.add_argument("--shared-eval-dir", default=None,
                        help="shared eval root (default: newest set covering the episode)")
    parser.add_argument("--thumb-width", type=int, default=240)
    parser.add_argument("--track-dt", type=float, default=0.2,
                        help="distractor track sampling period, sim seconds")
    parser.add_argument("--serve", type=int, nargs="?", const=DEFAULT_LIVE_PORT,
                        default=None, metavar="PORT",
                        help="serve locally; live mode defaults to port 8090")
    parser.add_argument(
        "--share", action="store_true",
        help="request a temporary public URL through share.viser.studio",
    )
    args = parser.parse_args()
    if args.prepare_cache and "all" in args.prepare_cache and args.prepare_cache != ["all"]:
        parser.error("--prepare-cache all cannot be combined with reconstruction names")
    apply_mode_defaults(args)
    args.rad_builder = None
    if args.build_lod:
        candidate = Path(args.build_lod).resolve()
        if not candidate.is_file() or not os.access(candidate, os.X_OK):
            parser.error("--build-lod is not executable: %s" % candidate)
        args.rad_builder = candidate
    elif args.rad:
        found = find_build_lod()
        if found:
            args.rad_builder = found
        else:
            print("warning: build-lod was not found; exporting PLY only")

    if args.live:
        if args.episode or args.out:
            parser.error("--live cannot be combined with --episode/--out")
        if not args.runs_dir and not args.rounds_dir:
            parser.error("--live needs --runs-dir or --rounds-dir")
        if args.runs_dir and args.rounds_dir:
            parser.error("choose either --runs-dir or --rounds-dir")
        if args.prune_cache and args.reconstruction is not None:
            parser.error("--prune-cache needs the complete catalog; omit --reconstruction")
        serve_live(args)
        return
    if args.prepare_cache is not None or args.prune_cache:
        parser.error("--prepare-cache/--prune-cache require live runs browsing")
    if not args.episode or not args.out:
        parser.error("static export needs --episode and --out")

    episode_dirs = [Path(e).resolve() for e in args.episode]
    for episode_dir in episode_dirs:
        if not (episode_dir / "manifest.json").exists():
            sys.exit("not an episode dir (no manifest.json): %s" % episode_dir)
        if not discover_reconstructions(episode_dir):
            print("warning: no reconstructions in %s; replay-only for that run" % episode_dir.name)

    shared_root = (
        Path(args.shared_eval_dir) if args.shared_eval_dir
        else default_shared_eval_root(episode_dirs[0])
    )
    manifest = export_bundle(
        episode_dirs,
        Path(args.out),
        reconstruction_runs=args.reconstruction,
        shared_eval_root=shared_root,
        thumb_width=args.thumb_width,
        track_dt=args.track_dt,
        template_dir=_REPO_ROOT / "webdemo",
        rad_builder=args.rad_builder,
    )
    print("exported %s bundle to %s (%d run%s):" % (
        manifest["mode"], args.out, len(manifest["runs"]),
        "" if len(manifest["runs"]) == 1 else "s"))
    for run in manifest["runs"]:
        total = sum(c["count"] for c in run["points"]["chunks"])
        print("  %-14s %d frames, %d points, reconstructions: %s" % (
            run["method"], len(run["frames"]), total,
            ", ".join(r["name"] for r in run["reconstructions"]) or "none"))
    print("shared eval: %s" % (shared_root or "none found"))

    if args.serve is not None:
        handler = functools.partial(
            RangeRequestHandler, directory=str(Path(args.out)))
        print("serving demo at http://127.0.0.1:%d" % args.serve)
        server = http.server.ThreadingHTTPServer(("127.0.0.1", args.serve), handler)
        serve_http_server(server, share=args.share)


if __name__ == "__main__":
    main()
