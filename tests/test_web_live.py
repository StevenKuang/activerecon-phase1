import functools
import http.server
import io
import json
import threading
import urllib.request
from types import SimpleNamespace
from pathlib import Path

import pytest

import activebench.web_live as web_live
from activebench.web_live import (
    LiveBundleCache,
    LiveExportJobs,
    RangeRequestHandler,
    discover_live_rounds,
    live_catalog,
)
from scripts.export_web_demo import (
    DEFAULT_LIVE_PORT,
    TerminalPrepareProgress,
    apply_mode_defaults,
    default_live_cache_dir,
    find_build_lod,
    start_share_tunnel,
)


def test_runs_dir_implies_live_rad_and_default_port():
    args = SimpleNamespace(
        runs_dir=["runs"], rounds_dir=None, live=False, serve=None,
        rad=True, build_lod=None,
    )

    apply_mode_defaults(args)

    assert args.live is True
    assert args.rad is True
    assert args.serve == DEFAULT_LIVE_PORT == 8090


def test_explicit_live_overrides_are_preserved():
    args = SimpleNamespace(
        runs_dir=[], rounds_dir="rounds", live=False, serve=9000,
        rad=False, build_lod=None,
    )

    apply_mode_defaults(args)

    assert args.live is True
    assert args.rad is False
    assert args.serve == 9000


def test_share_implies_a_local_server_for_static_exports():
    args = SimpleNamespace(
        runs_dir=[], rounds_dir=None, live=False, serve=None,
        share=True, rad=True, build_lod=None,
    )

    apply_mode_defaults(args)

    assert args.live is False
    assert args.serve == DEFAULT_LIVE_PORT


def test_share_tunnel_reports_public_url_without_owning_http_server():
    output = io.StringIO()
    created = []

    class FakeTunnel:
        def __init__(self, domain, port):
            created.append((domain, port))
            self.status = "ready"
            self.url = None

        def on_disconnect(self, _callback):
            pass

        def on_connect(self, callback):
            self.status = "connected"
            self.url = "https://share.example/viewer"
            callback(8)

        def get_status(self):
            return self.status

        def get_url(self):
            return self.url

    tunnel = start_share_tunnel(
        8090, stream=output, tunnel_factory=FakeTunnel)

    assert tunnel.get_status() == "connected"
    assert created == [("share.viser.studio", 8090)]
    assert "expires in 24 hours, max 8 clients" in output.getvalue()
    assert "https://share.example/viewer" in output.getvalue()


def test_live_cache_defaults_to_selected_campaign(tmp_path):
    runs = tmp_path / "runs"
    rounds = tmp_path / "rounds"

    assert default_live_cache_dir([runs], None) == (
        runs.resolve() / ".spark-web-cache")
    assert default_live_cache_dir([], rounds) == (
        rounds.resolve() / ".spark-web-cache")


def test_terminal_prepare_progress_reports_count_and_eta():
    output = io.StringIO()
    reporter = TerminalPrepareProgress(output)

    reporter({
        "status": "complete", "ready": 3, "total": 3,
        "cached": 2, "built": 1, "failed": 0, "progress": 100.0,
        "elapsed_seconds": 7.0, "eta_seconds": 0.0, "current": [],
    })

    line = output.getvalue()
    assert "3/3 ready (2 cached, 1 built)" in line
    assert "elapsed 00:07" in line
    assert "ETA ~00:00" in line


def test_build_lod_is_found_beside_active_python(monkeypatch, tmp_path):
    python = tmp_path / "bin" / "python"
    builder = python.parent / "build-lod"
    python.parent.mkdir()
    builder.write_text("#!/bin/sh\n")
    builder.chmod(0o755)
    monkeypatch.setattr("scripts.export_web_demo.shutil.which", lambda _name: None)
    monkeypatch.setattr("scripts.export_web_demo.sys.executable", str(python))

    assert find_build_lod() == builder.resolve()


def _episode(root: Path, group: str, method: str) -> Path:
    episode = root / group / method
    episode.mkdir(parents=True)
    (episode / "manifest.json").write_text(json.dumps({"method": {"name": method}}))
    return episode


def _reconstruction(episode: Path, name: str) -> None:
    path = episode / "reconstructions" / name / "gaussians.npz"
    path.parent.mkdir(parents=True)
    path.touch()


def _template(root: Path) -> Path:
    template = root / "template"
    template.mkdir()
    for name in ("index.html", "app.js", "style.css"):
        (template / name).write_text(name)
    return template


def test_live_catalog_is_flat_and_prefers_gsplat(tmp_path):
    runs = tmp_path / "runs"
    random = _episode(runs, "room__dyn__s0", "random")
    r3con = _episode(runs, "room__dyn__s0", "r3con-pano")
    _reconstruction(random, "anysplat")
    _reconstruction(random, "gsplat")
    _reconstruction(r3con, "gsplat")

    rounds = discover_live_rounds([runs])
    catalog = live_catalog(rounds)

    assert list(rounds) == ["adhoc"]
    assert catalog["rounds"] == [{"id": "adhoc", "label": "Ad-hoc"}]
    assert catalog["views"] == [
        {"round": "adhoc", "scene": "room", "difficulty": "dyn",
         "method": "r3con-pano", "reconstruction": "gsplat", "seed": "s0"},
        {"round": "adhoc", "scene": "room", "difficulty": "dyn",
         "method": "random", "reconstruction": "gsplat", "seed": "s0"},
        {"round": "adhoc", "scene": "room", "difficulty": "dyn",
         "method": "random", "reconstruction": "anysplat", "seed": "s0"},
    ]


def test_v6_catalog_hides_duplicate_scores_and_isolated_ablations(tmp_path):
    runs = tmp_path / "runs_campaign_v6"
    for method in ("r3con-pano", "gleam"):
        episode = _episode(runs, "interior_0007__d0__s0", method)
        for name in ("gsplat1600", "gsplat1600cube", "gsplat1600uni", "i3dgs1600_gtinit"):
            _reconstruction(episode, name)
    rounds = discover_live_rounds([runs])
    catalog = live_catalog(rounds)
    assert len(catalog["views"]) == 2
    assert {v["reconstruction"] for v in catalog["views"]} == {"gsplat1600"}
    archived = live_catalog(rounds, ["gsplat1600cube"])
    assert len(archived["views"]) == 2
    assert {v["reconstruction"] for v in archived["views"]} == {"gsplat1600cube"}


def test_view_metadata_carries_selected_metrics_and_frame_fractions(
    tmp_path, monkeypatch,
):
    class FakeReplay:
        def __init__(self, episode_dir):
            assert Path(episode_dir) == tmp_path
            self.frames = [SimpleNamespace(index=0), SimpleNamespace(index=2)]
            self.capture_entries = [
                {"index": 0, "distractor_pixel_fraction": 0.125},
                {"index": 2, "distractor_pixel_fraction": 0.5},
            ]

        def summary(self, reconstruction):
            return "selected %s metrics" % reconstruction

    monkeypatch.setattr(web_live, "EpisodeReplay", FakeReplay)

    metadata = LiveBundleCache._view_metadata(tmp_path, "gsplat")

    assert metadata == {
        "summary": "selected gsplat metrics",
        "distractor_pixel_fractions": [0.125, 0.5],
    }


def test_same_scene_panes_select_dimensions_independently(tmp_path):
    runs = tmp_path / "runs"
    d0 = _episode(runs, "room__d0__s0", "r3con-pano")
    dyn = _episode(runs, "room__dyn__s0", "r3con-pano")
    random = _episode(runs, "room__d0__s0", "random")
    seed1 = _episode(runs, "room__d0__s1", "r3con-pano")
    for episode in (d0, dyn, random, seed1):
        _reconstruction(episode, "gsplat")
    _reconstruction(d0, "worldmirror")
    cache = LiveBundleCache(discover_live_rounds([runs]), tmp_path / "cache")
    query = {
        "round": "adhoc", "scene": "room",
        "a_difficulty": "d0", "a_method": "r3con-pano",
        "a_reconstruction": "gsplat", "a_seed": "s0",
        "compare": "1",
        "b_difficulty": "dyn", "b_method": "r3con-pano",
        "b_reconstruction": "gsplat", "b_seed": "s0",
    }

    selection = cache.selection(query)

    assert selection["a"] == {
        "difficulty": "d0", "method": "r3con-pano",
        "reconstruction": "gsplat", "seed": "s0",
    }
    assert selection["b"] == {
        "difficulty": "dyn", "method": "r3con-pano",
        "reconstruction": "gsplat", "seed": "s0",
    }
    assert cache.resolve(query) == [d0, dyn]
    base_b = {
        "b_difficulty": "d0", "b_method": "r3con-pano",
        "b_reconstruction": "gsplat", "b_seed": "s0",
    }
    for key, value, expected in (
        ("difficulty", "dyn", dyn),
        ("method", "random", random),
        ("reconstruction", "worldmirror", d0),
        ("seed", "s1", seed1),
    ):
        changed = cache.selection({
            **query, **base_b, "b_" + key: value,
        })
        assert changed["b"][key] == value
        assert cache.resolve({**query, **base_b, "b_" + key: value}) == [d0, expected]
    cloned = cache.selection({**cache.default_query(), "compare": "1"})
    assert cloned["b"] == cloned["a"]
    inherited = cache.selection({
        **cache.default_query(), "compare": "1", "b_method": "random",
    })
    assert inherited["b"] == {**inherited["a"], "method": "random"}
    with pytest.raises(ValueError, match="B pane selection not found"):
        cache.selection({**query, "b_reconstruction": "worldmirror"})


def test_artifacts_are_reused_across_view_combinations(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    random = _episode(runs, "room__d0__s0", "random")
    r3con = _episode(runs, "room__d0__s0", "r3con-pano")
    for episode in (random, r3con):
        _reconstruction(episode, "gsplat")
        _reconstruction(episode, "anysplat")
    replay_calls = []
    reconstruction_calls = []

    def fake_replay(episode, out_dir, data_prefix, method, **_kwargs):
        replay_calls.append(Path(episode))
        return {
            "method": method,
            "frames": [{"thumb": data_prefix + "/thumbs/000.jpg"}],
            "points": {"bin": data_prefix + "/points.bin", "chunks": []},
            "contam": {"bin": data_prefix + "/contam.bin", "chunks": []},
            "distractors": [{"glb": data_prefix + "/distractors/a.glb"}],
            "reconstructions": [],
        }

    def fake_reconstruction(episode, out_dir, data_prefix, name, **_kwargs):
        reconstruction_calls.append((Path(episode), name))
        return {"name": name, "rad": data_prefix + "/splats/" + name + ".rad", "count": 7}

    monkeypatch.setattr(web_live, "export_replay", fake_replay)
    monkeypatch.setattr(web_live, "export_reconstruction", fake_reconstruction)
    cache = LiveBundleCache(
        discover_live_rounds([runs]), tmp_path / "cache",
        template_dir=_template(tmp_path),
    )
    default_query = cache.default_query()
    single = cache.export(default_query)
    assert cache.export(default_query) == single
    assert replay_calls == [r3con]
    assert reconstruction_calls == [(r3con, "gsplat")]

    clone_query = {**default_query, "compare": "1"}
    clone = cache.export(clone_query)
    clone_manifest = json.loads((clone / "manifest.json").read_text())
    assert clone_manifest["mode"] == "compare"
    assert replay_calls == [r3con]
    assert reconstruction_calls == [(r3con, "gsplat")]

    compare_reconstruction = {
        **clone_query,
        "b_difficulty": "d0", "b_method": "r3con-pano",
        "b_reconstruction": "anysplat", "b_seed": "s0",
    }
    cache.export(compare_reconstruction)
    assert replay_calls == [r3con]
    assert reconstruction_calls == [(r3con, "gsplat"), (r3con, "anysplat")]

    compare_method = {
        **clone_query,
        "b_difficulty": "d0", "b_method": "random",
        "b_reconstruction": "gsplat", "b_seed": "s0",
    }
    compared = cache.export(compare_method)
    manifest = json.loads((compared / "manifest.json").read_text())
    assert replay_calls == [r3con, random]
    assert reconstruction_calls[-1] == (random, "gsplat")
    assert manifest["runs"][0]["replay"].startswith("/artifacts/runs/")
    assert manifest["runs"][1]["reconstruction"].startswith(
        "/artifacts/reconstructions/")
    replay = json.loads((
        tmp_path / "cache" / manifest["runs"][0]["replay"].lstrip("/")
    ).read_text())
    reconstruction = json.loads((
        tmp_path / "cache" / manifest["runs"][1]["reconstruction"].lstrip("/")
    ).read_text())
    assert replay["points"]["bin"].startswith("/artifacts/runs/")
    assert reconstruction["rad"].startswith("/artifacts/reconstructions/")
    assert (compared / "app.js").read_text() == "app.js"


def test_prepare_builds_unique_filtered_artifacts_once(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    random = _episode(runs, "room__d0__s0", "random")
    r3con = _episode(runs, "room__d0__s0", "r3con-pano")
    for episode in (random, r3con):
        _reconstruction(episode, "gsplat")
        _reconstruction(episode, "anysplat")
    replay_calls = []
    reconstruction_calls = []

    def fake_replay(episode, _out_dir, data_prefix, method, **_kwargs):
        replay_calls.append(Path(episode))
        return {
            "method": method,
            "frames": [],
            "points": {"bin": data_prefix + "/points.bin", "chunks": []},
            "contam": {"bin": data_prefix + "/contam.bin", "chunks": []},
            "distractors": [],
            "reconstructions": [],
        }

    def fake_reconstruction(episode, _out_dir, data_prefix, name, **_kwargs):
        reconstruction_calls.append((Path(episode), name))
        return {
            "name": name,
            "rad": data_prefix + "/splats/model.rad",
            "count": 7,
        }

    monkeypatch.setattr(web_live, "export_replay", fake_replay)
    monkeypatch.setattr(web_live, "export_reconstruction", fake_reconstruction)
    cache = LiveBundleCache(discover_live_rounds([runs]), tmp_path / "cache")
    snapshots = []

    summary = cache.prepare(["gsplat"], progress=snapshots.append)

    assert summary["status"] == "complete"
    assert summary["total"] == 4
    assert summary["cached"] == 0
    assert summary["built"] == 4
    assert snapshots[0]["eta_seconds"] > 0
    assert snapshots[-1]["ready"] == 4
    assert set(replay_calls) == {random, r3con}
    assert set(reconstruction_calls) == {
        (random, "gsplat"), (r3con, "gsplat"),
    }
    assert (cache.cache_dir / "prepare-stats.json").is_file()

    index = json.loads((cache.cache_dir / "cache-index.json").read_text())
    assert len(index["artifacts"]) == 6
    assert sum(item["cached"] for item in index["artifacts"]) == 4
    gsplat = next(item for item in index["artifacts"]
                   if item.get("name") == "gsplat")
    assert gsplat["target"].startswith("artifacts/reconstructions/")
    assert gsplat["selections"][0]["reconstruction"] == "gsplat"

    again = cache.prepare(["gsplat"])
    assert again["cached"] == 4
    assert again["built"] == 0
    assert len(replay_calls) == 2
    assert len(reconstruction_calls) == 2


def test_prune_stale_preserves_current_artifacts(tmp_path):
    runs = tmp_path / "runs"
    episode = _episode(runs, "room__d0__s0", "random")
    _reconstruction(episode, "gsplat")
    cache = LiveBundleCache(discover_live_rounds([runs]), tmp_path / "cache")
    current = cache.artifact_plan(["gsplat"])[0]
    current_path = cache.cache_dir / current["target"]
    current_path.parent.mkdir(parents=True)
    current_path.write_text("{}")
    stale = cache.cache_dir / "artifacts" / "runs" / "obsolete"
    stale.mkdir(parents=True)
    (stale / "run.json").write_text("old")
    old_view = cache.cache_dir / "views" / "obsolete"
    old_view.mkdir(parents=True)
    (old_view / "manifest.json").write_text("old")

    result = cache.prune_stale()

    assert result["removed"] == 2
    assert result["bytes"] == 6
    assert current_path.is_file()
    assert not stale.exists()
    assert not old_view.exists()


def test_live_export_jobs_report_each_pane_and_deduplicate(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    random = _episode(runs, "room__d0__s0", "random")
    r3con = _episode(runs, "room__d0__s0", "r3con-pano")
    for episode in (random, r3con):
        _reconstruction(episode, "gsplat")
    replay_calls = []
    reconstruction_calls = []
    entered_rad = threading.Event()
    release_rad = threading.Event()

    def fake_replay(episode, _out_dir, data_prefix, method, progress=None, **_kwargs):
        replay_calls.append(Path(episode))
        if progress:
            progress(0.5, "Packing replay")
            progress(1.0, "Replay ready")
        return {
            "method": method,
            "frames": [{"thumb": data_prefix + "/thumbs/000.jpg"}],
            "points": {"bin": data_prefix + "/points.bin", "chunks": []},
            "contam": {"bin": data_prefix + "/contam.bin", "chunks": []},
            "distractors": [],
            "reconstructions": [],
        }

    def fake_reconstruction(
        episode, _out_dir, data_prefix, name, progress=None, **_kwargs,
    ):
        reconstruction_calls.append((Path(episode), name))
        if progress:
            progress(0.32, "Building streamable RAD")
        entered_rad.set()
        assert release_rad.wait(2)
        if progress:
            progress(1.0, "Reconstruction ready")
        return {"name": name, "rad": data_prefix + "/splats/model.rad", "count": 7}

    monkeypatch.setattr(web_live, "export_replay", fake_replay)
    monkeypatch.setattr(web_live, "export_reconstruction", fake_reconstruction)
    cache = LiveBundleCache(
        discover_live_rounds([runs]), tmp_path / "cache",
        template_dir=_template(tmp_path),
    )
    query = {
        **cache.default_query(), "compare": "1",
        "b_difficulty": "d0", "b_method": "random",
        "b_reconstruction": "gsplat", "b_seed": "s0",
    }
    jobs = LiveExportJobs(cache)

    job_id = jobs.start(query)
    assert entered_rad.wait(2)
    assert jobs.start(query) == job_id
    running = jobs.snapshot(job_id)
    assert running["status"] == "running"
    assert [pane["tag"] for pane in running["panes"]] == ["A", "B"]
    assert any(pane["components"]["reconstruction"] == pytest.approx(0.32)
               for pane in running["panes"])

    release_rad.set()
    complete = jobs.wait(job_id, 2)
    assert complete["status"] == "complete"
    assert complete["target"].startswith("/views/")
    assert [pane["progress"] for pane in complete["panes"]] == [100, 100]
    assert set(replay_calls) == {r3con, random}
    assert set(reconstruction_calls) == {(r3con, "gsplat"), (random, "gsplat")}
    assert jobs.start(query) == job_id


def test_live_cache_rejects_unknown_scene_and_exact_view(tmp_path):
    runs = tmp_path / "runs"
    episode = _episode(runs, "room__d0__s0", "random")
    _reconstruction(episode, "gsplat")
    cache = LiveBundleCache(discover_live_rounds([runs]), tmp_path / "cache")

    with pytest.raises(ValueError, match="scene not found"):
        cache.selection({"round": "adhoc", "scene": "missing"})
    with pytest.raises(ValueError, match="A pane selection not found"):
        cache.selection({
            "round": "adhoc", "scene": "room", "a_difficulty": "d0",
            "a_method": "missing", "a_reconstruction": "gsplat", "a_seed": "s0",
        })


def test_range_handler_serves_requested_bytes(tmp_path):
    (tmp_path / "artifacts").mkdir()
    (tmp_path / "artifacts" / "model.rad").write_bytes(b"0123456789")
    handler = functools.partial(RangeRequestHandler, directory=str(tmp_path))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        request = urllib.request.Request(
            "http://127.0.0.1:%d/artifacts/model.rad" % server.server_port,
            headers={"Range": "bytes=2-5"},
        )
        with urllib.request.urlopen(request) as response:
            assert response.status == 206
            assert response.headers["Content-Range"] == "bytes 2-5/10"
            assert "immutable" in response.headers["Cache-Control"]
            assert response.read() == b"2345"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
