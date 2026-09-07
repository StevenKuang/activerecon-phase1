"""Web-demo export: PLY conversion, chunk packing, manifest assembly."""

import json
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import activebench.web_export as web_export  # noqa: E402
from activebench.web_export import (  # noqa: E402
    SH_C0,
    build_rad_from_ply,
    discover_reconstructions,
    pack_point_chunks,
    ply_from_gaussians,
    run_manifest,
    sample_tracks,
)


def test_build_rad_uses_spark_output_convention(tmp_path, monkeypatch):
    ply = tmp_path / "gsplat.ply"
    ply.write_bytes(b"ply")
    calls = []

    def fake_run(command, cwd, check, **kwargs):
        calls.append((command, cwd, check, kwargs))
        ply.with_name("gsplat-lod.rad").write_bytes(b"rad")

    monkeypatch.setattr(web_export.subprocess, "run", fake_run)

    output = build_rad_from_ply(ply, tmp_path / "build-lod")

    assert output == tmp_path / "gsplat-lod.rad"
    assert calls[0][0][1:] == [str(ply), "--quick", "--max-sh=0"]
    assert calls[0][1:3] == (tmp_path, True)
    assert calls[0][3] == {
        "stdout": web_export.subprocess.PIPE,
        "stderr": web_export.subprocess.STDOUT,
        "text": True,
    }


def test_build_rad_passes_sh_degree_through(tmp_path, monkeypatch):
    ply = tmp_path / "gsplat.ply"
    ply.write_bytes(b"ply")
    calls = []

    def fake_run(command, cwd, check, **kwargs):
        calls.append(command)
        ply.with_name("gsplat-lod.rad").write_bytes(b"rad")

    monkeypatch.setattr(web_export.subprocess, "run", fake_run)
    build_rad_from_ply(ply, tmp_path / "build-lod", sh_degree=3)
    assert calls[0][-1] == "--max-sh=3"


def test_lod_fallback_preserves_sh3(tmp_path, monkeypatch):
    ply = tmp_path / "gsplat.ply"
    ply.write_bytes(b"ply")
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        if "--quick" in command:
            raise web_export.subprocess.CalledProcessError(-11, command, "builder crash")
        ply.with_name("gsplat-lod.rad").write_bytes(b"rad")

    monkeypatch.setattr(web_export.subprocess, "run", run)
    with pytest.warns(RuntimeWarning, match="SH3 preserved"):
        build_rad_from_ply(ply, tmp_path / "build-lod", sh_degree=3)
    assert [c[-2:] for c in commands] == [
        ["--quick", "--max-sh=3"], ["--quality", "--max-sh=3"]]


def test_raw_dc_survives_export_without_rgb_clamping(tmp_path):
    g = _gaussians(n=2, sh_degree=3)
    raw = np.array([[[-4., 0., 4.]], [[-2., 1., 3.]]], dtype=np.float32)
    np.savez(tmp_path / "model.npz", **g, sh0=raw)
    web_export.splats_npz_to_ply(tmp_path / "model.npz", tmp_path / "model.ply")
    _, columns = _parse_3dgs_ply((tmp_path / "model.ply").read_bytes())
    actual = np.stack([columns[f"f_dc_{i}"] for i in range(3)], axis=1)
    np.testing.assert_array_equal(actual, raw[:, 0, :])


def test_replay_export_does_not_auto_select_a_scoring_alias(tmp_path):
    from PIL import Image
    from activebench.replay import EpisodeReplay

    episode = tmp_path / "room__d0__s0" / "r3con-pano"
    episode.mkdir(parents=True)
    Image.new("RGB", (4, 4), (120, 80, 50)).save(episode / "frame.png")
    np.save(episode / "depth.npy", np.ones((4, 4), dtype=np.float32))
    (episode / "manifest.json").write_text(json.dumps({
        "method": {"name": "r3con-pano"}, "num_captures": 1,
        "clock": {"final_sim_time": 0},
        "captures": [{"index": 0, "time": 0, "pose": [0, 0, 0, 0, 0]}],
    }))
    (episode / "transforms.json").write_text(json.dumps({
        "w": 4, "h": 4, "fl_x": 2, "fl_y": 2, "cx": 2, "cy": 2,
        "frames": [{"file_path": "frame.png", "depth_path": "depth.npy",
                    "transform_matrix": np.eye(4).tolist()}],
    }))
    for name in ("gsplat1600", "gsplat1600cube"):
        folder = episode / "reconstructions" / name
        folder.mkdir(parents=True)
        (folder / "eval.json").write_text("{}")
    with pytest.raises(ValueError, match="multiple reconstruction"):
        EpisodeReplay(episode).summary(None)
    result = web_export.export_replay(episode, tmp_path / "out", "data", "r3con-pano")
    assert len(result["frames"]) == 1
    assert "PSNR" not in result["summary"]


def test_unversioned_rad_and_sh0_fallback_are_not_reused(tmp_path):
    model = tmp_path / "gaussians.npz"
    model.write_bytes(b"source")
    rad = tmp_path / "gaussians-lod.rad"
    rad.write_bytes(b"rad")
    assert not web_export.current_campaign_rad(model, rad, 3)
    web_export.write_rad_provenance(model, rad, 0)
    assert not web_export.current_campaign_rad(model, rad, 3)
    web_export.write_rad_provenance(model, rad, 3)
    assert web_export.current_campaign_rad(model, rad, 3)
    model.write_bytes(b"changed source")
    assert not web_export.current_campaign_rad(model, rad, 3)


def _parse_3dgs_ply(payload: bytes):
    """Minimal standard-3DGS PLY reader used only by these tests."""

    end = payload.index(b"end_header\n") + len(b"end_header\n")
    header = payload[:end].decode("ascii").splitlines()
    assert header[0] == "ply"
    assert header[1] == "format binary_little_endian 1.0"
    count = int(next(l.split()[-1] for l in header if l.startswith("element vertex")))
    names = [l.split()[-1] for l in header if l.startswith("property")]
    assert all(l.split()[1] == "float" for l in header if l.startswith("property"))
    data = np.frombuffer(payload[end:], dtype=np.float32).reshape(count, len(names))
    return names, {name: data[:, i] for i, name in enumerate(names)}


def _gaussians(n=4, sh_degree: int = 0):
    rng = np.random.default_rng(0)
    quats = rng.normal(size=(n, 4)).astype(np.float32)
    quats /= np.linalg.norm(quats, axis=1, keepdims=True)
    g = dict(
        centers=rng.uniform(-5, 5, (n, 3)).astype(np.float32),
        scales=rng.uniform(1e-4, 0.5, (n, 3)).astype(np.float32),
        quats_wxyz=quats,
        opacities=rng.uniform(0.1, 0.9, n).astype(np.float32),
        colors=rng.uniform(0, 1, (n, 3)).astype(np.float32),
    )
    if sh_degree > 0:
        coeffs = (sh_degree + 1) ** 2 - 1
        g["sh_rest"] = rng.normal(size=(n, coeffs, 3)).astype(np.float32)
        g["sh_degree"] = sh_degree
    return g


def test_ply_property_layout_is_standard_3dgs():
    names, _ = _parse_3dgs_ply(ply_from_gaussians(**_gaussians()))
    assert names == [
        "x", "y", "z", "nx", "ny", "nz",
        "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
        "scale_0", "scale_1", "scale_2",
        "rot_0", "rot_1", "rot_2", "rot_3",
    ]


def test_ply_property_layout_includes_full_sh3_band():
    g = _gaussians(n=3, sh_degree=3)
    names, cols = _parse_3dgs_ply(ply_from_gaussians(**g))
    # 17 base properties + 45 SH3 coefficients (15 coeffs * 3 channels).
    assert names[:17] == [
        "x", "y", "z", "nx", "ny", "nz",
        "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
        "scale_0", "scale_1", "scale_2",
        "rot_0", "rot_1", "rot_2", "rot_3",
    ]
    assert names[17:] == ["f_rest_%d" % k for k in range(45)]
    assert np.all(np.isfinite(cols["f_rest_0"]))


def test_ply_roundtrips_through_3dgs_activations():
    g = _gaussians(8)
    _, cols = _parse_3dgs_ply(ply_from_gaussians(**g))
    xyz = np.stack([cols["x"], cols["y"], cols["z"]], axis=1)
    np.testing.assert_allclose(xyz, g["centers"], rtol=1e-6)
    rgb = np.stack([cols["f_dc_%d" % i] for i in range(3)], axis=1) * SH_C0 + 0.5
    np.testing.assert_allclose(rgb, g["colors"], atol=1e-6)
    opacity = 1.0 / (1.0 + np.exp(-cols["opacity"]))
    np.testing.assert_allclose(opacity, g["opacities"], atol=1e-5)
    scales = np.exp(np.stack([cols["scale_%d" % i] for i in range(3)], axis=1))
    np.testing.assert_allclose(scales, g["scales"], rtol=1e-5)
    rot = np.stack([cols["rot_%d" % i] for i in range(4)], axis=1)
    np.testing.assert_allclose(rot, g["quats_wxyz"], atol=1e-6)


def test_ply_preserves_sh3_coefficients_in_standard_channel_major_order():
    g = _gaussians(n=5, sh_degree=3)
    _, cols = _parse_3dgs_ply(ply_from_gaussians(**g))
    # Independent Spark/INRIA reader contract: f_rest_{c*K + k}.
    # Do not round-trip using the exporter's own flattening convention.
    for k in range(15):
        for c in range(3):
            np.testing.assert_allclose(
                cols["f_rest_%d" % (c * 15 + k)], g["sh_rest"][:, k, c], rtol=1e-6
            )


def test_ply_sh_degree_zero_omits_f_rest_even_with_extra_coeffs():
    # An explicit sh_degree=0 must always produce a DC-only PLY (the SH0
    # fallback), even if sh_rest is accidentally supplied.
    g = _gaussians(n=2, sh_degree=3)
    payload = ply_from_gaussians(
        centers=g["centers"], scales=g["scales"], quats_wxyz=g["quats_wxyz"],
        opacities=g["opacities"], colors=g["colors"],
        sh_rest=g["sh_rest"], sh_degree=0,
    )
    names, _ = _parse_3dgs_ply(payload)
    assert "f_rest_0" not in names


def test_ply_sh_rest_wrong_shape_raises():
    g = _gaussians(n=4, sh_degree=3)
    bad = g["sh_rest"][:, :-1, :]  # wrong coefficient count
    with pytest.raises(ValueError, match="sh_rest must have shape"):
        ply_from_gaussians(
            centers=g["centers"], scales=g["scales"], quats_wxyz=g["quats_wxyz"],
            opacities=g["opacities"], colors=g["colors"],
            sh_rest=bad, sh_degree=3,
        )


def test_splats_npz_to_ply_returns_count_and_sh_degree(tmp_path):
    g = _gaussians(n=6, sh_degree=3)
    npz = tmp_path / "gaussians.npz"
    np.savez(npz, **g)
    count, sh = web_export.splats_npz_to_ply(npz, tmp_path / "out.ply")
    assert count == 6
    assert sh == 3


def test_splats_npz_to_ply_legacy_sh0_npz_falls_back(tmp_path):
    # Legacy NPZ without sh_rest/sh_degree fields must round-trip as SH0.
    g = _gaussians(n=4, sh_degree=0)
    npz = tmp_path / "legacy.npz"
    np.savez(npz, **g)
    count, sh = web_export.splats_npz_to_ply(npz, tmp_path / "out.ply")
    assert count == 4
    assert sh == 0
    names, _ = _parse_3dgs_ply((tmp_path / "out.ply").read_bytes())
    assert "f_rest_0" not in names


def test_splats_npz_to_ply_sh0_override_on_sh3_npz(tmp_path):
    g = _gaussians(n=3, sh_degree=3)
    npz = tmp_path / "gaussians.npz"
    np.savez(npz, **g)
    count, sh = web_export.splats_npz_to_ply(
        npz, tmp_path / "out.ply", sh_degree=0)
    assert sh == 0
    names, _ = _parse_3dgs_ply((tmp_path / "out.ply").read_bytes())
    assert "f_rest_0" not in names


def test_read_sh_degree_returns_recorded_value(tmp_path):
    g = _gaussians(n=2, sh_degree=3)
    npz = tmp_path / "gsplat.npz"
    np.savez(npz, **g)
    assert web_export.read_sh_degree(npz) == 3


def test_read_sh_degree_legacy_npz_returns_zero(tmp_path):
    g = _gaussians(n=2, sh_degree=0)
    npz = tmp_path / "legacy.npz"
    np.savez(npz, **g)
    assert web_export.read_sh_degree(npz) == 0


def test_read_sh_degree_handles_empty_or_corrupt_file(tmp_path):
    empty = tmp_path / "empty.npz"
    empty.touch()
    assert web_export.read_sh_degree(empty) == 0
    bogus = tmp_path / "bogus.npz"
    bogus.write_bytes(b"not a zip")
    assert web_export.read_sh_degree(bogus) == 0


def test_ply_survives_degenerate_scales_and_saturated_opacity():
    g = _gaussians(3)
    g["scales"][0] = 0.0
    g["opacities"][1] = 1.0
    g["opacities"][2] = 0.0
    payload = ply_from_gaussians(**g)
    _, cols = _parse_3dgs_ply(payload)
    for name, col in cols.items():
        assert np.all(np.isfinite(col)), name
    assert 1.0 / (1.0 + np.exp(-cols["opacity"][1])) > 0.999
    assert np.exp(cols["scale_0"][0]) < 1e-8


def test_pack_point_chunks_offsets_alignment_and_roundtrip():
    rng = np.random.default_rng(1)
    chunks = []
    for n in (5, 0, 3):  # odd counts force padding; empty chunk allowed
        chunks.append((
            rng.normal(size=(n, 3)).astype(np.float32),
            rng.integers(0, 256, (n, 3)).astype(np.uint8),
        ))
    payload, index = pack_point_chunks(chunks)
    assert [e["count"] for e in index] == [5, 0, 3]
    for entry, (pts, rgb) in zip(index, chunks):
        assert entry["offset"] % 4 == 0
        n = entry["count"]
        got_pts = np.frombuffer(
            payload, dtype=np.float32, count=n * 3, offset=entry["offset"]
        ).reshape(n, 3)
        got_rgb = np.frombuffer(
            payload, dtype=np.uint8, count=n * 3, offset=entry["offset"] + n * 12
        ).reshape(n, 3)
        np.testing.assert_array_equal(got_pts, pts)
        np.testing.assert_array_equal(got_rgb, rgb)


class _Traj:
    def pose_at(self, t):
        return np.array([t, 2 * t, 0.0]), 0.5 * t


class _FakeReplay:
    """Duck-typed EpisodeReplay covering what web_manifest reads."""

    class _I:
        width, height, fx, fy, cx, cy = 64, 48, 60.0, 60.0, 32.0, 24.0

    class _F:
        def __init__(self, i, t):
            self.index, self.time = i, t
            self.c2w_gl = np.eye(4) + 0.0 * i

    def __init__(self, tmp_path):
        self.episode_dir = tmp_path
        self.intrinsics = self._I()
        self.frames = [self._F(0, 0.0), self._F(1, 2.0), self._F(2, 5.0)]
        self.capture_entries = [
            {"index": 0, "distractor_pixel_fraction": 0.0},
            {"index": 1, "distractor_pixel_fraction": 0.125},
            {"index": 2, "distractor_pixel_fraction": 0.5},
        ]
        self.t_end = 5.0
        self.distractors = [("ghost", _Traj(), None)]

    def summary(self, reconstruction_run=None):
        return "fake summary"


def test_sample_tracks_covers_clock_and_shapes():
    replay = _FakeReplay(Path("/nonexistent"))
    tracks = sample_tracks(replay, dt=0.5)
    assert [t["name"] for t in tracks] == ["ghost"]
    samples = np.asarray(tracks[0]["track"], dtype=np.float64)
    assert tracks[0]["dt"] == 0.5
    assert len(samples) == 11  # 0..5 s inclusive at 0.5 s
    np.testing.assert_allclose(samples[4], [2.0, 4.0, 0.0, 1.0])  # t=2.0


def _run_manifest(replay, **overrides):
    kwargs = dict(
        method="r3con-pano",
        points_bin="data/r3con-pano/points.bin",
        contam_bin="data/r3con-pano/contam.bin",
        reconstructions=[{"name": "gsplat", "ply": "data/r3con-pano/splats/gsplat.ply", "count": 7}],
        point_index=[{"offset": 0, "count": 5}] * 3,
        contam_index=[{"offset": 0, "count": 0}] * 3,
        thumbs=["data/r3con-pano/thumbs/%03d.jpg" % i for i in range(3)],
        tracks=sample_tracks(replay, dt=1.0),
        eval_frusta=[{"c2w": list(range(16)), "stratum": "level"}],
        shared_frusta=[{"c2w": list(range(16)), "cls": "severe"}],
    )
    kwargs.update(overrides)
    return run_manifest(replay, **kwargs)


def test_run_manifest_shapes_and_reconstruction_decoupling(tmp_path):
    replay = _FakeReplay(tmp_path)
    run = _run_manifest(replay)
    assert run["method"] == "r3con-pano"
    assert run["t_end"] == 5.0
    assert [f["time"] for f in run["frames"]] == [0.0, 2.0, 5.0]
    assert [f["distractor_pixel_fraction"] for f in run["frames"]] == [
        0.0, 0.125, 0.5,
    ]
    assert all(len(f["c2w"]) == 16 for f in run["frames"])
    assert run["frames"][1]["thumb"] == "data/r3con-pano/thumbs/001.jpg"
    # per-run namespaced asset paths, so compare runs never collide
    assert run["points"]["bin"] == "data/r3con-pano/points.bin"
    # replay data must not depend on which reconstruction is active
    assert run["reconstructions"][0]["name"] == "gsplat"
    assert "reconstruction" not in run["frames"][0]
    json.dumps(run)  # everything JSON-serializable


def test_discover_reconstructions_leads_with_optimizer_backend(tmp_path):
    root = tmp_path / "reconstructions"
    for name in ("worldmirror", "anysplat", "gsplat"):
        (root / name).mkdir(parents=True)
        np.savez(root / name / "gaussians.npz", **_gaussians(1))
    names = list(discover_reconstructions(tmp_path))
    assert names[0] == "gsplat"  # preferred default regardless of glob order
    assert set(names) == {"gsplat", "anysplat", "worldmirror"}


def test_run_manifest_requires_matching_chunk_index(tmp_path):
    replay = _FakeReplay(tmp_path)
    with pytest.raises(ValueError):
        _run_manifest(replay, point_index=[{"offset": 0, "count": 5}])  # 1 entry for 3 frames


def test_export_bundle_modes_and_namespacing(tmp_path, monkeypatch):
    import activebench.web_export as web

    seen = []

    def fake_export_run(episode_dir, out_dir, data_prefix, method, **kw):
        seen.append((str(episode_dir), data_prefix, method))
        return {"method": method, "points": {"bin": "%s/points.bin" % data_prefix}}

    monkeypatch.setattr(web, "export_run", fake_export_run)

    single = web.export_bundle([tmp_path / "runs/scene/r3con-pano"], tmp_path / "s")
    assert single["mode"] == "single" and len(single["runs"]) == 1

    compare = web.export_bundle(
        [tmp_path / "runs/scene/r3con-pano", tmp_path / "runs/scene/random"],
        tmp_path / "c")
    assert compare["mode"] == "compare"
    assert [r["method"] for r in compare["runs"]] == ["r3con-pano", "random"]
    # distinct data prefixes so the two runs' assets never collide
    prefixes = [p for _e, p, _m in seen if "/c" not in p]  # crude filter is fine
    assert compare["runs"][0]["points"]["bin"] != compare["runs"][1]["points"]["bin"]
    assert json.loads((tmp_path / "c" / "manifest.json").read_text())["mode"] == "compare"


def test_export_bundle_accepts_per_run_reconstruction_filters(tmp_path, monkeypatch):
    import activebench.web_export as web

    seen = []

    def fake_export_run(_episode_dir, _out_dir, _data_prefix, method, **kwargs):
        seen.append((method, kwargs["reconstruction_runs"]))
        return {"method": method}

    monkeypatch.setattr(web, "export_run", fake_export_run)
    episodes = [tmp_path / "scene/a", tmp_path / "scene/b"]
    web.export_bundle(
        episodes, tmp_path / "out",
        per_run_reconstructions=[["gsplat"], ["worldmirror"]],
    )

    assert seen == [("a", ["gsplat"]), ("b", ["worldmirror"])]
    with pytest.raises(ValueError, match="must match episode dirs"):
        web.export_bundle(episodes, tmp_path / "bad", per_run_reconstructions=[["gsplat"]])


def test_export_bundle_disambiguates_repeated_method_name(tmp_path, monkeypatch):
    import activebench.web_export as web

    monkeypatch.setattr(
        web, "export_run",
        lambda episode_dir, out_dir, data_prefix, method, **kw: {
            "method": method, "prefix": data_prefix})
    manifest = web.export_bundle(
        [tmp_path / "a/random", tmp_path / "b/random"], tmp_path / "out")
    prefixes = [r["prefix"] for r in manifest["runs"]]
    assert prefixes == ["data/random", "data/random-2"]


# --- campaign RAD reuse fast-path tests ----------------------------------------


def test_campaign_rad_for_names_sibling_next_to_npz(tmp_path):
    npz = tmp_path / "reconstructions" / "gsplat" / "gaussians.npz"
    npz.parent.mkdir(parents=True)
    assert web_export.campaign_rad_for(npz) == npz.parent / "gaussians-lod.rad"


def _make_episode_with_npz(tmp_path, name="gsplat", sh_degree=3):
    """Create a minimal episode dir with a real gaussians.npz."""
    ep = tmp_path / "scene__d0__s0" / "r3con-pano"
    npz = ep / "reconstructions" / name / "gaussians.npz"
    npz.parent.mkdir(parents=True)
    g = _gaussians(n=5, sh_degree=sh_degree)
    np.savez(npz, **g)
    return ep, npz


def _patch_replay(monkeypatch, summary="fake summary"):
    """Stub EpisodeReplay so export tests don't need a real manifest.json."""

    class _FakeER:
        def __init__(self, episode_dir):
            pass

        def summary(self, reconstruction_run=None):
            return summary

    # export_reconstruction does ``from activebench.replay import EpisodeReplay``
    # inside the function body, so patch the source module.
    import activebench.replay as replay_mod

    monkeypatch.setattr(replay_mod, "EpisodeReplay", _FakeER)


def test_export_reconstruction_reuses_fresh_sibling_rad(tmp_path, monkeypatch):
    """Fast path: fresh sibling RAD → hardlinked into output, build_rad never called."""
    _patch_replay(monkeypatch)
    ep, npz = _make_episode_with_npz(tmp_path)
    rad = npz.parent / "gaussians-lod.rad"
    rad.write_bytes(b"FAKE-RAD-DATA")
    web_export.write_rad_provenance(npz, rad, 3)

    def fail_build(*args, **kwargs):
        raise AssertionError("build_rad_from_ply must not be called on reuse path")

    monkeypatch.setattr(web_export, "build_rad_from_ply", fail_build)

    out = tmp_path / "export"
    manifest = web_export.export_reconstruction(
        ep, out, "data/r3con-pano", "gsplat",
        rad_builder=Path("/fake/build-lod"),
    )

    dest = out / "data" / "r3con-pano" / "splats" / "gsplat-lod.rad"
    assert manifest["rad"] == "data/r3con-pano/splats/gsplat-lod.rad"
    assert "ply" not in manifest
    assert manifest["count"] == 5
    assert manifest["sh_degree"] == 3
    assert dest.read_bytes() == b"FAKE-RAD-DATA"
    assert (tmp_path / "export" / "data" / "r3con-pano" / "splats" / "gsplat.ply").exists() is False


def test_export_reconstruction_stale_rad_falls_through(tmp_path, monkeypatch):
    """Stale RAD (older than npz) → slow path builds from scratch."""
    _patch_replay(monkeypatch)
    ep, npz = _make_episode_with_npz(tmp_path)
    rad = npz.parent / "gaussians-lod.rad"
    rad.write_bytes(b"STALE-RAD")

    # Make RAD older than npz.
    import os
    older_ns = npz.stat().st_mtime_ns - 10_000_000_000
    os.utime(rad, ns=(older_ns, older_ns))

    calls = []

    def fake_build(ply_path, builder, sh_degree=0):
        calls.append(("build", sh_degree))
        out_rad = ply_path.with_name(ply_path.stem + "-lod.rad")
        out_rad.write_bytes(b"BUILT-RAD")
        return out_rad

    monkeypatch.setattr(web_export, "build_rad_from_ply", fake_build)

    out = tmp_path / "export"
    manifest = web_export.export_reconstruction(
        ep, out, "data/r3con-pano", "gsplat",
        rad_builder=Path("/fake/build-lod"),
    )

    assert calls == [("build", 3)]
    dest = out / "data" / "r3con-pano" / "splats" / "gsplat-lod.rad"
    assert dest.read_bytes() == b"BUILT-RAD"
    assert "ply" not in manifest
    assert manifest["count"] == 5


def test_export_reconstruction_no_sibling_rad_takes_slow_path(tmp_path, monkeypatch):
    """No sibling RAD → slow path builds from scratch."""
    _patch_replay(monkeypatch)
    ep, npz = _make_episode_with_npz(tmp_path)
    calls = []

    def fake_build(ply_path, builder, sh_degree=0):
        calls.append(("build", sh_degree))
        out_rad = ply_path.with_name(ply_path.stem + "-lod.rad")
        out_rad.write_bytes(b"BUILT-RAD")
        return out_rad

    monkeypatch.setattr(web_export, "build_rad_from_ply", fake_build)

    out = tmp_path / "export"
    manifest = web_export.export_reconstruction(
        ep, out, "data/r3con-pano", "gsplat",
        rad_builder=Path("/fake/build-lod"),
    )

    assert calls == [("build", 3)]
    dest = out / "data" / "r3con-pano" / "splats" / "gsplat-lod.rad"
    assert dest.read_bytes() == b"BUILT-RAD"
    assert "ply" not in manifest


def test_export_reconstruction_no_rad_builder_skips_reuse_even_with_sibling(
    tmp_path, monkeypatch,
):
    """Sibling RAD exists but rad_builder=None → slow path (no RAD produced)."""
    _patch_replay(monkeypatch)
    ep, npz = _make_episode_with_npz(tmp_path)
    rad = npz.parent / "gaussians-lod.rad"
    rad.write_bytes(b"FAKE-RAD-DATA")

    out = tmp_path / "export"
    manifest = web_export.export_reconstruction(
        ep, out, "data/r3con-pano", "gsplat",
        rad_builder=None,
    )

    assert "rad" not in manifest
    assert "ply" in manifest
