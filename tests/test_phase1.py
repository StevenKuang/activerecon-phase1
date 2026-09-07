"""Release selection, protocol isolation and safe resume checks."""
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def module(name):
    spec = importlib.util.spec_from_file_location("phase1_" + name, ROOT / "scripts/phase1" / (name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_no_pairing_across_protocol_groups_or_seeds():
    report = module("report")
    row = dict(group="mesh", scene="same", method="random", seed=0, condition="d0",
               psnr_severe=10., psnr_clean=20., psnr_shared=15., path_length_m=1., actual_duration_s=300.)
    assert report.paired_rows([row, {**row, "group": "gs", "condition": "dyn"}]) == []
    assert report.paired_rows([row, {**row, "seed": 1, "condition": "dyn"}]) == []
    pair = report.paired_rows([row, {**row, "condition": "dyn", "psnr_severe": 11., "psnr_clean": 23.}])[0]
    assert pair["regional_contrast"] == -2.
    assert pair["delta_severe"] > 0.  # Negative contrast does not mean an absolute loss.
    with pytest.raises(ValueError, match="Duplicate"):
        report.paired_rows([row, row])


def test_frozen_gavis_override_is_per_group():
    import json
    runner = module("run")
    cells = json.loads((ROOT / "phase1/campaign.json").read_text())["cells"]
    assert len(cells) == 64
    assert sum(c["status"] == "retained" for c in cells) == 60
    for group, budget, points in [("mesh", 300, 10000), ("gs", 270, 2000)]:
        cell = next(c for c in cells if c["group"] == group and c["method"] == "gavis")
        config = runner.cell_config(cell, ROOT / "phase1")
        assert config["episode"]["max_sim_time"] == budget
        assert cell["method_options"]["depth_init_pts_per_view"] == points
        assert config["habitat"]["width"] == 640
        assert config["episode"]["collision"] == ("navmesh" if group == "mesh" else "none")


def test_portable_paths_are_explicit_and_unknown_variables_fail(monkeypatch):
    runner = module("run")
    monkeypatch.setenv("HABITAT_GS_ROOT", "/relocated/data")
    assert runner.resolve({"scene": "${HABITAT_GS_ROOT}/room.ply"}) == {"scene": "/relocated/data/room.ply"}
    with pytest.raises(KeyError):
        runner.resolve("${UNCONFIGURED_PHASE1_VARIABLE}/room.ply")


def test_runtime_paths_in_worker_environments(monkeypatch):
    from activebench.runtime import conda_python, external_repo, relocated_asset_path
    monkeypatch.setenv("ACTIVEBENCH_ENVS_DIR", "/custom/envs")
    monkeypatch.setenv("R3CON_REPO", "/custom/r3")
    assert conda_python("r3con") == "/custom/envs/r3con/bin/python"
    assert external_repo("R3CON") == "/custom/r3"
    monkeypatch.setenv("HABITAT_GS_ROOT", "/custom/simulator")
    assert relocated_asset_path("/home/previous-user/Projects/habitat-gs/data/chair.json") == Path("/custom/simulator/data/chair.json")


def test_replay_summary_uses_verified_psnr_without_replacing_historical_geometry(tmp_path):
    import json
    import shutil
    from activebench.replay import EpisodeReplay
    evidence = ROOT / "phase1/evidence/gs/interior_0007__d0__s0/gleam"
    rec = tmp_path / "reconstructions/gsplat"
    rec.mkdir(parents=True)
    for name in ("eval.json", "eval_shared.json"):
        shutil.copy2(evidence / name, rec / name)
    replay = EpisodeReplay.__new__(EpisodeReplay)
    replay.manifest = json.loads((evidence / "manifest.json").read_text())
    replay.episode_dir = tmp_path
    historical = replay.summary("gsplat")
    (rec / "psnr_verification.json").write_text(json.dumps(dict(
        psnr=33.3, class_psnr={"severe": {"psnr": 22.2}, "clean": {"psnr": 44.4}})))
    verified = replay.summary("gsplat")
    assert "recon PSNR 33.3" in verified and "sev/clean 22.2/44.4" in verified
    assert historical.split("cmp@5cm")[1].split(" | ")[0] == verified.split("cmp@5cm")[1].split(" | ")[0]
