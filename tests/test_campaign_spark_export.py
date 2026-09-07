"""Campaign resumes must use the same faithful Spark export as the viewer."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from activebench import web_export


def _campaign():
    path = Path(__file__).parents[1] / "scripts" / "run_campaign.py"
    spec = importlib.util.spec_from_file_location("campaign_spark_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _model(tmp_path):
    episode = tmp_path / "room__d0__s0" / "random"
    model = episode / "reconstructions" / "gsplat1600" / "gaussians.npz"
    model.parent.mkdir(parents=True)
    np.savez(model, centers=np.zeros((1, 3)), scales=np.ones((1, 3)),
             quats_wxyz=np.array([[1., 0., 0., 0.]]), opacities=np.array([.5]),
             colors=np.ones((1, 3)) * .5, sh_rest=np.zeros((1, 15, 3)), sh_degree=3)
    builder = tmp_path / "build-lod"
    builder.write_bytes(b"test builder")
    rad = model.with_name("gaussians-lod.rad")
    return episode, model, builder, rad


def test_resume_rebuilds_legacy_ply_and_ignores_old_failure_marker(tmp_path, monkeypatch):
    episode, model, builder, rad = _model(tmp_path)
    ply = model.with_suffix(".ply")
    ply.write_bytes(b"old incorrectly ordered SH export")
    marker = Path(str(rad) + ".failed")
    marker.write_text(json.dumps({"sh3": "old quality crash", "sh0": "old crash"}))
    calls = []

    def build(path, executable, sh_degree):
        assert path.read_bytes().startswith(b"ply\n")
        calls.append(sh_degree)
        rad.write_bytes(b"current RAD")
        return rad

    monkeypatch.setattr(web_export, "build_rad_from_ply", build)
    campaign = _campaign()
    campaign.run_spark_export(episode, "gsplat1600", builder)
    assert calls == [3]
    assert not marker.exists()
    assert web_export.current_campaign_rad(model, rad, 3)
    campaign.run_spark_export(episode, "gsplat1600", builder)
    assert calls == [3], "current versioned RAD should be reused"


def test_campaign_failure_never_retries_sh3_as_sh0(tmp_path, monkeypatch):
    episode, model, builder, rad = _model(tmp_path)
    calls = []

    def fail(path, executable, sh_degree):
        calls.append(sh_degree)
        raise RuntimeError("both same-SH LoD builders failed")

    monkeypatch.setattr(web_export, "build_rad_from_ply", fail)
    campaign = _campaign()
    with pytest.raises(RuntimeError, match="same-SH"):
        campaign.run_spark_export(episode, "gsplat1600", builder)
    assert calls == [3]
    assert model.with_suffix(".ply").is_file()
    campaign.run_spark_export(episode, "gsplat1600", builder)
    assert calls == [3], "matching failure signature should suppress repeated work"
    builder.write_bytes(b"updated builder")
    with pytest.raises(RuntimeError):
        campaign.run_spark_export(episode, "gsplat1600", builder)
    assert calls == [3, 3], "an updated converter must invalidate an older failure"
