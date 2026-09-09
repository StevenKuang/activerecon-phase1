import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import yaml

from activebench import registry
from activebench.api import Observation, mask_observation_pose
from activebench.campaign import check_resume, resolve_methods, select_configs, validate_assets
from activebench.common.camera import CameraIntrinsics, CameraPose
from activebench.common.image import save_rgb_png
from activebench.configuration import digest, file_digest, load_yaml, reference_scene
from activebench.rpc import AgentProcessProxy

ROOT = Path(__file__).resolve().parents[1]


def config():
    return {"habitat": {"scene_path": "/unused.glb", "width": 64, "height": 48},
            "episode": {"max_captures": 10, "max_sim_time": 6.0, "capture_cost": 0.0,
                        "reconstruction_interval": 1.0},
            "seed": 0, "campaign": {"scene": "room", "difficulty": "d0", "protocol": "test"}}


def test_external_agent_has_identical_actions_through_rpc(tmp_path):
    name = str(ROOT / "examples/methods/spin_agent.py") + ":build_agent"
    local = registry.build_agent(name, {"turn_degrees": 45})
    assert registry.default_env(name) is None
    image = np.zeros((48, 64, 3), dtype=np.uint8)
    path = tmp_path / "rgb.png"
    save_rgb_png(path, image)
    observation = mask_observation_pose(Observation(rgb=image, rgb_path=str(path),
        pose=CameraPose.from_xyz_yaw_pitch([0, 1, 0]), time=0.0,
        intrinsics=CameraIntrinsics.from_hfov(64, 48, 90)))
    proxy = AgentProcessProxy(name, {"turn_degrees": 45}, sys.executable,
                              stderr_log=tmp_path / "worker.log")
    try:
        local.reset(7)
        proxy.reset(7)
        assert proxy.info().pose_access == "none"
        assert proxy.act(observation).to_dict() == local.act(observation).to_dict()
        assert observation.pose is None
    finally:
        proxy.close()


def test_bad_external_agent_contract_is_reported(tmp_path):
    path = tmp_path / "bad.py"
    path.write_text("def build(options):\n    return object()\n")
    with pytest.raises(TypeError, match="must implement"):
        registry.build_agent(str(path) + ":build", {})


def test_method_file_resolves_relative_factory_and_environment(tmp_path):
    (tmp_path / "agent.py").write_text("def build(options): pass\n")
    method_file = tmp_path / "methods.yaml"
    method_file.write_text(yaml.safe_dump({"schema_version": 1, "methods": {
        "mine": {"agent": "agent.py:build", "environment": "my-conda", "options": {"k": 1}}}}))
    method = resolve_methods(["mine"], method_file, {"mine": {"k": 2}})["mine"]
    assert method["agent"] == str(tmp_path / "agent.py") + ":build"
    assert method["environment"] == "my-conda"
    assert method["options"] == {"k": 2}
    assert method["source_sha256"] == file_digest(tmp_path / "agent.py")


def test_typo_and_benchmark_owned_method_options_fail():
    with pytest.raises(KeyError, match="unknown agent"):
        resolve_methods(["typo"], None)
    with pytest.raises(ValueError, match="benchmark-owned"):
        resolve_methods(["random"], None, {"random": {"seed": 123}})
    with pytest.raises(ValueError, match="names must"):
        resolve_methods(["../escape"], None)


def test_config_selection_is_metadata_based_and_duplicates_fail(tmp_path):
    payload = config()
    (tmp_path / "arbitrary.yaml").write_text(yaml.safe_dump(payload))
    assert select_configs(tmp_path, scenes=["room"])[0][0] == "room__d0__s0"
    with pytest.raises(ValueError, match="no episode"):
        select_configs(tmp_path, scenes=["typo"])
    with pytest.raises(ValueError, match="absent"):
        select_configs(tmp_path, scenes=["room", "typo"])
    (tmp_path / "copy.yaml").write_text(yaml.safe_dump(payload))
    with pytest.raises(ValueError, match="duplicate"):
        select_configs(tmp_path)


def test_env_variables_expand_and_missing_ones_fail(tmp_path, monkeypatch):
    monkeypatch.setenv("ACTIVEBENCH_TEST_ASSET", "/assets/scene.glb")
    path = tmp_path / "cfg.yaml"
    path.write_text("path: ${ACTIVEBENCH_TEST_ASSET}\n")
    assert load_yaml(path)["path"] == "/assets/scene.glb"
    monkeypatch.delenv("ACTIVEBENCH_TEST_ASSET")
    with pytest.raises(ValueError, match="unresolved"):
        load_yaml(path)


def test_resume_rejects_changed_identity_and_untracked_outputs(tmp_path):
    (tmp_path / "manifest.json").write_text("{}")
    with pytest.raises(ValueError, match="untracked"):
        check_resume(tmp_path, "new")
    (tmp_path / "benchmark-run.json").write_text(json.dumps({"identity": "old"}))
    check_resume(tmp_path, "old")
    with pytest.raises(ValueError, match="changed"):
        check_resume(tmp_path, "new")


def test_reference_assets_are_verified_and_bound_to_scene(tmp_path):
    files = {"surface/room.npz": b"surface", "eval/room__s0/transforms_eval_shared.json": b"{}"}
    for rel, data in files.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
    payload = config()
    record = {"schema_version": 1, "groups": {"room__s0": {"scene_identity": digest(reference_scene(payload))}},
              "files": {rel: file_digest(tmp_path / rel) for rel in files}}
    (tmp_path / "assets.json").write_text(json.dumps(record))
    selected = [("room__d0__s0", Path("unused"), payload)]
    validate_assets(tmp_path, selected)
    payload["habitat"]["scene_path"] = "/other.glb"
    with pytest.raises(ValueError, match="do not match"):
        validate_assets(tmp_path, selected)
    payload["habitat"]["scene_path"] = "/unused.glb"
    (tmp_path / "surface/room.npz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        validate_assets(tmp_path, selected)


def test_summary_rejects_mixed_protocol_and_preserves_missing_cells(tmp_path):
    spec = importlib.util.spec_from_file_location("summarize_benchmark", ROOT / "scripts/summarize_benchmark.py")
    summary = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(summary)
    for name, seconds in (("a", 6), ("b", 8)):
        p = tmp_path / (name + "__d0__s0") / "random"
        p.mkdir(parents=True)
        payload = config()
        payload["campaign"]["scene"] = name
        payload["episode"]["max_sim_time"] = seconds
        (p / "benchmark-run.json").write_text(json.dumps({"scene": name, "condition": "d0", "seed": 0,
                "alias": "random", "config": payload, "recipe": {"run_name": "gsplat"}}))
    with pytest.raises(ValueError, match="mixed"):
        summary.load_rows(tmp_path)
    rows = summary.load_rows(tmp_path, ["a"])
    values, pairs = summary.summarize(rows)
    assert values[0]["planned"] == 1 and values[0]["complete"] == 0
    assert values[0]["psnr_n"] == 0 and pairs == []


def test_campaign_records_whole_plan_and_exits_nonzero_on_failure(tmp_path, monkeypatch):
    sys.path.insert(0, str(ROOT / "scripts"))
    import run_campaign as stages
    from activebench.campaign import main
    configs = tmp_path / "configs"
    configs.mkdir()
    asset = tmp_path / "room.glb"
    asset.write_bytes(b"test preflight only; simulator is mocked")
    out = tmp_path / "runs"
    for name in ("a", "b"):
        payload = config()
        payload["habitat"]["scene_path"] = str(asset)
        payload["campaign"]["scene"] = name
        (configs / (name + ".yaml")).write_text(yaml.safe_dump(payload))
    monkeypatch.setattr(stages, "_sim_python", lambda source: sys.executable)

    def failed_episode(*args, **kwargs):
        assert (out / "b__d0__s0/random/benchmark-run.json").is_file()
        return False, 1

    monkeypatch.setattr(stages, "run_episode", failed_episode)
    monkeypatch.setattr(sys, "argv", ["run_campaign.py", "--configs-dir", str(configs), "--out-dir", str(out),
                                     "--acquisition-only", "--execute"])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 1
    status = json.loads((out / "campaign-status.json").read_text())
    assert status["planned"] == 2 and len(status["failures"]) == 2


def test_summary_audits_actual_world_not_only_requested_config(tmp_path):
    spec = importlib.util.spec_from_file_location("summary_audit", ROOT / "scripts/summarize_benchmark.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name, objects in (("a", []), ("b", [{"name": "unexpected-distractor"}])):
        directory = tmp_path / "room__d0__s0" / name
        directory.mkdir(parents=True)
        receipt = {"scene": "room", "condition": "d0", "seed": 0, "alias": name,
                   "config": config(), "recipe": {"run_name": "gsplat"}}
        (directory / "benchmark-run.json").write_text(json.dumps(receipt))
        manifest = {"num_captures": 0, "clock": {"final_sim_time": 0}, "captures": [],
                    "method": {}, "distractors": objects}
        (directory / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="recorded worlds differ"):
        module.load_rows(tmp_path)
