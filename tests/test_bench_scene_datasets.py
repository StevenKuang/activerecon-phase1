from pathlib import Path

import pytest
import yaml

from activebench.scene_datasets import (
    inventory,
    load_scene_catalog,
    select_scene_candidates,
)


def _catalog(tmp_path: Path) -> Path:
    payload = {
        "schema_version": 1,
        "suites": {
            "smoke": [
                {"dataset": "files", "split": "smoke"},
                {"dataset": "handles", "split": "smoke"},
            ]
        },
        "datasets": {
            "files": {
                "capability": "habitat",
                "root": "files",
                "discovery": {"scene_glob": "*/*.glb", "name_from": "parent"},
                "splits": {"smoke": ["alpha"]},
            },
            "handles": {
                "capability": "habitat",
                "root": "handles",
                "scene_dataset_config_file": "dataset.json",
                "scene_paths_are_handles": True,
                "scenes": [{"name": "apt_0", "scene_path": "apt_0"}],
                "splits": {"smoke": ["apt_0"]},
            },
            "offline": {
                "capability": "posed_capture",
                "root": "offline",
                "asset_glob": "*.json",
            },
        },
    }
    path = tmp_path / "catalog.yaml"
    path.write_text(yaml.safe_dump(payload))
    return path


def test_suite_resolves_file_scenes_and_scene_handles(tmp_path):
    (tmp_path / "files/alpha").mkdir(parents=True)
    (tmp_path / "files/alpha/alpha.glb").touch()
    (tmp_path / "handles").mkdir()
    (tmp_path / "handles/dataset.json").write_text("{}")
    catalog = load_scene_catalog(_catalog(tmp_path))

    scenes = select_scene_candidates(catalog, tmp_path, suite="smoke")

    assert [scene["name"] for scene in scenes] == ["alpha", "apt_0"]
    assert scenes[0]["scene_path"] == str(tmp_path / "files/alpha/alpha.glb")
    assert scenes[1]["scene_path"] == "apt_0"
    assert scenes[1]["scene_dataset_config_file"] == str(
        tmp_path / "handles/dataset.json"
    )


def test_dataset_split_reports_missing_licensed_scenes(tmp_path):
    catalog = load_scene_catalog(_catalog(tmp_path))

    with pytest.raises(FileNotFoundError, match="alpha"):
        select_scene_candidates(catalog, tmp_path, datasets=["files"], split="smoke")


def test_inventory_keeps_offline_assets_out_of_habitat_candidates(tmp_path):
    (tmp_path / "offline").mkdir()
    (tmp_path / "offline/transforms.json").touch()
    catalog = load_scene_catalog(_catalog(tmp_path))

    status = {entry.name: entry for entry in inventory(catalog, tmp_path)}

    assert status["offline"].installed
    assert status["offline"].capability == "posed_capture"
    assert select_scene_candidates(
        catalog, tmp_path, datasets=["offline"], split="all"
    ) == []
