"""Dataset-aware scene discovery for ActiveBench.

The catalog records what an asset can do. Only ``habitat`` datasets can enter
an active benchmark; mesh-only and posed-capture datasets remain visible in
the inventory without being mistaken for simulator scenes.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import yaml


@dataclass(frozen=True)
class SceneCandidate:
    dataset: str
    name: str
    scene_path: str
    scene_dataset_config_file: str = "default"

    def as_dict(self) -> Dict[str, str]:
        return {
            "dataset": self.dataset,
            "name": self.name,
            "scene_path": self.scene_path,
            "scene_dataset_config_file": self.scene_dataset_config_file,
        }


@dataclass(frozen=True)
class DatasetStatus:
    name: str
    capability: str
    installed: bool
    scene_count: int
    root: Path
    detail: str


def load_scene_catalog(path: Path) -> Dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text())
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("scene dataset catalog must use schema_version: 1")
    if not isinstance(payload.get("datasets"), dict):
        raise ValueError("scene dataset catalog must define datasets")
    return payload


def _dataset_root(spec: Dict[str, Any], data_root: Path) -> Path:
    root = Path(str(spec.get("root", "."))).expanduser()
    return root if root.is_absolute() else data_root / root


def _resolve_config(value: str, root: Path) -> str:
    if value == "default":
        return value
    path = Path(value).expanduser()
    return str(path if path.is_absolute() else root / path)


def _explicit_scenes(
    dataset: str, spec: Dict[str, Any], root: Path
) -> List[SceneCandidate]:
    config = _resolve_config(str(spec.get("scene_dataset_config_file", "default")), root)
    candidates = []
    for scene in spec.get("scenes", []):
        path_value = str(scene.get("scene_path", scene["name"]))
        if not spec.get("scene_paths_are_handles", False):
            path = Path(path_value).expanduser()
            path_value = str(path if path.is_absolute() else root / path)
        candidates.append(
            SceneCandidate(
                dataset=dataset,
                name=str(scene["name"]),
                scene_path=path_value,
                scene_dataset_config_file=_resolve_config(
                    str(scene.get("scene_dataset_config_file", config)), root
                ),
            )
        )
    return candidates


def _discovered_scenes(
    dataset: str, spec: Dict[str, Any], root: Path
) -> List[SceneCandidate]:
    discovery = spec.get("discovery")
    if not discovery or not root.exists():
        return []
    config = _resolve_config(str(spec.get("scene_dataset_config_file", "default")), root)
    candidates = []
    for path in sorted(root.glob(str(discovery["scene_glob"]))):
        if not path.is_file():
            continue
        if discovery.get("name_from") == "parent":
            name = path.parent.name
        elif discovery.get("name_from") == "grandparent":
            name = path.parent.parent.name
        else:
            name = path.name.split(".", 1)[0]
        candidates.append(SceneCandidate(dataset, name, str(path), config))
    return candidates


def dataset_scenes(
    catalog: Dict[str, Any], dataset: str, data_root: Path
) -> List[SceneCandidate]:
    try:
        spec = catalog["datasets"][dataset]
    except KeyError as exc:
        raise KeyError("unknown scene dataset: %s" % dataset) from exc
    if spec.get("capability") != "habitat":
        return []
    root = _dataset_root(spec, Path(data_root))
    by_name = {
        scene.name: scene
        for scene in _discovered_scenes(dataset, spec, root)
        + _explicit_scenes(dataset, spec, root)
    }
    return [by_name[name] for name in sorted(by_name)]


def _select_split(
    scenes: Sequence[SceneCandidate], spec: Dict[str, Any], split: str
) -> List[SceneCandidate]:
    if split == "all":
        return list(scenes)
    splits = spec.get("splits", {})
    if split not in splits:
        raise KeyError("dataset has no split %r" % split)
    names = list(splits[split])
    by_name = {scene.name: scene for scene in scenes}
    missing = [name for name in names if name not in by_name]
    if missing:
        raise FileNotFoundError("split scenes are not installed: %s" % ", ".join(missing))
    return [by_name[name] for name in names]


def select_scene_candidates(
    catalog: Dict[str, Any],
    data_root: Path,
    datasets: Optional[Iterable[str]] = None,
    split: str = "all",
    suite: Optional[str] = None,
) -> List[Dict[str, str]]:
    if bool(datasets) == bool(suite):
        raise ValueError("select exactly one of datasets or suite")
    selections = []
    if suite:
        try:
            selections = catalog["suites"][suite]
        except KeyError as exc:
            raise KeyError("unknown scene suite: %s" % suite) from exc
    else:
        selections = [{"dataset": name, "split": split} for name in datasets or []]

    result = []
    seen = set()
    for selection in selections:
        dataset = str(selection["dataset"])
        spec = catalog["datasets"][dataset]
        scenes = dataset_scenes(catalog, dataset, data_root)
        for scene in _select_split(scenes, spec, str(selection.get("split", "all"))):
            key = (scene.dataset, scene.name)
            if key not in seen:
                result.append(scene.as_dict())
                seen.add(key)
    return result


def inventory(catalog: Dict[str, Any], data_root: Path) -> List[DatasetStatus]:
    statuses = []
    for name, spec in catalog["datasets"].items():
        root = _dataset_root(spec, Path(data_root))
        capability = str(spec.get("capability", "unknown"))
        if capability == "habitat":
            scenes = dataset_scenes(catalog, name, data_root)
            if spec.get("scene_paths_are_handles", False):
                marker = _resolve_config(
                    str(spec.get("scene_dataset_config_file", "default")), root
                )
                installed = marker != "default" and Path(marker).is_file()
            else:
                installed = any(Path(scene.scene_path).is_file() for scene in scenes)
            count = sum(
                1
                for scene in scenes
                if spec.get("scene_paths_are_handles", False)
                or Path(scene.scene_path).is_file()
            )
        else:
            pattern = str(spec.get("asset_glob", "*"))
            count = sum(1 for path in root.glob(pattern) if path.is_file()) if root.exists() else 0
            installed = count > 0
        statuses.append(
            DatasetStatus(
                name=name,
                capability=capability,
                installed=installed,
                scene_count=count,
                root=root,
                detail=str(spec.get("install", {}).get("note", "")),
            )
        )
    return statuses
