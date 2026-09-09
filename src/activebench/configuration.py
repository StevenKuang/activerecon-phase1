"""Portable YAML inputs and content identities for new benchmark campaigns."""

import hashlib
import json
import os
import re
from pathlib import Path

import yaml


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                     separators=(",", ":")).encode()).hexdigest()


def file_digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def expand_values(value):
    if isinstance(value, dict):
        return {k: expand_values(v) for k, v in value.items()}
    if isinstance(value, list):
        return [expand_values(v) for v in value]
    if isinstance(value, str):
        expanded = os.path.expandvars(os.path.expanduser(value))
        if re.search(r"\$(?:\{[A-Za-z_]\w*\}|[A-Za-z_]\w*)", expanded):
            raise ValueError("unresolved environment variable in %r" % value)
        return expanded
    return value


def load_yaml(path):
    value = expand_values(yaml.safe_load(Path(path).read_text()))
    if not isinstance(value, dict):
        raise ValueError("YAML must contain a mapping: %s" % path)
    return value


def safe_name(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise ValueError("names must contain letters, digits, dots, hyphens or underscores: %r" % value)
    if "__" in value:
        raise ValueError("double underscores are reserved as cell separators: %s" % value)
    return value


def cell_identity(payload):
    from activebench.episode import EpisodeSpec
    EpisodeSpec.from_dict(payload)
    campaign = payload.get("campaign", {})
    scene = safe_name(campaign.get("scene"))
    condition = safe_name(campaign.get("difficulty"))
    if not campaign.get("protocol"):
        raise ValueError("campaign.protocol is required (name your comparison protocol)")
    seed = int(payload.get("seed", 0))
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    return scene, condition, seed


def reference_scene(payload):
    """Surface/camera references depend on the clean scene, never on a policy."""
    return {"habitat": payload["habitat"], "start_pose": payload.get("start_pose")}


def input_file_hashes(payload, cache=None):
    """Hash directly referenced assets and adjacent stage navmeshes.

    Dataset-config handles can reference additional files internally; their
    dataset distribution/version must also be preserved by the experimenter.
    """
    cache = {} if cache is None else cache
    paths = []
    habitat = payload["habitat"]
    for key in ("scene_path", "scene_dataset_config_file", "physics_config_file"):
        value = habitat.get(key)
        if value and value != "default":
            path = Path(value).expanduser()
            if not path.is_file():
                if key == "scene_path" and not path.suffix and habitat.get("scene_dataset_config_file", "default") != "default":
                    continue  # Habitat scene-dataset handle, e.g. apt_0.
                raise FileNotFoundError("missing simulator asset: %s" % path)
            paths.append(path)
            if key == "scene_path":
                candidates = [path.with_suffix(".navmesh")]
                if path.name.endswith(".gs.ply"):
                    candidates.append(path.with_name(path.name.removesuffix(".gs.ply") + ".navmesh"))
                paths.extend(p for p in candidates if p.is_file())
    for obj in payload.get("distractors", []):
        value = obj.get("object_template", "")
        if Path(value).suffix == ".json":
            paths.append(Path(value).expanduser())
    result = {}
    for path in paths:
        key = str(path.resolve())
        if key not in cache:
            cache[key] = file_digest(path)
        result[key] = cache[key]
    return result


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)
