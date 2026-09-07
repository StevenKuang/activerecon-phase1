"""Machine-local locations, configurable without editing benchmark code."""
import os
from pathlib import Path


def external_repo(name: str) -> str:
    return str(Path(os.environ.get(name.upper() + "_REPO", str(Path.home() / "Projects" / name))).expanduser())


def conda_envs_dir() -> Path:
    return Path(os.environ.get("ACTIVEBENCH_ENVS_DIR", str(Path.home() / "miniconda3/envs"))).expanduser()


def conda_python(name: str) -> str:
    return str(conda_envs_dir() / name / "bin/python")


def relocated_asset_path(value: str) -> Path:
    """Resolve archived simulator paths through explicit local root settings."""
    path = Path(value).expanduser()
    for dirname, variable in (("habitat-sim", "HABITAT_SIM_ROOT"), ("habitat-gs", "HABITAT_GS_ROOT")):
        marker = "/Projects/" + dirname + "/"
        if marker in str(path):
            root = Path(os.environ.get(variable, str(Path.home() / "Projects" / dirname))).expanduser()
            return root / str(path).split(marker, 1)[1]
    return path
