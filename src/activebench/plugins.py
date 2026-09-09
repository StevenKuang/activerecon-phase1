"""Load external ActiveAgent factories without importing them in the launcher."""

import importlib
import importlib.util
from pathlib import Path
from typing import Any, Dict

from activebench.api import MethodInfo


def normalize_factory(value: str) -> str:
    """Accept ``package.module:factory`` or ``/path/agent.py:factory``."""
    module, separator, function = value.rpartition(":")
    if not separator or not module or not function.isidentifier():
        raise ValueError("external agent must be package.module:factory or /path/agent.py:factory")
    if module.endswith(".py"):
        path = Path(module).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        module = str(path)
    elif not all(part.isidentifier() for part in module.split(".")):
        raise ValueError("invalid Python module in external agent: %s" % module)
    return module + ":" + function


def build_external(value: str, options: Dict[str, Any]):
    module_name, function = normalize_factory(value).rsplit(":", 1)
    if module_name.endswith(".py"):
        spec = importlib.util.spec_from_file_location("activebench_external_agent", module_name)
        module = importlib.util.module_from_spec(spec)
        # Dataclasses and other introspection need a registered module.
        import sys
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    else:
        module = importlib.import_module(module_name)
    factory = getattr(module, function)
    if not callable(factory):
        raise TypeError("external factory is not callable: %s" % value)
    agent = factory(dict(options))
    for name in ("info", "reset", "act"):
        if not callable(getattr(agent, name, None)):
            raise TypeError("external agent must implement info(), reset(seed, task), act(observation)")
    info = agent.info()
    if not isinstance(info, MethodInfo):
        raise TypeError("agent.info() must return activebench.api.MethodInfo")
    if info.pose_access not in ("gt", "none"):
        raise ValueError("MethodInfo.pose_access must be 'gt' or 'none'")
    return agent
