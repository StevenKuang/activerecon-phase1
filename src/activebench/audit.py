"""Cross-method fairness audits over recorded campaign runs.

Benchmark fairness requires that every method evaluated on one campaign
config saw the exact same world: identical distractor scripts (count, assets,
trajectories) and the same agent start pose. This holds by construction —
both live in the shared config YAML — but the guarantee must be *checked*,
not assumed, so regressions (e.g. a runner change that reseeds trajectories
per episode) are caught the moment they land in a run directory.
"""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List


def world_fingerprint(manifest: Dict[str, Any]) -> str:
    """Short stable id of the world one episode ran in.

    Covers the distractor scripts and the agent start pose: two episodes with
    equal fingerprints experienced identical scene dynamics and initial
    conditions, so their results are directly comparable.
    """

    captures = manifest.get("captures") or []
    start_pose = captures[0]["pose"] if captures else None
    payload = json.dumps(
        {"distractors": manifest.get("distractors", []), "start_pose": start_pose},
        sort_keys=True,
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:10]


def audit_cross_method_consistency(runs_dir: Path) -> List[str]:
    """Violation messages for configs whose methods saw different worlds.

    Walks the campaign layout ``<runs_dir>/<config>/<method>/manifest.json``
    and returns one message per config where fingerprints disagree (empty
    list = all consistent).
    """

    violations: List[str] = []
    for config_dir in sorted(p for p in Path(runs_dir).iterdir() if p.is_dir()):
        by_fingerprint: Dict[str, List[str]] = {}
        for manifest_path in sorted(config_dir.glob("*/manifest.json")):
            manifest = json.loads(manifest_path.read_text())
            by_fingerprint.setdefault(world_fingerprint(manifest), []).append(
                manifest_path.parent.name
            )
        if len(by_fingerprint) > 1:
            groups = "; ".join(
                "%s: %s" % (fp, ",".join(methods))
                for fp, methods in sorted(by_fingerprint.items())
            )
            violations.append("%s saw different worlds (%s)" % (config_dir.name, groups))
    return violations
