import json

from activebench.audit import audit_cross_method_consistency, world_fingerprint


def write_manifest(path, distractors, start_pose):
    path.mkdir(parents=True)
    (path / "manifest.json").write_text(
        json.dumps(
            {
                "captures": [{"index": 0, "time": 0.0, "pose": start_pose}],
                "distractors": distractors,
            }
        )
    )


DISTRACTOR = {
    "name": "obj0",
    "object_template": "skillet.object_config.json",
    "trajectory": {"type": "waypoint_patrol", "waypoints": [[0, 0, 0], [1, 0, 0]], "speed": 0.3},
}
START = [2.0, 1.5, 0.5, 0.0, 0.0]


class TestWorldFingerprint:
    def test_stable_across_key_order(self):
        a = {"captures": [{"pose": START}], "distractors": [DISTRACTOR]}
        reordered = json.loads(json.dumps(a))
        reordered["distractors"][0] = dict(reversed(list(DISTRACTOR.items())))
        assert world_fingerprint(a) == world_fingerprint(reordered)

    def test_sensitive_to_trajectory_and_start_pose(self):
        base = {"captures": [{"pose": START}], "distractors": [DISTRACTOR]}
        moved = json.loads(json.dumps(base))
        moved["distractors"][0]["trajectory"]["speed"] = 0.4
        shifted = json.loads(json.dumps(base))
        shifted["captures"][0]["pose"] = [2.1, 1.5, 0.5, 0.0, 0.0]
        prints = {world_fingerprint(m) for m in (base, moved, shifted)}
        assert len(prints) == 3

    def test_empty_distractors_still_fingerprints(self):
        assert world_fingerprint({"captures": [], "distractors": []})


class TestCrossMethodAudit:
    def test_consistent_config_passes(self, tmp_path):
        config = tmp_path / "scene__d1__s0"
        write_manifest(config / "gavis", [DISTRACTOR], START)
        write_manifest(config / "fisherrf", [DISTRACTOR], START)
        assert audit_cross_method_consistency(tmp_path) == []

    def test_divergent_world_reported_with_methods(self, tmp_path):
        config = tmp_path / "scene__d1__s0"
        write_manifest(config / "gavis", [DISTRACTOR], START)
        other = json.loads(json.dumps(DISTRACTOR))
        other["trajectory"]["speed"] = 0.9
        write_manifest(config / "fisherrf", [other], START)
        violations = audit_cross_method_consistency(tmp_path)
        assert len(violations) == 1
        assert "scene__d1__s0" in violations[0]
        assert "gavis" in violations[0] and "fisherrf" in violations[0]

    def test_ignores_stray_files(self, tmp_path):
        (tmp_path / "summary.csv").write_text("x")
        config = tmp_path / "scene__d1__s0"
        write_manifest(config / "gavis", [DISTRACTOR], START)
        assert audit_cross_method_consistency(tmp_path) == []
