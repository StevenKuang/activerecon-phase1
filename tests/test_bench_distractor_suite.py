import numpy as np
import pytest

from activebench.distractor_suite import (
    DIFFICULTIES,
    generate_distractors,
    generate_episode_config,
)

POOL = [
    {"handle": "objA.object_config.json", "diag": 0.8},
    {"handle": "objB.object_config.json", "diag": 1.0},
    {"handle": "objC.object_config.json", "diag": 1.2},
]


def scene(name="test_scene", extent=(8.0, 3.0, 10.0)):
    return {
        "name": name,
        "scene_path": "/tmp/fake.glb",
        "aabb_min": [0.0, 0.0, 0.0],
        "aabb_max": list(extent),
        "start_pose": [4.0, 1.4, 5.0, 0.0, 0.0],
        "has_ceiling": True,
        "navigable_area_m2": extent[0] * extent[2],
        "patrol_routes": [
            {
                "clearance_m": 1.0,
                "waypoints": [
                    [0.8, 0.0, 1.0],
                    [0.8, 0.0, extent[2] - 1.0],
                ],
            },
            {
                "clearance_m": 1.0,
                "waypoints": [
                    [extent[0] - 0.8, 0.0, 1.0],
                    [extent[0] - 0.8, 0.0, extent[2] - 1.0],
                ],
            },
            {
                "clearance_m": 1.0,
                "waypoints": [
                    [1.0, 0.0, 0.8],
                    [extent[0] - 1.0, 0.0, 0.8],
                ],
            },
            {
                "clearance_m": 1.0,
                "waypoints": [
                    [1.0, 0.0, extent[2] - 0.8],
                    [extent[0] - 1.0, 0.0, extent[2] - 0.8],
                ],
            },
        ],
    }


class TestGenerateDistractors:
    def test_d0_is_empty(self):
        assert generate_distractors(scene(), "d0", POOL) == []

    def test_deterministic_per_key(self):
        a = generate_distractors(scene(), "d2", POOL, seed=1)
        b = generate_distractors(scene(), "d2", POOL, seed=1)
        assert a == b
        c = generate_distractors(scene(), "d2", POOL, seed=2)
        assert a != c
        d = generate_distractors(scene("other_scene"), "d2", POOL, seed=1)
        assert a != d

    def test_count_scales_with_area(self):
        small = generate_distractors(scene(extent=(5.0, 3.0, 6.0)), "d2", POOL)  # 30 m2
        large = generate_distractors(scene(extent=(12.0, 3.0, 10.0)), "d2", POOL)  # 120 m2
        assert len(small) < len(large)
        assert len(large) == 15
        assert len(large) <= DIFFICULTIES["d2"]["max_count"]

    def test_d1_slower_and_sparser_than_d2(self):
        d1 = generate_distractors(scene(), "d1", POOL)
        d2 = generate_distractors(scene(), "d2", POOL)
        assert len(d1) < len(d2)
        assert np.mean([x["trajectory"]["speed"] for x in d1]) < np.mean(
            [x["trajectory"]["speed"] for x in d2]
        )
        assert max(x["trajectory"]["speed"] for x in d1) < 0.5

    def test_routes_use_audited_waypoints_and_fit_object_clearance(self):
        s = scene()
        audited = {
            tuple(tuple(point) for point in route["waypoints"])
            for route in s["patrol_routes"]
        }
        for entry in generate_distractors(s, "d2", POOL):
            points = np.array(entry["trajectory"]["waypoints"])
            lift = points[0, 1]
            floor_route = tuple(tuple(point) for point in (points - [0.0, lift, 0.0]))
            assert floor_route in audited
            assert entry["object_diag_m"] >= 0.8
            assert entry["trajectory"]["type"] == "waypoint_patrol"
            assert (points[:, 0] >= 0.0).all() and (points[:, 0] <= 8.0).all()
            assert (points[:, 2] >= 0.0).all() and (points[:, 2] <= 10.0).all()

    def test_d2_has_speeds_below_and_above_agent(self):
        speeds = [
            entry["trajectory"]["speed"]
            for entry in generate_distractors(scene(), "d2", POOL)
        ]
        assert min(speeds) < 0.5 < max(speeds)
        assert np.mean(speeds) < 0.65

    def test_missing_audited_routes_rejected(self):
        s = scene()
        del s["patrol_routes"]
        with pytest.raises(ValueError, match="patrol_routes"):
            generate_distractors(s, "d2", POOL)

    def test_unknown_difficulty_rejected(self):
        with pytest.raises(KeyError):
            generate_distractors(scene(), "d9", POOL)


class TestEpisodeConfig:
    def test_config_shape_and_strata(self):
        cfg = generate_episode_config(
            scene(), {"width": 640, "height": 480, "hfov": 75.178}, "d1", POOL, seed=3
        )
        assert cfg["habitat"]["hfov"] == pytest.approx(75.178)
        assert cfg["seed"] == 3 and cfg["trajectory_seed"] == 3
        # Per-episode eval sets are gone; shared eval sets replaced them.
        assert "eval" not in cfg
        assert cfg["campaign"]["difficulty"] == "d1"
        assert cfg["campaign"]["protocol"] == "mission-time-v2"
        assert cfg["episode"]["max_sim_time"] == 60.0
        assert cfg["episode"]["reconstruction_interval"] == 1.0
        assert cfg["episode"]["capture_cost"] == 0.0
        assert len(cfg["distractors"]) == len(generate_distractors(scene(), "d1", POOL, 3))
        # Round-trips through EpisodeSpec.
        from activebench.episode import EpisodeSpec

        spec = EpisodeSpec.from_dict(cfg)
        # The old per-episode eval block is tolerated but ignored.
        assert not hasattr(spec, "num_eval_poses")
        assert len(spec.scene.distractors) == len(cfg["distractors"])


class TestDynDifficulty:
    """Single dynamic level for v3 rounds: few, large, mixed-speed objects."""

    def suite_entry(self):
        import copy

        # Reuse the module-level fixture if one exists; otherwise build one.
        entry = {
            "name": "toy_scene",
            "scene_path": "/tmp/toy.glb",
            "aabb_min": [0.0, 0.0, 0.0],
            "aabb_max": [12.0, 3.5, 10.0],
            "navigable_area_m2": 80.0,
            "start_pose": [1.0, 1.5, 1.0, 0.0, 0.0],
            "patrol_routes": [
                {"waypoints": [[1.0, 0.0, 1.0], [9.0, 0.0, 1.0]], "clearance_m": 1.4},
                {"waypoints": [[1.0, 0.0, 5.0], [9.0, 0.0, 5.0]], "clearance_m": 1.3},
                {"waypoints": [[2.0, 0.0, 8.0], [8.0, 0.0, 8.0]], "clearance_m": 1.2},
                {"waypoints": [[5.0, 0.0, 2.0], [5.0, 0.0, 9.0]], "clearance_m": 1.5},
            ],
        }
        return copy.deepcopy(entry)

    def pool(self):
        return [
            {"handle": "obj_%02d" % i, "diag": diag}
            for i, diag in enumerate(
                [0.65, 0.75, 0.92, 1.0, 1.1, 1.25, 1.4, 1.6, 1.9]
            )
        ]

    def test_dyn_uses_only_large_objects(self):
        from activebench.distractor_suite import generate_distractors

        distractors = generate_distractors(self.suite_entry(), "dyn", self.pool(), seed=0)
        assert 4 <= len(distractors) <= 6
        for d in distractors:
            assert d["object_diag_m"] >= 0.9

    def test_dyn_speeds_span_the_range_and_differ(self):
        from activebench.distractor_suite import generate_distractors

        distractors = generate_distractors(self.suite_entry(), "dyn", self.pool(), seed=0)
        speeds = [d["trajectory"]["speed"] for d in distractors]
        assert len(set(speeds)) == len(speeds)  # stratified: all distinct
        assert min(speeds) < 0.5 < max(speeds)  # straddles the agent speed
        assert all(0.15 <= s <= 0.8 for s in speeds)

    def test_dyn_is_deterministic_per_seed(self):
        from activebench.distractor_suite import generate_distractors

        a = generate_distractors(self.suite_entry(), "dyn", self.pool(), seed=3)
        b = generate_distractors(self.suite_entry(), "dyn", self.pool(), seed=3)
        c = generate_distractors(self.suite_entry(), "dyn", self.pool(), seed=4)
        assert a == b
        assert a != c

    def test_d1_draws_unchanged_by_widened_pool(self):
        from activebench.distractor_suite import generate_distractors

        narrow = [o for o in self.pool() if 0.6 <= o["diag"] < 1.8]
        widened = self.pool()  # includes a 1.9m object d1 must ignore
        assert generate_distractors(
            self.suite_entry(), "d1", narrow, seed=0
        ) == generate_distractors(self.suite_entry(), "d1", widened, seed=0)

    def test_stream_flag_marks_v3_protocol(self):
        from activebench.distractor_suite import generate_episode_config

        intrinsics = {"width": 640, "height": 480, "hfov": 75.178}
        v3 = generate_episode_config(
            self.suite_entry(), intrinsics, "dyn", self.pool(),
            stream_observations=True,
        )
        assert v3["episode"]["stream_observations"] is True
        assert v3["campaign"]["protocol"] == "video-stream-v3"
        v2 = generate_episode_config(self.suite_entry(), intrinsics, "d0", self.pool())
        assert v2["episode"]["stream_observations"] is False
        assert v2["campaign"]["protocol"] == "mission-time-v2"
