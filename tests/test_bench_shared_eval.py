import numpy as np
import pytest

from activebench.eval.shared_eval import (
    SelectionResult,
    candidate_orientations,
    certify_clean,
    farthest_point_indices,
    select_two_pole,
    sweep_centers,
)


class TestFarthestPoint:
    def test_spreads_over_line(self):
        points = np.array([[0.0, 0, 0], [0.1, 0, 0], [5.0, 0, 0], [10.0, 0, 0]])
        chosen = farthest_point_indices(points, 3)
        assert chosen[0] == 0
        assert chosen[1] == 3  # farthest from 0
        assert chosen[2] == 2  # maximizes min distance to {0, 10}

    def test_deterministic_and_bounded(self):
        rng = np.random.default_rng(0)
        points = rng.random((50, 3))
        assert farthest_point_indices(points, 10) == farthest_point_indices(points, 10)
        assert farthest_point_indices(points, 99) == farthest_point_indices(points, 50)
        assert farthest_point_indices(np.zeros((0, 3)), 5) == []


class TestCandidateOrientations:
    def test_strata_cycle_and_pitch_ranges(self):
        orientations = candidate_orientations(8, np.random.default_rng(0))
        strata = [s for _, _, s in orientations]
        assert strata.count("level") == 4
        assert strata.count("lookup") == 2 and strata.count("lookdown") == 2
        for _, pitch, stratum in orientations:
            if stratum == "lookup":
                assert np.deg2rad(40) <= pitch <= np.deg2rad(70)
            elif stratum == "lookdown":
                assert np.deg2rad(-70) <= pitch <= np.deg2rad(-40)
            else:
                assert abs(pitch) <= 0.35


class FakeTrajectory:
    def __init__(self, a, b, speed=1.0):
        self.a, self.b = np.asarray(a, float), np.asarray(b, float)

    def pose_at(self, t):
        alpha = min(t / 10.0, 1.0)
        return self.a + alpha * (self.b - self.a), 0.0


class TestCleanCertification:
    def test_far_region_is_clean(self):
        visible = np.array([[10.0, 0.0, 10.0], [10.5, 0.0, 10.0]])
        centers, radii = sweep_centers(
            [FakeTrajectory([0, 0, 0], [1, 0, 0])], [0.5], t_end=10.0
        )
        assert certify_clean(visible, centers, radii)

    def test_swept_through_region_is_dirty_even_between_endpoints(self):
        # Trajectory passes x in [0, 1]; visible surface sits at x=0.55 which is
        # only reached mid-sweep — fine dt catches it.
        visible = np.array([[0.55, 0.0, 0.0]])
        centers, radii = sweep_centers(
            [FakeTrajectory([0, 0, 0], [1, 0, 0])], [0.3], t_end=10.0
        )
        assert not certify_clean(visible, centers, radii)

    def test_margin_is_conservative(self):
        visible = np.array([[0.0, 0.0, 1.0]])  # 1m from path, sphere r=0.5
        centers, radii = sweep_centers(
            [FakeTrajectory([0, 0, 0], [1, 0, 0])], [0.5], t_end=10.0
        )
        assert certify_clean(visible, centers, radii, margin=0.25)
        assert not certify_clean(visible, centers, radii, margin=0.75)

    def test_no_distractors_always_clean(self):
        assert certify_clean(np.zeros((5, 3)), np.zeros((0, 3)), np.zeros(0))


def grid_positions(n):
    xs = np.arange(n, dtype=float)
    return np.column_stack([xs, np.zeros(n), np.zeros(n)])


class TestSelectTwoPole:
    def make_stats(self, n=20):
        fracs = np.zeros(n)
        fracs[:8] = 0.5  # candidates 0-7 severe
        clean = np.zeros(n, dtype=bool)
        clean[10:] = True  # candidates 10-19 clean
        return {"d2": fracs}, {"d2": clean}

    def test_quotas_and_labels(self):
        fracs, clean = self.make_stats()
        result = select_two_pole(grid_positions(20), fracs, clean, 4, 4)
        severe = [i for i in result.indices if result.labels[i]["d2"] == "severe"]
        clean_sel = [i for i in result.indices if result.labels[i]["d2"] == "clean"]
        assert len(severe) == 4 and all(i < 8 for i in severe)
        assert len(clean_sel) == 4 and all(i >= 10 for i in clean_sel)
        assert result.shortfalls == []

    def test_shortfall_recorded_not_fatal(self):
        fracs, clean = self.make_stats()
        result = select_two_pole(grid_positions(20), fracs, clean, 10, 4)
        assert any("severe[d2]" in s for s in result.shortfalls)
        severe = [i for i in result.indices if result.labels[i]["d2"] == "severe"]
        assert len(severe) == 8  # took everything available

    def test_clean_requires_all_difficulties(self):
        n = 12
        fracs = {"d1": np.zeros(n), "d2": np.zeros(n)}
        clean = {
            "d1": np.array([True] * 6 + [False] * 6),
            "d2": np.array([False] * 3 + [True] * 9),
        }
        result = select_two_pole(grid_positions(n), fracs, clean, 0, 3)
        assert all(3 <= i < 6 for i in result.indices)

    def test_severe_overlap_reused_across_difficulties(self):
        n = 10
        fracs = {"d1": np.zeros(n), "d2": np.zeros(n)}
        fracs["d1"][:4] = 0.5
        fracs["d2"][:4] = 0.5  # identical pools: picks should be shared
        clean = {k: np.zeros(n, dtype=bool) for k in fracs}
        result = select_two_pole(grid_positions(n), fracs, clean, 3, 0)
        assert len(result.indices) == 3
        for index in result.indices:
            assert result.labels[index] == {"d1": "severe", "d2": "severe"}

    def test_mixed_label_for_between_pole_views(self):
        n = 6
        fracs = {"d2": np.array([0.5, 0.05, 0.0, 0.0, 0.0, 0.0])}
        clean = {"d2": np.array([False, False, False, True, True, True])}
        result = select_two_pole(grid_positions(n), fracs, clean, 2, 2)
        # index 1 (frac 5%, not certified clean) may only ever be "mixed" —
        # and must not be picked by either pole.
        assert 1 not in result.indices

    def test_result_type(self):
        fracs, clean = self.make_stats()
        assert isinstance(
            select_two_pole(grid_positions(20), fracs, clean, 1, 1), SelectionResult
        )


class TestViewerHelpers:
    def test_shared_eval_set_dir_derives_group(self, tmp_path):
        from activebench.replay import shared_eval_set_dir

        episode = tmp_path / "runs" / "room__d2__s0" / "gavis"
        episode.mkdir(parents=True)
        group = tmp_path / "shared" / "room__s0"
        group.mkdir(parents=True)
        assert shared_eval_set_dir(episode, tmp_path / "shared") is None  # no transforms yet
        (group / "transforms_eval_shared.json").write_text("{}")
        assert shared_eval_set_dir(episode, tmp_path / "shared") == group
        flat = tmp_path / "runs" / "no-group-name" / "gavis"
        flat.mkdir(parents=True)
        assert shared_eval_set_dir(flat, tmp_path / "shared") is None

    def test_shared_view_class_falls_back_to_hardest(self):
        from activebench.replay import shared_view_class

        frame = {"occlusion": {
            "d1": {"frac": 0.0, "class": "clean"},
            "d2": {"frac": 0.4, "class": "severe"},
        }}
        assert shared_view_class(frame, "d1") == "clean"
        assert shared_view_class(frame, "d2") == "severe"
        assert shared_view_class(frame, "d0") == "severe"  # hardest label fallback
        assert shared_view_class({}, "d2") == "mixed"


class TestResolveSharedEval:
    def test_follows_run_resolution(self, tmp_path):
        import json as _json

        from activebench.eval.reconstruction import (
            reconstruction_artifacts,
            resolve_shared_eval,
        )

        assert resolve_shared_eval(tmp_path) is None
        (tmp_path / "retrain_eval.json").write_text("{}")
        assert resolve_shared_eval(tmp_path) is None  # legacy eval but no shared result
        (tmp_path / "eval_shared.json").write_text("{}")
        assert resolve_shared_eval(tmp_path) == tmp_path / "eval_shared.json"

        named = reconstruction_artifacts(tmp_path, "vanilla-3dgs")
        named.eval_path.parent.mkdir(parents=True)
        named.eval_path.write_text("{}")
        assert resolve_shared_eval(tmp_path) is None  # named run wins, has no shared yet
        shared = named.eval_path.parent / "eval_shared.json"
        shared.write_text(_json.dumps({"class_psnr": {}}))
        assert resolve_shared_eval(tmp_path) == shared
        assert resolve_shared_eval(tmp_path, "legacy") == tmp_path / "eval_shared.json"


class TestRouteBiasedSampling:
    def test_aim_at_matches_convention(self):
        from activebench.eval.shared_eval import aim_at

        yaw, pitch = aim_at([0, 0, 0], [0, 0, -5])  # -Z target: yaw 0
        assert yaw == pytest.approx(0.0) and pitch == pytest.approx(0.0)
        yaw, _ = aim_at([0, 0, 0], [5, 0, 0])  # +X target
        assert yaw == pytest.approx(-np.pi / 2)
        _, pitch = aim_at([0, 0, 0], [0, 3, -3])  # above: positive pitch (looks up)
        assert pitch == pytest.approx(np.pi / 4)
        assert aim_at([1, 2, 3], [1, 2, 3]) == (0.0, 0.0)

    def test_route_biased_positions_ring(self):
        from activebench.eval.shared_eval import route_biased_positions

        xs = np.linspace(0.0, 20.0, 41)
        nav = np.column_stack([xs, np.zeros_like(xs), np.zeros_like(xs)])
        route = np.array([[10.0, 1.1, 0.0]])
        picked = route_biased_positions(nav, route, count=6)
        distances = np.abs(nav[picked][:, 0] - 10.0)
        assert len(picked) == 6
        assert (distances >= 0.8).all() and (distances <= 3.0).all()

    def test_route_biased_positions_empty_cases(self):
        from activebench.eval.shared_eval import route_biased_positions

        nav = np.zeros((5, 3))
        assert route_biased_positions(nav, np.zeros((0, 3)), 4) == []
        assert route_biased_positions(np.zeros((0, 3)), np.ones((1, 3)), 4) == []
        # all nav points sit ON the route (< far_enough): nothing eligible
        assert route_biased_positions(np.ones((5, 3)), np.ones((1, 3)), 4) == []


def test_clean_pick_excludes_lens_crossing_views():
    """Certified-clean but render-occluded views must not fill clean quota."""

    n = 8
    fracs = {"dyn": np.array([0.4, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])}
    # Views 0-1: distractor crossed near the lens — geometrically certified
    # clean (far from surfaces) yet heavily occluded in the render.
    clean = {"dyn": np.array([True, True, True, True, True, False, False, False])}
    result = select_two_pole(grid_positions(n), fracs, clean, severe_quota=2, clean_quota=3)
    clean_picked = [i for i in result.indices if result.labels[i]["dyn"] == "clean"]
    assert sorted(clean_picked) == [2, 3, 4]
