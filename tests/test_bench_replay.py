from pathlib import Path

import numpy as np
import pytest

CAMPAIGN = Path(__file__).resolve().parent.parent / "runs_campaign"
EPISODE = CAMPAIGN / "van_gogh_room__d2__s0" / "r3con-pano"

pytestmark = pytest.mark.skipif(
    not (EPISODE / "manifest.json").exists(), reason="campaign run data not present"
)


def make_replay():
    from activebench.replay import EpisodeReplay

    return EpisodeReplay(EPISODE, point_stride=12)


class TestEpisodeReplay:
    def test_frames_and_monotonic_capture_index(self):
        replay = make_replay()
        assert len(replay.frames) == replay.manifest["num_captures"]
        last = -1
        for t in np.linspace(0.0, replay.t_end + 1.0, 60):
            k = replay.capture_index_at(t)
            assert k >= last
            last = k
        assert last == len(replay.frames) - 1
        assert replay.capture_index_at(-5.0) == 0

    def test_camera_pose_interpolates_between_captures(self):
        replay = make_replay()
        a, b = replay.frames[3], replay.frames[4]
        mid = replay.camera_pose_at(0.5 * (a.time + b.time))
        # Midpoint lies between the two captures (within the segment bbox).
        lo = np.minimum(a.pose.position, b.pose.position) - 1e-6
        hi = np.maximum(a.pose.position, b.pose.position) + 1e-6
        assert (mid.position >= lo).all() and (mid.position <= hi).all()
        # Exact at capture times, clamped at the end.
        np.testing.assert_allclose(replay.camera_pose_at(a.time).position, a.pose.position)
        np.testing.assert_allclose(
            replay.camera_pose_at(replay.t_end + 99).position, replay.frames[-1].pose.position
        )

    def test_chunks_load_and_cache(self):
        replay = make_replay()
        chunk = replay.chunk(0)
        assert len(chunk.points) > 100
        assert chunk.colors.shape == chunk.points.shape
        assert replay.chunk(0) is chunk  # cached

    def test_distractors_reconstruct_and_move(self):
        replay = make_replay()
        assert len(replay.distractors) == len(replay.manifest["distractors"])
        p0 = replay.distractor_poses_at(0.0)
        p9 = replay.distractor_poses_at(9.0)
        assert any(
            np.linalg.norm(np.asarray(a[1]) - np.asarray(b[1])) > 0.2 for a, b in zip(p0, p9)
        )

    def test_summary_mentions_scores(self):
        text = make_replay().summary()
        assert "captures" in text and "PSNR" in text


class TestRunIndex:
    def test_scan_finds_campaign_layout(self):
        from activebench.replay import RunIndex

        index = RunIndex.scan([CAMPAIGN])
        assert "van_gogh_room" in index.scenes()
        assert "d2" in index.difficulties("van_gogh_room")
        seeds = index.seeds("van_gogh_room", "d2")
        assert seeds[:3] == ["s0", "s1", "s2"]
        methods = index.methods("van_gogh_room", "d2", "s0")
        assert "r3con-pano" in methods and "gavis" in methods
        assert index.episode_dir("van_gogh_room", "d2", "s0", "r3con-pano") == EPISODE
