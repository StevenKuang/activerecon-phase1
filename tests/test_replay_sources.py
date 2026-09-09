from types import SimpleNamespace

import numpy as np
import pytest

from activebench.replay import EpisodeReplay


def test_corrupt_replay_rgb_reports_the_exact_frame(tmp_path):
    rgb = tmp_path / "frame_00042.png"
    rgb.write_bytes(b"corrupt image")
    depth = tmp_path / "depth_00042.npy"
    np.save(depth, np.ones((4, 4), dtype=np.float32))
    replay = EpisodeReplay.__new__(EpisodeReplay)
    replay._chunks = [None]
    replay.point_stride = 1
    replay.frames = [SimpleNamespace(index=42, rgb_path=rgb, depth_path=depth)]
    with pytest.raises(OSError, match="RGB frame 42: .*frame_00042.png"):
        replay.chunk(0)


def test_summary_allows_short_reconstruction_without_overall_completeness(tmp_path):
    import json

    reconstruction = tmp_path / "reconstructions/gsplat"
    reconstruction.mkdir(parents=True)
    (reconstruction / "eval.json").write_text(json.dumps({
        "appearance_per_stratum": {"all": {"psnr": 9.5}},
        "geometry": {"num_recon_points": 0, "bins": {}},
    }))
    replay = EpisodeReplay.__new__(EpisodeReplay)
    replay.episode_dir = tmp_path
    replay.manifest = {"method": {"name": "example"}, "num_captures": 2,
                       "clock": {"final_sim_time": 1.0}, "seed": 0, "distractors": []}
    text = replay.summary("gsplat")
    assert "PSNR 9.5" in text
    assert "cmp@5cm unavailable" in text
