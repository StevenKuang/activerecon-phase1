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
