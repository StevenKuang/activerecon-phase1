from activebench.eval.dataset import training_transforms_path


def test_training_stream_falls_back_to_agent_captures(tmp_path):
    legacy = tmp_path / "transforms.json"
    legacy.write_text("{}")
    assert training_transforms_path(tmp_path) == legacy


def test_training_stream_prefers_uniform_trajectory_samples(tmp_path):
    legacy = tmp_path / "transforms.json"
    sampled = tmp_path / "transforms_reconstruction.json"
    legacy.write_text("{}")
    sampled.write_text("{}")
    assert training_transforms_path(tmp_path) == sampled
