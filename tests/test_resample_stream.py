import importlib.util
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SPEC = importlib.util.spec_from_file_location(
    "resample_stream", _REPO_ROOT / "scripts" / "resample_stream.py"
)
assert _SPEC is not None and _SPEC.loader is not None
resample_stream = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(resample_stream)


def test_resample_grid_includes_endpoints_and_counts():
    times = resample_stream.resample_grid(300.0, 5.0)
    assert len(times) == 1501
    assert times[0] == 0.0 and times[-1] == 300.0
    assert abs(times[1] - 0.2) < 1e-12
    # 1 Hz grid is a subset: every 5th sample lands on integer seconds.
    assert all(abs(times[5 * k] - k) < 1e-9 for k in range(301))


def test_resample_grid_at_source_rate_is_identity():
    times = resample_stream.resample_grid(60.0, 1.0)
    assert len(times) == 61 and times[-1] == 60.0


def test_patched_manifest_keeps_world_and_adds_provenance():
    source = {
        "method": {"name": "r3con-pano"},
        "distractors": [{"name": "d0"}],
        "clock": {"final_sim_time": 300.0},
        "reconstruction": {"interval_s": 1.0, "streamed_to_agent": True,
                           "num_frames": 301, "frames": ["old"]},
    }
    log = [{"index": 0, "time": 0.0}]
    patched = resample_stream.patched_manifest(source, log, 5.0, "transforms_stream.json", "/src/ep")

    # World identity (audit inputs) and method metadata are preserved.
    assert patched["distractors"] == source["distractors"]
    assert patched["method"] == source["method"]
    assert patched["clock"] == source["clock"]
    rec = patched["reconstruction"]
    assert rec["interval_s"] == 0.2
    assert rec["num_frames"] == 1 and rec["frames"] is log
    # Honest provenance: the agent never saw these frames.
    assert rec["streamed_to_agent"] is False
    assert rec["resampled_from"] == "/src/ep"
    assert rec["source_interval_s"] == 1.0
    # The source manifest is not mutated.
    assert source["reconstruction"]["streamed_to_agent"] is True


def test_patched_manifest_records_spatial_resampling():
    source = {
        "method": {"name": "r3con-pano"},
        "reconstruction": {"interval_s": 1.0},
    }
    patched = resample_stream.patched_manifest(
        source,
        [],
        1.0,
        "transforms_stream.json",
        "/src/ep",
        resolution=(1280, 960),
        source_resolution=(640, 480),
    )
    rec = patched["reconstruction"]
    assert rec["resolution"] == {"width": 1280, "height": 960}
    assert rec["source_resolution"] == {"width": 640, "height": 480}
