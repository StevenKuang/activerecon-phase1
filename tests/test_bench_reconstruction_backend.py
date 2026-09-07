import json

import numpy as np
from PIL import Image

from activebench.eval.reconstruction import (
    GaussianSplats,
    ReconstructionCamera,
    ReconstructionRender,
)
from activebench.eval.retrain import RetrainConfig, run_retrain_eval


class _FakeReconstruction:
    metadata = {"backend": "fake-feedforward", "weights": "test"}

    def __init__(self, points, image_shape):
        self._points = points
        self._image_shape = image_shape
        self.render_cameras = []

    def render(self, c2w_cv, camera=None):
        self.render_cameras.append(camera)
        if camera is None:
            height, width = self._image_shape
        else:
            height, width = camera.height, camera.width
        return ReconstructionRender(
            rgb=np.zeros((height, width, 3), dtype=np.float32),
            depth=np.ones((height, width), dtype=np.float32),
        )

    def gaussian_splats(self):
        count = len(self._points)
        return GaussianSplats(
            centers=self._points.astype(np.float32),
            scales=np.full((count, 3), 0.01, dtype=np.float32),
            quats_wxyz=np.tile([1.0, 0.0, 0.0, 0.0], (count, 1)).astype(np.float32),
            opacities=np.full(count, 0.9, dtype=np.float32),
            colors=np.zeros((count, 3), dtype=np.float32),
        )


class _FakeBackend:
    name = "fake-feedforward"

    def __init__(self, points):
        self._points = points
        self.call = None

    def reconstruct(self, dataset, *, iterations, seed, options):
        self.call = (dataset, iterations, seed, options)
        self.reconstruction = _FakeReconstruction(
            self._points, (dataset.height, dataset.width)
        )
        return self.reconstruction


def _write_episode(episode_dir, width=16, height=16):
    transform = np.eye(4).tolist()
    train_frames = []
    for index in range(3):
        rgb_path = "train_%d.png" % index
        depth_path = "train_%d.npy" % index
        Image.fromarray(np.zeros((height, width, 3), dtype=np.uint8)).save(
            episode_dir / rgb_path
        )
        np.save(episode_dir / depth_path, np.ones((height, width), dtype=np.float32))
        train_frames.append(
            {
                "file_path": rgb_path,
                "depth_path": depth_path,
                "transform_matrix": transform,
            }
        )
    payload = {"w": width, "h": height, "fl_x": 12.0, "fl_y": 12.0, "frames": train_frames}
    (episode_dir / "transforms_reconstruction.json").write_text(json.dumps(payload))

    Image.fromarray(np.zeros((height, width, 3), dtype=np.uint8)).save(
        episode_dir / "eval.png"
    )
    np.save(episode_dir / "eval.npy", np.ones((height, width), dtype=np.float32))
    eval_payload = {
        "frames": [
            {
                "file_path": "eval.png",
                "depth_path": "eval.npy",
                "transform_matrix": transform,
                "stratum": "level",
            }
        ]
    }
    (episode_dir / "transforms_eval.json").write_text(json.dumps(eval_payload))


def test_tier2_evaluator_accepts_framework_neutral_backend(tmp_path):
    episode_dir = tmp_path / "episode"
    episode_dir.mkdir()
    _write_episode(episode_dir)
    points = np.array([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.0, 0.2, 0.0]])
    samples_path = tmp_path / "surface.npz"
    np.savez(samples_path, points=points, normals=np.tile([0.0, 1.0, 0.0], (3, 1)))
    backend = _FakeBackend(points)
    config = RetrainConfig(
        backend="fake-feedforward", run_name="fake-v1", train_iterations=7,
        backend_options={"checkpoint": "test"}, save_renders=True,
    )

    result = run_retrain_eval(episode_dir, samples_path, config, backend_impl=backend)

    assert backend.call[1:] == (7, 0, {"checkpoint": "test"})
    assert result["appearance_per_stratum"]["all"]["psnr"] == 99.0
    assert result["geometry"]["completeness@0.05"] == 1.0
    artifacts = episode_dir / "reconstructions/fake-v1"
    assert (artifacts / "eval.json").exists()
    assert (artifacts / "gaussians.npz").exists()
    assert result["config"]["backend_metadata"]["weights"] == "test"


def test_tier2_evaluator_scores_on_a_shared_eval_set(tmp_path):
    episode_dir = tmp_path / "toy__dyn__s0" / "fake"
    episode_dir.mkdir(parents=True)
    _write_episode(episode_dir)
    shared_dir = tmp_path / "shared" / "toy__s0"
    (shared_dir / "gt").mkdir(parents=True)
    transform = np.eye(4).tolist()
    frames = []
    for index, (cls, frac) in enumerate(
        [("severe", 0.4), ("severe", 0.3), ("clean", 0.0), ("clean", 0.0)]
    ):
        rgb_rel = "gt/eval_%04d.png" % index
        depth_rel = "gt/eval_depth_%04d.npy" % index
        Image.fromarray(np.zeros((16, 16, 3), dtype=np.uint8)).save(shared_dir / rgb_rel)
        np.save(shared_dir / depth_rel, np.ones((16, 16), dtype=np.float32))
        frames.append({
            "file_path": rgb_rel,
            "depth_path": depth_rel,
            "transform_matrix": transform,
            "stratum": "level",
            "occlusion": {"dyn": {"frac": frac, "class": cls}},
        })
    (shared_dir / "transforms_eval_shared.json").write_text(json.dumps({
        "w": 16, "h": 16, "fl_x": 8.0, "fl_y": 8.0, "cx": 8.0, "cy": 8.0,
        "frames": frames,
    }))
    points = np.zeros((4, 3))
    np.save(tmp_path / "samples.npy", points)  # placeholder; samples npz below
    np.savez(tmp_path / "samples.npz", points=points, normals=np.tile([0.0, 1.0, 0.0], (4, 1)))

    backend = _FakeBackend(points)
    result = run_retrain_eval(
        episode_dir, tmp_path / "samples.npz",
        RetrainConfig(backend="fake", run_name="fake-run"),
        backend_impl=backend,
        eval_set_dir=shared_dir,
    )
    assert result["eval_set"] == str(shared_dir)
    shared_out = episode_dir / "reconstructions" / "fake-run" / "eval_shared.json"
    payload = json.loads(shared_out.read_text())
    assert payload["difficulty"] == "dyn"
    assert payload["class_psnr"]["dyn"]["severe"]["n"] == 2
    assert payload["class_psnr"]["dyn"]["clean"]["n"] == 2
    assert len(payload["views"]) == 4


def test_tier2_evaluator_renders_at_shared_eval_resolution(tmp_path):
    episode_dir = tmp_path / "toy__d0__s0" / "fake"
    episode_dir.mkdir(parents=True)
    _write_episode(episode_dir, width=16, height=16)
    shared_dir = tmp_path / "shared" / "toy__s0"
    shared_dir.mkdir(parents=True)
    transform = np.eye(4).tolist()
    Image.fromarray(np.zeros((24, 32, 3), dtype=np.uint8)).save(
        shared_dir / "eval.png"
    )
    np.save(shared_dir / "eval.npy", np.ones((24, 32), dtype=np.float32))
    (shared_dir / "transforms_eval_shared.json").write_text(json.dumps({
        "w": 32, "h": 24, "fl_x": 20.0, "fl_y": 20.0, "cx": 16.0, "cy": 12.0,
        "frames": [{
            "file_path": "eval.png",
            "depth_path": "eval.npy",
            "transform_matrix": transform,
            "stratum": "level",
        }],
    }))
    points = np.zeros((4, 3))
    samples = tmp_path / "samples.npz"
    np.savez(samples, points=points, normals=np.tile([0.0, 1.0, 0.0], (4, 1)))
    backend = _FakeBackend(points)

    result = run_retrain_eval(
        episode_dir,
        samples,
        RetrainConfig(
            backend="fake",
            run_name="resolution-test",
            evaluation_resolutions=((16, 12),),
        ),
        backend_impl=backend,
        eval_set_dir=shared_dir,
    )

    camera = backend.reconstruction.render_cameras[0]
    assert camera == ReconstructionCamera(32, 24, 20.0, 20.0, 16.0, 12.0)
    assert result["train_resolution"] == [16, 16]
    assert result["eval_resolution"] == [32, 24]
    assert set(result["appearance_by_resolution"]) == {"32x24", "16x12"}
    assert result["appearance_by_resolution"]["16x12"]["all"]["psnr"] == 99.0
