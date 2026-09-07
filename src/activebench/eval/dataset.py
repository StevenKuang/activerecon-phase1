"""Dataset-stream selection shared by coverage and standardized retraining."""

from pathlib import Path


def training_transforms_path(episode_dir: Path) -> Path:
    """Prefer the uniform trajectory stream, oldest-name fallbacks last.

    ``transforms_stream.json`` is the current name (the recorded 1 Hz video
    stream); ``transforms_reconstruction.json`` covers v2/v3-era episodes and
    ``transforms.json`` the v1 decision-frames protocol.
    """

    episode_dir = Path(episode_dir)
    for name in ("transforms_stream.json", "transforms_reconstruction.json"):
        candidate = episode_dir / name
        if candidate.exists():
            return candidate
    return episode_dir / "transforms.json"
