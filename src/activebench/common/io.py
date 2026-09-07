"""I/O helpers."""

import json
from pathlib import Path
from typing import Any, Dict


def ensure_dir(path: Path) -> Path:
    """Create a directory if needed and return it."""

    path.mkdir(parents=True, exist_ok=True)
    return path


def load_yaml(path: Path) -> Dict[str, Any]:
    """Load a YAML mapping with PyYAML."""

    import yaml

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError("YAML file must contain a mapping at the top level: %s" % path)
    return dict(data)


def load_json(path: Path) -> Dict[str, Any]:
    """Load a JSON file."""

    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return dict(data)


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    """Write indented JSON."""

    if path.parent:
        ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def _drop_page_cache(path: Path) -> None:
    """Evict ``path``'s clean pages so the next read reaches the device.

    Best effort: POSIX_FADV_DONTNEED is advisory and absent on some
    platforms, in which case verification still runs, just against whatever
    the cache holds.
    """

    import os

    try:
        fd = os.open(str(path), os.O_RDONLY)
    except OSError:
        return
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    except (AttributeError, OSError):
        pass
    finally:
        os.close(fd)


def save_npz_compressed(path: Path, **arrays: Any) -> Path:
    """Write a compressed npz atomically, durably, and verified.

    Three failure classes, all seen on this machine:

    - a campaign crash mid-write leaves a truncated archive that fails with
      BadZipFile on the NEXT analysis pass (one v2 legacy artifact was lost
      this way) — hence temp file + rename;
    - data sitting in the page cache when the box goes down — hence fsync of
      both the file and its directory;
    - **silent write-time corruption** on the local NVMe, the same class the
      PNG writer already guards against. A 691 MB gsplat artifact came back
      with bad CRC-32 on two different members after a clean 30k retrain, so
      a 27-minute reconstruction was lost at the scoring step.

    Verification is ``testzip()`` rather than a byte comparison: an npz is a
    zip, so every member already carries a CRC-32, and checking those streams
    the archive instead of holding a second copy of a multi-GB payload in
    memory. Retries re-run the whole write so a fresh block allocation can
    sidestep a transiently bad region.

    The verification read must bypass the page cache or it proves nothing.
    ``fsync`` guarantees the data reached the device; it does not make a
    subsequent read come *from* the device, so a naive read-back re-reads the
    correct copy still in RAM and passes while the platter holds garbage.
    That is not hypothetical here: an artifact verified this way at write time
    failed CRC half an hour later, with two reads agreeing on the same bad
    bytes. ``posix_fadvise(DONTNEED)`` drops the (now clean) pages first.
    """

    import os
    import tempfile
    import zipfile

    import numpy as np

    path = Path(path)
    ensure_dir(path.parent)
    try:
        parent_fd = os.open(str(path.parent), os.O_RDONLY | os.O_DIRECTORY)
    except (OSError, NotImplementedError):
        parent_fd = None

    attempts = 3
    last_error: Any = None
    try:
        for attempt in range(attempts):
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="wb", dir=path.parent, prefix=".%s." % path.name,
                    suffix=".tmp", delete=False,
                ) as handle:
                    temporary = Path(handle.name)
                    np.savez_compressed(handle, **arrays)
                    handle.flush()
                    os.fsync(handle.fileno())
                temporary.replace(path)
                temporary = None
                if parent_fd is not None:
                    os.fsync(parent_fd)
                _drop_page_cache(path)
                with zipfile.ZipFile(path) as archive:
                    corrupt = archive.testzip()
                if corrupt is None:
                    return path
                last_error = IOError(
                    "npz verification failed for %s: bad CRC for %r (attempt %d/%d)"
                    % (path, corrupt, attempt + 1, attempts)
                )
            except (OSError, zipfile.BadZipFile) as exc:
                last_error = exc
            finally:
                if temporary is not None and temporary.exists():
                    temporary.unlink()
            print("[io] %s; rewriting" % last_error, flush=True)
        raise IOError("could not write a verifiable npz at %s: %s" % (path, last_error))
    finally:
        if parent_fd is not None:
            os.close(parent_fd)

