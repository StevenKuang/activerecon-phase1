"""Write-integrity checks for the compressed-npz artifact writer.

Every Tier-2 artifact goes through ``save_npz_compressed``. On this machine a
691 MB gsplat artifact came back with bad CRC-32 on two different members
after a clean 30k retrain — silent NVMe corruption, the same class the PNG
writer already guards against — which threw away 27 minutes of reconstruction
at the scoring step. These pin the guard.
"""

import zipfile
from pathlib import Path

import numpy as np
import pytest

from activebench.common.io import save_npz_compressed


def test_round_trips_arrays(tmp_path):
    path = tmp_path / "artifact.npz"
    means = np.random.default_rng(0).normal(size=(1000, 3)).astype(np.float32)
    opacities = np.linspace(0.0, 1.0, 1000, dtype=np.float32)

    save_npz_compressed(path, means=means, opacities=opacities)

    with np.load(path) as payload:
        np.testing.assert_array_equal(payload["means"], means)
        np.testing.assert_array_equal(payload["opacities"], opacities)


def test_leaves_no_temp_files(tmp_path):
    path = tmp_path / "artifact.npz"
    save_npz_compressed(path, values=np.arange(16))
    assert [p.name for p in tmp_path.iterdir()] == ["artifact.npz"]


def test_verifies_crc_and_retries_a_corrupt_write(tmp_path, monkeypatch, capsys):
    """A member whose CRC does not match must trigger a rewrite, not a pass."""

    path = tmp_path / "artifact.npz"
    calls = {"n": 0}
    real_testzip = zipfile.ZipFile.testzip

    def flaky_testzip(self):
        calls["n"] += 1
        # Fail the first write the way a bad block does: a named member.
        return "values.npy" if calls["n"] == 1 else real_testzip(self)

    monkeypatch.setattr(zipfile.ZipFile, "testzip", flaky_testzip)
    save_npz_compressed(path, values=np.arange(64))

    assert calls["n"] == 2  # one rejected write, one accepted
    with np.load(path) as payload:
        np.testing.assert_array_equal(payload["values"], np.arange(64))
    assert "bad CRC" in capsys.readouterr().out


def test_raises_when_corruption_persists(tmp_path, monkeypatch):
    """Persistent corruption must fail loudly, not leave a bad artifact behind.

    Scoring reads the artifact back, so a silently-kept bad file would surface
    much later as an unexplained BadZipFile in an analysis pass.
    """

    monkeypatch.setattr(zipfile.ZipFile, "testzip", lambda self: "values.npy")
    with pytest.raises(IOError, match="could not write a verifiable npz"):
        save_npz_compressed(tmp_path / "artifact.npz", values=np.arange(16))


def test_verification_read_bypasses_the_page_cache(tmp_path, monkeypatch):
    """The read-back must be forced to the device, or it proves nothing.

    fsync gets the bytes to the platter but leaves a correct copy in RAM, so a
    naive read-back re-reads that copy and passes while the device holds
    garbage — which is exactly how a corrupt artifact once passed verification
    at write time and failed CRC half an hour later.
    """

    import activebench.common.io as io_module

    dropped = []
    real_drop = io_module._drop_page_cache
    monkeypatch.setattr(
        io_module, "_drop_page_cache",
        lambda p: (dropped.append(Path(p)), real_drop(p))[1],
    )

    path = tmp_path / "artifact.npz"
    io_module.save_npz_compressed(path, values=np.arange(32))
    assert dropped == [path]
