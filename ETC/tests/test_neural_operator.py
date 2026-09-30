"""Real-model and independent equation checks for all neural-operator levels."""
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

MODULE_DIR = Path(__file__).resolve().parents[2] / "02_challenges/04_neural_operators"
sys.path.insert(0, str(MODULE_DIR))
from generate_data import generate_batch_fourier_series, generate_splits, make_unit_square_grid, spectral_residual
from operator_training import load_data


@pytest.fixture(autouse=True)
def bounded_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def test_exact_periodic_reaction_diffusion_and_both_spatial_axes():
    x, y = make_unit_square_grid(32)
    assert x.max().item() == y.max().item() == 31 / 32
    f, u = generate_batch_fourier_series(3, 32, 6, generator=torch.Generator().manual_seed(11))
    relative_residual = spectral_residual(u, f).norm() / f.norm()
    assert relative_residual.item() < 1e-11
    # Check a cross mode against an analytic answer independent of the generator.
    exact = torch.sin(4 * torch.pi * x) * torch.cos(6 * torch.pi * y)
    forcing = (1 + (2 * torch.pi) ** 2 * (2 ** 2 + 3 ** 2)) * exact
    torch.testing.assert_close(spectral_residual(exact, forcing), torch.zeros_like(exact), atol=1e-9, rtol=0)
    assert f.diff(dim=-1).abs().max() > 0.1
    assert f.diff(dim=-2).abs().max() > 0.1


def test_splits_independent_reproducible_and_reject_old_data(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    kwargs = dict(train_samples=5, val_samples=3, test_samples=3, grid_size=16, max_mode=3, seed=10)
    manifest = generate_splits(first, **kwargs)
    generate_splits(second, **kwargs)
    pairs = load_data(first)
    assert [tuple(f.shape) for f, u in pairs] == [(5, 1, 16, 16), (3, 1, 16, 16), (3, 1, 16, 16)]
    assert len({split["seed"] for split in manifest["splits"]}) == 3
    for split in ("train", "val", "test"):
        with h5py.File(first / f"{split}.hdf5") as a, h5py.File(second / f"{split}.hdf5") as b:
            np.testing.assert_array_equal(a["f"][:], b["f"][:])
    assert not torch.equal(pairs[0][0][0], pairs[1][0][0])
    assert not torch.equal(pairs[1][0][0], pairs[2][0][0])
    with pytest.raises(FileExistsError):
        generate_splits(first, **kwargs)
    with h5py.File(first / "test.hdf5", "a") as handle:
        handle.attrs["schema"] = "legacy_poisson"
    with pytest.raises(ValueError, match="wrong dataset schema"):
        load_data(first)
