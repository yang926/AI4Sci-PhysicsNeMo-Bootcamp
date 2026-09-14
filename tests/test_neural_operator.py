"""Real-model and independent equation checks for all neural-operator levels."""
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

MODULE_DIR = Path(__file__).resolve().parents[1] / "challenge/neural_operator"
sys.path.insert(0, str(MODULE_DIR))
from generate_data import generate_batch_fourier_series, generate_splits, make_unit_square_grid, spectral_residual
from operator_training import load_data, reference_afno, reference_fno, reference_physics, run


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


@pytest.mark.parametrize("method", ["fno", "afno"])
def test_real_physicsnemo_models_forward_and_backward(method):
    torch.manual_seed(0)
    config = ({"latent_channels": 8, "num_fno_layers": 2, "num_fno_modes": 4,
               "padding": 0, "coord_features": False, "decoder_layer_size": 8}
              if method == "fno" else {"patch_size": [4, 4], "embed_dim": 16, "depth": 2, "num_blocks": 4})
    model = (reference_fno if method == "fno" else reference_afno)(config, 16)
    assert model.__class__.__module__.startswith("physicsnemo.models")
    forcing, solution = generate_batch_fourier_series(2, 16, 3, dtype=torch.float32)
    prediction = model(forcing[:, None])
    assert prediction.shape == solution[:, None].shape
    loss = (prediction - solution[:, None]).square().mean()
    loss.backward()
    gradients = [param.grad for param in model.parameters() if param.grad is not None]
    assert gradients and all(torch.isfinite(grad).all() for grad in gradients)
    assert sum(grad.abs().sum().item() for grad in gradients) > 0


def test_physicsinformer_matches_fft_and_backpropagates():
    f, u = generate_batch_fourier_series(2, 16, 3, dtype=torch.float32)
    # Use a deliberately inaccurate prediction so the physical loss has a gradient.
    prediction = (u[:, None] * 0.8).requires_grad_()
    physics = reference_physics("cpu")
    residual = physics.forward({"u": prediction, "f": f[:, None]})["reaction_diffusion"]
    torch.testing.assert_close(residual, spectral_residual(prediction, f[:, None]), atol=5e-5, rtol=5e-5)
    residual.square().mean().backward()
    assert prediction.grad is not None and torch.isfinite(prediction.grad).all()
    assert prediction.grad.abs().sum() > 0


@pytest.mark.parametrize("level", [1, 2, 3])
def test_reference_training_writes_finite_artifacts(tmp_path, level):
    import yaml
    data_dir = tmp_path / "data"
    generate_splits(data_dir, train_samples=6, val_samples=2, test_samples=2, grid_size=16, max_mode=3)
    model = ({"latent_channels": 8, "num_fno_layers": 2, "num_fno_modes": 4,
              "padding": 0, "coord_features": False, "decoder_layer_size": 8}
             if level != 2 else {"patch_size": [4, 4], "embed_dim": 16, "depth": 2, "num_blocks": 4})
    config = {"data": {"grid_size": 16}, "model": model,
              "training": {"steps": 2, "batch_size": 2, "learning_rate": 0.001, "cpu_threads": 2}}
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    output = tmp_path / "output"
    metrics = run(level, reference_fno, argv=["--reference", "--device", "cpu", "--steps", "2",
                  "--data-dir", str(data_dir), "--config", str(config_path), "--output-dir", str(output)])
    assert metrics["steps"] == 2 and not metrics["convergence_claim"]
    for filename in ("metrics.json", "loss.csv", "model.pt", "predictions.npz", "preview.png"):
        assert (output / filename).stat().st_size > 0
    assert json.loads((output / "metrics.json").read_text())["status"] == "completed"
    from ai4sci.run_validation import validate_artifacts
    checked = validate_artifacts(output, steps=2, device="cpu", expected_version="2.2.2", seed=42)
    assert checked["checkpoint_seed"] == 42 and checked["checkpoint_physicsnemo_version"] == "2.2.2"
    if level == 1:
        with pytest.raises(ValueError, match="Recorded seed does not match"):
            validate_artifacts(output, steps=2, device="cpu", expected_version="2.2.2", seed=43)
        checkpoint = torch.load(output / "model.pt", map_location="cpu", weights_only=True)
        checkpoint["seed"] = 43
        torch.save(checkpoint, output / "model.pt")
        with pytest.raises(ValueError, match="Checkpoint seed does not match"):
            validate_artifacts(output, steps=2, device="cpu", expected_version="2.2.2", seed=42)
        checkpoint["seed"] = 42
        checkpoint["physicsnemo_version"] = "0.0.0-wrong-version"
        torch.save(checkpoint, output / "model.pt")
        with pytest.raises(ValueError, match="Checkpoint PhysicsNeMo version"):
            validate_artifacts(output, steps=2, device="cpu", expected_version="2.2.2", seed=42)
    with np.load(output / "predictions.npz") as predictions:
        assert all(np.isfinite(predictions[key]).all() for key in predictions.files)
