"""Numerical, memory and configuration regressions for the operator lessons."""
import copy
import math
import sys
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "02_challenges/04_neural_operators"))
from generate_data import (SCHEMA, generate_and_save_dataset, generate_batch_fourier_series,
                           generate_splits, spectral_residual)
from operator_training import (BatchedPhysicsInformer, evaluate, load_pair, physics_residual,
                               reference_afno, run, validate_config)


STATS = {"f_mean": 0., "f_std": 1., "u_mean": 0., "u_std": 1.}


class TwiceInput(torch.nn.Module):
    def forward(self, values):
        return 2 * values


def evaluation_loader(batch_size):
    forcing = torch.arange(1., 6.)[:, None, None, None].expand(-1, 1, 4, 4).clone()
    return DataLoader(TensorDataset(forcing, 3 * forcing), batch_size=batch_size)


@pytest.mark.parametrize("batch_size", [1, 2, 3, 5])
def test_metrics_weight_grid_points_not_batches(batch_size):
    metrics, arrays = evaluate(TwiceInput(), evaluation_loader(batch_size), "cpu", STATS)
    assert metrics == pytest.approx({"rmse": math.sqrt(11), "relative_l2": 1 / 3,
                                     "pde_rmse_fft": math.sqrt(11)})
    assert arrays["u_pred"].shape == (5, 1, 4, 4)
    np.testing.assert_array_equal(arrays["u_pred"], 2 * arrays["f"])


def test_metrics_only_does_not_accumulate_prediction_arrays():
    with patch("operator_training.torch.cat", side_effect=AssertionError("must stream")):
        metrics, arrays = evaluate(TwiceInput(), evaluation_loader(2), "cpu", STATS,
                                  collect_predictions=False)
    assert arrays is None
    assert metrics["rmse"] == pytest.approx(math.sqrt(11))


def test_metric_sums_do_not_overflow_float32():
    forcing = torch.full((1, 1, 4, 4), 1e20)
    loader = DataLoader(TensorDataset(forcing, 3 * forcing))
    metrics, _ = evaluate(TwiceInput(), loader, "cpu", STATS, collect_predictions=False)
    assert all(math.isfinite(value) for value in metrics.values())
    assert metrics["rmse"] == pytest.approx(1e20)


def test_empty_evaluation_has_actionable_error():
    fields = torch.empty(0, 1, 4, 4)
    with pytest.raises(ValueError, match="empty dataset"):
        evaluate(TwiceInput(), DataLoader(TensorDataset(fields, fields)), "cpu", STATS)


def test_nonfinite_denormalization_fails_before_export():
    with pytest.raises(FloatingPointError, match="Non-finite evaluation"):
        evaluate(TwiceInput(), evaluation_loader(2), "cpu", dict(STATS, u_std=float("inf")))


class FakePhysics:
    def __init__(self, residual):
        self.residual = residual

    def forward(self, fields):
        return {"reaction_diffusion": self.residual}


def test_physics_residual_cannot_broadcast_away_batch_or_grid():
    fields = torch.zeros(2, 1, 4, 4)
    with pytest.raises(ValueError, match="preserve every"):
        physics_residual(FakePhysics(torch.zeros(1, 1, 4, 4)), fields, fields)
    with pytest.raises(FloatingPointError, match="Non-finite physics"):
        physics_residual(FakePhysics(torch.full_like(fields, float("nan"))), fields, fields)


def test_batched_informer_rejects_empty_batch():
    fields = torch.empty(0, 1, 4, 4)
    with pytest.raises(ValueError, match="matching"):
        BatchedPhysicsInformer(None).forward({"u": fields, "f": fields})


def test_spectral_residual_rectangular_grid_and_autograd():
    x = torch.arange(9, dtype=torch.float64)[:, None] / 9
    y = torch.arange(12, dtype=torch.float64)[None, :] / 12
    solution = torch.sin(6 * torch.pi * x) * torch.cos(4 * torch.pi * y)
    forcing = (1 + (2 * torch.pi) ** 2 * 13) * solution
    torch.testing.assert_close(spectral_residual(solution, forcing),
                               torch.zeros_like(solution), atol=2e-11, rtol=0)
    prediction = (0.9 * solution).requires_grad_()
    spectral_residual(prediction, forcing).square().mean().backward()
    assert prediction.grad is not None and torch.isfinite(prediction.grad).all()
    assert prediction.grad.abs().sum() > 0


def test_spectral_residual_handles_reaction_constant_and_nyquist():
    field = torch.full((8, 8), 3., dtype=torch.float64)
    torch.testing.assert_close(spectral_residual(field, field), torch.zeros_like(field))
    nyquist = (-1.) ** torch.arange(8, dtype=torch.float64)[:, None].expand(8, 8)
    forcing = (1 + (8 * torch.pi) ** 2) * nyquist
    torch.testing.assert_close(spectral_residual(nyquist, forcing),
                               torch.zeros_like(nyquist), atol=1e-11, rtol=0)


def test_spectral_residual_rejects_shape_and_dtype_broadcasts():
    fields = torch.zeros(2, 1, 4, 4)
    with pytest.raises(ValueError, match="matching fields"):
        spectral_residual(fields, fields[:1])
    with pytest.raises(ValueError, match="float32/float64"):
        spectral_residual(fields, fields.double())
    with pytest.raises(ValueError, match="float32/float64"):
        spectral_residual(fields.long(), fields.long())


def test_odd_grid_allows_last_mode_below_nyquist():
    forcing, solution = generate_batch_fourier_series(2, 9, 4,
                                                     generator=torch.Generator().manual_seed(8))
    assert spectral_residual(solution, forcing).norm() / forcing.norm() < 1e-12


@pytest.mark.parametrize("setting", [{"grid_size": 3}, {"grid_size": 4.5},
                                     {"max_mode": 4}, {"batch_size": 0},
                                     {"batch_size": True}, {"val_samples": 0},
                                     {"test_samples": -1}])
def test_invalid_generator_settings_leave_no_partial_dataset(tmp_path, setting):
    destination = tmp_path / "data"
    kwargs = dict(train_samples=2, val_samples=1, test_samples=1, grid_size=8, max_mode=2)
    kwargs.update(setting)
    with pytest.raises(ValueError):
        generate_splits(destination, **kwargs)
    assert not destination.exists()


def test_single_dataset_invalid_split_leaves_no_file(tmp_path):
    filename = tmp_path / "data" / "unknown.hdf5"
    with pytest.raises(ValueError, match="split must"):
        generate_and_save_dataset(filename, 1, grid_size=8, max_mode=2, split="unknown")
    assert not filename.parent.exists()


@pytest.mark.parametrize("dtype", ["i4", "f8"])
def test_loader_rejects_noncanonical_dtype(tmp_path, dtype):
    filename = tmp_path / "train.hdf5"
    with h5py.File(filename, "w") as handle:
        handle.attrs.update(schema=SCHEMA, split="train", seed=1)
        handle.create_dataset("f", data=np.ones((2, 1, 4, 4), dtype=dtype))
        handle.create_dataset("u", data=np.ones((2, 1, 4, 4), dtype=dtype))
    with pytest.raises(ValueError, match="expected float32"):
        load_pair(filename, "train")


@pytest.fixture
def config():
    return {"data": {"grid_size": 8}, "model": {},
            "training": {"steps": 2, "batch_size": 2, "learning_rate": .001, "cpu_threads": 2}}


@pytest.mark.parametrize("name,value", [("batch_size", 0), ("batch_size", 1.5),
                                       ("cpu_threads", True), ("learning_rate", 0),
                                       ("learning_rate", float("nan")),
                                       ("physics_weight", -1), ("physics_weight", float("inf"))])
def test_invalid_training_settings_fail_before_model_build(config, name, value):
    config["training"][name] = value
    with pytest.raises(ValueError, match=name):
        validate_config(config, 2)


@pytest.mark.parametrize("invalid", [None, [], {}, {"data": {}, "model": {}, "training": []}])
def test_invalid_yaml_structure_has_actionable_error(invalid):
    with pytest.raises(ValueError, match="data, model and training mappings"):
        validate_config(invalid, 2)


def test_config_validation_does_not_mutate_settings_and_allows_zero_physics_weight(config):
    config["training"]["physics_weight"] = 0
    original = copy.deepcopy(config)
    validate_config(config, 2)
    assert config == original
    with pytest.raises(ValueError, match="steps"):
        validate_config(config, 1.5)


@pytest.mark.parametrize("patch_size", [[0, 4], [True, 4], [2.5, 4], [4], 4, [3, 4]])
def test_afno_invalid_patch_settings_fail_before_import(patch_size):
    with pytest.raises(ValueError, match="patch|divisible"):
        reference_afno({"patch_size": patch_size}, 8)


def test_failed_write_can_retry_and_normalization_uses_train_only(tmp_path, config):
    import yaml

    data_dir = tmp_path / "data"
    generate_splits(data_dir, train_samples=3, val_samples=2, test_samples=2,
                    grid_size=8, max_mode=2)
    (train_f, train_u), _ = load_pair(data_dir / "train.hdf5", "train")
    # Deliberately change only held-out amplitudes: normalization must remain
    # derived from train data, not from the combined three-way distribution.
    for split in ("val", "test"):
        with h5py.File(data_dir / f"{split}.hdf5", "a") as handle:
            for name in ("f", "u"):
                handle[name][...] = 100 * handle[name][:]
    config["training"]["steps"] = 1
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    output = tmp_path / "output"
    argv = ["--device", "cpu", "--data-dir", str(data_dir), "--config", str(config_path),
            "--output-dir", str(output)]
    model_builder = lambda model_config, grid_size: torch.nn.Conv2d(1, 1, 1)
    with patch("operator_training.torch.save", side_effect=OSError("simulated disk failure")):
        with pytest.raises(OSError, match="simulated disk failure"):
            run(1, model_builder, argv=argv)
    assert not output.exists()
    assert not list(tmp_path.glob(".output-*"))
    metrics = run(1, model_builder, argv=argv)
    assert metrics["normalization_train_only"] == pytest.approx({
        "f_mean": train_f.mean().item(), "f_std": train_f.std(unbiased=False).item(),
        "u_mean": train_u.mean().item(), "u_std": train_u.std(unbiased=False).item()})
    checkpoint = torch.load(output / "model.pt", map_location="cpu", weights_only=True)
    assert checkpoint["metrics"] == metrics
    assert {"metrics.json", "model.pt", "loss.csv", "predictions.npz", "preview.png"} == {
        path.name for path in output.iterdir()}


def test_broken_output_symlink_is_preserved_and_not_followed(tmp_path):
    output, target = tmp_path / "output", tmp_path / "missing"
    output.symlink_to(target, target_is_directory=True)
    with pytest.raises(FileExistsError, match="previous results are preserved"):
        run(1, None, argv=["--output-dir", str(output)])
    assert output.is_symlink() and not target.exists()
