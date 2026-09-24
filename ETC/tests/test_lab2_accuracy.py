"""Projectile accuracy contracts; full seeded FP32 runs are explicitly opt-in.

AI4SCI_RUN_CONVERGENCE=1 enables the 5000-step regression checks. Select CUDA
with AI4SCI_CONVERGENCE_DEVICE=cuda; the default small tests check execution.
"""
import importlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "01_labs/02_projectile/source_code"
sys.path.insert(0, str(SOURCE))
lab = importlib.import_module("projectile")
SMALL_CONFIG = {"steps": 4, "batch_size": 8, "learning_rate": .001,
                "layer_size": 8, "num_layers": 1}


@pytest.fixture(scope="module", autouse=True)
def bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(min(2, previous))
    yield
    torch.set_num_threads(previous)


class KnownTrajectory(torch.nn.Module):
    def __init__(self, offset=0., velocity_error=0.):
        super().__init__()
        self.offset, self.velocity_error = offset, velocity_error

    def forward(self, t):
        return lab.analytical(t) + self.offset + self.velocity_error * t


def test_exact_trajectory_passes_and_initial_errors_are_independent_checks():
    physics = lab.informer(lab.ProjectileEquation(), "cpu")
    exact = lab.evaluate(KnownTrajectory(), physics, "cpu")
    assert lab.accuracy_checks(exact)["passed"]
    offset = lab.evaluate(KnownTrajectory(offset=.11), physics, "cpu")
    velocity = lab.evaluate(KnownTrajectory(velocity_error=.06), physics, "cpu")
    for result in (offset, velocity):
        assert result["pde_rmse"] < 1e-6
        assert not lab.accuracy_checks(result)["passed"]
    assert offset["initial_position_max_abs"] == pytest.approx(.11)
    assert velocity["initial_velocity_max_abs"] == pytest.approx(.06, abs=3e-6)


@pytest.mark.parametrize("metric", ["solution_rmse", "solution_max_abs", "pde_rmse",
                                   "initial_position_max_abs", "initial_velocity_max_abs"])
@pytest.mark.parametrize("bad", [True, None, -1., float("nan"), float("inf"), 100.])
def test_accuracy_rejects_invalid_and_inaccurate_errors(metric, bad):
    valid = {key: 0. for key in ("solution_rmse", "solution_max_abs", "pde_rmse",
                               "initial_position_max_abs", "initial_velocity_max_abs")}
    assert lab.accuracy_checks(valid)["passed"]
    valid[metric] = bad
    result = lab.accuracy_checks(valid)
    assert not result["passed"]
    assert not next(check for check in result["checks"] if check["metric"] == metric)["passed"]


def test_training_uses_only_ode_and_initial_conditions_and_updates_every_step(monkeypatch):
    torch.manual_seed(42)
    model = lab.ProjectileModel(SMALL_CONFIG).float()
    physics = lab.informer(lab.ProjectileEquation(), "cpu")

    def forbidden(*args, **kwargs):
        raise AssertionError("Analytical trajectories cannot be used for training")

    monkeypatch.setattr(lab, "analytical", forbidden)
    monkeypatch.setattr(lab, "evaluate", forbidden)
    original_loss, calls = lab.loss_terms, []

    def counted_loss(*args):
        terms = original_loss(*args)
        calls.append(terms)
        return terms

    monkeypatch.setattr(lab, "loss_terms", counted_loss)
    initial = [parameter.detach().clone() for parameter in model.parameters()]
    history = lab.optimize_projectile(model, physics, SMALL_CONFIG, "cpu")
    assert len(calls) == len(history) == 4
    assert [row["step"] for row in history] == [1, 2, 3, 4]
    rates = [row["learning_rate"] for row in history]
    assert rates[0] == .001 and all(a > b > 0 for a, b in zip(rates, rates[1:]))
    assert all(math.isfinite(value) for row in history for value in row.values())
    assert all(parameter.dtype == torch.float32 for parameter in model.parameters())
    assert any(not torch.equal(before, after) for before, after in zip(initial, model.parameters()))


def assert_reload(output):
    checkpoint = torch.load(output / "model.pt", map_location="cpu", weights_only=True)
    cfg = checkpoint["config"]
    assert cfg["lab2_dtype"] == "float32" and cfg["lab2_recipe"] == "adam_cosine_fp32_v1"
    assert all(value.dtype == torch.float32 for value in checkpoint["model_state_dict"].values()
               if value.is_floating_point())
    model = lab.ProjectileModel(cfg).float()
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    with np.load(output / "predictions.npz", allow_pickle=False) as saved:
        with torch.no_grad():
            actual = model(torch.from_numpy(saved["t"]))
        np.testing.assert_allclose(actual.numpy(), saved["prediction"], rtol=2e-5, atol=2e-5)


def test_short_cli_is_reloadable_but_not_convergence(tmp_path, monkeypatch):
    import yaml
    config = tmp_path / "small.yaml"
    config.write_text(yaml.safe_dump(SMALL_CONFIG), encoding="utf-8")
    output = tmp_path / "short"
    monkeypatch.setattr(sys, "argv", ["projectile.py", "--device", "cpu", "--steps", "4",
                                     "--config", str(config), "--output-dir", str(output)])
    lab.main()
    metrics = json.loads((output / "metrics.json").read_text())
    assert not metrics["accuracy"]["passed"]
    assert metrics["training_recipe"]["optimizer_step_calls"] == 4
    assert metrics["training_recipe"]["reference_targets_used_for_training"] is False
    assert_reload(output)


@pytest.mark.skipif(os.environ.get("AI4SCI_RUN_CONVERGENCE") != "1",
                    reason="Full lesson accuracy is opt-in, separate from execution tests")
@pytest.mark.parametrize("seed", [42, 43, 7])
def test_full_lesson_accuracy_fp32(seed, tmp_path):
    device = os.environ.get("AI4SCI_CONVERGENCE_DEVICE", "cpu")
    output = tmp_path / f"seed-{seed}"
    subprocess.run([sys.executable, str(SOURCE / "projectile.py"), "--device", device,
                    "--seed", str(seed), "--steps", "5000", "--output-dir", str(output)],
                   cwd=ROOT, check=True,
                   env={**os.environ, "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"})
    metrics = json.loads((output / "metrics.json").read_text())
    assert metrics["steps"] == 5000 and metrics["seed"] == seed
    assert metrics["accuracy"]["passed"], metrics["accuracy"]
    assert lab.accuracy_checks(metrics["heldout_after"])["passed"]
    assert math.isfinite(metrics["extrapolation_rmse"])
    assert_reload(output)
