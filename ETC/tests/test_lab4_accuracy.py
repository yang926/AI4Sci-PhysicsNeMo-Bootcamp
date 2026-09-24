"""Independent Lab 4 physics checks and opt-in FP32 training regressions.

The fast tests verify equations, evaluation coverage, and training contracts.
Run the published three-seed budget with AI4SCI_RUN_CONVERGENCE=1; optionally
set AI4SCI_CONVERGENCE_DEVICE=cuda. A short execution run is not convergence.
"""
import copy
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
flow = importlib.import_module("01_labs.04_navier_stokes.source_code.navier_stokes")
SMALL = {"steps": 4, "batch_size": 8, "learning_rate": .001,
         "layer_size": 8, "num_layers": 1}


@pytest.fixture(scope="module", autouse=True)
def bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(min(2, previous))
    yield
    torch.set_num_threads(previous)


class ExactFlow(torch.nn.Module):
    def forward(self, xy, time):
        # Independently stated solution, including a time-dependent gauge.
        k = 2 * torch.pi / flow.LENGTH
        x, y = k * (xy[:, :1] - flow.LOWER), k * (xy[:, 1:] - flow.LOWER)
        decay = torch.exp(-2 * .01 * k * k * time)
        return torch.cat((-x.cos() * y.sin() * decay,
                          x.sin() * y.cos() * decay,
                          -.25 * ((2 * x).cos() + (2 * y).cos()) * decay.square() + 3 * time), 1)


def test_reference_satisfies_all_equations_and_periodic_derivatives_in_fp32():
    physics = flow.informer(flow.NavierStokes(.01), "cpu")
    metrics = flow.evaluate(ExactFlow(), physics, "cpu", True)
    assert [row["time"] for row in metrics["per_time"]] == list(flow.EVALUATION_TIMES)
    assert metrics["pde_rmse"] < 3e-6
    assert metrics["periodic_gradient_max_abs"] < 2e-5
    assert metrics["synthetic_pressure_gauge_aligned_rmse"] < 5e-7
    assert metrics["synthetic_pressure_rmse"] > 1.5
    assert metrics["initial_data_rmse"] < 1e-7
    assert flow.accuracy_checks(metrics)["passed"]


def test_pressure_gauge_is_aligned_independently_at_each_time():
    reference = torch.zeros(3, 7, 3)
    prediction = reference.clone()
    prediction[:, :, 2] = torch.tensor([1., -3., 2.])[:, None]
    errors = flow.solution_errors(prediction, reference)
    assert errors["velocity_rmse"] == 0
    assert errors["pressure_gauge_aligned_rmse"] == 0
    assert errors["pressure_rmse"] == pytest.approx((14 / 3) ** .5)
    assert errors["pressure_offset_rmse"] == errors["pressure_rmse"]


def test_zero_flow_cannot_pass_on_small_pde_residual_alone():
    class Zero(torch.nn.Module):
        def forward(self, xy, time):
            return (0 * (xy.pow(3).sum(1, keepdim=True) + time.pow(3))).expand(-1, 3)

    metrics = flow.evaluate(Zero(), flow.informer(flow.NavierStokes(.01), "cpu"), "cpu", True)
    assert metrics["pde_rmse"] == 0
    assert metrics["per_time"][0]["synthetic_velocity_rmse"] == pytest.approx(.5, abs=1e-6)
    assert not flow.accuracy_checks(metrics)["passed"]


def good_metrics():
    return {"initial_data_rmse": 1e-4, "per_time": [
        {"time": time, "synthetic_velocity_rmse": 1e-4,
         "synthetic_pressure_gauge_aligned_rmse": 1e-4, "pde_rmse": 1e-4,
         "periodic_value_max_abs": 1e-6, "periodic_gradient_max_abs": 1e-6}
        for time in flow.EVALUATION_TIMES]}


@pytest.mark.parametrize("bad", [.021, -.01, True, float("nan"), float("inf"), None])
def test_accuracy_rejects_bad_metric_at_one_time(bad):
    metrics = good_metrics()
    assert flow.accuracy_checks(metrics)["passed"]
    metrics["per_time"][-1]["synthetic_velocity_rmse"] = bad
    assert not flow.accuracy_checks(metrics)["passed"]


def test_missing_evaluation_time_cannot_pass():
    metrics = good_metrics()
    metrics["per_time"].pop()
    with pytest.raises(ValueError, match="six evaluation times"):
        flow.accuracy_checks(metrics)


def test_boolean_time_cannot_stand_in_for_zero():
    metrics = good_metrics()
    metrics["per_time"][0]["time"] = False
    with pytest.raises(ValueError, match="six evaluation times"):
        flow.accuracy_checks(metrics)


def test_training_uses_only_initial_reference_and_logs_real_step_calls(monkeypatch):
    torch.manual_seed(42)
    model = flow.PeriodicFlow(SMALL)
    physics = flow.informer(flow.NavierStokes(.01), "cpu")
    reference = flow.taylor_green
    reference_times = []

    def initial_only(xy, time, *args, **kwargs):
        assert torch.count_nonzero(time) == 0
        reference_times.append(time.detach())
        return reference(xy, time, *args, **kwargs)

    def forbidden(*args, **kwargs):
        raise AssertionError("Training must not evaluate against a future reference")

    monkeypatch.setattr(flow, "taylor_green", initial_only)
    monkeypatch.setattr(flow, "evaluate", forbidden)
    original_loss, calls = flow.loss_terms, []

    def counted_loss(*args, **kwargs):
        calls.append(kwargs.get("points"))
        return original_loss(*args, **kwargs)

    monkeypatch.setattr(flow, "loss_terms", counted_loss)
    before = copy.deepcopy(model.state_dict())
    history = flow.optimize_lab(model, physics, SMALL, "cpu")
    assert reference_times
    assert [row["phase"] for row in history] == ["adam", "lbfgs", "lbfgs", "lbfgs"]
    assert [row["step"] for row in history] == [1, 2, 3, 4]
    assert sum(row["closure_evaluations"] for row in history) == len(calls)
    assert calls[0] is None and all(points is calls[1] for points in calls[1:])
    assert len(calls[1][0]) == 2048 and len(calls[1][2]) == 1024
    assert all(value.dtype == torch.float32 for value in model.state_dict().values())
    assert any(not torch.equal(value, before[name]) for name, value in model.state_dict().items())


def test_initial_loss_weight_preserves_provided_observations(monkeypatch):
    model = flow.PeriodicFlow(SMALL)
    coords = torch.tensor([[-.3, .2], [.1, -.4]])
    target = torch.tensor([[.4, -.3, .2], [-.2, .1, .6]])
    monkeypatch.setattr(flow, "residuals", lambda *args: {"zero": torch.tensor(0.)})
    points = (coords, torch.ones(2, 1), coords, target)
    actual = flow.loss_terms(model, None, 2, "cpu", points=points)["initial_data"]
    expected = 10 * (model(coords, torch.zeros(2, 1)) - target).square().mean()
    torch.testing.assert_close(actual, expected)


@pytest.mark.skipif(os.environ.get("AI4SCI_RUN_CONVERGENCE") != "1",
                    reason="Set AI4SCI_RUN_CONVERGENCE=1 for full measured training")
@pytest.mark.parametrize("seed", [42, 43, 7])
def test_measured_fp32_lesson_convergence(seed, tmp_path):
    device = os.environ.get("AI4SCI_CONVERGENCE_DEVICE", "cpu")
    assert device in ("cpu", "cuda")
    output = tmp_path / f"lab4-{seed}"
    completed = subprocess.run(
        [sys.executable, str(ROOT / "01_labs/04_navier_stokes/source_code/navier_stokes.py"),
         "--smoke-data", "--device", device, "--steps", "3000", "--seed", str(seed),
         "--output-dir", str(output)], cwd=ROOT, capture_output=True, text=True, timeout=1800)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    metrics = json.loads((output / "metrics.json").read_text())
    assert flow.accuracy_checks(metrics["heldout_after"])["passed"], metrics["heldout_after"]
    assert metrics["accuracy"]["passed"]
    assert metrics["training_recipe"]["dtype"] == "float32"
    assert not metrics["training_recipe"]["positive_time_reference_targets_used_for_training"]
    checkpoint = torch.load(output / "model.pt", map_location="cpu", weights_only=True)
    model = flow.PeriodicFlow(checkpoint["config"])
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    with np.load(output / "predictions.npz", allow_pickle=False) as arrays, torch.no_grad():
        xy = torch.from_numpy(arrays["xy"])
        prediction = torch.stack([model(xy, torch.full_like(xy[:, :1], float(time)))
                                  for time in arrays["times"]])
        np.testing.assert_allclose(prediction.numpy(), arrays["prediction"], rtol=2e-4, atol=2e-5)
