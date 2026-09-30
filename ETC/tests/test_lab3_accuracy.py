"""Material-resolved heat-conduction checks and opt-in FP32 convergence runs."""
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
bar = importlib.import_module("01_labs.03_heat_conduction.source_code.diffusion_bar")
SMALL = {"steps": 3, "batch_size": 8, "learning_rate": .001,
         "layer_size": 8, "num_layers": 1}


@pytest.fixture(scope="module", autouse=True)
def bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(min(previous, 2))
    yield
    torch.set_num_threads(previous)


def physics(device="cpu"):
    return (bar.informer(bar.Diffusion("u_1", "D1"), device),
            bar.informer(bar.Diffusion("u_2", bar.D2), device),
            bar.informer(bar.DiffusionInterface(), device))


@pytest.mark.parametrize("parameterized", [False, True])
def test_endpoint_transform_and_training_never_use_the_reference(parameterized, monkeypatch):
    torch.manual_seed(42)
    model = bar.CompositeBar(SMALL, parameterized)
    d = torch.tensor([[5.], [10.], [25.]])
    left, _ = model(torch.zeros_like(d), d)
    _, right = model(torch.full_like(d, 2.), d)
    assert torch.equal(left, torch.zeros_like(d))
    assert torch.equal(right, torch.full_like(d, 100.))

    def forbidden(*args, **kwargs):
        raise AssertionError("The analytical reference must be evaluation-only")

    monkeypatch.setattr(bar, "analytical", forbidden)
    monkeypatch.setattr(bar, "evaluate", forbidden)
    original_loss, observed = bar.loss_terms, []

    def counted_loss(*args, **kwargs):
        terms = original_loss(*args, **kwargs)
        observed.append({name: float(value.detach()) for name, value in terms.items()})
        return terms

    monkeypatch.setattr(bar, "loss_terms", counted_loss)
    initial = [p.detach().clone() for p in model.parameters()]
    history = bar.optimize_lab(model, physics(), SMALL, "cpu")
    assert [row["step"] for row in history] == [1, 2, 3]
    assert [row["phase"] for row in history] == ["adam", "adam", "lbfgs"]
    assert all(row["closure_evaluations"] >= 1 for row in history)
    offset = 0
    for row in history:
        assert {name: row[name] for name in observed[offset]} == observed[offset]
        offset += row["closure_evaluations"]
    assert offset == len(observed)
    assert any(not torch.equal(a, b) for a, b in zip(initial, model.parameters()))
    assert all(p.dtype == torch.float32 for p in model.parameters())


def good_metrics(parameterized=True):
    return {"per_conductivity": [
        {"D1": d, **{metric: limit / 10 for metric, limit in bar.ACCURACY_LIMITS.items()}}
        for d in bar.evaluation_parameters(parameterized)]}


@pytest.mark.parametrize("metric", list(bar.ACCURACY_LIMITS))
def test_every_material_and_interface_gate_applies_away_from_preview(metric):
    metrics = good_metrics()
    assert bar.accuracy_checks(metrics, True)["passed"]
    # D1=13.5 is absent from both the three preview curves and training grid.
    metrics["per_conductivity"][17][metric] = bar.ACCURACY_LIMITS[metric] * 2
    quality = bar.accuracy_checks(metrics, True)
    assert not quality["passed"]
    assert any(check.get("D1") == 13.5 and check["metric"] == metric and not check["passed"]
               for check in quality["checks"])


@pytest.mark.parametrize("bad", [None, {}, [None], [], [{"D1": "5"}]])
def test_missing_or_malformed_family_cannot_pass(bad):
    assert not bar.accuracy_checks({"per_conductivity": bad}, True)["passed"]


def test_duplicate_parameter_cannot_replace_a_missing_endpoint():
    metrics = good_metrics()
    metrics["per_conductivity"][-1] = metrics["per_conductivity"][0].copy()
    assert not bar.accuracy_checks(metrics, True)["passed"]


@pytest.mark.parametrize("bad", [None, float("nan"), float("inf"), True, -.01])
def test_invalid_error_is_not_a_pass(bad):
    metrics = good_metrics(False)
    metrics["per_conductivity"][0]["left_relative_rmse"] = bad
    assert not bar.accuracy_checks(metrics)["passed"]


@pytest.mark.skipif(os.environ.get("AI4SCI_RUN_CONVERGENCE") != "1",
                    reason="Set AI4SCI_RUN_CONVERGENCE=1 for six full FP32 training runs")
@pytest.mark.parametrize("parameterized", [False, True])
@pytest.mark.parametrize("seed", [42, 43, 7])
def test_default_budget_converges_and_checkpoint_reproduces_fields(tmp_path, parameterized, seed):
    device = os.environ.get("AI4SCI_CONVERGENCE_DEVICE", "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    script = "diffusion_bar_parameterized.py" if parameterized else "diffusion_bar.py"
    output = tmp_path / "run"
    subprocess.run([sys.executable, str(ROOT / "01_labs/03_heat_conduction/source_code" / script),
                    "--device", device, "--seed", str(seed), "--output-dir", str(output)],
                   cwd=ROOT, check=True, capture_output=True, text=True)
    metrics = json.loads((output / "metrics.json").read_text())
    assert metrics["steps"] == bar.DEFAULT_STEPS == 300
    assert metrics["accuracy"]["passed"]
    assert bar.accuracy_checks(metrics["heldout_after"], parameterized)["passed"]
    checkpoint = torch.load(output / "model.pt", map_location=device, weights_only=True)
    assert checkpoint["config"]["lab3_parameterized"] == parameterized
    assert checkpoint["config"]["lab3_dtype"] == "float32"
    model = bar.CompositeBar(checkpoint["config"], parameterized).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    assert all(p.dtype == torch.float32 for p in model.parameters())
    assert bar.accuracy_checks(bar.evaluate(model, physics(device), device), parameterized)["passed"]
    with np.load(output / "predictions.npz", allow_pickle=False) as arrays:
        assert all(array.dtype == np.float32 for array in arrays.values())
        x = torch.as_tensor(arrays["x"], device=device)
        with torch.no_grad():
            for index, d in enumerate(arrays["D1"]):
                left, right = model(x, torch.full_like(x, float(d)))
                actual = torch.where(x <= 1, left, right).cpu().numpy()
                np.testing.assert_allclose(actual, arrays["prediction"][index], rtol=1e-6, atol=2e-5)
