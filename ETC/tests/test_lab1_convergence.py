"""Lab 1 training contracts and opt-in, measured lesson-accuracy regressions.

The default tests use a few updates and do not claim convergence. Run the full
three-mode/three-seed checks with AI4SCI_RUN_CONVERGENCE=1. Set
AI4SCI_CONVERGENCE_DEVICE=cuda to run that separate suite on a GPU.
"""
import csv
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
sys.path.insert(0, str(ROOT))
basic = importlib.import_module("01_labs.01_pinn.source_code.pinn_basics")
from ETC.course_materials.run_validation import validate_artifacts

MODES = ("forward", "parameterized", "inverse")
SMALL_CONFIG = {"steps": 4, "batch_size": 8, "learning_rate": 0.001,
                "layer_size": 8, "num_layers": 1}


@pytest.fixture(scope="module", autouse=True)
def bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(min(2, previous))
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_inverse_zero_boundaries_are_exact_for_arbitrary_weights(dtype):
    model = basic.BasicPINN(SMALL_CONFIG, "inverse", dtype=dtype)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.uniform_(-2, 2)
    x = torch.tensor([[0.0], [1.0]], dtype=dtype)
    assert torch.equal(model(x), torch.zeros_like(x))
    assert model.solution.frequencies.dtype == dtype
    assert model.source.frequencies.dtype == dtype


def test_transformed_observation_loss_uses_only_solution_and_supplied_data():
    model = basic.BasicPINN(SMALL_CONFIG, "inverse", dtype=torch.float64)
    # Non-analytical data makes this a test of the supplied observations, not a
    # second implementation of the reference function.
    x = torch.tensor([[0.0], [.1], [.35], [.7], [1.0]], dtype=torch.float64)
    observed = torch.tensor([[5.0], [-.08], [-.04], [.03], [-9.0]], dtype=torch.float64)
    interior_x, interior_u = x[1:-1], observed[1:-1]
    expected = basic.INVERSE_DATA_WEIGHT * (
        (model(interior_x) - interior_u) / (interior_x * (1 - interior_x))
    ).square().mean()
    actual = basic.observation_loss(model, x, observed)
    torch.testing.assert_close(actual, expected, rtol=1e-13, atol=1e-13)
    actual.backward()
    solution_gradients = [parameter.grad for parameter in model.solution.parameters()]
    assert all(gradient is not None and torch.isfinite(gradient).all()
               for gradient in solution_gradients)
    assert any(torch.count_nonzero(gradient) for gradient in solution_gradients)
    assert all(parameter.grad is None for parameter in model.source.parameters())
    with torch.no_grad():
        for parameter in model.source.parameters():
            parameter.add_(100)
    torch.testing.assert_close(basic.observation_loss(model, x, observed), actual.detach(),
                               rtol=0, atol=0)
    with pytest.raises(ValueError, match="interior"):
        basic.observation_loss(model, x[[0, -1]], observed[[0, -1]])


@pytest.mark.parametrize("mode", MODES)
def test_optimizer_has_no_reference_access_and_records_real_closures(mode, monkeypatch):
    torch.manual_seed(42)
    model = basic.BasicPINN(SMALL_CONFIG, mode, dtype=torch.float64)
    physics = basic.informer(basic.Poisson1D(mode == "inverse"), "cpu")
    x = torch.linspace(.01, .99, 100, dtype=torch.float64)[:, None]
    observations = (x, basic.inverse_reference(x)[0]) if mode == "inverse" else None

    def forbidden(*args, **kwargs):
        raise AssertionError("Training must not call an analytical reference or evaluation")

    for name in ("inverse_reference", "analytical", "evaluate", "evaluate_at_length"):
        monkeypatch.setattr(basic, name, forbidden)
    original_loss = basic.loss_terms
    calls = []

    def counted_loss(*args, **kwargs):
        calls.append(kwargs.get("points"))
        return original_loss(*args, **kwargs)

    monkeypatch.setattr(basic, "loss_terms", counted_loss)
    initial = [parameter.detach().clone() for parameter in model.parameters()]
    history = basic.optimize_lab(model, physics, SMALL_CONFIG, "cpu", observations)
    assert [row["step"] for row in history] == [1, 2, 3, 4]
    assert [row["phase"] for row in history] == ["lbfgs"] * 4
    assert all(type(row["closure_evaluations"]) is int and row["closure_evaluations"] >= 1
               for row in history)
    assert sum(row["closure_evaluations"] for row in history) == len(calls)
    assert calls[0] is not None and all(points is calls[0] for points in calls)
    x, lengths = calls[0]
    assert x.shape == lengths.shape == (SMALL_CONFIG["batch_size"] * (17 if mode == "parameterized" else 1), 1)
    assert all(row.keys() == history[0].keys() for row in history)
    assert all(math.isfinite(value) for row in history for key, value in row.items()
               if key != "phase")
    assert any(not torch.equal(before, after) for before, after in zip(initial, model.parameters()))


@pytest.mark.parametrize("mode", MODES)
def test_evaluation_accepts_float64_without_changing_global_default(mode):
    previous_dtype = torch.get_default_dtype()
    model = basic.BasicPINN(SMALL_CONFIG, mode, dtype=torch.float64)
    physics = basic.informer(basic.Poisson1D(mode == "inverse"), "cpu")
    metrics = basic.evaluate(model, physics, "cpu")
    assert torch.get_default_dtype() == previous_dtype
    assert next(model.parameters()).dtype == torch.float64
    records = metrics.get("per_length", [metrics])
    assert all(math.isfinite(value) for record in records for value in record.values())
    assert not basic.accuracy_checks(metrics, mode)["passed"]
    if mode == "inverse":
        assert metrics["boundary_max_abs"] == 0
    if mode == "parameterized":
        assert [record["length"] for record in records] == [1., 1.25, 1.5, 1.75, 2.]


def good_errors():
    return {"solution_rmse": 1e-5, "pde_rmse": 1e-4, "boundary_max_abs": 0.,
            "source_rmse": .01, "source_max_abs": .02}


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("cli_steps", [None, 12])
def test_partial_yaml_keeps_lab1_defaults_and_cli_step_precedence(mode, cli_steps, tmp_path, monkeypatch):
    config = tmp_path / "partial.yaml"
    config.write_text("learning_rate: 0.002\n", encoding="utf-8")
    output = tmp_path / "never_started"
    argv = ["pinn_basics.py", "--mode", mode, "--device", "cpu",
            "--config", str(config), "--output-dir", str(output)]
    if cli_steps is not None:
        argv.extend(("--steps", str(cli_steps)))
    monkeypatch.setattr(sys, "argv", argv)
    original_setup = basic.setup
    recorded = {}

    class SetupChecked(Exception):
        pass

    def inspect_setup(*args, **kwargs):
        recorded["config"], _ = original_setup(*args, **kwargs)
        raise SetupChecked

    monkeypatch.setattr(basic, "setup", inspect_setup)
    with pytest.raises(SetupChecked):
        basic.main()
    cfg = recorded["config"]
    assert cfg["learning_rate"] == .002
    assert cfg["steps"] == (1000 if cli_steps is None else cli_steps)
    assert cfg["batch_size"] == (32 if mode == "parameterized" else 256)
    assert (cfg["num_layers"], cfg["layer_size"]) == ((2, 32) if mode == "inverse" else (3, 64))
    assert not output.exists()


@pytest.mark.parametrize("name,value", [("source_rmse", .68), ("source_max_abs", .2),
                                        ("source_rmse", float("nan")),
                                        ("source_max_abs", float("inf"))])
def test_inverse_quality_rejects_bad_source_despite_accurate_solution(name, value):
    metrics = good_errors()
    assert basic.accuracy_checks(metrics, "inverse")["passed"]
    metrics[name] = value
    result = basic.accuracy_checks(metrics, "inverse")
    assert not result["passed"]
    assert any(check["metric"] == name and not check["passed"] for check in result["checks"])


@pytest.mark.parametrize("name,value", [("solution_rmse", .002), ("pde_rmse", .02),
                                        ("boundary_max_abs", .002)])
def test_parameterized_quality_checks_each_length_not_only_pooled_error(name, value):
    records = [{"length": length, **good_errors()} for length in (1., 1.25, 1.5, 1.75, 2.)]
    metrics = {**good_errors(), "per_length": records}
    assert basic.accuracy_checks(metrics, "parameterized")["passed"]
    records[-1][name] = value
    result = basic.accuracy_checks(metrics, "parameterized")
    assert not result["passed"]
    assert any(check["length"] == 2. and check["metric"] == name and not check["passed"]
               for check in result["checks"])


def assert_reloaded_predictions(output):
    checkpoint = torch.load(output / "model.pt", map_location="cpu", weights_only=True)
    cfg = checkpoint["config"]
    dtype = getattr(torch, cfg["lab1_dtype"])
    # Check checkpoint reproduction on the execution device. CPU and CUDA FP32
    # reductions can differ even when both satisfy the lesson accuracy limits.
    device = torch.device(checkpoint["device"])
    model = basic.BasicPINN(cfg, cfg["lab1_mode"], dtype=dtype).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    metrics = json.loads((output / "metrics.json").read_text())
    assert cfg["lab1_mode"] == metrics["mode"]
    assert dtype == torch.float32
    assert cfg["lab1_recipe"] == "lbfgs_fp32_v3"
    assert all(value.dtype == torch.float32 for value in checkpoint["model_state_dict"].values()
               if torch.is_tensor(value) and value.is_floating_point())
    with np.load(output / "predictions.npz", allow_pickle=False) as saved:
        x = torch.as_tensor(saved["x"], dtype=dtype, device=device)
        length = torch.full_like(x, metrics["validation_length"])
        with torch.no_grad():
            prediction = model(x, length).cpu().numpy()
            np.testing.assert_allclose(prediction, saved["prediction"], rtol=1e-5, atol=1e-7)
            if model.mode == "inverse":
                np.testing.assert_allclose(model.source(x).cpu().numpy(), saved["source_prediction"],
                                           rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("mode", MODES)
def test_short_cli_publishes_reloadable_model_and_honest_step_metadata(mode, tmp_path, monkeypatch):
    import yaml

    config = tmp_path / "small.yaml"
    config.write_text(yaml.safe_dump(SMALL_CONFIG), encoding="utf-8")
    output = tmp_path / mode
    monkeypatch.setattr(sys, "argv", ["pinn_basics.py", "--mode", mode, "--device", "cpu",
                                     "--steps", "4", "--seed", "42", "--config", str(config),
                                     "--output-dir", str(output)])
    basic.main()
    result = validate_artifacts(output, steps=4, device="cpu", expected_version="2.2.2", seed=42)
    assert result["passed"]
    metrics = result["metrics"]
    assert not metrics["accuracy"]["passed"]  # Four steps are an execution test only.
    recipe = metrics["training_recipe"]
    assert recipe["optimizer"] == "full-batch L-BFGS"
    assert recipe["learning_rate"] == SMALL_CONFIG["learning_rate"]
    assert recipe["fixed_training_points"] == SMALL_CONFIG["batch_size"] * (17 if mode == "parameterized" else 1)
    assert recipe["adam_step_calls"] == 0 and recipe["lbfgs_step_calls"] == 4
    assert recipe["source_targets_used_for_training"] is False
    with (output / "loss.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [int(row["step"]) for row in rows] == [1, 2, 3, 4]
    assert recipe["closure_evaluations"] == sum(int(row["closure_evaluations"]) for row in rows)
    assert metrics["final_minibatch_loss_before_update"] == float(rows[-1]["loss"])
    assert_reloaded_predictions(output)


@pytest.mark.skipif(os.environ.get("AI4SCI_RUN_CONVERGENCE") != "1",
                    reason="Set AI4SCI_RUN_CONVERGENCE=1 for full, separately timed training")
@pytest.mark.parametrize("seed", [42, 43, 7])
@pytest.mark.parametrize("mode", MODES)
def test_measured_lesson_convergence_at_published_budget(mode, seed, tmp_path):
    device = os.environ.get("AI4SCI_CONVERGENCE_DEVICE", "cpu")
    assert device in ("cpu", "cuda"), "AI4SCI_CONVERGENCE_DEVICE must be cpu or cuda"
    output = tmp_path / f"{mode}-{seed}"
    completed = subprocess.run(
        [sys.executable, str(ROOT / "01_labs/01_pinn/source_code/pinn_basics.py"),
         "--mode", mode, "--device", device, "--steps", "1000", "--seed", str(seed),
         "--output-dir", str(output)],
        cwd=ROOT, capture_output=True, text=True, timeout=1200,
        env=dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", MPLBACKEND="Agg"),
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    result = validate_artifacts(output, steps=1000, device=device,
                                expected_version="2.2.2", seed=seed)
    assert result["passed"]
    metrics = result["metrics"]
    assert metrics["accuracy"]["passed"], json.dumps(metrics["accuracy"], indent=2)
    assert basic.accuracy_checks(metrics["heldout_after"], mode)["passed"]
    assert_reloaded_predictions(output)
