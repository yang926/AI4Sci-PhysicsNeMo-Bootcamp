"""Absolute physical errors, family coverage, and FP32 artifact provenance."""
from argparse import Namespace
import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from ETC.course_materials import run_validation as runner
from ETC.runtime import labs


def good_metrics(case):
    errors = {"solution_rmse": 1e-5, "pde_rmse": 0., "boundary_max_abs": 0.}
    metrics = {"heldout_before": {"solution_rmse": 1.}, "heldout_after": errors,
               "accuracy": {"passed": True, "checks": []}}
    if case == "pinn_inverse":
        errors.update(source_rmse=.001, source_max_abs=.002)
    elif case == "pinn_parameterized":
        errors["per_length"] = [{"length": length, **errors}
                                for length in (1., 1.25, 1.5, 1.75, 2.)]
    elif case == "projectile":
        errors.update(solution_max_abs=.001, initial_position_max_abs=.001,
                      initial_velocity_max_abs=.001)
    elif case in ("diffusion", "diffusion_parameterized"):
        parameters = [5. + .5 * i for i in range(41)] if case.endswith("parameterized") else [10.]
        errors["per_conductivity"] = [{"D1": value, "left_relative_rmse": .001,
            "right_relative_rmse": .001, "profile_max_abs": .01, "boundary_max_abs": .01,
            "temperature_jump_abs": .001, "flux_jump_relative": .001,
            "left_normalized_pde_rmse": 0., "right_normalized_pde_rmse": 0.}
            for value in parameters]
    elif case == "navier_stokes":
        metrics["data_kind"] = "synthetic_taylor_green"
        metrics["heldout_before"] = {"synthetic_velocity_rmse": 1.}
        # Deliberately no generic solution_rmse: flow uses separate velocity and pressure errors.
        errors.pop("solution_rmse")
        errors.update(synthetic_velocity_rmse=.001, initial_data_rmse=.001)
        errors["per_time"] = [{"time": time, "synthetic_velocity_rmse": .001,
            "synthetic_pressure_gauge_aligned_rmse": .001, "pde_rmse": 0.,
            "periodic_value_max_abs": 0., "periodic_gradient_max_abs": 0.}
            for time in (0., .13, .37, .61, .83, 1.)]
    return metrics


def corrupt_error(metrics, case, value):
    errors = metrics["heldout_after"]
    if case == "pinn_inverse":
        errors["source_rmse"] = value
    elif case == "pinn_parameterized":
        errors["per_length"][-1]["boundary_max_abs"] = value
    elif case == "pinn_forward":
        errors["boundary_max_abs"] = value
    elif case == "projectile":
        errors["initial_velocity_max_abs"] = value
    elif case in ("diffusion", "diffusion_parameterized"):
        errors["per_conductivity"][-1]["left_relative_rmse"] = value
    else:
        errors["per_time"][-1]["synthetic_velocity_rmse"] = value


@pytest.mark.parametrize("case", runner.LAB_CASES)
def test_all_seven_lab_modes_recompute_accuracy_from_measured_errors(case):
    assert case in runner.CONVERGENCE_CASES
    metrics = good_metrics(case)
    metrics["accuracy"]["passed"] = False  # A saved claim must not decide the result either way.
    original = copy.deepcopy(metrics)
    result = runner.improvement(metrics, case, .99)
    assert result["passed"] and result["accuracy"]["passed"]
    assert metrics == original
    if case == "navier_stokes":
        assert result["metric"] == "heldout_synthetic_velocity_rmse"
        assert "synthetic" in result["scope"]


@pytest.mark.parametrize("case", runner.LAB_CASES)
@pytest.mark.parametrize("bad", [True, -.1, float("nan"), float("inf"), 1.])
def test_small_pde_and_pooled_error_cannot_hide_bad_physics_or_forged_pass(case, bad):
    metrics = good_metrics(case)
    corrupt_error(metrics, case, bad)
    result = runner.improvement(metrics, case, .99)
    assert result["actual_after_before_ratio"] < .99
    assert metrics["accuracy"]["passed"]
    assert not result["accuracy"]["passed"] and not result["passed"]


@pytest.mark.parametrize("metric", ["initial_position_max_abs", "initial_velocity_max_abs"])
def test_projectile_requires_both_initial_conditions_despite_zero_acceleration_residual(metric):
    metrics = good_metrics("projectile")
    metrics["heldout_after"][metric] = .2
    assert not runner.improvement(metrics, "projectile", .99)["passed"]


@pytest.mark.parametrize("case", ["diffusion", "diffusion_parameterized"])
@pytest.mark.parametrize("corruption", ["missing", "empty", "duplicate"])
def test_diffusion_requires_complete_material_parameter_coverage(case, corruption):
    metrics = good_metrics(case)
    errors = metrics["heldout_after"]
    if corruption == "missing":
        errors.pop("per_conductivity")
    elif corruption == "empty":
        errors["per_conductivity"] = []
    else:
        errors["per_conductivity"].append(copy.deepcopy(errors["per_conductivity"][0]))
    assert not runner.improvement(metrics, case, .99)["passed"]


@pytest.mark.parametrize("records", [None, [], [{"time": 0.}]])
def test_flow_requires_all_six_times(records):
    metrics = good_metrics("navier_stokes")
    metrics["heldout_after"]["per_time"] = records
    with pytest.raises(ValueError, match="all six evaluation times"):
        runner.improvement(metrics, "navier_stokes", .99)


def test_flow_requires_initial_data_and_gauge_aligned_pressure():
    metrics = good_metrics("navier_stokes")
    metrics["heldout_after"]["initial_data_rmse"] = .1
    assert not runner.improvement(metrics, "navier_stokes", .99)["passed"]
    metrics = good_metrics("navier_stokes")
    metrics["heldout_after"]["per_time"][2]["synthetic_pressure_gauge_aligned_rmse"] = .1
    assert not runner.improvement(metrics, "navier_stokes", .99)["passed"]


@pytest.mark.parametrize("kind", [None, "upstream_data_lat_legacy_normalization"])
def test_flow_accuracy_is_only_for_the_synthetic_fixture(kind):
    metrics = good_metrics("navier_stokes")
    metrics["data_kind"] = kind
    with pytest.raises(ValueError, match="synthetic_taylor_green"):
        runner.improvement(metrics, "navier_stokes", .99)


@pytest.mark.parametrize("case", runner.LAB_CASES)
@pytest.mark.parametrize("budget", [None, 17])
def test_cli_preserves_default_and_explicit_budget_for_every_lab(case, budget, tmp_path, monkeypatch):
    manifest = json.loads(runner.MANIFEST.read_text())
    commands = []
    metrics = good_metrics(case)
    corrupt_error(metrics, case, 1.)
    monkeypatch.setattr(runner, "versions", lambda: {"packages": {"nvidia-physicsnemo": manifest["physicsnemo"]}})
    monkeypatch.setattr(runner, "source_snapshot", lambda output: {})

    def execute(command, output, label, timeout):
        commands.append(command)
        return {"label": label, "passed": True, "exit_code": 0}

    monkeypatch.setattr(runner, "execute", execute)
    monkeypatch.setattr(runner, "validate_artifacts", lambda *args, **kwargs:
                        {"passed": True, "metrics": metrics})
    output = tmp_path / case
    argv = ["--suite", "convergence", "--case", case, "--output-dir", str(output)]
    if budget is not None:
        argv.extend(("--convergence-steps", str(budget)))
    assert runner.main(argv) == 1  # Execution succeeded; physical accuracy did not.
    assert len(commands) == 1
    assert commands[0][commands[0].index("--steps") + 1] == str(500 if budget is None else budget)
    report = json.loads((output / "report.json").read_text())
    assert not report["passed"] and not report["runs"][0]["improvement"]["accuracy"]["passed"]


def make_artifacts(tmp_path, dtype_claim="float32"):
    output = tmp_path / "artifacts"
    args = Namespace(output_dir=output, seed=42)
    cfg = {"steps": 2, "test_dtype": "float32"}
    metrics = {"heldout_before": {"objective": 1.}, "heldout_after": {"objective": .5}}
    if dtype_claim is not None:
        metrics["training_recipe"] = {"dtype": dtype_claim}
    labs.save_run(args, cfg, torch.nn.Linear(1, 1).float(),
                  [{"step": 1, "loss": 1.}, {"step": 2, "loss": .5}],
                  {"prediction": np.zeros((2, 1), dtype=np.float32)}, metrics,
                  lambda plt, arrays: plt.subplots()[0])
    return output


def validate(path):
    return runner.validate_artifacts(path, steps=2, device="cpu", expected_version="2.2.2", seed=42)


@pytest.mark.parametrize("corruption", ["weights", "predictions", "dtype_metadata"])
def test_declared_fp32_is_checked_against_actual_artifacts(tmp_path, corruption):
    output = make_artifacts(tmp_path)
    assert validate(output)["passed"]
    if corruption == "predictions":
        np.savez_compressed(output / "predictions.npz", prediction=np.zeros((2, 1), dtype=np.float64))
    else:
        checkpoint = torch.load(output / "model.pt", map_location="cpu", weights_only=True)
        if corruption == "weights":
            checkpoint["model_state_dict"] = {key: value.double()
                                               for key, value in checkpoint["model_state_dict"].items()}
        else:
            checkpoint["config"]["test_dtype"] = "float64"
        torch.save(checkpoint, output / "model.pt")
    with pytest.raises(ValueError, match="FP32"):
        validate(output)


def test_artifacts_without_dtype_claim_keep_existing_challenge_contract(tmp_path):
    output = make_artifacts(tmp_path, dtype_claim=None)
    np.savez_compressed(output / "predictions.npz", prediction=np.zeros((2, 1), dtype=np.float64))
    assert validate(output)["passed"]


def call_history(kind):
    rows = [{"step": str(step), "loss": ".01"} for step in (1, 2, 3)]
    if kind == "adam":
        return rows, {"name": "adam_cosine_fp32_v1", "optimizer_step_calls": 3}
    for row, phase, closures in zip(rows, ("adam", "lbfgs", "lbfgs"), (1, 3, 2)):
        row.update(phase=phase, closure_evaluations=str(closures))
    if kind == "split":
        recipe = {"optimizer": "Adam then full-batch L-BFGS",
                  "adam_step_calls": 1, "lbfgs_step_calls": 2, "closure_evaluations": 6}
    else:
        recipe = {"optimizer": "adam_then_lbfgs", "step_unit": "optimizer_calls",
                  "adam_steps": 1, "closure_evaluations": 6}
    return rows, recipe


@pytest.mark.parametrize("kind", ["adam", "split", "bar"])
def test_per_call_recipe_accepts_exact_step_and_closure_counts(kind):
    rows, recipe = call_history(kind)
    runner.validate_optimizer_history(rows, recipe, 3)


@pytest.mark.parametrize("kind", ["adam", "split", "bar"])
@pytest.mark.parametrize("bad", ["missing", "duplicate", "fractional", "reordered"])
def test_per_call_recipe_rejects_incomplete_or_inexact_step_sequence(kind, bad):
    rows, recipe = call_history(kind)
    if bad == "missing":
        rows.pop(1)
    elif bad == "duplicate":
        rows[1]["step"] = "1"
    elif bad == "fractional":
        rows[-1]["step"] = "3.5"  # Previously truncated to 3 by int(float(...)).
    else:
        rows[0], rows[1] = rows[1], rows[0]
    with pytest.raises(ValueError, match="step"):
        runner.validate_optimizer_history(rows, recipe, 3)


@pytest.mark.parametrize("kind", ["split", "bar"])
@pytest.mark.parametrize("bad", ["phase", "sum", "fractional", "zero", "adam_twice", "missing"])
def test_lbfgs_recipe_requires_actual_phases_and_closure_sum(kind, bad):
    rows, recipe = call_history(kind)
    if bad == "phase":
        rows[0]["phase"] = "lbfgs"
    elif bad == "sum":
        recipe["closure_evaluations"] = 7
    elif bad == "fractional":
        rows[-1]["closure_evaluations"] = "2.5"
    elif bad == "zero":
        rows[-1]["closure_evaluations"] = "0"
    elif bad == "adam_twice":
        rows[0]["closure_evaluations"] = "2"
        recipe["closure_evaluations"] = 7
    else:
        rows[-1].pop("closure_evaluations")
    with pytest.raises(ValueError, match="phase|closure"):
        runner.validate_optimizer_history(rows, recipe, 3)


@pytest.mark.parametrize("kind,key", [("adam", "optimizer_step_calls"),
                                     ("split", "adam_step_calls"),
                                     ("split", "lbfgs_step_calls"),
                                     ("bar", "adam_steps")])
@pytest.mark.parametrize("bad", [None, True, -1, 4, 1.5])
def test_recipe_call_counts_must_be_valid_and_match_history(kind, key, bad):
    rows, recipe = call_history(kind)
    recipe[key] = bad
    with pytest.raises(ValueError, match="call|step"):
        runner.validate_optimizer_history(rows, recipe, 3)


def test_legacy_sparse_history_does_not_acquire_a_per_call_contract():
    rows = [{"step": "1", "loss": "1"}, {"step": "100", "loss": ".1"}]
    runner.validate_optimizer_history(rows, {}, 100)
    runner.validate_optimizer_history(rows, {"dtype": "float32"}, 100)


def test_artifact_validation_enforces_new_recipe_call_counts(tmp_path):
    output = make_artifacts(tmp_path)
    metrics_path = output / "metrics.json"
    metrics = json.loads(metrics_path.read_text())
    metrics["training_recipe"].update(name="adam_cosine_fp32_v1", optimizer_step_calls=2)
    metrics_path.write_text(json.dumps(metrics))
    assert validate(output)["passed"]
    metrics["training_recipe"]["optimizer_step_calls"] = 3
    metrics_path.write_text(json.dumps(metrics))
    with pytest.raises(ValueError, match="optimizer_step_calls"):
        validate(output)
