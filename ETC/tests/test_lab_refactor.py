"""Lab setup, complete result publishing and parameter-family evaluation."""
from argparse import Namespace
import importlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ETC.runtime import labs

basic = importlib.import_module("01_labs.01_pinn.source_code.pinn_basics")
flow = importlib.import_module("01_labs.04_navier_stokes.source_code.navier_stokes")


def arguments(tmp_path, **overrides):
    return Namespace(**dict({"output_dir": tmp_path / "run", "config": None,
                             "steps": 2, "device": "cpu", "seed": 42}, **overrides))


@pytest.mark.parametrize("key,value", [("steps", True), ("batch_size", 0),
    ("batch_size", 2.5), ("num_layers", -1), ("layer_size", "16"),
    ("learning_rate", 0), ("learning_rate", float("nan")),
    ("learning_rate", float("inf")), ("learning_rate", True)])
def test_invalid_config_fails_before_creating_outputs(tmp_path, key, value):
    import yaml
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({key: value}))
    args = arguments(tmp_path, config=config, steps=None)
    with pytest.raises(ValueError, match=key):
        labs.setup(args)
    assert not args.output_dir.exists()


@pytest.mark.parametrize("contents", ["[]", "[1, 2]", "12", "true"])
def test_yaml_requires_a_mapping(tmp_path, contents):
    config = tmp_path / "config.yaml"
    config.write_text(contents)
    with pytest.raises(ValueError, match="mapping"):
        labs.setup(arguments(tmp_path, config=config))


@pytest.mark.parametrize("seed", [-1, 2**32, True, 1.5])
def test_invalid_seed_has_actionable_error(tmp_path, seed):
    with pytest.raises(ValueError, match="seed"):
        labs.setup(arguments(tmp_path, seed=seed))


def test_setup_does_not_publish_an_empty_run(tmp_path):
    args = arguments(tmp_path)
    cfg, device = labs.setup(args)
    assert cfg["steps"] == 2 and device.type == "cpu"
    assert not args.output_dir.exists()


def test_optional_defaults_do_not_leak_to_other_lab_calls(tmp_path):
    """The common helper receives values, not a Lab-number special case."""
    args = arguments(tmp_path, steps=None)
    previous_dtype = torch.get_default_dtype()
    custom, _ = labs.setup(args, defaults={"steps": 3000, "layer_size": 32, "num_layers": 2})
    ordinary, _ = labs.setup(args)
    assert custom["steps"] == 3000 and custom["layer_size"] == 32
    assert ordinary == {"steps": 2000, "batch_size": 128, "learning_rate": .001,
                        "layer_size": 64, "num_layers": 3}
    custom["layer_size"] = 999
    another, _ = labs.setup(args)
    assert another == ordinary
    assert torch.get_default_dtype() == previous_dtype


def test_optional_defaults_reject_unknown_settings(tmp_path):
    with pytest.raises(ValueError, match="Unknown Lab defaults"):
        labs.setup(arguments(tmp_path), defaults={"lab_number": 1})


def save_inputs(tmp_path):
    args = arguments(tmp_path)
    cfg, _ = labs.setup(args)
    model = torch.nn.Linear(1, 1)
    history = [{"step": 1, "loss": 1.}, {"step": 2, "loss": .5}]
    arrays = {"x": np.array([0., 1.]), "prediction": np.array([1., 2.])}
    metrics = {"heldout_before": {"objective": 1.}, "heldout_after": {"objective": .5}}
    return args, cfg, model, history, arrays, metrics


def test_saving_publishes_complete_data_and_preserves_input_metrics(tmp_path):
    inputs = save_inputs(tmp_path)
    original = json.dumps(inputs[-1], sort_keys=True)
    labs.save_run(*inputs)
    assert {p.name for p in inputs[0].output_dir.iterdir()} == {
        "predictions.npz", "model.pt", "loss.csv", "metrics.json"}
    assert json.dumps(inputs[-1], sort_keys=True) == original
    assert json.loads((inputs[0].output_dir / "metrics.json").read_text())["final_loss"] == .5


def test_one_step_result_satisfies_artifact_validation(tmp_path):
    from ETC.course_materials.run_validation import validate_artifacts
    inputs = list(save_inputs(tmp_path))
    inputs[1]["steps"] = 1
    inputs[3] = inputs[3][:1]
    labs.save_run(*inputs, plot=lambda plt, arrays: plt.subplots()[0])
    result = validate_artifacts(inputs[0].output_dir, steps=1, device="cpu",
                                expected_version="2.2.2", seed=42)
    assert result["passed"]


def test_plot_failure_does_not_leave_a_completed_or_partial_run(tmp_path):
    inputs = save_inputs(tmp_path)
    def failing_plot(plt, arrays):
        raise RuntimeError("plot failed")
    with pytest.raises(RuntimeError, match="plot failed"):
        labs.save_run(*inputs, plot=failing_plot)
    assert not inputs[0].output_dir.exists()
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("bad", [np.array([]), np.array([np.nan])])
def test_invalid_arrays_are_not_saved(tmp_path, bad):
    inputs = list(save_inputs(tmp_path))
    inputs[4] = {"prediction": bad}
    with pytest.raises(FloatingPointError, match="nonempty and finite"):
        labs.save_run(*inputs)
    assert not inputs[0].output_dir.exists()


@pytest.mark.parametrize("connected", [False, True])
def test_optimizer_rejects_a_loss_unrelated_to_the_model(connected):
    model = torch.nn.Linear(1, 1)
    with pytest.raises(ValueError, match="disconnected"):
        labs.optimize(model, lambda: {"physics": torch.tensor(1., requires_grad=connected)},
                      {"steps": 1, "learning_rate": .001})


class ExactFamily(torch.nn.Module):
    mode = "parameterized"
    def __init__(self, wrong_away_from_preview=False):
        super().__init__()
        self.wrong_away_from_preview = wrong_away_from_preview
    def forward(self, x, length):
        value = basic.analytical(x, length)
        return value + (length - 1.5 if self.wrong_away_from_preview else 0)


def test_parameterized_evaluation_covers_endpoints_and_interior_lengths():
    physics = labs.informer(basic.Poisson1D(), "cpu")
    actual = basic.evaluate(ExactFamily(), physics, "cpu")
    assert [row["length"] for row in actual["per_length"]] == [1., 1.25, 1.5, 1.75, 2.]
    assert actual["objective"] < 1e-10 and actual["solution_rmse"] < 1e-7


def test_midpoint_only_correct_model_fails_parameter_family_check():
    physics = labs.informer(basic.Poisson1D(), "cpu")
    actual = basic.evaluate(ExactFamily(True), physics, "cpu")
    assert actual["per_length"][2]["solution_rmse"] < 1e-7
    assert actual["solution_rmse"] == pytest.approx(math.sqrt(.125), abs=1e-6)
    assert actual["objective"] > .1


@pytest.mark.parametrize("params", [{"nu": -1.}, {"nu": float("nan")},
                                    {"nu": .01, "rho": 0.}])
def test_flow_coefficients_must_be_physical(params):
    with pytest.raises(ValueError, match="nu|rho"):
        flow.NavierStokes(**params)


@pytest.mark.parametrize("kwargs", [{"velocity_scale": 0}, {"pressure_scale": -1},
                                    {"velocity_scale": float("inf")}])
def test_data_scale_validation_precedes_file_loading(kwargs):
    with pytest.raises(ValueError, match="scale"):
        flow.read_wf_data(data_path="missing-test-data.npy", **kwargs)


@pytest.mark.parametrize("fields", [torch.empty(0, 3), torch.full((2, 3), float("nan"))])
def test_empty_or_nonfinite_flow_errors_are_not_accuracy_metrics(fields):
    with pytest.raises(ValueError, match="nonempty and finite"):
        flow.solution_errors(fields, fields)
