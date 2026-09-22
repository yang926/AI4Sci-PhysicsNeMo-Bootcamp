"""Explicit PINN contracts and failure-safe local result publication."""
from copy import deepcopy
import importlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ETC.runtime.artifacts import staged_output
from ETC.runtime import pinn


CONFIG = {"training": {"steps": 2, "learning_rate": .001},
          "model": {"width": 8, "layers": 2},
          "samples": {"interior": 8, "boundary": 8, "initial": 8}}


@pytest.mark.parametrize("equations", [None, {}, [], {"": 0}])
def test_incomplete_equation_returns_have_clear_errors(equations):
    with pytest.raises(ValueError, match="Equations must be a nonempty mapping"):
        pinn.create_informer(SimpleNamespace(equations=equations), "cpu")


@pytest.mark.parametrize("level", [1, 2])
def test_climate_rejects_silently_ignored_parameter_typos(level):
    climate = importlib.import_module(f"02_challenges.03_climate.climate_l{level}")
    with pytest.raises(ValueError, match="Unknown physics parameters"):
        climate.ClimatePDE(reference=True, params={"kapppa": .1})
    with pytest.raises(ValueError, match="finite numbers"):
        climate.ClimatePDE(reference=True, params={"u0": "0.0"})


@pytest.mark.parametrize("level", [1, 2, 3])
@pytest.mark.parametrize("speed", [True, "1.0"])
def test_wave_rejects_nonnumeric_speeds(level, speed):
    wave = importlib.import_module(f"02_challenges.01_wave.wave_l{level}")
    with pytest.raises(ValueError, match="wave speed c must be positive"):
        wave.WaveEquation2D(reference=True, c=speed)


@pytest.mark.parametrize("section,key,value", [
    ("training", "steps", True), ("training", "steps", 1.5),
    ("training", "steps", 0), ("training", "learning_rate", "0.001"),
    ("training", "learning_rate", float("nan")),
    ("training", "learning_rate", True), ("model", "width", 0),
    ("samples", "interior", False),
])
def test_invalid_config_has_a_named_error(section, key, value):
    config = deepcopy(CONFIG)
    config[section][key] = value
    with pytest.raises(ValueError, match=section + "." + key):
        pinn.validate_config(config)


def test_missing_config_and_override_are_explicit():
    with pytest.raises(ValueError, match="training must contain"):
        pinn.validate_config({})
    config = deepcopy(CONFIG)
    del config["model"]["width"]
    with pytest.raises(ValueError, match="Missing config value: model.width"):
        pinn.validate_config(config)
    assert pinn.validate_config(CONFIG, steps=3) == 3
    assert pinn.validate_config(CONFIG) == 2


class Field(torch.nn.Module):
    def __init__(self, function):
        super().__init__()
        self.function = function

    def forward(self, inputs):
        return self.function(inputs)


@pytest.mark.parametrize("function", [
    lambda x: x[:, 0], lambda x: x[:, :1], lambda x: x[:, :0],
    lambda x: torch.ones(1, 2), lambda x: {"u": x[:, :1]},
])
def test_field_shape_cannot_broadcast_or_hide_a_missing_field(function):
    with pytest.raises(ValueError, match="Model output must have shape"):
        pinn.evaluate_fields(Field(function), torch.ones(3, 2), None, ["u", "v"])


def test_field_names_and_time_shape_are_checked():
    model = Field(lambda x: x)
    xy = torch.ones(3, 2)
    with pytest.raises(ValueError, match="unique strings"):
        pinn.evaluate_fields(model, xy, None, ["u", "u"])
    with pytest.raises(ValueError, match="time must have shape"):
        pinn.evaluate_fields(model, xy, torch.ones(3), ["u"])


def test_field_validation_preserves_autograd():
    xy = torch.ones(3, 2, requires_grad=True)
    fields = pinn.evaluate_fields(Field(lambda x: x.square()), xy, None, ["u", "v"])
    (fields["u"] + fields["v"]).sum().backward()
    torch.testing.assert_close(xy.grad, torch.full_like(xy, 2))


@pytest.mark.parametrize("losses,error", [
    ({}, ValueError), ({"pde": torch.ones(2)}, ValueError),
    ({"pde": 1.}, ValueError), ({"pde": torch.tensor(float("nan"))}, FloatingPointError),
    ({"pde": torch.tensor(float("inf"))}, FloatingPointError),
])
def test_invalid_losses_never_append_successful_history(losses, error):
    history = []
    with pytest.raises(error):
        pinn.record_step(1, losses, history)
    assert history == []


def test_checked_loss_preserves_gradient():
    value = torch.tensor(2., requires_grad=True)
    total = pinn.record_step(1, {"pde": value.square()}, [])
    total.backward()
    assert value.grad == 4


def test_heldout_preserves_rng_and_rejects_nonfinite_values():
    before = torch.random.get_rng_state().clone()
    def sampled_loss(*args):
        return {"pde": torch.rand(()).square()}
    first = pinn.heldout_losses(sampled_loss, None, None, {}, torch.device("cpu"), 15)
    second = pinn.heldout_losses(sampled_loss, None, None, {}, "cpu", 15)
    torch.testing.assert_close(first["pde"], second["pde"])
    assert torch.equal(before, torch.random.get_rng_state())
    with pytest.raises(FloatingPointError, match="Non-finite loss"):
        pinn.heldout_losses(lambda *args: {"pde": torch.tensor(float("nan"))},
                            None, None, {}, "cpu", 15)
    assert torch.equal(before, torch.random.get_rng_state())


@pytest.mark.parametrize("reference,error", [
    (torch.ones(3), "shapes must match"),
    (torch.full((3, 1), float("nan")), "finite fields"),
])
def test_single_time_reference_is_not_broadcast(reference, error):
    with pytest.raises(ValueError, match=error):
        pinn.reference_errors(Field(lambda x: x[:, :1]), torch.ones(3, 2),
                              None, ["u"], reference)


def test_reference_availability_must_be_consistent_even_when_initially_absent():
    def reference(xy, time):
        return None if time[0, 0] == 0 else xy[:, :1]
    with pytest.raises(ValueError, match="availability must not change"):
        pinn.reference_errors_over_time(Field(lambda x: x[:, :1]), torch.ones(3, 2),
                                        1., ["u"], reference)


def test_staged_output_publishes_only_when_complete(tmp_path):
    destination = tmp_path / "run"
    with staged_output(destination) as staging:
        (staging / "metrics.json").write_text("{}")
        assert not destination.exists()
    assert (destination / "metrics.json").read_text() == "{}"
    assert list(tmp_path.iterdir()) == [destination]


def test_staged_output_cleans_failed_writes(tmp_path):
    destination = tmp_path / "run"
    with pytest.raises(RuntimeError, match="interrupted write"):
        with staged_output(destination) as staging:
            (staging / "partial.txt").write_text("incomplete")
            raise RuntimeError("interrupted write")
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("kind", ["file", "directory", "broken_symlink"])
def test_staged_output_preserves_every_kind_of_occupied_path(tmp_path, kind):
    destination = tmp_path / "run"
    if kind == "file":
        destination.write_text("existing")
    elif kind == "directory":
        destination.mkdir()
    else:
        destination.symlink_to(tmp_path / "missing")
    with pytest.raises(FileExistsError):
        with staged_output(destination):
            pytest.fail("An existing path must not be accepted")
    assert destination.exists() or destination.is_symlink()
    if kind == "file":
        assert destination.read_text() == "existing"


def test_staged_output_does_not_replace_a_late_writer(tmp_path):
    destination = tmp_path / "run"
    with pytest.raises(FileExistsError):
        with staged_output(destination) as staging:
            (staging / "ours").write_text("new")
            destination.mkdir()
            (destination / "theirs").write_text("existing")
    assert [path.name for path in destination.iterdir()] == ["theirs"]
    assert list(tmp_path.iterdir()) == [destination]


def test_staged_output_rolls_back_a_failed_publish(tmp_path, monkeypatch):
    destination = tmp_path / "run"
    rename = Path.rename
    calls = 0
    def failing_second_move(source, target):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("publication failed")
        return rename(source, target)
    monkeypatch.setattr(Path, "rename", failing_second_move)
    with pytest.raises(OSError, match="publication failed"):
        with staged_output(destination) as staging:
            (staging / "a").write_text("first")
            (staging / "b").write_text("second")
    assert list(tmp_path.iterdir()) == []


def result_args(destination):
    return SimpleNamespace(output_dir=destination, seed=42, steps=1, device="cpu", reference=True)


def test_nonfinite_prediction_does_not_leave_a_result_directory(tmp_path):
    model = Field(lambda x: torch.full_like(x[:, :1], float("nan")))
    with pytest.raises(FloatingPointError, match="Non-finite final predictions"):
        pinn.save_results(result_args(tmp_path / "run"), CONFIG, model,
                          [{"step": 1, "phase": "training", "total": 1.}],
                          torch.ones(3, 2), None, ["u"], {})
    assert list(tmp_path.iterdir()) == []
    assert model.training


def test_successful_save_preserves_mode_and_metrics_input(tmp_path):
    model = torch.nn.Linear(2, 1)
    metrics = {"example_rmse": .25}
    destination = tmp_path / "run"
    pinn.save_results(result_args(destination), CONFIG, model,
                      [{"step": 1, "phase": "training", "total": 1.}],
                      torch.ones(3, 2), None, ["u"], metrics)
    assert model.training
    assert metrics == {"example_rmse": .25}
    saved = json.loads((destination / "metrics.json").read_text())
    assert saved["assessment"]["official_score"] is None
    assert "heldout_before" not in saved and "heldout_after" not in saved
    assert {"model.pt", "predictions.npz", "metrics.json", "loss.csv"} <= {p.name for p in destination.iterdir()}


def test_saved_heldout_comparison_uses_canonical_rows_without_mutating_inputs(tmp_path):
    history = [
        {"step": 0, "phase": "heldout_before_training", "total": 3., "pde": 2., "boundary": 1.},
        {"step": 1, "phase": "training_minibatch_before_update", "total": 8., "pde": 6., "boundary": 2.},
        {"step": 1, "phase": "heldout_after_training", "total": .5, "pde": .2, "boundary": .3},
    ]
    metrics = {"evaluation_equations": "provided_reference_equations"}
    original_history, original_metrics = deepcopy(history), deepcopy(metrics)
    destination = tmp_path / "run"
    pinn.save_results(result_args(destination), CONFIG, torch.nn.Linear(2, 1),
                      history, torch.ones(3, 2), None, ["u"], metrics)
    saved = json.loads((destination / "metrics.json").read_text())
    assert saved["heldout_before"] == {
        "objective": 3., "loss_components": {"pde": 2., "boundary": 1.}}
    assert saved["heldout_after"] == {
        "objective": .5, "loss_components": {"pde": .2, "boundary": .3}}
    assert "not unweighted RMSE" in saved["metric_definition"]
    assert "canonical equations" in saved["metric_definition"]
    assert history == original_history and metrics == original_metrics


def test_reference_preview_includes_every_coupled_field(tmp_path, monkeypatch):
    import matplotlib.pyplot as plt
    subplots = plt.subplots
    captured = []
    def capture(*args, **kwargs):
        result = subplots(*args, **kwargs)
        captured.append(result[1])
        return result
    monkeypatch.setattr(plt, "subplots", capture)
    xy = torch.tensor([[.2, .3], [.5, .7], [.8, .4]])
    model = Field(lambda x: x)
    pinn.save_results(result_args(tmp_path / "run"), CONFIG, model,
                      [{"step": 1, "phase": "training", "total": 1.}],
                      xy, None, ["Ta", "To"], {}, reference=xy)
    assert captured[0].shape == (2, 3)
    assert [ax.get_title() for ax in captured[0][1]] == [
        "Reference To", "Prediction To", "Absolute error To"]
