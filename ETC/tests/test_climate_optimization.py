"""Numerical contracts for Climate optimization, without an exercise answer key.

The polynomial fixtures below have different initial and boundary conditions
from the assigned sine-mode exercises. They test physical derivatives and the
optimizer machinery; they do not certify a teaching run's convergence.
"""
from copy import deepcopy
import importlib
import math
from pathlib import Path
import sys

import pytest
import torch
from sympy import Function, symbols

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ETC.runtime import climate as optimization
from ETC.runtime.exercises import conditions
from ETC.runtime.pinn import create_informer, residuals
from physicsnemo.sym.eq.pde import PDE


CONFIG = {
    "training": {"steps": 17, "learning_rate": 0.001},
    "model": {"width": 8, "layers": 2},
    "samples": {"interior": 32, "initial": 17, "boundary": 17},
}
TIME_END = 2 * math.pi


@pytest.fixture(autouse=True)
def bounded_cpu_and_preserved_rng():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    with torch.random.fork_rng(devices=[]):
        yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("initial_count,boundary_count", [(17, 17), (1, 1)])
def test_collocation_covers_physical_endpoints_and_is_rng_independent(initial_count, boundary_count):
    config = deepcopy(CONFIG)
    config["samples"].update(initial=initial_count, boundary=boundary_count)
    before = torch.random.get_rng_state().clone()
    first = optimization.collocation_points(config, "cpu", TIME_END, multiplier=1)
    assert torch.equal(before, torch.random.get_rng_state())
    torch.rand(11)  # An unrelated notebook draw must not change polish points.
    second = optimization.collocation_points(config, "cpu", TIME_END, multiplier=1)
    for name, value in first.items():
        torch.testing.assert_close(value, second[name], rtol=0, atol=0)
        assert value.dtype == torch.float32
        assert torch.isfinite(value).all()

    xy, times = first["interior_xy"], first["interior_t"]
    assert len(xy) == config["samples"]["interior"]
    assert ((xy > 0) & (xy < math.pi)).all()
    assert ((times > 0) & (times < TIME_END)).all()

    initial = first["initial_xy"]
    boundary = torch.cat((first["boundary_xy"], first["boundary_t"]), dim=1)
    assert len(initial) >= config["samples"]["initial"]
    assert len(boundary) >= config["samples"]["boundary"]
    on_edge = torch.isclose(boundary[:, :2], torch.tensor(0.0)) | torch.isclose(
        boundary[:, :2], torch.tensor(math.pi))
    assert on_edge.any(dim=1).all()
    for x in (0.0, math.pi):
        for y in (0.0, math.pi):
            assert torch.isclose(initial, torch.tensor([x, y])).all(dim=1).any()
            for t in (0.0, TIME_END):
                assert torch.isclose(boundary, torch.tensor([x, y, t])).all(dim=1).any()


class PolynomialNetwork(torch.nn.Module):
    """A manufactured heat field in the network's normalized coordinates."""

    def __init__(self, diffusivities):
        super().__init__()
        self.amplitude = torch.nn.Parameter(torch.tensor(1.0))
        self.diffusivities = diffusivities

    def forward(self, normalized):
        # Reconstruct physical coordinates independently of ClimateModel.
        x = (normalized[:, :1] + 1) * (math.pi / 2)
        y = (normalized[:, 1:2] + 1) * (math.pi / 2)
        t = (normalized[:, 2:3] + 1) * (TIME_END / 2)
        return self.amplitude * torch.cat([
            x.square() + y.square() + 4 * kappa * t
            for kappa in self.diffusivities
        ], dim=1)


class ManufacturedHeat(PDE):
    def __init__(self, names, diffusivities):
        self.dim = 2
        x, y, t = symbols("x y t")
        fields = [Function(name)(x, y, t) for name in names]
        self.equations = {
            "manufactured_" + name: field.diff(t) - kappa * (field.diff(x, 2) + field.diff(y, 2))
            for name, field, kappa in zip(names, fields, diffusivities)
        }


@pytest.mark.parametrize("names,diffusivities", [
    (["T"], [1.0]), (["Ta", "To"], [1.0, 0.5]),
])
def test_normalized_model_retains_physical_spatial_and_time_derivatives(names, diffusivities):
    model = optimization.ClimateModel(PolynomialNetwork(diffusivities), TIME_END)
    xy = torch.tensor([[0.2, 0.7], [1.1, 2.3], [2.7, 0.4]], requires_grad=True)
    times = torch.tensor([[0.0], [0.4], [TIME_END]], requires_grad=True)
    inputs = torch.cat((xy, times), dim=1)
    predicted = model(inputs)
    assert predicted.dtype == torch.float32
    assert model.coordinate_extent.dtype == torch.float32
    first = torch.autograd.grad(predicted[:, 0].sum(), inputs, create_graph=True)[0]
    assert first.dtype == torch.float32
    expected = torch.cat((2 * xy, torch.full_like(times, 4 * diffusivities[0])), dim=1)
    torch.testing.assert_close(first, expected, rtol=2e-6, atol=2e-6)
    informer = create_informer(ManufacturedHeat(names, diffusivities), "cpu")
    checked = residuals(model, informer, xy, times, names)
    for value in checked.values():
        torch.testing.assert_close(value, torch.zeros_like(value), rtol=0, atol=1e-5)


@pytest.mark.parametrize("level,names,diffusivities", [
    (1, ["T"], [1.0]), (2, ["Ta", "To"], [1.0, 0.5]),
])
def test_fixed_collocation_loss_uses_no_sampling_rng(level, names, diffusivities, monkeypatch):
    module = importlib.import_module(f"02_challenges.03_climate.climate_l{level}")
    model = optimization.ClimateModel(PolynomialNetwork(diffusivities), TIME_END)
    informer = create_informer(ManufacturedHeat(names, diffusivities), "cpu")
    exercise = conditions(lambda x, y, t: {
        **{"initial_" + name: x**2 + y**2 for name in names},
        **{"boundary_" + name: x**2 + y**2 + 4 * kappa * t
           for name, kappa in zip(names, diffusivities)},
    })
    points = optimization.collocation_points(CONFIG, "cpu", TIME_END)

    def unexpected_sampling(*args, **kwargs):
        raise AssertionError("A fixed-point closure must not redraw coordinates or times")

    monkeypatch.setattr(module, "sample_square", unexpected_sampling)
    monkeypatch.setattr(module, "sample_time", unexpected_sampling)
    before = torch.random.get_rng_state().clone()
    first = module.loss_terms(model, informer, CONFIG, "cpu", exercise, points=points)
    second = module.loss_terms(model, informer, CONFIG, "cpu", exercise, points=points)
    assert torch.equal(before, torch.random.get_rng_state())
    for name in first:
        torch.testing.assert_close(first[name], second[name], rtol=0, atol=0)
        assert first[name].item() < 1e-8


@pytest.mark.parametrize("names,amplitudes", [(["T"], [2.0]), (["Ta", "To"], [2.0, -6.0])])
def test_temperature_errors_show_decay_amplitude_without_rescaling_absolute_error(names, amplitudes):
    offset = 0.125

    class OffsetField(torch.nn.Module):
        def forward(self, inputs):
            return inputs.new_tensor(amplitudes)[None, :] * torch.exp(-inputs[:, 2:3]) + offset

    def comparison(coordinates, times):
        return coordinates.new_tensor(amplitudes)[None, :] * torch.exp(-times)

    coordinates = torch.tensor([[0.0, 0.0], [0.3, 0.8], [math.pi, math.pi]])
    result = optimization.temperature_errors(OffsetField(), coordinates, TIME_END, names, comparison)
    initial_rms = math.sqrt(sum(value**2 for value in amplitudes) / len(amplitudes))
    assert result["rmse"] == pytest.approx(offset, rel=1e-5)
    times = [row["time"] for row in result["per_time"]]
    assert 0.25 in times and 0.5 in times and 1.0 in times
    assert times[0] == 0.0 and times[-1] == TIME_END
    for row in result["per_time"]:
        expected_max = max(abs(value) for value in amplitudes) * math.exp(-row["time"])
        assert row["reference_max"] == pytest.approx(expected_max, rel=1e-6)
        assert row["rmse"] == pytest.approx(offset, rel=1e-5)
        assert row["initial_scale_rmse"] == pytest.approx(offset / initial_rms, rel=1e-5)
    # The new fixed initial scale must not hide the original late-time relative error.
    assert result["per_time"][-1]["relative_l2"] > result["per_time"][0]["relative_l2"] * 100


def test_temperature_errors_do_not_fabricate_a_comparison_when_unavailable():
    class NoForecast(torch.nn.Module):
        def forward(self, inputs):
            raise AssertionError("No expression means there is no reference-error comparison")

    result = optimization.temperature_errors(NoForecast(), torch.ones(3, 2), TIME_END,
                                             ["T"], lambda coordinates, times: None)
    assert result == {}


@pytest.mark.parametrize("total", [1, 2, 9, 10, 200, 3000, 5000])
def test_budget_respects_short_execution_checks_and_total_override(total):
    adam, lbfgs = optimization.optimization_budget(total)
    assert adam + lbfgs == total
    assert adam >= 0 and lbfgs >= 0
    if total < 10:
        assert lbfgs == 0
    else:
        assert adam >= 1 and lbfgs >= 1
        assert adam <= 1500


@pytest.mark.parametrize("total", [0, -1, True, 1.5, "200"])
def test_invalid_budgets_cannot_silently_skip_or_expand_training(total):
    with pytest.raises(ValueError):
        optimization.optimization_budget(total)


def test_two_step_execution_check_never_allocates_polish_points(monkeypatch):
    model = torch.nn.Linear(1, 1)
    calls = []

    def unexpected_collocation(*args, **kwargs):
        raise AssertionError("A two-step check must not allocate L-BFGS points")

    monkeypatch.setattr(optimization, "collocation_points", unexpected_collocation)

    def loss():
        calls.append(1)
        return {"manufactured_fit": model(torch.ones(2, 1)).square().mean()}

    adam, lbfgs = optimization.optimization_budget(2)
    stats = optimization.optimize_climate(model, loss, CONFIG, "cpu", TIME_END,
                                         adam_steps=adam, lbfgs_steps=lbfgs)
    assert len(calls) == 2
    assert stats["lbfgs_closure_evaluations"] == 0
    assert stats["lbfgs_samples"] == {}


@pytest.mark.parametrize("max_iter", [0, -1, True, 1.5, "3"])
def test_invalid_inner_iteration_limit_fails_before_training(max_iter):
    def unexpected_loss():
        raise AssertionError("An invalid iteration limit must be rejected before training")

    with pytest.raises(ValueError, match="lbfgs_max_iter must be a positive integer"):
        optimization.optimize_climate(torch.nn.Linear(1, 1), unexpected_loss, CONFIG,
                                      "cpu", TIME_END, adam_steps=1, lbfgs_steps=1,
                                      lbfgs_max_iter=max_iter)


@pytest.mark.parametrize("max_iter", [2, 3])
def test_tiny_cpu_optimization_reduces_independent_error_and_checkpoint_reloads(tmp_path, monkeypatch, max_iter):
    torch.manual_seed(7)
    network = torch.nn.Linear(3, 1)
    torch.nn.init.zeros_(network.weight)
    torch.nn.init.zeros_(network.bias)
    model = optimization.ClimateModel(network, TIME_END)
    config = deepcopy(CONFIG)
    history = []
    lbfgs_instances = []
    lbfgs_constructor = torch.optim.LBFGS

    def tracked_lbfgs(*args, **kwargs):
        optimizer = lbfgs_constructor(*args, **kwargs)
        lbfgs_instances.append(optimizer)
        return optimizer

    monkeypatch.setattr(torch.optim, "LBFGS", tracked_lbfgs)

    def target(inputs):
        return 2 * inputs[:, :1] - 3 * inputs[:, 1:2] + 0.5 * inputs[:, 2:3]

    def losses(*, points=None):
        if points is None:
            inputs = torch.rand(32, 3) * torch.tensor([math.pi, math.pi, TIME_END])
        else:
            inputs = torch.cat((points["interior_xy"], points["interior_t"]), dim=1)
        return {"manufactured_fit": (model(inputs) - target(inputs)).square().mean()}

    def record(step, phase, values):
        assert all(not value.requires_grad for value in values.values())
        history.append((step, phase, {name: value.detach().item() for name, value in values.items()}))

    # These points include domain extremes and never enter the training closure.
    verification = torch.tensor([[0.0, 0.0, 0.0], [math.pi, math.pi, TIME_END],
                                 [0.2, 1.9, 0.4], [2.3, 0.7, 5.0]])
    before = (model(verification) - target(verification)).square().mean().item()
    stats = optimization.optimize_climate(model, losses, config, "cpu", TIME_END,
                                         adam_steps=5, lbfgs_steps=12, multiplier=1,
                                         lbfgs_max_iter=max_iter, callback=record)
    after = (model(verification) - target(verification)).square().mean().item()
    assert after < before * 1e-4
    assert [row[0] for row in history] == list(range(1, 18))
    assert {row[1] for row in history} == {"adam_before_update", "lbfgs_after_update"}
    assert stats["adam_steps"] == 5 and stats["lbfgs_steps"] == 12
    assert stats["lbfgs_closure_evaluations"] >= 12
    actual_settings = lbfgs_instances[0].param_groups[0]
    assert stats["lbfgs_max_iter"] == actual_settings["max_iter"] == max_iter
    assert stats["lbfgs_history_size"] == actual_settings["history_size"] == 100
    assert stats["lbfgs_objective_scale"] == 1000
    assert all(parameter.dtype == torch.float32 for parameter in model.parameters())
    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())

    checkpoint = tmp_path / "manufactured.pt"
    torch.save({"state_dict": model.state_dict(), "time_end": TIME_END}, checkpoint)
    saved = torch.load(checkpoint, weights_only=True)
    reloaded = optimization.ClimateModel(torch.nn.Linear(3, 1), saved["time_end"])
    reloaded.load_state_dict(saved["state_dict"])
    torch.testing.assert_close(reloaded(verification), model(verification), rtol=0, atol=0)
