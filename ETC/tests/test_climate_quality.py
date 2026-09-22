"""CPU checks for the coupled teaching problem and its independent reference.

These tests check equations and evaluation behavior, not training convergence.
"""
import importlib
import math
from pathlib import Path
import sys
from unittest import mock

import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from ETC.runtime.pinn import create_evaluation_informer, create_informer, residuals

climate = importlib.import_module("02_challenges.03_climate.climate_l2")


class AnalyticField(torch.nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params

    def forward(self, inputs):
        return climate.exact_reference(inputs[:, :2], inputs[:, 2:3], self.params)


@pytest.fixture(autouse=True)
def bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def test_default_coupling_is_active_and_matches_the_notebook_config():
    config_path = ROOT / "02_challenges/03_climate/conf/config_coupled.yaml"
    config = yaml.safe_load(config_path.read_text())
    assert config["physics"] == climate.DEFAULT_PHYSICS
    assert climate.DEFAULT_PHYSICS["gamma0"] == .5


@pytest.mark.parametrize("kappa_a,kappa_o,gamma", [
    (1., .5, 0.), (1., .5, .5), (1., .5, 3.),
    (.7, .7, 0.), (.7, .7, .5), (.2, 2., .1),
])
def test_coupled_solution_satisfies_both_pdes_initial_and_boundary(kappa_a, kappa_o, gamma):
    params = {**climate.DEFAULT_PHYSICS, "kappa_a": kappa_a,
              "kappa_o": kappa_o, "gamma0": gamma}
    xy = torch.tensor([[.3, .8], [1.2, 2.1], [2.4, .7]], dtype=torch.float64)
    time = torch.tensor([[.05], [.7], [2.1]], dtype=torch.float64)
    model = AnalyticField(params)
    informer = create_evaluation_informer(climate.ClimatePDE, "cpu", params=params)
    result = residuals(model, informer, xy, time, climate.FIELD_NAMES)
    for value in result.values():
        torch.testing.assert_close(value, torch.zeros_like(value), atol=1e-7, rtol=0)

    initial = climate.exact_reference(xy, torch.zeros_like(time), params)
    expected = (torch.sin(xy[:, :1]) * torch.sin(xy[:, 1:2])).expand(-1, 2)
    torch.testing.assert_close(initial, expected, atol=1e-14, rtol=1e-14)
    for axis in (0, 1):
        for edge in (0., math.pi):
            boundary = xy.clone()
            boundary[:, axis] = edge
            value = climate.exact_reference(boundary, time, params)
            torch.testing.assert_close(value, torch.zeros_like(value), atol=1e-14, rtol=0)


def test_eigendecomposition_agrees_with_independent_matrix_exponential():
    params = climate.DEFAULT_PHYSICS.copy()
    xy = torch.tensor([[.3, .8], [1.2, 2.1], [2.4, .7]], dtype=torch.float64)
    time = torch.tensor([[0.], [.7], [climate.TIME_END]], dtype=torch.float64)
    generator = torch.tensor([[-2.5, .5], [.5, -1.5]], dtype=torch.float64)
    amplitudes = (torch.matrix_exp(time[..., None] * generator)
                  @ torch.ones(2, 1, dtype=torch.float64)).squeeze(-1)
    expected = torch.sin(xy[:, :1]) * torch.sin(xy[:, 1:2]) * amplitudes
    torch.testing.assert_close(climate.exact_reference(xy, time, params), expected,
                               atol=1e-14, rtol=1e-12)


def test_zero_coupling_limit_recovers_independent_decay_without_singularity():
    xy = torch.tensor([[.3, .8], [1.2, 2.1]], dtype=torch.float64)
    time = torch.tensor([[.2], [1.3]], dtype=torch.float64)
    params = {**climate.DEFAULT_PHYSICS, "gamma0": 0.}
    initial = torch.sin(xy[:, :1]) * torch.sin(xy[:, 1:2])
    expected = torch.cat((initial * torch.exp(-2 * time),
                          initial * torch.exp(-time)), dim=1)
    torch.testing.assert_close(climate.exact_reference(xy, time, params), expected)
    near_zero = climate.exact_reference(xy, time, {**params, "gamma0": 1e-12})
    torch.testing.assert_close(near_zero, expected, atol=1e-12, rtol=1e-12)


def test_exchange_moves_both_fields_toward_each_other():
    xy = torch.full((1, 2), math.pi / 2, dtype=torch.float64)
    time = torch.tensor([[.6]], dtype=torch.float64)
    coupled = climate.exact_reference(xy, time, climate.DEFAULT_PHYSICS)
    uncoupled = climate.exact_reference(
        xy, time, {**climate.DEFAULT_PHYSICS, "gamma0": 0.})
    # Atmosphere diffuses faster; the warmer ocean supplies heat to it.
    assert uncoupled[0, 0] < coupled[0, 0] < coupled[0, 1] < uncoupled[0, 1]


@pytest.mark.parametrize("name", ["u0", "v0", "lam_a", "Q_a0", "Q_o0"])
def test_reference_is_disabled_when_its_assumptions_do_not_hold(name):
    params = {**climate.DEFAULT_PHYSICS, name: .1}
    assert climate.exact_reference(torch.ones(2, 2), torch.ones(2, 1), params) is None


@pytest.mark.parametrize("params,error", [
    ({"gamma0": -.1}, "nonnegative"),
    ({"gamma0": float("nan")}, "finite"),
    ({"kappa_a": 0.}, "positive"),
    ({"kappa_o": -1.}, "positive"),
])
def test_nonphysical_exchange_and_diffusion_parameters_are_rejected(params, error):
    with pytest.raises(ValueError, match=error):
        climate.ClimatePDE(reference=True, params=params)


def test_evaluation_detects_omitted_exchange_even_if_student_training_residual_is_zero():
    params = climate.DEFAULT_PHYSICS.copy()
    uncoupled = {**params, "gamma0": 0.}
    xy = torch.tensor([[.7, 1.1], [1.2, 1.4]], dtype=torch.float64)
    time = torch.tensor([[.4], [.8]], dtype=torch.float64)
    model = AnalyticField(uncoupled)

    def wrong_student_equations(x, y, t, fields, parameters):
        return climate.reference_equations(x, y, t, fields,
                                           {**parameters, "gamma0": 0.})

    with mock.patch.object(climate, "student_equations", wrong_student_equations):
        training_informer = create_informer(climate.ClimatePDE(params=params), "cpu")
        evaluation_informer = create_evaluation_informer(climate.ClimatePDE, "cpu", params=params)
    train = residuals(model, training_informer, xy, time, climate.FIELD_NAMES)
    evaluation = residuals(model, evaluation_informer, xy, time, climate.FIELD_NAMES)
    assert all(float(value.detach().abs().max()) < 1e-12 for value in train.values())
    assert all(float(value.detach().abs().max()) > .01 for value in evaluation.values())
    # The exchange mismatch contributes equal and opposite residuals.
    torch.testing.assert_close(evaluation["atmosphere"], -evaluation["ocean"])
