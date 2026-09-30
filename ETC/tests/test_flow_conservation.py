"""Algebra checks of optional periodic integral penalties; no forecast claims."""

import math

import pytest
import torch

from ETC.runtime.flow_conservation import conservation_terms, prepare_conservation


@pytest.fixture(autouse=True)
def bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(min(2, previous))
    yield
    torch.set_num_threads(previous)


def prepared(**kwargs):
    options = dict(device="cpu", dtype=torch.float64, lower=-math.pi, length=2 * math.pi)
    options.update(kwargs)
    return prepare_conservation(torch.tensor([[3., 4., 700.], [3., 4., -90.]]), **options)


class PeriodicMode(torch.nn.Module):
    def __init__(self, rate, pressure=0.):
        super().__init__()
        self.rate = rate
        self.pressure = pressure

    def forward(self, xy, time):
        x, y = xy[:, :1], xy[:, 1:]
        amplitude = torch.exp(-self.rate * time)
        u = x.sin() * y.cos() * amplitude
        v = -x.cos() * y.sin() * amplitude
        p = self.pressure * (x**2 + y * time)
        return torch.cat((u, v, p), dim=1)


class Constant(torch.nn.Module):
    def forward(self, xy, time):
        return torch.tensor([3., -4., 100.], device=xy.device, dtype=xy.dtype).expand(len(xy), 3)


class UniformTime(torch.nn.Module):
    def __init__(self, rate=1.):
        super().__init__()
        self.rate = rate

    def forward(self, xy, time):
        u = torch.exp(-self.rate * time)
        return torch.cat((u, 2 * u, 0 * u), dim=1)


def test_preparation_matches_experiment_and_does_not_modify_training_fields():
    fields = torch.tensor([[3., 4., 90.], [1., -2., 10.]], requires_grad=True)
    original = fields.detach().clone()
    result = prepare_conservation(fields, "cpu", lower=-.72, length=1.44)
    axis = -.72 + 1.44 * (torch.arange(32) + .5) / 32
    xx, yy = torch.meshgrid(axis, axis, indexing="xy")
    torch.testing.assert_close(result.xy, torch.stack((xx.flatten(), yy.flatten()), dim=1), rtol=0, atol=0)
    torch.testing.assert_close(result.energy_scale, .5 * fields.detach()[:, :2].square().sum(-1).mean())
    assert result.xy.shape == (1024, 2)
    assert result.xy.dtype == result.energy_scale.dtype == torch.float32
    assert not result.xy.requires_grad and not result.energy_scale.requires_grad
    assert (result.xy > -.72).all() and (result.xy < .72).all()
    torch.testing.assert_close(fields, original, rtol=0, atol=0)
    assert fields.grad is None


@pytest.mark.parametrize("time", [0., .37, 1.])
def test_periodic_divergence_free_mode_cancels_energy_rate_and_dissipation(time):
    nu = .17
    terms = conservation_terms(PeriodicMode(rate=2 * nu), prepared(), nu, time=time)
    assert terms["energy_balance"].item() < 1e-28
    assert terms["mean_momentum"].item() < 1e-28


def test_constant_flow_has_zero_terms_even_without_coordinate_dependencies():
    terms = conservation_terms(Constant(), prepared(), nu=.01, time=.4)
    assert all(value.item() == 0 for value in terms.values())
    sum(terms.values()).backward()


def test_unbalanced_periodic_decay_has_the_analytic_nonzero_penalty():
    nu, rate, time = .17, 1.3, .4
    state = prepared()
    terms = conservation_terms(PeriodicMode(rate), state, nu, time=time)
    # <E> = .25 exp(-2 r t), and nu<|grad u|²+|grad v|²> = nu exp(-2 r t).
    expected = ((nu - .5 * rate) * math.exp(-2 * rate * time) / state.energy_scale.item()) ** 2
    assert terms["energy_balance"].item() == pytest.approx(expected, rel=1e-12)
    assert terms["mean_momentum"].item() < 1e-28


@pytest.mark.parametrize("grid_size", [8, 32])
def test_uniform_temporal_decay_flags_both_identities_without_grid_size_factor(grid_size):
    rate, time = .6, .3
    state = prepared(grid_size=grid_size)
    terms = conservation_terms(UniformTime(rate), state, nu=.01, time=time)
    expected_energy_rate = -5 * rate * math.exp(-2 * rate * time)
    expected_mean_rate_squared = 5 * rate**2 * math.exp(-2 * rate * time)
    assert terms["energy_balance"].item() == pytest.approx((expected_energy_rate / 12.5) ** 2, rel=1e-12)
    assert terms["mean_momentum"].item() == pytest.approx(expected_mean_rate_squared / 25, rel=1e-12)


def test_penalties_retain_finite_nonzero_parameter_gradients():
    class LearnedRate(UniformTime):
        def __init__(self):
            super().__init__()
            self.rate = torch.nn.Parameter(torch.tensor(.6))

    model = LearnedRate()
    terms = conservation_terms(model, prepared(dtype=torch.float32), nu=.01, time=.3)
    sum(terms.values()).backward()
    assert model.rate.grad is not None
    assert torch.isfinite(model.rate.grad)
    assert model.rate.grad.abs() > 0


def test_uniform_acceleration_has_the_analytic_parameter_gradient():
    class LinearVelocity(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.amplitude = torch.nn.Parameter(torch.tensor(1., dtype=torch.float64))

        def forward(self, xy, time):
            return self.amplitude * torch.cat((2 * time, -time, 0 * time), dim=1)

    state = prepare_conservation([[2., 0., 0.]], "cpu", dtype=torch.float64,
                                 lower=-.72, length=1.44)
    model = LinearVelocity()
    terms = conservation_terms(model, state, nu=.01, time=.4)
    assert terms["energy_balance"].item() == pytest.approx(1., rel=1e-12)
    assert terms["mean_momentum"].item() == pytest.approx(1.25, rel=1e-12)
    sum(terms.values()).backward()
    assert model.amplitude.grad.item() == pytest.approx(6.5, rel=1e-12)


def test_pressure_is_irrelevant_to_the_velocity_integral_penalties():
    state = prepared()
    ordinary = conservation_terms(PeriodicMode(.6, pressure=0), state, nu=.01, time=.3)
    large = conservation_terms(PeriodicMode(.6, pressure=1e12), state, nu=.01, time=.3)
    for key in ordinary:
        torch.testing.assert_close(ordinary[key], large[key], rtol=0, atol=0)
    changed = prepare_conservation([[3., 4., -1e9]], "cpu", dtype=torch.float64,
                                   lower=-math.pi, length=2 * math.pi)
    torch.testing.assert_close(state.energy_scale, changed.energy_scale, rtol=0, atol=0)


def test_initial_data_energy_is_only_a_scale_and_does_not_impose_a_target():
    # A spatially and temporally constant model incurs no penalty even when its
    # energy differs from the supplied training energy.
    state = prepare_conservation([[30., 40., 0.]], "cpu", lower=-.72, length=1.44)
    terms = conservation_terms(Constant(), state, nu=.01, time=.7)
    assert state.energy_scale.item() == 1250
    assert all(value.item() == 0 for value in terms.values())


def test_omitted_time_uses_one_random_shared_time_with_independent_rows(monkeypatch):
    class Recorded(UniformTime):
        def forward(self, xy, time):
            self.time = time
            return super().forward(xy, time)

    calls = []

    def fixed_rand(*shape, **kwargs):
        calls.append(shape)
        return torch.full(shape, .3, **kwargs)

    monkeypatch.setattr(torch, "rand", fixed_rand)
    model = Recorded(.6)
    state = prepared()
    actual = conservation_terms(model, state, nu=.01)
    expected = conservation_terms(UniformTime(.6), state, nu=.01, time=.3)
    assert calls == [(1, 1)]
    assert model.time.shape == (1024, 1)
    assert model.time.is_leaf and model.time.requires_grad
    assert model.time.stride(0) != 0
    assert torch.unique(model.time).numel() == 1
    for key in actual:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)


@pytest.mark.parametrize("fields", [[], [[1., 2.]], [[0., 0., 1.]], [[math.nan, 1., 2.]],
                                    [[1., 2., math.inf]], [[1e30, 2., 0.]],
                                    [[True, False, True]], [[1j, 1., 1.]]])
def test_invalid_initial_fields_or_energy_scale_are_rejected(fields):
    with pytest.raises(ValueError):
        prepare_conservation(fields, "cpu", lower=-.72, length=1.44)


@pytest.mark.parametrize("change", [{"lower": math.nan}, {"length": 0.}, {"length": -1.},
                                    {"length": math.inf}, {"grid_size": 1}, {"grid_size": 3.5},
                                    {"grid_size": True}, {"dtype": torch.int64}])
def test_invalid_grid_settings_are_rejected(change):
    with pytest.raises(ValueError):
        prepared(**change)


@pytest.mark.parametrize("nu", [-.01, math.nan, math.inf, True])
def test_invalid_viscosity_is_rejected(nu):
    with pytest.raises(ValueError, match="nu"):
        conservation_terms(Constant(), prepared(), nu, time=.3)


@pytest.mark.parametrize("time", [-.1, 1.1, math.nan, math.inf, True, 1j, [0., .3]])
def test_invalid_shared_time_is_rejected(time):
    with pytest.raises(ValueError, match="time"):
        conservation_terms(Constant(), prepared(), nu=.01, time=time)
