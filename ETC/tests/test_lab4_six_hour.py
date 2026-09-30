"""The opt-in six-hour recipe must preserve units and disclose changed inputs."""
import copy
import importlib
import math
import sys

import numpy as np
import pytest
import torch

flow = importlib.import_module('ETC.reference_labs.04_navier_stokes.source_code.navier_stokes')


@pytest.fixture(autouse=True)
def bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(min(2, previous))
    yield
    torch.set_num_threads(previous)


def config(**changes):
    return dict(steps=2, batch_size=8, learning_rate=.001, layer_size=8, num_layers=1,
                lab4_architecture='upstream_silu_weight_norm', lab4_feature_bands=[1, 2, 4, 8],
                lab4_field_center=[-.01, -.02, 5.3], lab4_field_scale=[.1, .09, .02],
                lab4_recipe='six_hour_projected_conservation_adam_fp32_v1',
                lab4_time_feature_scale=.1, **changes)


def initial_grid():
    axis = flow.LOWER + flow.LENGTH * np.arange(8) / 8
    x, y = np.meshgrid(axis, axis, indexing='xy')
    phase_x = 2 * math.pi * (x - flow.LOWER) / flow.LENGTH
    phase_y = 2 * math.pi * (y - flow.LOWER) / flow.LENGTH
    xy = np.column_stack((x.ravel(), y.ravel())).astype(np.float32)
    fields = np.column_stack(((np.sin(phase_x) * np.cos(phase_y) + .2 * np.sin(phase_x)).ravel(),
                              (-np.cos(phase_x) * np.sin(phase_y)).ravel(),
                              (5.3 + .1 * np.cos(phase_x)).ravel())).astype(np.float32)
    return xy, fields


def test_time_feature_scale_has_correct_physical_chain_rule():
    class TimeFeatures(torch.nn.Module):
        def forward(self, features):
            tau = features[:, -1:]
            return torch.cat((tau, tau**2, tau.sin()), dim=1)

    model = flow.PeriodicFlow(config()).double()
    model.network = TimeFeatures()
    t = torch.tensor([[0.], [.02], [.1]], dtype=torch.float64, requires_grad=True)
    xy = torch.zeros(3, 2, dtype=torch.float64)
    prediction = model(xy, t)
    torch.testing.assert_close(flow.derivative(prediction[:, :1], t),
                               torch.full_like(t, float(model.scale[0]) / .1))
    torch.testing.assert_close(flow.derivative(prediction[:, 1:2], t),
                               2 * float(model.scale[1]) * t / .1**2)


@pytest.mark.parametrize('value', [0., -1., True, float('nan'), float('inf')])
def test_invalid_time_feature_scale_is_rejected(value):
    cfg = config()
    cfg['lab4_time_feature_scale'] = value
    with pytest.raises(ValueError, match='Time feature scale'):
        flow.PeriodicFlow(cfg)


def test_evaluation_horizon_does_not_mutate_global_times():
    torch.manual_seed(1)
    model = flow.PeriodicFlow(config())
    physics = flow.informer(flow.NavierStokes(flow.REAL_NU), 'cpu',
                            supplied_derivatives=('u__t', 'v__t'))
    initial = tuple(torch.from_numpy(value) for value in initial_grid())
    original_times = flow.EVALUATION_TIMES
    metrics = flow.evaluate(model, physics, 'cpu', False, initial, grid_size=4, time_horizon=.1)
    assert [row['time'] for row in metrics['per_time']] == [value * .1 for value in original_times]
    assert flow.EVALUATION_TIMES is original_times
    default = flow.evaluate(model, physics, 'cpu', False, initial, grid_size=4)
    assert [row['time'] for row in default['per_time']] == list(original_times)


def test_six_hour_optimizer_matches_reviewed_loop_and_rng_order():
    torch.manual_seed(42)
    cfg = config()
    production = flow.PeriodicFlow(cfg)
    reference = copy.deepcopy(production)
    initial = tuple(torch.from_numpy(value) for value in initial_grid())
    physics = flow.informer(flow.NavierStokes(flow.REAL_NU), 'cpu',
                            supplied_derivatives=('u__t', 'v__t'))
    torch.manual_seed(4242)
    production_history = flow.optimize_original(production, physics, cfg, 'cpu', initial)
    production_rng = torch.get_rng_state()
    torch.manual_seed(4242)
    prepared = flow.prepare_conservation(initial[1], 'cpu', lower=flow.LOWER, length=flow.LENGTH, grid_size=32)
    optimizer = torch.optim.Adam(reference.parameters(), lr=.001)
    reference_history = []
    for step in range(1, cfg['steps'] + 1):
        for group in optimizer.param_groups:
            group['lr'] = .001 * .95**((step - 1) / 3000)
        optimizer.zero_grad(set_to_none=True)
        x, t, ix, target = flow.training_points(cfg['batch_size'], 'cpu', initial)
        terms = flow.original_loss_terms(reference, physics, (x, t * .1, ix, target))
        qtime = torch.rand((), device='cpu') * .1
        terms.update(flow.conservation_terms(reference, prepared, flow.REAL_NU, time=qtime))
        loss = sum(terms.values())
        loss.backward()
        optimizer.step()
        reference_history.append({key: float(value.detach()) for key, value in terms.items()})
    assert torch.equal(production_rng, torch.get_rng_state())
    for key, value in production.state_dict().items():
        torch.testing.assert_close(value, reference.state_dict()[key], rtol=0, atol=0)
    for actual, expected in zip(production_history, reference_history):
        for key, value in expected.items():
            assert actual[key] == value
    assert {'energy_balance', 'mean_momentum'} <= set(production_history[-1])


def test_main_six_hour_discloses_derived_fields_and_keeps_raw_preview(tmp_path, monkeypatch):
    xy, raw = initial_grid()
    source = tmp_path / 'source.npy'
    np.save(source, raw.T.reshape(3, 8, 8))
    observed = {}

    def setup(args, **kwargs):
        return dict(steps=1, batch_size=4, learning_rate=.001, layer_size=8, num_layers=1), 'cpu'

    def evaluate(model, physics, device, synthetic, initial, **kwargs):
        assert not synthetic and kwargs['time_horizon'] == .1
        assert kwargs['grid_size'] == 64
        return dict(objective=1., pde_rmse=.2, initial_data_rmse=.1)

    def optimize(model, physics, cfg, device, initial):
        observed['cfg'] = dict(cfg)
        observed['training_fields'] = initial[1]
        return [dict(step=1, learning_rate=.001, loss=1., physics=.2, initial_data=.3,
                     energy_balance=.4, mean_momentum=.1)]

    def forbidden(*args, **kwargs):
        raise AssertionError('Derived original-data recipe must not use the analytic fixture')

    monkeypatch.setattr(sys, 'argv', ['navier_stokes.py', '--recipe', 'six_hour',
                                     '--data-path', str(source), '--device', 'cpu',
                                     '--output-dir', str(tmp_path / 'run')])
    monkeypatch.setattr(flow, 'setup', setup)
    monkeypatch.setattr(flow, 'read_wf_data', lambda **kwargs: (xy, raw))
    monkeypatch.setattr(flow, 'evaluate', evaluate)
    monkeypatch.setattr(flow, 'optimize_original', optimize)
    monkeypatch.setattr(flow, 'optimize_lab', forbidden)
    monkeypatch.setattr(flow, 'taylor_green', forbidden)
    monkeypatch.setattr(flow, 'save_run', lambda args, cfg, model, history, arrays, metrics, plot:
                        observed.update(arrays=arrays, metrics=metrics))
    flow.main()
    cfg, arrays, metrics = observed['cfg'], observed['arrays'], observed['metrics']
    assert cfg['lab4_time_feature_scale'] == cfg['lab4_time_horizon'] == .1
    assert metrics['data_kind'] == flow.DERIVED_DATA_KIND
    assert metrics['time_scale_hours'] == 60 and metrics['horizon_hours'] == 6
    assert metrics['nu'] == flow.REAL_NU
    assert metrics['weather_forecast_validated'] is False
    assert metrics['training_recipe']['conservation_weight'] == 1.
    assert metrics['training_recipe']['conservation_quadrature_grid'] == 32
    assert metrics['initial_condition_transformation']['source_context']['sha256']
    torch.testing.assert_close(arrays['times'], torch.linspace(0, .1, 11))
    torch.testing.assert_close(arrays['times_hours'], torch.linspace(0, 6, 11))
    np.testing.assert_array_equal(arrays['raw_initial_xy'], xy)
    np.testing.assert_array_equal(arrays['raw_initial_fields'], raw)
    assert not np.array_equal(arrays['initial_fields'], raw)
    assert 'reference' not in arrays
