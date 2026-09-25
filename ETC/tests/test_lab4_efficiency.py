"""Efficient representations must preserve physical fields and periodicity."""
import importlib
import pytest
import torch

flow = importlib.import_module("01_labs.04_navier_stokes.source_code.navier_stokes")


@pytest.fixture(autouse=True)
def bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(min(2, previous))
    yield
    torch.set_num_threads(previous)


def configuration(**overrides):
    return {"lab4_architecture": "upstream_silu_weight_norm", "num_layers": 2,
            "layer_size": 16, "lab4_feature_bands": [1, 2, 4, 8],
            "lab4_field_center": [0.01, -0.02, 5.3],
            "lab4_field_scale": [0.13, 0.11, 0.067], **overrides}


def test_multiscale_network_retains_periodicity_of_values_and_derivatives():
    torch.manual_seed(42)
    model = flow.PeriodicFlow(configuration())
    for time in (0., .37, 1.):
        metrics = flow.periodic_errors(model, "cpu", time)
        assert metrics["periodic_value_max_abs"] < 2e-6
        assert metrics["periodic_gradient_max_abs"] < 2e-5
    assert all(tensor.dtype == torch.float32 for tensor in model.state_dict().values())


def test_output_transform_returns_physical_fields_before_residual_evaluation():
    class Constant(torch.nn.Module):
        def forward(self, features):
            return torch.tensor([1., -2., 3.]).expand(len(features), 3)

    model = flow.PeriodicFlow(configuration())
    model.network = Constant()
    xy = torch.randn(7, 2)
    result = model(xy, torch.zeros(7, 1))
    expected = torch.tensor([.01 + .13, -.02 - .22, 5.3 + .201])
    torch.testing.assert_close(result, expected.expand_as(result))


def test_time_derivative_includes_output_scale():
    class LinearTime(torch.nn.Module):
        def forward(self, features):
            return features[:, -1:].expand(-1, 3)

    model = flow.PeriodicFlow(configuration())
    model.network = LinearTime()
    time = torch.rand(7, 1, requires_grad=True)
    result = model(torch.randn(7, 2), time)
    for column, scale in enumerate(configuration()["lab4_field_scale"]):
        torch.testing.assert_close(flow.derivative(result[:, column:column + 1], time),
                                   torch.full_like(time, scale))


@pytest.mark.parametrize("bands", [[], [1, 1], [0, 1], [1, 2.5], [True, 2], "1,2"])
def test_fractional_or_invalid_frequencies_cannot_change_periodic_boundary(bands):
    with pytest.raises(ValueError, match="positive integers"):
        flow.PeriodicFlow(configuration(lab4_feature_bands=bands))


@pytest.mark.parametrize("overrides", [
    {"lab4_field_center": [0., 0.]}, {"lab4_field_scale": [1., 0., 1.]},
    {"lab4_field_scale": [1., float("nan"), 1.]},
    {"lab4_field_center": [1., float("inf"), 1.]},
])
def test_invalid_field_transform_is_rejected(overrides):
    with pytest.raises(ValueError, match="finite u/v/p"):
        flow.PeriodicFlow(configuration(**overrides))


def test_initial_validation_is_disjoint_and_not_used_for_field_scaling():
    x, y = torch.meshgrid(torch.arange(17), torch.arange(19), indexing="xy")
    xy = torch.stack((x.flatten(), y.flatten()), dim=1).float()
    index = torch.arange(len(xy), dtype=torch.float32)
    fields = torch.stack((index, index ** 2, index + 3), dim=1)
    (train_xy, train_fields), (valid_xy, valid_fields) = flow.split_initial_observations((xy, fields), 4)
    assert len(valid_xy) == 16 and len(train_xy) == len(xy) - 16
    assert not (train_xy[:, None] == valid_xy[None]).all(-1).any()
    torch.testing.assert_close(fields, torch.stack((index, index ** 2, index + 3), dim=1))
    cfg = {}
    flow.configure_efficient_representation(cfg, (train_xy, train_fields))
    torch.testing.assert_close(torch.tensor(cfg["lab4_field_center"]), train_fields.mean(0))
    torch.testing.assert_close(torch.tensor(cfg["lab4_field_scale"]), train_fields.std(0))
    assert not torch.equal(torch.tensor(cfg["lab4_field_center"]), fields.mean(0))


@pytest.mark.parametrize("grid_size", [0, 1, True, 2.5, "32"])
def test_invalid_initial_validation_size_is_rejected(grid_size):
    with pytest.raises(ValueError, match="grid_size"):
        flow.split_initial_observations((None, None), grid_size)
