"""Protect the original-data student lesson separately from analytic regressions."""
import importlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
flow = importlib.import_module("01_labs.04_navier_stokes.source_code.navier_stokes")
CONF = ROOT / "01_labs/04_navier_stokes/source_code/conf"


@pytest.fixture(autouse=True)
def bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(min(2, previous))
    yield
    torch.set_num_threads(previous)


def original_grid(nx=7, ny=5):
    x, y = torch.meshgrid(torch.linspace(-.72, .719, nx),
                          torch.linspace(-.72, .719, ny), indexing="xy")
    xy = torch.stack((x.ravel(), y.ravel()), dim=1)
    # Every sample has a unique signature, exposing interpolation or transposition.
    identifier = torch.arange(nx * ny, dtype=torch.float32)
    field = torch.stack((identifier, -2 * identifier, 100 + identifier), dim=1)
    return xy, field


def forbid_fixture(*args, **kwargs):
    raise AssertionError("The original student lesson must not use the analytic fixture")


@pytest.mark.parametrize("size, expected_shape", [(3, (3, 3)), (128, (5, 7))])
def test_original_sampler_keeps_cartesian_rows_and_columns_without_interpolation(size, expected_shape):
    xy, field = original_grid()
    selected_xy, selected_field = flow.sample_initial_field((xy, field), size)
    ny, nx = expected_shape
    assert selected_xy.shape == (nx * ny, 2)
    assert torch.unique(selected_xy[:, 0]).numel() == nx
    assert torch.unique(selected_xy[:, 1]).numel() == ny
    for coordinate, value in zip(selected_xy, selected_field):
        index = torch.nonzero((xy == coordinate).all(dim=1)).flatten()
        assert index.numel() == 1
        torch.testing.assert_close(value, field[index.item()], rtol=0, atol=0)
    grid = selected_xy.reshape(ny, nx, 2)
    torch.testing.assert_close(grid[:, :, 0], grid[0:1, :, 0].expand(ny, nx))
    torch.testing.assert_close(grid[:, :, 1], grid[:, :1, 1].expand(ny, nx))
    if size == 3:
        assert set(selected_field[:, 0].tolist()) == {0, 3, 6, 14, 17, 20, 28, 31, 34}


@pytest.mark.parametrize("size", [0, 1, True, 2.5])
def test_original_sampler_rejects_invalid_size(size):
    with pytest.raises(ValueError, match="grid_size"):
        flow.sample_initial_field(original_grid(), size)


def test_original_sampler_rejects_diagonal_and_misordered_points():
    xy, field = original_grid()
    with pytest.raises(ValueError, match="grid"):
        flow.sample_initial_field((xy[[0, 8, 16]], field[[0, 8, 16]]), 3)
    with pytest.raises(ValueError, match="ordering"):
        flow.sample_initial_field((xy.flip(0), field.flip(0)), 3)


def test_original_loss_keeps_initial_batch_sum_and_pde_area_scaling(monkeypatch):
    calls = []

    class ConstantModel(torch.nn.Module):
        def forward(self, xy, time):
            calls.append(time.detach().clone())
            return torch.tensor([1., -2., 3.]).expand(len(xy), 3)

    monkeypatch.setattr(flow, "residuals", lambda field, xy, time, physics: {
        "continuity": torch.ones(len(xy), 1),
        "momentum_x": torch.full((len(xy), 1), 2.),
        "momentum_y": torch.full((len(xy), 1), 3.),
    })
    xy, _ = original_grid()
    pde_xy = xy[:4]
    initial_xy = xy[:2]
    points = (pde_xy, torch.full((4, 1), .6), initial_xy, torch.zeros(2, 3))
    terms = flow.original_loss_terms(ConstantModel(), object(), points)
    assert terms["physics"].item() == pytest.approx(flow.LENGTH**2 * (1 + 4 + 9))
    assert terms["initial_data"].item() == 2 * (1 + 4 + 9)
    assert torch.equal(calls[0], points[1])
    assert not calls[1].any(), "All observed targets belong to the initial time"
    doubled = (*points[:2], initial_xy.repeat(2, 1), torch.zeros(4, 3))
    doubled_terms = flow.original_loss_terms(ConstantModel(), object(), doubled)
    assert doubled_terms["initial_data"] == 2 * terms["initial_data"]
    assert doubled_terms["physics"] == terms["physics"]


def test_original_training_uses_observations_only_and_updates_fp32_adam(monkeypatch):
    torch.manual_seed(42)
    cfg = {"steps": 3, "batch_size": 4, "learning_rate": .001,
           "num_layers": 1, "layer_size": 8,
           "lab4_architecture": "upstream_silu_weight_norm"}
    xy, values = original_grid()
    initial = (xy, values / 100)
    model = flow.PeriodicFlow(cfg).float()
    physics = flow.informer(flow.NavierStokes(flow.REAL_NU), "cpu",
                            supplied_derivatives=("u__t", "v__t"))
    monkeypatch.setattr(flow, "taylor_green", forbid_fixture)
    monkeypatch.setattr(flow, "evaluate", forbid_fixture)
    monkeypatch.setattr(torch.optim, "LBFGS", forbid_fixture)
    original_points, original_step = flow.training_points, torch.optim.Adam.step
    samples, snapshots = [], [torch.cat([p.detach().flatten() for p in model.parameters()])]

    def points(batch_size, device, initial_data, **kwargs):
        assert initial_data is initial
        result = original_points(batch_size, device, initial_data, **kwargs)
        for point, target in zip(result[2], result[3]):
            match = torch.nonzero((initial[0] == point).all(dim=1)).flatten()
            assert match.numel() == 1
            torch.testing.assert_close(target, initial[1][match.item()], rtol=0, atol=0)
        samples.append(result)
        return result

    def step(optimizer, *args, **kwargs):
        result = original_step(optimizer, *args, **kwargs)
        snapshots.append(torch.cat([p.detach().flatten() for p in model.parameters()]))
        return result

    monkeypatch.setattr(flow, "training_points", points)
    monkeypatch.setattr(torch.optim.Adam, "step", step)
    history = flow.optimize_original(model, physics, cfg, "cpu", initial)
    assert len(samples) == len(history) == 3
    assert all(not torch.equal(samples[i][0], samples[i + 1][0]) for i in range(2))
    assert [row["step"] for row in history] == [1, 2, 3]
    assert [row["learning_rate"] for row in history] == pytest.approx(
        [.001 * .95 ** (i / 3000) for i in range(3)])
    assert history[-1]["learning_rate"] < history[0]["learning_rate"]
    assert all(torch.isfinite(tensor).all() and tensor.dtype == torch.float32 for tensor in snapshots)
    assert all(not torch.equal(before, after) for before, after in zip(snapshots, snapshots[1:]))
    assert all(np.isfinite(list(row.values())).all() for row in history)


def test_original_training_rejects_missing_observations_before_optimizer():
    with pytest.raises(ValueError, match="initial field"):
        flow.optimize_original(None, None, {}, "cpu", None)


def test_original_and_synthetic_configurations_stay_separate():
    original = yaml.safe_load((CONF / "config.yaml").read_text())
    synthetic = yaml.safe_load((CONF / "synthetic_fixture.yaml").read_text())
    assert original == {"steps": 50000, "batch_size": 2048, "learning_rate": .001,
                        "layer_size": 256, "num_layers": 6}
    assert synthetic == {"steps": 3000, "batch_size": 128, "learning_rate": .001,
                         "layer_size": 64, "num_layers": 3}
    manifest = json.loads((ROOT / "ETC/course_materials/course_manifest.json").read_text())
    run = next(run for run in manifest["runs"] if run["id"] == "navier_stokes")
    assert "--smoke-data" not in run["args"]


def test_original_architecture_requests_silu_and_weight_normalization(monkeypatch):
    seen = {}

    def network(**kwargs):
        seen.update(kwargs)
        return torch.nn.Identity()

    monkeypatch.setattr(flow, "FullyConnected", network)
    monkeypatch.setattr(flow, "mlp", forbid_fixture)
    cfg = yaml.safe_load((CONF / "config.yaml").read_text())
    cfg["lab4_architecture"] = "upstream_silu_weight_norm"
    flow.PeriodicFlow(cfg)
    assert seen == {"in_features": 5, "out_features": 3, "layer_size": 256,
                    "num_layers": 6, "activation_fn": "silu", "weight_norm": True}


def test_default_main_selects_original_input_and_records_original_recipe(tmp_path, monkeypatch):
    observed = {}
    xy, fields = original_grid()

    def setup(args, **kwargs):
        observed["default_config"] = args.config
        observed["configured_values"] = yaml.safe_load(args.config.read_text())
        return {"steps": 1, "batch_size": 4, "learning_rate": .001,
                "num_layers": 1, "layer_size": 8}, "cpu"

    class TinyModel(torch.nn.Module):
        def __init__(self, cfg):
            super().__init__()
            observed["architecture"] = cfg["lab4_architecture"]
        def forward(self, coordinates, time):
            return torch.cat((coordinates, time), dim=1)

    def read_data(*, data_path):
        assert data_path is None
        observed["read_data"] = True
        return xy.numpy(), fields.numpy()

    def optimize(model, physics, cfg, device, initial_data):
        torch.testing.assert_close(initial_data[0], xy)
        torch.testing.assert_close(initial_data[1], fields)
        observed["original_optimizer"] = True
        return [{"step": 1, "learning_rate": .001, "loss": 1., "physics": .2, "initial_data": .8}]

    def evaluate(model, physics, device, synthetic, initial_data, *, initial_batch_size):
        assert synthetic is False and initial_data is not None
        assert initial_batch_size == 4
        return {"objective": 1., "pde_rmse": .2, "initial_data_rmse": .3}

    def save(args, cfg, model, history, arrays, metrics, plot):
        observed.update(arrays=arrays, metrics=metrics)

    monkeypatch.setattr(sys, "argv", ["navier_stokes.py", "--device", "cpu", "--steps", "1",
                                       "--output-dir", str(tmp_path / "out")])
    monkeypatch.setenv("AI4SCI_SMOKE_DATA", "1")  # The retired UI switch must have no effect.
    monkeypatch.setattr(flow, "setup", setup)
    monkeypatch.setattr(flow, "PeriodicFlow", TinyModel)
    monkeypatch.setattr(flow, "read_wf_data", read_data)
    monkeypatch.setattr(flow, "optimize_original", optimize)
    monkeypatch.setattr(flow, "optimize_lab", forbid_fixture)
    monkeypatch.setattr(flow, "taylor_green", forbid_fixture)
    monkeypatch.setattr(flow, "accuracy_checks", forbid_fixture)
    monkeypatch.setattr(flow, "informer", lambda *args, **kwargs: object())
    monkeypatch.setattr(flow, "evaluate", evaluate)
    monkeypatch.setattr(flow, "save_run", save)
    flow.main()
    assert observed["default_config"] == CONF / "config.yaml"
    assert observed["configured_values"]["steps"] == 50000
    assert observed["architecture"] == "upstream_silu_weight_norm"
    assert observed["read_data"] and observed["original_optimizer"]
    metrics, arrays = observed["metrics"], observed["arrays"]
    assert metrics["data_kind"] == "upstream_data_lat_legacy_normalization"
    assert metrics["weather_forecast_validated"] is False
    assert metrics["time_unit"] == "hours" and metrics["time_scale_hours"] == 60
    assert metrics["training_recipe"]["optimizer"] == "Adam"
    assert metrics["training_recipe"]["name"] == "upstream_adam_fp32_v1"
    assert metrics["training_recipe"]["positive_time_reference_targets_used_for_training"] is False
    assert "accuracy" not in metrics and "reference" not in arrays
    assert arrays["prediction"].shape == (11, 128 * 128, 3)
    torch.testing.assert_close(arrays["initial_xy"], xy)
    torch.testing.assert_close(arrays["initial_fields"], fields)
    torch.testing.assert_close(arrays["times_hours"], torch.arange(0, 61, 6, dtype=torch.float32))


def test_missing_original_data_fails_without_synthetic_replacement(tmp_path, monkeypatch):
    monkeypatch.setattr(flow, "taylor_green", forbid_fixture)
    with pytest.raises(FileNotFoundError, match="Missing original data"):
        flow.read_wf_data(data_path=tmp_path / "missing.npy")


@pytest.mark.parametrize("suite", ["smoke", "convergence"])
def test_validation_runner_does_not_substitute_fixture_for_student_smoke(suite, tmp_path, monkeypatch):
    from ETC.course_materials import run_validation as runner

    manifest = json.loads(runner.MANIFEST.read_text())
    commands, checked = [], []
    requested_steps = 7

    def execute(command, output, label, timeout):
        commands.append(command)
        return {"label": label, "passed": True, "exit_code": 0}

    def validate(path, *, steps, device, expected_version, seed):
        assert steps == requested_steps and device == "cpu" and seed == 42
        assert expected_version == manifest["physicsnemo"]
        kind = "synthetic_taylor_green" if suite == "convergence" else "upstream_data_lat_legacy_normalization"
        return {"passed": True, "metrics": {"data_kind": kind}}

    def improvement(metrics, case, ratio):
        assert suite == "convergence" and case == "navier_stokes"
        assert metrics["data_kind"] == "synthetic_taylor_green"
        checked.append(case)
        # A deliberately failed accuracy gate must not erase the data-scope record.
        return {"passed": False}

    monkeypatch.setattr(runner, "versions", lambda: {
        "packages": {"nvidia-physicsnemo": manifest["physicsnemo"]}})
    monkeypatch.setattr(runner, "source_snapshot", lambda output: {})
    monkeypatch.setattr(runner, "execute", execute)
    monkeypatch.setattr(runner, "validate_artifacts", validate)
    monkeypatch.setattr(runner, "improvement", improvement)
    output = tmp_path / suite
    status = runner.main(["--suite", suite, "--case", "navier_stokes", "--device", "cpu",
                          "--steps", str(requested_steps), "--convergence-steps", str(requested_steps),
                          "--output-dir", str(output)])
    assert len(commands) == 1
    command = commands[0]
    assert command[command.index("--steps") + 1] == str(requested_steps)
    assert ("--smoke-data" in command) == (suite == "convergence")
    report = json.loads((output / "report.json").read_text())
    record = report["runs"][0]
    assert record["suite"] == suite and record["case"] == "navier_stokes"
    if suite == "smoke":
        assert status == 0 and not checked
        assert record["data_scope"] == "original data_lat.npy student lesson; no future ground truth"
    else:
        assert status == 1 and checked == ["navier_stokes"]
        assert record["data_scope"] == "separate synthetic Taylor-Green regression"
        assert record["improvement"]["passed"] is False
