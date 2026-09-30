"""CPU-only contract tests; fake AFNO, no checkpoints or weather downloads.

These verify wiring and validation, not meteorological skill. The loader tests
use broadcast views rather than allocating complete weather states. The tiny
rollout bypasses the full-grid loader only after that loader is tested directly.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types

import numpy as np
import pytest
import torch


HERE = Path(__file__).resolve().parents[2] / "01_labs/04_weather_forecasting/source_code"
OFFICIAL_VARIABLES = [
    "u10m", "v10m", "t2m", "sp", "msl", "t850", "u1000", "v1000", "z1000",
    "u850", "v850", "z850", "u500", "v500", "z500", "t500", "z50", "r500",
    "r850", "tcwv", "u100m", "v100m", "u250", "v250", "z250", "t250",
]


@pytest.fixture
def runner(monkeypatch):
    """Import the runner without importing PhysicsNeMo or initializing CUDA."""
    fake_afno = types.ModuleType("physicsnemo.models.afno")
    fake_afno.AFNO = types.SimpleNamespace(from_checkpoint=None)
    monkeypatch.setitem(sys.modules, "physicsnemo.models.afno", fake_afno)
    downloader_spec = importlib.util.spec_from_file_location("weather_model_under_test", HERE / "download_model.py")
    downloader = importlib.util.module_from_spec(downloader_spec)
    downloader_spec.loader.exec_module(downloader)
    monkeypatch.setitem(sys.modules, "download_model", downloader)
    spec = importlib.util.spec_from_file_location("forecast_under_test", HERE / "run_forecast.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_official_channel_order_and_grid_are_not_silently_remapped(runner):
    assert runner.VARIABLES == OFFICIAL_VARIABLES
    assert runner.REVISION == "c67a63995f6c8e0e557eb3d791f32f437e9b02d5"
    np.testing.assert_array_equal(runner.LATITUDE, 90.0 - 0.25 * np.arange(720))
    np.testing.assert_array_equal(runner.LONGITUDE, 0.25 * np.arange(1440))


class _BroadcastState(np.ndarray):
    # load_initial copies the source. Keeping this read-only view avoids a
    # 108-MB allocation while exercising its actual finite/shape/dtype checks.
    def copy(self, *args, **kwargs):
        return self


def state_view(value=0.0, dtype=np.float32, shape=(26, 720, 1440)):
    return np.broadcast_to(np.array(value, dtype=dtype), shape).view(_BroadcastState)


class _Saved(dict):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def initial_payload(runner):
    return _Saved(state=state_view(), variables=np.array(runner.VARIABLES),
                  latitude=runner.LATITUDE.copy(), longitude=runner.LONGITUDE.copy(),
                  time=np.array("2022-09-25T00:00:00", dtype="datetime64[s]"))


def mock_initial_load(monkeypatch, runner, payload):
    calls = []

    def read(path, **kwargs):
        assert kwargs == {"allow_pickle": False}
        calls.append(path)
        return payload

    monkeypatch.setattr(runner.np, "load", read)
    return calls


def test_initial_full_grid_contract_and_exact_hour(runner, monkeypatch):
    payload = initial_payload(runner)
    calls = mock_initial_load(monkeypatch, runner, payload)
    state, timestamp = runner.load_initial("initial-only.npz")
    assert state.shape == (26, 720, 1440)
    assert state.dtype == np.float32
    assert timestamp == np.datetime64("2022-09-25T00", "h")
    assert calls == ["initial-only.npz"]


@pytest.mark.parametrize("state", [
    state_view(shape=(25, 720, 1440)),
    state_view(shape=(26, 721, 1440)),
    state_view(dtype=np.float64),
    state_view(value=np.nan),
    state_view(value=np.inf),
])
def test_initial_rejects_bad_state(runner, monkeypatch, state):
    payload = initial_payload(runner)
    payload["state"] = state
    mock_initial_load(monkeypatch, runner, payload)
    with pytest.raises(ValueError, match="finite FP32"):
        runner.load_initial("unused")


@pytest.mark.parametrize("problem", ["channel_order", "latitude_reversed", "south_pole", "longitude_shift"])
def test_initial_rejects_coordinate_or_channel_remapping(runner, monkeypatch, problem):
    payload = initial_payload(runner)
    if problem == "channel_order":
        payload["variables"][[0, 1]] = payload["variables"][[1, 0]]
    elif problem == "latitude_reversed":
        payload["latitude"] = payload["latitude"][::-1]
    elif problem == "south_pole":
        payload["latitude"] = np.linspace(90, -90, 720)
    else:
        payload["longitude"] = np.arange(-180, 180, 0.25)
    mock_initial_load(monkeypatch, runner, payload)
    with pytest.raises(ValueError):
        runner.load_initial("unused")


@pytest.mark.parametrize("timestamp", [
    np.array("NaT", dtype="datetime64[s]"),
    np.array(["2022-09-25T00"], dtype="datetime64[h]"),
    np.array("2022-09-25T00"),
    np.array("2022-09-25T03", dtype="datetime64[h]"),
    np.array("2022-09-25T00:30", dtype="datetime64[m]"),
    np.array("2022-09-25T00:00:00.001", dtype="datetime64[ms]"),
])
def test_initial_rejects_invalid_or_fractional_hour(runner, monkeypatch, timestamp):
    payload = initial_payload(runner)
    payload["time"] = timestamp
    mock_initial_load(monkeypatch, runner, payload)
    with pytest.raises(ValueError):
        runner.load_initial("unused")


def model_artifacts(runner, monkeypatch, folder, mean=None, std=None):
    (folder / "fcn.mdlus").write_bytes(b"fake checkpoint; never deserialized")
    np.save(folder / "global_means.npy", np.zeros((1, 26, 1, 1)) if mean is None else mean)
    np.save(folder / "global_stds.npy", np.ones((1, 26, 1, 1)) if std is None else std)
    files = {name: (p.stat().st_size, hashlib.sha256(p.read_bytes()).hexdigest())
             for name in ("fcn.mdlus", "global_means.npy", "global_stds.npy")
             for p in [folder / name]}
    monkeypatch.setattr(runner, "FILES", files)


class _FakeLoadedModel:
    def __init__(self):
        self.device = None
        self.evaluating = False

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        self.evaluating = True
        return self


def test_strict_cpu_load_and_float32_normalization(runner, monkeypatch, tmp_path):
    model_artifacts(runner, monkeypatch, tmp_path)
    model, calls = _FakeLoadedModel(), []

    def load(path, **kwargs):
        calls.append((path, kwargs))
        return model

    monkeypatch.setattr(runner.AFNO, "from_checkpoint", load)
    actual, mean, std = runner.load_model(tmp_path, "cpu")
    assert actual is model and model.device == "cpu" and model.evaluating
    assert calls == [(str(tmp_path / "fcn.mdlus"), {"strict": True})]
    assert mean.dtype == std.dtype == torch.float32
    assert mean.shape == std.shape == (1, 26, 1, 1)
    assert mean.device.type == std.device.type == "cpu"


@pytest.mark.parametrize("name", ["fcn.mdlus", "global_means.npy", "global_stds.npy"])
def test_all_hashes_checked_before_checkpoint_deserialization(runner, monkeypatch, tmp_path, name):
    model_artifacts(runner, monkeypatch, tmp_path)
    path = tmp_path / name
    original = path.read_bytes()
    path.write_bytes(bytes([original[0] ^ 1]) + original[1:])  # same length, wrong digest
    calls = []
    monkeypatch.setattr(runner.AFNO, "from_checkpoint", lambda *a, **k: calls.append(a))
    with pytest.raises(ValueError, match="Pinned artifact mismatch"):
        runner.load_model(tmp_path, "cpu")
    assert calls == []


@pytest.mark.parametrize("mean,std", [
    (np.zeros((26,)), np.ones((26,))),
    (np.full((1, 26, 1, 1), np.nan), np.ones((1, 26, 1, 1))),
    (np.zeros((1, 26, 1, 1)), np.zeros((1, 26, 1, 1))),
    (np.zeros((1, 26, 1, 1)), np.full((1, 26, 1, 1), np.inf)),
])
def test_invalid_normalization_rejected(runner, monkeypatch, tmp_path, mean, std):
    model_artifacts(runner, monkeypatch, tmp_path, mean, std)
    monkeypatch.setattr(runner.AFNO, "from_checkpoint", lambda *a, **k: _FakeLoadedModel())
    with pytest.raises(ValueError, match="normalization"):
        runner.load_model(tmp_path, "cpu")


def configure_tiny_rollout(runner, monkeypatch, tmp_path, *, nonfinite=False):
    """No full-grid arrays, actual model, future truth, or CUDA calls."""
    initial_path = tmp_path / "initial.npz"
    initial_path.write_bytes(b"initial-only fixture; loader separately tested")
    state = np.arange(26 * 2 * 3, dtype=np.float32).reshape(26, 2, 3)
    calls = {"initial": [], "model_inputs": []}
    mean = torch.full((1, 26, 1, 1), 5.0)
    std = torch.full((1, 26, 1, 1), 3.0)

    def load_initial(path):
        calls["initial"].append(path)
        return state.copy(), np.datetime64("2022-09-25T00", "h")

    def model(x):
        assert x.shape == (1, 26, 2, 3) and x.device.type == "cpu"
        assert not torch.is_grad_enabled()
        calls["model_inputs"].append(x.clone())
        return torch.full_like(x, float("nan") if nonfinite else 2.0)

    def forbidden_cuda(*args, **kwargs):
        raise AssertionError("CPU rollout attempted a CUDA operation")

    monkeypatch.setattr(runner, "load_initial", load_initial)
    monkeypatch.setattr(runner, "load_model", lambda folder, device: (model, mean, std))
    monkeypatch.setattr(runner, "LATITUDE", np.array([90.0, 89.75]))
    monkeypatch.setattr(runner, "LONGITUDE", np.array([0.0, 0.25, 0.5]))
    monkeypatch.setattr(runner.torch, "set_num_threads", lambda _: None)
    monkeypatch.setattr(runner.torch, "set_default_dtype", lambda _: None)
    for name in ("is_available", "synchronize", "reset_peak_memory_stats", "max_memory_allocated", "get_device_name"):
        monkeypatch.setattr(runner.torch.cuda, name, forbidden_cuda)
    output = tmp_path / "output"
    monkeypatch.setattr(sys, "argv", ["run_forecast.py", "--initial", str(initial_path),
                                     "--output", str(output), "--device", "cpu", "--steps", "2"])
    return initial_path, output, state, calls


def test_initial_only_autoregression_normalization_next_state_and_times(runner, monkeypatch, tmp_path):
    initial_path, output, state, calls = configure_tiny_rollout(runner, monkeypatch, tmp_path)
    runner.main()
    assert calls["initial"] == [initial_path]
    assert len(calls["model_inputs"]) == 2
    np.testing.assert_allclose(calls["model_inputs"][0].numpy()[0], (state - 5) / 3)
    # Model output 2 becomes physical state 2*3+5=11, not initial+11.
    # The second step receives the full predicted 26-channel state, not truth.
    assert torch.all(calls["model_inputs"][1] == 2)
    with np.load(output / "forecast.npz", allow_pickle=False) as saved:
        assert saved["prediction"].shape == (3, 4, 2, 3)
        np.testing.assert_array_equal(saved["prediction"][0], state[runner.OUTPUT_INDICES])
        assert np.all(saved["prediction"][1:] == 11)
        assert saved["variables"].tolist() == ["u10m", "v10m", "msl", "t2m"]
        assert saved["lead_hours"].tolist() == [0, 6, 12]
        np.testing.assert_array_equal(saved["times"], np.array(
            ["2022-09-25T00", "2022-09-25T06", "2022-09-25T12"], dtype="datetime64[h]"))
    report = json.loads((output / "runtime.json").read_text())
    assert report["future_verification_data_used_as_input"] is False
    assert report["smoke_only"] is False
    assert report["initial_sha256"] == hashlib.sha256(initial_path.read_bytes()).hexdigest()
    assert report["cuda_peak_allocated_gib"] is None
    assert report["variables"] == runner.VARIABLES
    assert len(report["channel_ranges"]) == 2
    statistics = report["initial_normalized_statistics"]
    assert list(statistics) == runner.VARIABLES
    for index, name in enumerate(runner.VARIABLES):
        expected = (state[index] - 5) / 3
        assert statistics[name]["mean"] == pytest.approx(float(expected.mean()))
        assert statistics[name]["min"] == pytest.approx(float(expected.min()))
        assert statistics[name]["max"] == pytest.approx(float(expected.max()))
        assert np.isfinite(statistics[name]["std"])


def test_nonfinite_forecast_rejected_without_output(runner, monkeypatch, tmp_path):
    _, output, _, _ = configure_tiny_rollout(runner, monkeypatch, tmp_path, nonfinite=True)
    with pytest.raises(FloatingPointError, match="lead 6h"):
        runner.main()
    assert not output.exists()


def test_future_verification_argument_not_accepted(runner, monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", ["run_forecast.py", "--output", str(tmp_path / "unused"),
                                     "--truth", "future.npz", "--device", "cpu"])
    with pytest.raises(SystemExit) as exc:
        runner.main()
    assert exc.value.code == 2
