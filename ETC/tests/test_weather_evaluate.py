"""Small CPU tests; no pretrained model, ERA5 download, or GPU is needed."""
import copy
import json
import importlib.util
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

WEATHER_SOURCE = Path(__file__).resolve().parents[2] / "01_labs/04_weather_forecasting/source_code"
_spec = importlib.util.spec_from_file_location("weather_evaluation_under_test", WEATHER_SOURCE / "evaluate_weather.py")
weather = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(weather)


def tiny_case():
    latitude = 90. - np.arange(18) * 10.
    longitude = np.arange(36) * 10.
    reference = np.empty((9, 4, 18, 36), dtype=np.float32)
    for frame in range(9):
        reference[frame, 0] = frame
        reference[frame, 1] = -frame
        reference[frame, 2] = 100000. + 100. * frame
        reference[frame, 3] = 273.15 + frame
    return dict(prediction=reference.copy(), reference=reference,
                variables=np.array(weather.VARIABLES), lead_hours=weather.LEAD_HOURS.copy(),
                times=np.datetime64("2022-09-01T00", "h") + weather.LEAD_HOURS.astype("timedelta64[h]"),
                latitude=latitude, longitude=longitude)


def write_pair(directory, data):
    metadata = {key: value for key, value in data.items() if key not in ("prediction", "reference")}
    forecast, truth = directory / "forecast.npz", directory / "truth.npz"
    np.savez(forecast, prediction=data["prediction"], **metadata)
    np.savez(truth, reference=data["reference"], **metadata)
    return forecast, truth


def test_cosine_area_weights_and_zero_pole():
    latitude = np.array([90., 60., 0.])
    values = np.array([[1e20, 1e20], [4., 4.], [1., 1.]])
    assert weather.weighted_mean(values, latitude) == pytest.approx(2.)
    assert weather.weighted_mean(np.ones((3, 2)) * 7., latitude) == pytest.approx(7.)
    with pytest.raises(ValueError, match="positive-area"):
        weather.weighted_mean(np.ones((2, 2)), [90., -90.])


def test_scalar_rmse_bias_and_persistence_skill():
    metrics = weather.scalar_metrics(np.full((2, 3), 3.), np.full((2, 3), 1.),
                                    np.full((2, 3), 5.), np.array([45., 0.]))
    assert metrics["rmse"] == pytest.approx(2.)
    assert metrics["bias_prediction_minus_reanalysis"] == pytest.approx(2.)
    assert metrics["persistence_rmse"] == pytest.approx(4.)
    assert metrics["mse_skill_vs_persistence"] == pytest.approx(.75)
    identical = weather.scalar_metrics(np.ones((2, 3)), np.ones((2, 3)), np.ones((2, 3)), [45., 0.])
    assert identical["mse_skill_vs_persistence"] is None
    json.dumps(identical, allow_nan=False)


def test_wind_vector_and_speed_are_different():
    truth = np.broadcast_to(np.array([3., 4.])[:, None, None], (2, 2, 3))
    reverse = -truth
    result = weather.wind_metrics(reverse, truth, truth, [45., 0.])
    assert result["vector_rmse"] == pytest.approx(10.)
    assert result["speed"]["rmse"] == pytest.approx(0.)
    assert result["vector_mse_skill_vs_persistence"] is None


def test_report_all_leads_si_and_display_units():
    data = tiny_case()
    data["prediction"][1:, 0] += 3.
    data["prediction"][1:, 1] += 4.
    data["prediction"][1:, 2] += 200.
    data["prediction"][1:, 3] += 2.
    report = weather.evaluate(data)
    assert report["forecast_lead_hours"] == list(range(6, 49, 6))
    for region in ("global", "east_asia"):
        assert len(report["metrics"][region]) == 8
        for row in report["metrics"][region]:
            assert row["variables"]["msl"]["rmse"] == pytest.approx(200.)
            assert row["variables"]["msl"]["display_rmse"] == pytest.approx(2.)
            assert row["variables"]["t2m"]["rmse"] == pytest.approx(2.)
            assert row["wind_10m"]["vector_rmse"] == pytest.approx(5.)
    json.dumps(report, allow_nan=False)
    np.testing.assert_allclose(weather.plot_field(data["reference"], "msl")[0], 1000.)
    np.testing.assert_allclose(weather.plot_field(data["reference"], "t2m")[0], 0., atol=1e-5)


def test_load_pair_and_exact_initial_state(tmp_path):
    data = tiny_case()
    forecast, truth = write_pair(tmp_path, data)
    loaded = weather.load_pair(forecast, truth)
    np.testing.assert_array_equal(loaded["prediction"], data["prediction"])
    data["prediction"][0, 0, 3, 2] = np.float32(.001)
    forecast, truth = write_pair(tmp_path, data)
    with pytest.raises(ValueError, match="exactly"):
        weather.load_pair(forecast, truth)


@pytest.mark.parametrize("defect", ["nonfinite", "integer_fields", "channel_order", "missing_lead", "time_spacing", "latitude_order", "longitude_order"])
def test_invalid_contract_rejected(tmp_path, defect):
    data = tiny_case()
    if defect == "nonfinite":
        data["prediction"][3, 2, 1, 1] = np.nan
    elif defect == "integer_fields":
        data["prediction"] = data["prediction"].astype(np.int64)
    elif defect == "channel_order":
        data["variables"] = data["variables"][::-1]
    elif defect == "missing_lead":
        data["lead_hours"][4] = 25
    elif defect == "time_spacing":
        data["times"][4] += np.timedelta64(1, "h")
    elif defect == "latitude_order":
        data["latitude"] = data["latitude"][::-1]
    elif defect == "longitude_order":
        data["longitude"] = np.roll(data["longitude"], 1)
    forecast, truth = write_pair(tmp_path, data)
    with pytest.raises(ValueError):
        weather.load_pair(forecast, truth)


def test_mismatched_metadata_rejected(tmp_path):
    data = tiny_case()
    forecast, truth = write_pair(tmp_path, data)
    metadata = {key: value for key, value in data.items() if key not in ("prediction", "reference")}
    metadata["times"] = metadata["times"] + np.timedelta64(6, "h")
    np.savez(truth, reference=data["reference"], **metadata)
    with pytest.raises(ValueError, match="times mismatch"):
        weather.load_pair(forecast, truth)


def test_regional_metrics_are_not_global_metrics():
    data = tiny_case()
    data["prediction"][1:, 0, :, :5] += 20.
    report = weather.evaluate(data)
    assert report["metrics"]["global"][0]["variables"]["u10m"]["rmse"] > 0.
    assert report["metrics"]["east_asia"][0]["variables"]["u10m"]["rmse"] == 0.


def acceptance_document():
    return {
        "initial_time_utc": "2022-09-01T00:00:00Z", "forecast_leads_hours": list(range(6, 49, 6)),
        "scope": "Test case only", "acceptance": {"global_and_east_asia_24h_and_48h": {
            "wind_vector_rmse_m_s_max": 6., "mean_sea_level_pressure_rmse_hpa_max": 5.,
            "temperature_2m_rmse_kelvin_max": 3., "wind_vector_mse_less_than_persistence": True,
            "pressure_mse_less_than_persistence": True, "temperature_mse_less_than_persistence": True,
        }},
    }


def test_acceptance_uses_fixed_thresholds_and_strict_persistence():
    data = tiny_case()
    criteria = acceptance_document()
    report = weather.evaluate(data)
    assert weather.assess_acceptance(report, criteria)["measured_forecast_criteria_passed"]
    data["prediction"][1:] = data["reference"][0]
    result = weather.assess_acceptance(weather.evaluate(data), criteria)
    assert not result["measured_forecast_criteria_passed"]
    persistence_checks = [row for row in result["checks"] if row["criterion"].endswith("less_than_persistence")]
    assert len(persistence_checks) == 12
    assert all(not row["passed"] for row in persistence_checks)
    altered = copy.deepcopy(criteria)
    altered["initial_time_utc"] = "2022-09-02T00:00:00Z"
    assert not weather.assess_acceptance(report, altered)["measured_forecast_criteria_passed"]


def test_geographic_lines_do_not_join_across_dateline(tmp_path):
    path = tmp_path / "coastline.geojson"
    path.write_text(json.dumps({"type": "LineString", "coordinates": [[-10, 10], [-1, 11], [1, 12], [10, 13]]}))
    lines = weather.geographic_lines(path)
    assert len(lines) == 2
    assert all(np.max(np.abs(np.diff(line[:, 0]))) < 180 for line in lines)


def test_small_cpu_cli_and_no_overwrite(tmp_path):
    forecast, truth = write_pair(tmp_path, tiny_case())
    output = tmp_path / "evaluation"
    command = [sys.executable, str(Path(weather.__file__)), "--forecast", str(forecast),
               "--truth", str(truth), "--output-dir", str(output), "--no-plots"]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    report = json.loads((output / "evaluation.json").read_text())
    assert report["forecast_lead_hours"] == list(range(6, 49, 6))
    assert report["figures"] == []
    assert subprocess.run(command, capture_output=True, text=True).returncode != 0


def test_cpu_plot_smoke(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    data = tiny_case()
    report = weather.evaluate(data)
    paths = weather.plot_time_curves(report, tmp_path)
    paths += weather.plot_maps(data, tmp_path, fields=("wind_speed",))
    assert len(paths) == 4
    assert all(Path(path).is_file() and Path(path).stat().st_size > 1000 for path in paths)
