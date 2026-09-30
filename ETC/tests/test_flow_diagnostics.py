"""CPU-only checks of reference-free statistics on saved velocity frames."""

import json

import numpy as np
import pytest

from ETC.runtime.flow_diagnostics import summarize_flow


def grid():
    x, y = np.meshgrid(np.arange(8) / 8, np.arange(6) / 6, indexing="xy")
    return np.column_stack((x.ravel(), y.ravel()))


def constant_flow():
    xy = grid()
    prediction = np.broadcast_to([3., 4., 100.], (3, len(xy), 3)).copy()
    return prediction, np.array([0., .3, 1.]), xy


def test_constant_flow_reports_speed_energy_and_spatial_means():
    prediction, times, xy = constant_flow()
    summary = summarize_flow(prediction, times, xy)
    assert summary["baseline"] == "predicted_t0"
    assert len(summary["per_time"]) == len(times)
    for row, time in zip(summary["per_time"], times):
        assert row == {"time": time, "speed_rms": 5., "mean_kinetic_energy": 12.5,
                       "mean_u": 3., "mean_v": 4., "fluctuating_kinetic_energy": 0.,
                       "speed_retention_ratio": 1., "energy_retention_ratio": 1.,
                       "fluctuating_energy_retention_ratio": None}
    json.dumps(summary, allow_nan=False)


def test_decay_retention_uses_predicted_zero_frame_and_squares_amplitude():
    prediction, times, xy = constant_flow()
    factors = np.array([1., .5, .1])
    prediction[:, :, :2] *= factors[:, None, None]
    rows = summarize_flow(prediction, times, xy)["per_time"]
    np.testing.assert_allclose([row["speed_rms"] for row in rows], 5 * factors)
    np.testing.assert_allclose([row["speed_retention_ratio"] for row in rows], factors)
    np.testing.assert_allclose([row["energy_retention_ratio"] for row in rows], factors**2)


def test_velocity_rotation_preserves_energy_without_claiming_accuracy():
    xy = grid()
    u, v = np.sin(2 * np.pi * xy[:, 0]), np.cos(2 * np.pi * xy[:, 1])
    first = np.column_stack((u, v, np.zeros(len(xy))))
    rotated = np.column_stack((-v, u, np.ones(len(xy)) * 1e6))
    summary = summarize_flow(np.stack((first, rotated)), [0., 1.], xy)
    assert not np.allclose(first[:, :2], rotated[:, :2])
    rows = summary["per_time"]
    assert rows[1]["energy_retention_ratio"] == pytest.approx(1.)
    assert rows[1]["fluctuating_energy_retention_ratio"] == pytest.approx(1.)
    assert rows[0]["fluctuating_kinetic_energy"] == pytest.approx(rows[0]["mean_kinetic_energy"])
    assert "passed" not in summary


def test_fluctuating_energy_removes_mean_and_is_translation_invariant():
    xy = grid()
    first = np.column_stack((np.sin(2 * np.pi * xy[:, 0]),
                             np.cos(2 * np.pi * xy[:, 1]), np.zeros(len(xy))))
    shifted = first + np.array([3., 4., 17.])
    rows = summarize_flow(np.stack((first, shifted)), [0., 1.], xy)["per_time"]
    assert rows[1]["mean_kinetic_energy"] - rows[0]["mean_kinetic_energy"] == pytest.approx(12.5)
    assert rows[1]["fluctuating_kinetic_energy"] == pytest.approx(rows[0]["fluctuating_kinetic_energy"])
    assert rows[1]["mean_u"] == pytest.approx(3.)
    assert rows[1]["mean_v"] == pytest.approx(4.)


def test_zero_baseline_ratios_are_undefined_even_if_later_flow_is_nonzero():
    prediction, times, xy = constant_flow()
    prediction[0] = 0
    summary = summarize_flow(prediction, times, xy)
    assert summary["per_time"][0]["mean_kinetic_energy"] == 0
    for row in summary["per_time"]:
        assert row["speed_retention_ratio"] is None
        assert row["energy_retention_ratio"] is None
        assert row["fluctuating_energy_retention_ratio"] is None
    json.dumps(summary, allow_nan=False)


def test_uniform_grid_order_does_not_change_statistics():
    prediction, times, xy = constant_flow()
    prediction[:, :, 0] = xy[:, 0]
    order = np.random.default_rng(23).permutation(len(xy))
    first = summarize_flow(prediction, times, xy)["per_time"]
    second = summarize_flow(prediction[:, order], times, xy[order])["per_time"]
    for expected, actual in zip(first, second):
        assert actual == pytest.approx(expected)


def test_actual_export_float32_spacing_is_accepted():
    axis = np.linspace(-.720, .720, 129, dtype=np.float32)[:-1]
    x, y = np.meshgrid(axis, axis, indexing="xy")
    xy = np.column_stack((x.ravel(), y.ravel()))
    summary = summarize_flow(np.zeros((1, len(xy), 3), dtype=np.float32), [0.], xy)
    assert summary["per_time"][0]["speed_rms"] == 0


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("key", ["prediction", "times", "xy"])
def test_nonfinite_values_are_rejected(key, bad):
    prediction, times, xy = constant_flow()
    values = {"prediction": prediction, "times": times, "xy": xy}
    values[key].flat[-1] = bad
    with pytest.raises(ValueError, match="finite"):
        summarize_flow(**values)


@pytest.mark.parametrize("change", ["fields", "frames", "times", "coordinates", "empty"])
def test_invalid_shapes_are_rejected(change):
    prediction, times, xy = constant_flow()
    if change == "fields":
        prediction = prediction[:, :, :2]
    elif change == "frames":
        prediction = prediction[0]
    elif change == "times":
        times = times[:, None]
    elif change == "coordinates":
        xy = xy[:, :1]
    else:
        prediction, times = prediction[:0], times[:0]
    with pytest.raises(ValueError):
        summarize_flow(prediction, times, xy)


@pytest.mark.parametrize("times", [[.1, .3, 1.], [0., 1., .5], [0., .5, .5], [0., -.5, 1.]])
def test_ambiguous_time_baseline_or_order_is_rejected(times):
    prediction, _, xy = constant_flow()
    with pytest.raises(ValueError, match="start at zero and be strictly increasing"):
        summarize_flow(prediction, times, xy)


@pytest.mark.parametrize("change", ["duplicate", "nonuniform", "missing"])
def test_grid_must_be_complete_uniform_and_without_duplicates(change):
    prediction, times, xy = constant_flow()
    if change == "duplicate":
        xy[1] = xy[0]
    elif change == "nonuniform":
        xy[xy[:, 0] == .125, 0] = .1
    else:
        xy, prediction = xy[:-1], prediction[:, :-1]
    with pytest.raises(ValueError, match="grid|uniform"):
        summarize_flow(prediction, times, xy)


@pytest.mark.parametrize("kind", ["complex", "boolean", "string"])
def test_nonreal_or_nonnumeric_values_are_rejected(kind):
    prediction, times, xy = constant_flow()
    prediction = prediction.astype({"complex": complex, "boolean": bool, "string": str}[kind])
    with pytest.raises(ValueError, match="real numeric"):
        summarize_flow(prediction, times, xy)


def test_finite_inputs_that_overflow_statistics_are_rejected():
    prediction, times, xy = constant_flow()
    prediction[:] = 1e300
    with pytest.raises(ValueError, match="overflowed"):
        summarize_flow(prediction, times, xy)
