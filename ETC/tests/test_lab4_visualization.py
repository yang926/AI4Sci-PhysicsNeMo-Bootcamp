"""The student view must show original inputs, real predictions and honest units."""

import ast
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from ETC.runtime import flow_visualization as view


ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = ROOT / "ETC/reference_labs/04_navier_stokes/Lab_4_Navier_Stokes.ipynb"


def arrays():
    x, y = np.meshgrid(np.linspace(-.72, .72, 8), np.linspace(-.72, .72, 6), indexing="xy")
    xy = np.column_stack((x.ravel(), y.ravel())).astype(np.float32)
    initial = np.column_stack((1 + xy[:, 1], -.5 * xy[:, 0], xy[:, 0] - xy[:, 1]))
    prediction = np.stack([initial * (1 + index / 20) for index in range(11)])
    return {"xy": xy, "prediction": prediction, "initial_xy": xy,
            "initial_fields": initial, "times": np.linspace(0, 1, 11),
            "times_hours": np.linspace(0, 60, 11)}


def save(tmp_path, data=None, kind=view.ORIGINAL_DATA_KIND):
    (tmp_path / "metrics.json").write_text(json.dumps({"data_kind": kind}))
    np.savez(tmp_path / "predictions.npz", **(arrays() if data is None else data))
    return tmp_path


def test_original_frames_load_without_inventing_future_targets(tmp_path):
    data = view.load_original_flow(save(tmp_path))
    assert set(data) == set(arrays())
    for key, value in arrays().items():
        np.testing.assert_array_equal(data[key], value)


@pytest.mark.parametrize("kind", ["synthetic_taylor_green", "", None])
def test_synthetic_or_unidentified_saved_results_are_rejected(tmp_path, kind):
    with pytest.raises(ValueError, match="original-data"):
        view.load_original_flow(save(tmp_path, kind=kind))


@pytest.mark.parametrize("key", ["initial_xy", "initial_fields", "times_hours"])
def test_older_result_requires_explicit_rerun(tmp_path, key):
    data = arrays()
    del data[key]
    with pytest.raises(ValueError, match="older result"):
        view.load_original_flow(save(tmp_path, data))


def test_future_reference_is_not_presented_as_original_weather(tmp_path):
    data = arrays()
    data["reference"] = data["prediction"]
    with pytest.raises(ValueError, match="future analytical reference"):
        view.load_original_flow(save(tmp_path, data))


@pytest.mark.parametrize("change", ["ten_frames", "wrong_hours", "nan", "wrong_grid", "wrong_shape"])
def test_bad_or_ambiguous_frames_fail_instead_of_being_mislabeled(tmp_path, change):
    data = arrays()
    if change == "ten_frames":
        data["times"] = data["times"][:-1]
    elif change == "wrong_hours":
        data["times_hours"] = data["times"]
    elif change == "nan":
        data["prediction"][0, 0, 0] = np.nan
    elif change == "wrong_grid":
        data["xy"] = data["xy"][::-1].copy()
    else:
        data["prediction"] = data["prediction"][:, :, :2]
    with pytest.raises(ValueError):
        view.load_original_flow(save(tmp_path, data))


def test_animation_updates_all_frames_with_fixed_scales_and_direction():
    data = arrays()
    # Original input ends at .719; inference includes the periodic .720 edge.
    data["initial_xy"] = data["initial_xy"] * .999
    fig, animation = view.build_flow_animation(data)
    left, right = fig.axes[:2]
    initial_surface, predicted_surface = left.collections[0], right.collections[0]
    scale = predicted_surface.get_clim()
    initial_colors = initial_surface.get_array().copy()
    assert scale == initial_surface.get_clim()
    assert left.collections[1].scale == right.collections[1].scale
    assert scale[0] == 0
    assert "normalized" in right.get_xlabel()
    assert "fixed" in left.get_title()
    assert len(list(animation.new_frame_seq())) == 11
    for frame in range(11):
        animation._func(frame)
        assert f"{frame * 6} hours" in right.get_title()
        assert predicted_surface.get_clim() == scale
        np.testing.assert_allclose(predicted_surface.get_array().ravel(),
                                   np.hypot(data["prediction"][frame, :, 0],
                                            data["prediction"][frame, :, 1]))
        np.testing.assert_array_equal(initial_surface.get_array(), initial_colors)
        np.testing.assert_allclose(right.collections[1].U, data["prediction"][frame, :, 0])
        np.testing.assert_allclose(right.collections[1].V, data["prediction"][frame, :, 1])
        fig.canvas.draw()
        title_bounds = fig._suptitle.get_window_extent(fig.canvas.get_renderer())
        assert 0 <= title_bounds.y0 < title_bounds.y1 <= fig.bbox.height
    plt.close(fig)


def test_html_player_has_all_frames_and_closes_figure():
    previous = plt.get_fignums()
    html = view.animate_flow(arrays()).data
    assert "frames[10]" in html
    assert 'type="range"' in html
    assert "data:image/png;base64," in html
    assert plt.get_fignums() == previous


def test_original_input_plot_handles_zero_flow_and_uses_normalized_axes():
    data = arrays()
    fig = view.plot_initial_flow(data["xy"], np.zeros_like(data["initial_fields"]))
    fig.canvas.draw()
    assert fig.axes[0].get_xlabel() == "x (normalized)"
    assert fig.axes[0].collections[0].get_clim()[1] > 0
    plt.close(fig)


def decaying_predictions(final_speed_fraction=.1):
    data = arrays()
    amplitude = np.linspace(1., final_speed_fraction, len(data["times"]))
    data["prediction"] = data["prediction"][0][None] * amplitude[:, None, None]
    return data


def test_flow_diagnostics_show_actual_weakening_and_predicted_initial_baseline():
    data = decaying_predictions()
    before = {key: value.copy() for key, value in data.items()}
    # Changing observations must not alter the predicted-frame energy baseline.
    data["initial_fields"] = data["initial_fields"] * 100
    fig = view.plot_flow_diagnostics(data)
    try:
        speed, energy = fig.axes
        np.testing.assert_allclose(speed.lines[0].get_xdata(), data["times_hours"])
        assert speed.lines[0].get_ydata()[-1] == pytest.approx(speed.lines[0].get_ydata()[0] * .1)
        assert {line.get_label() for line in energy.lines} == {
            "Total kinetic energy", "Fluctuating kinetic energy"}
        for line in energy.lines:
            assert line.get_ydata()[0] == pytest.approx(1)
            assert line.get_ydata()[-1] == pytest.approx(.01)
        assert "normalized" in speed.get_ylabel()
        assert "predicted initial" in energy.get_ylabel()
        assert all("hours" in ax.get_xlabel() for ax in fig.axes)
        assert "not forecast accuracy" in fig._suptitle.get_text()
        note = "\n".join(text.get_text() for text in fig.texts)
        assert "Review flag" in note and "1.0%" in note
        assert "heuristic, not a physics threshold" in note
        fig.canvas.draw()
    finally:
        plt.close(fig)
    for key in ("prediction", "times", "times_hours", "xy"):
        np.testing.assert_array_equal(data[key], before[key])


def test_diagnostic_summary_distinguishes_mean_flow_and_fluctuations_without_future_targets():
    data = arrays()
    fluctuations = data["prediction"][0].copy()
    fluctuations[:, :2] -= fluctuations[:, :2].mean(axis=0)
    for index, amplitude in enumerate(np.linspace(1, .1, len(data["times"]))):
        data["prediction"][index] = fluctuations * amplitude
        data["prediction"][index, :, 0] += 3
    inputs = {key: data[key] for key in ("prediction", "times", "times_hours", "xy")}
    previous = plt.get_fignums()
    html = view.flow_diagnostic_summary(inputs).data
    assert "1.0% fluctuating kinetic energy retained" in html
    assert "Mean velocity (u, v)" in html
    assert "At 60 hours" in html and "predicted 0-hour frame" in html
    assert "not forecast accuracy" in html
    assert "preserved energy alone" in html.lower()
    assert "Review flag" not in html  # Constant mean flow retains most total energy.
    assert plt.get_fignums() == previous
    inputs["reference"] = object()
    inputs["initial_fields"] = object()
    assert view.flow_diagnostic_summary(inputs).data == html


def test_diagnostic_summary_prominently_flags_strong_decay_without_stopping():
    html = view.flow_diagnostic_summary(decaying_predictions()).data
    assert "Review flag: strong weakening" in html
    assert "1.0% total kinetic energy retained" in html
    assert "below 50%" in html
    assert "heuristic review flag, not a physics threshold" in html
    assert "not forecast accuracy" in html


def test_moderate_decay_is_not_flagged_as_crossing_the_review_threshold():
    html = view.flow_diagnostic_summary(decaying_predictions(.8)).data
    assert "64.0% total kinetic energy retained" in html
    assert "Review flag" not in html


def test_zero_flow_has_explicit_undefined_retention_and_finite_speed_plot():
    data = arrays()
    data["prediction"][:] = 0
    html = view.flow_diagnostic_summary(data).data
    assert "undefined (zero predicted initial energy)" in html
    assert "Review flag" not in html
    fig = view.plot_flow_diagnostics(data)
    try:
        np.testing.assert_array_equal(fig.axes[0].lines[0].get_ydata(), 0)
        assert not fig.axes[1].lines
        assert "undefined" in fig.axes[1].texts[0].get_text()
        fig.canvas.draw()
    finally:
        plt.close(fig)


@pytest.mark.parametrize("hours", [np.arange(10), np.full(11, np.nan), np.arange(11)[::-1], np.arange(11) + 1])
def test_diagnostic_hours_are_validated_before_a_figure_is_created(hours):
    data = arrays()
    data["times_hours"] = hours
    previous = plt.get_fignums()
    with pytest.raises(ValueError, match="hours"):
        view.plot_flow_diagnostics(data)
    assert plt.get_fignums() == previous


def test_diagnostic_plot_closes_its_figure_if_render_preparation_fails(monkeypatch):
    previous = plt.get_fignums()

    def fail(*args, **kwargs):
        raise RuntimeError("plot failed")

    monkeypatch.setattr(plt.Axes, "plot", fail)
    with pytest.raises(RuntimeError, match="plot failed"):
        view.plot_flow_diagnostics(arrays())
    assert plt.get_fignums() == previous


def test_notebook_derives_initial_flow_from_original_array_and_provides_complete_playback():
    notebook = json.loads(NOTEBOOK.read_text())
    sources = ["".join(cell["source"]) for cell in notebook["cells"]]
    code = [source for cell, source in zip(notebook["cells"], sources) if cell["cell_type"] == "code"]
    for source in code:
        ast.parse(source)
    all_source = "\n".join(sources)
    for forbidden in ("SMOKE_DATA", "--smoke-data", "Taylor", "flow_final.csv"):
        assert forbidden not in all_source
    assert "plot_initial_flow(initial_xy, initial_fields)" in all_source
    assert 'Video(str(LAB / "images/paraview.webm"), embed=True' in all_source
    assert "not the output of this run" in all_source
    video_index = next(index for index, source in enumerate(sources) if "display(Video(" in source)
    training_index = next(index for index, source in enumerate(sources) if "subprocess.run(" in source)
    assert video_index < training_index
    assert (NOTEBOOK.parent / "images/paraview.webm").is_file()
    assert "display(animate_flow(flow))" in all_source
    assert "export_paraview(OUTPUT)" in all_source
    assert 'export["archive"]' in all_source
    assert "flow.pvd" in all_source and "11 frames" in all_source
    assert "not a future reference" in all_source
    training = next(source for source in code if "subprocess.run(" in source)
    assert training.index("load_derived_flow(OUTPUT)") < training.index('RUN_COMPLETED["navier_stokes"] = True')
    assert 'os.environ.get("AI4SCI_STEPS", "3000")' in "\n".join(code)
    assert "3,000" in all_source and "2,048 samples per constraint" in all_source
    assert "--recipe upstream" in all_source and "50,000" in all_source
    assert "six 256-unit hidden layers" in all_source
    assert "L-BFGS" not in all_source
    assert "multiplied by 10" not in all_source
