"""The six-hour derived-input lesson keeps distinct labels and honest time units."""

import json
from pathlib import Path
import xml.etree.ElementTree as ET
from zipfile import ZipFile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from ETC.runtime import flow_visualization as view
from ETC.runtime.paraview import export_paraview


def save_derived(path, *, change=None, metadata=None):
    x, y = np.meshgrid(np.linspace(-.72, .54, 8), np.linspace(-.72, .48, 6), indexing="xy")
    xy = np.column_stack((x.ravel(), y.ravel())).astype(np.float32)
    initial = np.column_stack((1 + xy[:, 1], -.5 * xy[:, 0], 5.3 + .01 * xy[:, 0]))
    arrays = {"xy": xy, "initial_xy": xy, "initial_fields": initial,
              "times": np.linspace(0, .1, 11, dtype=np.float32),
              "times_hours": np.linspace(0, 6, 11, dtype=np.float32),
              "prediction": np.stack([initial + [i * .01, -i * .005, 0] for i in range(11)])}
    if change is not None:
        change(arrays)
    np.savez(path / "predictions.npz", **arrays)
    details = {"data_kind": view.DERIVED_DATA_KIND, "time_unit": "hours", "time_scale_hours": 60}
    details.update(metadata or {})
    (path / "metrics.json").write_text(json.dumps(details))
    return arrays


def test_derived_loader_retains_values_and_supplies_explicit_input_labels(tmp_path):
    saved = save_derived(tmp_path)
    loaded = view.load_derived_flow(tmp_path)
    for key, value in saved.items():
        np.testing.assert_array_equal(loaded[key], value)
    assert loaded["initial_label"] == "Derived incompressible input at 0 hours (fixed)"
    assert "Derived-input six-hour" in loaded["evolution_label"]
    with pytest.raises(ValueError, match="original-data"):
        view.load_original_flow(tmp_path)


@pytest.mark.parametrize("kind", [view.ORIGINAL_DATA_KIND, "synthetic_taylor_green", None])
def test_derived_loader_rejects_other_problems(tmp_path, kind):
    save_derived(tmp_path, metadata={"data_kind": kind})
    with pytest.raises(ValueError, match="derived-initial-condition"):
        view.load_derived_flow(tmp_path)


@pytest.mark.parametrize("change", [
    lambda a: a.pop("initial_fields"),
    lambda a: a.pop("times_hours"),
    lambda a: a.update(times=np.linspace(0, 1, 11)),
    lambda a: a.update(times_hours=np.linspace(0, 60, 11)),
    lambda a: a.update(times=np.linspace(0, .1, 10)),
    lambda a: a.update(prediction=a["prediction"][:-1]),
    lambda a: a.update(prediction=a["prediction"].astype(complex)),
    lambda a: a.update(prediction=np.full_like(a["prediction"], np.nan)),
    lambda a: a.update(initial_xy=a["initial_xy"][::-1]),
    lambda a: a.update(reference=a["prediction"]),
])
def test_invalid_or_ambiguous_derived_media_is_rejected(tmp_path, change):
    save_derived(tmp_path, change=change)
    figures = plt.get_fignums()
    with pytest.raises(ValueError):
        view.load_derived_flow(tmp_path)
    assert plt.get_fignums() == figures


def test_animation_labels_derived_input_and_updates_to_six_hours(tmp_path):
    save_derived(tmp_path)
    data = view.load_derived_flow(tmp_path)
    fig, animation = view.build_flow_animation(data)
    try:
        left, right = fig.axes[:2]
        initial_colors = left.collections[0].get_array().copy()
        scale = right.collections[0].get_clim()
        assert "Derived incompressible" in left.get_title()
        assert "Derived-input six-hour" in fig._suptitle.get_text()
        assert "Original" not in fig._suptitle.get_text()
        assert len(list(animation.new_frame_seq())) == 11
        animation._func(10)
        assert right.get_title() == "PINN prediction: 6 hours"
        np.testing.assert_array_equal(left.collections[0].get_array(), initial_colors)
        assert right.collections[0].get_clim() == scale == left.collections[0].get_clim()
        np.testing.assert_allclose(right.collections[0].get_array().ravel(),
                                   np.linalg.norm(data["prediction"][-1, :, :2], axis=1))
        fig.canvas.draw()
    finally:
        plt.close(fig)
    fig = view.plot_initial_flow(data["initial_xy"], data["initial_fields"],
                                 title="Derived incompressible initial input at 0 hours")
    try:
        assert fig.axes[0].get_title() == "Derived incompressible initial input at 0 hours"
    finally:
        plt.close(fig)


def test_derived_paraview_exports_six_hour_frames_with_pressure_provenance(tmp_path):
    arrays = save_derived(tmp_path)
    result = export_paraview(tmp_path)
    entries = ET.parse(result["pvd"]).findall("./Collection/DataSet")
    np.testing.assert_array_equal([float(e.attrib["timestep"]) for e in entries], arrays["times_hours"])
    readme = result["readme"].read_text()
    assert "Saved range: 0 to 6" in readme
    assert "Helmholtz-projected" in readme
    assert "derived incompressible pressure, not the original weather pressure" in readme
    assert "not a validated weather forecast" in readme
    assert "Original bootcamp data" not in readme
    with ZipFile(result["archive"]) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        assert manifest["data_kind"] == view.DERIVED_DATA_KIND
        assert manifest["time_unit"] == "hours"
        assert manifest["weather_forecast_validated"] is False
        assert manifest["initial_conditions"] == "periodic Helmholtz-projected wind"
        assert "recomputed incompressible pressure" in manifest["pressure_semantics"]
        assert len([name for name in archive.namelist() if name.endswith(".vti")]) == 11
    last = ET.parse(result["directory"] / entries[-1].attrib["file"])
    field_time = last.find("./ImageData/FieldData/DataArray[@Name='normalized_time']")
    assert float(field_time.text) == pytest.approx(.1)
    pressure = last.find("./ImageData/Piece/PointData/DataArray[@Name='p']")
    np.testing.assert_array_equal(np.fromstring(pressure.text, sep=" "), arrays["prediction"][-1, :, 2])


@pytest.mark.parametrize("metadata", [
    {"time_unit": "dimensionless"}, {"time_scale_hours": 6}, {"original_time_scale_hours": 6},
])
def test_contradictory_derived_time_metadata_is_rejected_by_both_consumers(tmp_path, metadata):
    save_derived(tmp_path, metadata=metadata)
    for consumer in (view.load_derived_flow, export_paraview):
        with pytest.raises(ValueError, match="time metadata"):
            consumer(tmp_path)
    assert not list(tmp_path.glob("paraview-*"))


@pytest.mark.parametrize("change", [
    lambda a: a.pop("times_hours"),
    lambda a: a.update(times=np.linspace(0, 1, 11)),
    lambda a: a.update(times_hours=np.linspace(0, 60, 11)),
])
def test_derived_paraview_never_infers_a_six_hour_scale_from_wrong_arrays(tmp_path, change):
    save_derived(tmp_path, change=change)
    with pytest.raises(ValueError, match="eleven explicit frames"):
        export_paraview(tmp_path)
    assert not list(tmp_path.glob("paraview-*"))
