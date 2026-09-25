"""Check ParaView topology, time semantics, vector fields, and safe publication."""
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
from zipfile import ZipFile

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from ETC.runtime.paraview import export_paraview


def save_run(path, *, kind="upstream_data_lat_legacy_normalization", **overrides):
    xx, yy = np.meshgrid(np.linspace(-.72, .72, 4), np.linspace(-.5, .5, 3), indexing="xy")
    xy = np.column_stack((xx.ravel(), yy.ravel()))
    times = np.array([0, .5, 1.], dtype=np.float32)
    prediction = np.stack([np.column_stack((xy[:, 0] + t, 2 * xy[:, 1] - t,
                                            3 * xy[:, 0] - xy[:, 1])) for t in times])
    arrays = {"xy": xy, "times": times, "prediction": prediction}
    arrays.update(overrides)
    np.savez(path / "predictions.npz", **arrays)
    (path / "metrics.json").write_text(json.dumps({"data_kind": kind}))
    return arrays


def arrays_from_vti(path):
    return {node.attrib["Name"]: np.fromstring(node.text, sep=" ").reshape(
        -1, int(node.attrib["NumberOfComponents"]))
        for node in ET.parse(path).findall("./ImageData/Piece/PointData/DataArray")}


def test_full_series_has_connected_grid_and_correct_vectors(tmp_path):
    original = save_run(tmp_path)
    result = export_paraview(tmp_path)
    entries = ET.parse(result["pvd"]).findall("./Collection/DataSet")
    assert [float(node.attrib["timestep"]) for node in entries] == [0, 30, 60]
    for i, entry in enumerate(entries):
        path = result["directory"] / entry.attrib["file"]
        grid = ET.parse(path).find("ImageData")
        assert grid.attrib["WholeExtent"] == "0 3 0 2 0 0"
        assert grid.find("Piece").attrib["Extent"] == "0 3 0 2 0 0"
        point_data = grid.find("./Piece/PointData")
        assert point_data.attrib == {"Scalars": "speed", "Vectors": "velocity"}
        arrays = arrays_from_vti(path)
        for j, name in enumerate(("u", "v", "p")):
            np.testing.assert_array_equal(arrays[name][:, 0], original["prediction"][i, :, j])
        np.testing.assert_array_equal(arrays["velocity"][:, :2], original["prediction"][i, :, :2])
        assert not arrays["velocity"][:, 2].any()
        np.testing.assert_allclose(arrays["speed"][:, 0], np.linalg.norm(arrays["velocity"], axis=1))
    readme = result["readme"].read_text()
    assert "Time unit: hours" in readme and "not a validated weather forecast" in readme
    script = result["viewer_script"].read_text()
    compile(script, "view_flow.py", "exec")
    assert 'AutomaticRescaleRangeMode = "Never"' in script
    assert 'arrows.OrientationArray = ["POINTS", "velocity"]' in script
    assert 'CameraViewUp = [0, 1, 0]' in script
    with ZipFile(result["archive"]) as bundle:
        names = bundle.namelist()
        assert set(names) == {"flow.pvd", "view_flow.py", "README.txt", "manifest.json",
                              "flow_0000.vti", "flow_0001.vti", "flow_0002.vti"}
        assert all(not Path(name).is_absolute() and ".." not in Path(name).parts for name in names)
        for entry in entries:
            assert entry.attrib["file"] in names


def test_exact_hours_are_used_and_old_outputs_remain_supported(tmp_path):
    times = np.linspace(0, 1, 11, dtype=np.float32)
    prediction = np.zeros((11, 12, 3))
    save_run(tmp_path, times=times, times_hours=np.linspace(0, 60, 11), prediction=prediction)
    result = export_paraview(tmp_path)
    entries = ET.parse(result["pvd"]).findall("./Collection/DataSet")
    assert [float(node.attrib["timestep"]) for node in entries] == list(range(0, 61, 6))


def test_reorders_shuffled_grid_without_scrambling_fields(tmp_path):
    original = save_run(tmp_path)
    order = np.random.default_rng(4).permutation(len(original["xy"]))
    save_run(tmp_path, xy=original["xy"][order], prediction=original["prediction"][:, order])
    result = export_paraview(tmp_path)
    arrays = arrays_from_vti(result["directory"] / "flow_0001.vti")
    np.testing.assert_array_equal(arrays["velocity"][:, :2], original["prediction"][1, :, :2])


def test_synthetic_time_is_not_mislabeled_as_weather_hours(tmp_path):
    save_run(tmp_path, kind="synthetic_taylor_green")
    result = export_paraview(tmp_path)
    entries = ET.parse(result["pvd"]).findall("./Collection/DataSet")
    assert [float(node.attrib["timestep"]) for node in entries] == [0, .5, 1]
    assert "Time unit: dimensionless" in result["readme"].read_text()


@pytest.mark.parametrize("invalid", [
    {"times": np.array([0, 1, .5])}, {"times": np.array([0, .5, .5])},
    {"times": np.array([0, .5, 2])}, {"times": np.array([0, np.nan, 1])},
    {"prediction": np.zeros((3, 12, 2))}, {"prediction": np.full((3, 12, 3), np.inf)},
    {"xy": np.zeros((12, 2))}, {"xy": np.ones((12, 3))},
    {"times_hours": np.array([0, 6, 12])},
    {"prediction": np.zeros((3, 12, 3), dtype=complex)},
])
def test_bad_arrays_rejected_before_any_export_is_created(tmp_path, invalid):
    save_run(tmp_path, **invalid)
    with pytest.raises(ValueError):
        export_paraview(tmp_path)
    assert not list(tmp_path.glob("paraview-*"))
    assert not list(tmp_path.glob(".paraview-*"))


@pytest.mark.parametrize("irregular", [False, True])
def test_duplicate_or_nonuniform_coordinates_rejected(tmp_path, irregular):
    data = save_run(tmp_path)
    xy = data["xy"].copy()
    if irregular:
        xy[xy[:, 0] == xy[1, 0], 0] += .01
    else:
        xy[0] = xy[1]
    save_run(tmp_path, xy=xy)
    with pytest.raises(ValueError, match="grid|Cartesian|uniform"):
        export_paraview(tmp_path)
    assert not list(tmp_path.glob("paraview-*"))


def test_unknown_kind_rejected_and_reexports_preserve_earlier_files(tmp_path):
    save_run(tmp_path, kind="unknown")
    with pytest.raises(ValueError, match="data_kind"):
        export_paraview(tmp_path)
    save_run(tmp_path)
    first = export_paraview(tmp_path)
    original_zip = first["archive"].read_bytes()
    first["readme"].write_text("User notes to keep")
    second = export_paraview(tmp_path)
    assert first["directory"] != second["directory"]
    assert first["readme"].read_text() == "User notes to keep"
    assert first["archive"].read_bytes() == original_zip


def test_vtk_reader_round_trip_when_available(tmp_path):
    vtk = pytest.importorskip("vtk")
    original = save_run(tmp_path)
    result = export_paraview(tmp_path)
    reader = vtk.vtkXMLImageDataReader()
    reader.SetFileName(str(result["directory"] / "flow_0001.vti"))
    reader.Update()
    image = reader.GetOutput()
    assert image.GetDimensions() == (4, 3, 1)
    assert image.GetNumberOfPoints() == 12
    assert image.GetNumberOfCells() == 6  # Connected quads, not a cloud of vertices.
    for i, xy in enumerate(original["xy"]):
        np.testing.assert_allclose(image.GetPoint(i), [*xy, 0])
        np.testing.assert_allclose(image.GetPointData().GetVectors().GetTuple(i),
                                   [*original["prediction"][1, i, :2], 0])
