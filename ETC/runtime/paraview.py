"""Export saved Lab 4 predictions for ParaView, without a VTK dependency."""
import json
import math
from pathlib import Path
import uuid
import xml.etree.ElementTree as ET
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np

from ETC.runtime.artifacts import staged_output


def _numeric(value, name):
    array = np.asarray(value)
    if array.dtype.kind not in "fiu" or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite real numbers")
    return array.astype(np.float64)


def _load(run_dir):
    with np.load(run_dir / "predictions.npz", allow_pickle=False) as saved:
        xy = _numeric(saved["xy"], "xy")
        times = _numeric(saved["times"], "times")
        prediction = _numeric(saved["prediction"], "prediction")
        hours = _numeric(saved["times_hours"], "times_hours") if "times_hours" in saved else None
    if xy.ndim != 2 or xy.shape[1] != 2 or len(xy) < 4:
        raise ValueError("xy must be a nonempty two-dimensional Cartesian grid with shape (N, 2)")
    if (times.ndim != 1 or not len(times) or np.any(np.diff(times) <= 0)
            or times.min() < 0 or times.max() > 1):
        raise ValueError("times must be strictly increasing normalized values in [0, 1]")
    if prediction.shape != (len(times), len(xy), 3):
        raise ValueError("prediction must have shape (number of times, number of points, 3) for u, v, p")
    xs, ys = np.unique(xy[:, 0]), np.unique(xy[:, 1])
    if len(xs) < 2 or len(ys) < 2 or len(xs) * len(ys) != len(xy):
        raise ValueError("xy must contain a complete rectangular Cartesian grid")
    order = np.lexsort((xy[:, 0], xy[:, 1]))  # VTK: x varies fastest, then y.
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    expected = np.column_stack((xx.ravel(), yy.ravel()))
    if not np.array_equal(xy[order], expected):
        raise ValueError("xy contains missing or duplicate Cartesian grid points")
    for axis in (xs, ys):
        spacing = (axis[-1] - axis[0]) / (len(axis) - 1)
        if not np.allclose(np.diff(axis), spacing, rtol=2e-5, atol=1e-8 * abs(spacing)):
            raise ValueError("VTI export requires uniformly spaced Cartesian coordinates")
    metadata = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    kind = metadata.get("data_kind")
    if kind == "upstream_data_lat_legacy_normalization":
        if hours is not None:
            if hours.shape != times.shape or not np.allclose(hours, times * 60, rtol=1e-6, atol=1e-6):
                raise ValueError("times_hours must match normalized times multiplied by 60")
            if np.any(np.diff(hours) <= 0):
                raise ValueError("times_hours must be strictly increasing")
        else:
            hours = times * 60
        exported_times, unit = hours, "hours"
    elif kind == "synthetic_taylor_green":
        if hours is not None:
            raise ValueError("Synthetic Taylor-Green times are dimensionless, not hours")
        exported_times, unit = times, "dimensionless"
    else:
        raise ValueError(f"Unknown Lab 4 data_kind: {kind!r}; cannot infer time units safely")
    return xs, ys, times, exported_times, unit, kind, prediction[:, order, :]


def _array(parent, name, values, components=1):
    element = ET.SubElement(parent, "DataArray", type="Float64", Name=name,
                            NumberOfComponents=str(components), format="ascii")
    element.text = " ".join(format(float(value), ".17g") for value in np.asarray(values).ravel())


def _write_xml(path, root):
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def _write_frame(path, xs, ys, field, time, normalized_time):
    extent = f"0 {len(xs) - 1} 0 {len(ys) - 1} 0 0"
    root = ET.Element("VTKFile", type="ImageData", version="0.1", byte_order="LittleEndian")
    grid = ET.SubElement(root, "ImageData", WholeExtent=extent,
                         Origin=f"{xs[0]:.17g} {ys[0]:.17g} 0",
                         Spacing=f"{(xs[-1] - xs[0]) / (len(xs) - 1):.17g} "
                                 f"{(ys[-1] - ys[0]) / (len(ys) - 1):.17g} 1")
    field_data = ET.SubElement(grid, "FieldData")
    for name, value in (("TimeValue", time), ("normalized_time", normalized_time)):
        _array(field_data, name, [value])
        field_data[-1].set("NumberOfTuples", "1")
    piece = ET.SubElement(grid, "Piece", Extent=extent)
    point_data = ET.SubElement(piece, "PointData", Scalars="speed", Vectors="velocity")
    for index, name in enumerate(("u", "v", "p")):
        _array(point_data, name, field[:, index])
    _array(point_data, "velocity", np.column_stack((field[:, :2], np.zeros(len(field)))), 3)
    _array(point_data, "speed", np.hypot(field[:, 0], field[:, 1]))
    ET.SubElement(piece, "CellData")
    _write_xml(path, root)


def _viewer_script(xs, ys, speed_range):
    """The optional script runs inside ParaView's Python Shell, not Jupyter."""
    low, high = speed_range
    if low == high:
        high = low + max(1.0, abs(low)) * 1e-6
    span = max(float(np.ptp(xs)), float(np.ptp(ys)))
    center = [float((xs[0] + xs[-1]) / 2), float((ys[0] + ys[-1]) / 2), 0.0]
    return f'''"""Load this script in ParaView's Python Shell with runpy.run_path(...)."""
from pathlib import Path
from paraview.simple import (PVDReader, GetActiveViewOrCreate, Show, ColorBy,
                             GetColorTransferFunction, Glyph, GetAnimationScene,
                             SetActiveSource, Render)

reader = PVDReader(FileName=str(Path(__file__).resolve().with_name("flow.pvd")))
reader.UpdatePipeline()
view = GetActiveViewOrCreate("RenderView")
surface = Show(reader, view)
surface.Representation = "Surface"
ColorBy(surface, ("POINTS", "speed"))
color_map = GetColorTransferFunction("speed")
color_map.RescaleTransferFunction({low!r}, {high!r})
color_map.AutomaticRescaleRangeMode = "Never"
surface.SetScalarBarVisibility(view, True)

arrows = Glyph(Input=reader, GlyphType="Arrow")
arrows.OrientationArray = ["POINTS", "velocity"]
arrows.ScaleArray = ["POINTS", "velocity"]
arrows.VectorScaleMode = "Scale by Magnitude"
arrows.GlyphMode = "Every Nth Point"
arrows.Stride = {max(1, math.ceil(len(xs) * len(ys) / 350))}
arrows.ScaleFactor = {0.08 * span / max(high, 1e-12)!r}
arrow_display = Show(arrows, view)
ColorBy(arrow_display, None)
arrow_display.DiffuseColor = [0.12, 0.12, 0.12]

scene = GetAnimationScene()
scene.UpdateAnimationUsingDataTimeSteps()
scene.GoToFirst()
view.CameraParallelProjection = 1
view.CameraFocalPoint = {center!r}
view.CameraPosition = {[center[0], center[1], 3 * span]!r}
view.CameraViewUp = [0, 1, 0]
view.CameraParallelScale = {0.6 * span!r}
SetActiveSource(reader)
Render()
print("Use Play to animate. Time units and normalized fields are explained in README.txt.")
'''


def export_paraview(run_dir):
    """Export every saved time into a new, self-contained ParaView ZIP bundle.

    No training, dependency installation, or existing output replacement occurs.
    Return paths named directory, pvd, archive, readme, and viewer_script.
    """
    run_dir = Path(run_dir).expanduser().resolve(strict=True)
    xs, ys, times, exported_times, unit, kind, prediction = _load(run_dir)
    speeds = np.hypot(prediction[..., 0], prediction[..., 1])
    speed_range = [float(speeds.min()), float(speeds.max())]
    if not np.isfinite(speeds).all():
        raise ValueError("Velocity magnitudes must be finite")
    destination = run_dir / f"paraview-{uuid.uuid4().hex[:12]}"
    with staged_output(destination) as staging:
        collection_root = ET.Element("VTKFile", type="Collection", version="0.1",
                                     byte_order="LittleEndian")
        collection = ET.SubElement(collection_root, "Collection")
        for i, (normalized_time, time, field) in enumerate(zip(times, exported_times, prediction)):
            filename = f"flow_{i:04d}.vti"
            _write_frame(staging / filename, xs, ys, field, time, normalized_time)
            ET.SubElement(collection, "DataSet", timestep=format(float(time), ".17g"),
                          group="", part="0", file=filename)
        _write_xml(staging / "flow.pvd", collection_root)
        (staging / "view_flow.py").write_text(_viewer_script(xs, ys, speed_range), encoding="utf-8")
        description = ("Original bootcamp data, described upstream as ERA5-derived. "
                       "The original 60-hour scale is retained; this is not a validated weather forecast."
                       if unit == "hours" else
                       "Synthetic Taylor-Green validation case, not the original weather-data exercise.")
        (staging / "README.txt").write_text(
            "Lab 4: ParaView time series\n\n"
            "1. Download and extract flow_paraview.zip on your computer.\n"
            "2. In ParaView, File > Open > flow.pvd > Apply. Keep the VTI files beside it.\n"
            "3. Select Surface, color by speed, and view from +Z. Press Play to animate.\n"
            "4. For arrows and a fixed color range: View > Python Shell, then run:\n"
            '   import runpy; runpy.run_path("/full/path/to/view_flow.py", run_name="__main__")\n\n'
            f"Time unit: {unit}. Saved range: {exported_times[0]:g} to {exported_times[-1]:g}.\n"
            "Coordinates and u, v, p are the model's normalized values.\n"
            "velocity = (u, v, 0); speed = sqrt(u*u + v*v).\n"
            "Do not label these arrays m/s or Pa: legacy pressure conversion remains unvalidated.\n"
            f"{description}\n\n"
            "The script fixes the speed color range over ALL saved frames, so colors are comparable.\n"
            "Arrows show the instantaneous velocity direction and relative magnitude, not particle tracks.\n"
            "Changing data in ParaView does not retrain the neural network.\n", encoding="utf-8")
        (staging / "manifest.json").write_text(json.dumps({
            "data_kind": kind, "time_unit": unit, "times": exported_times.tolist(),
            "normalized_times": times.tolist(), "field_units": "normalized",
            "grid_dimensions": [len(xs), len(ys), 1], "speed_range": speed_range,
        }, indent=2) + "\n", encoding="utf-8")
        with ZipFile(staging / "flow_paraview.zip", "w", compression=ZIP_DEFLATED) as archive:
            for path in sorted(staging.iterdir()):
                if path.suffix != ".zip":
                    archive.write(path, arcname=path.name)
    return {"directory": destination, "pvd": destination / "flow.pvd",
            "archive": destination / "flow_paraview.zip", "readme": destination / "README.txt",
            "viewer_script": destination / "view_flow.py"}
