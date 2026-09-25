"""Saved-run teaching views for Labs 2 and 3; no training or VTK dependency."""
from contextlib import contextmanager
import json
import math
from pathlib import Path
import uuid
import xml.etree.ElementTree as ET
from urllib.parse import urlsplit
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np

from ETC.runtime.artifacts import staged_output


@contextmanager
def training_writer(run_dir):
    """Stream real optimizer-call losses; keep logs even if training fails."""
    from torch.utils.tensorboard import SummaryWriter

    run_dir = Path(run_dir)
    log_dir = run_dir.parent / "_tensorboard" / run_dir.name
    if log_dir.exists():
        raise FileExistsError(f"TensorBoard logs already exist: {log_dir}")
    with SummaryWriter(log_dir=str(log_dir), flush_secs=5) as writer:
        def record(row):
            step = row["step"]
            for name, value in row.items():
                if name != "step" and isinstance(value, (int, float)) and math.isfinite(value):
                    writer.add_scalar(f"training/{name}", value, step)
        yield record


_TENSORBOARDS = {}


def tensorboard_link(log_dir, *, base_url=None):
    """Start TensorBoard only when explicitly requested, bound to loopback.

    The returned browser URL goes through the existing authenticated Jupyter
    server, not a public raw port. jupyter-server-proxy must be enabled there.
    """
    import html
    import os
    from IPython.display import HTML
    from jupyter_server.serverapp import list_running_servers

    log_dir = Path(log_dir).resolve(strict=True)
    if base_url is None:
        base_url = os.environ.get("JUPYTERHUB_SERVICE_PREFIX")
        if not base_url:
            bases = {server.get("base_url", "/") for server in list_running_servers()}
            if len(bases) != 1:
                raise RuntimeError("Cannot identify the Jupyter base URL; pass base_url explicitly.")
            base_url = bases.pop()
    parsed = urlsplit(base_url)
    if (parsed.scheme or parsed.netloc or parsed.query or parsed.fragment
            or not base_url.startswith("/") or ".." in base_url.split("/")):
        raise ValueError("base_url must be this Jupyter server's absolute URL path")
    if log_dir not in _TENSORBOARDS:
        from tensorboard import program
        server = program.TensorBoard()
        server.configure(argv=["tensorboard", "--logdir", str(log_dir), "--load_fast=false",
                               "--host", "127.0.0.1", "--port", "0"])
        local_url = server.launch()
        _TENSORBOARDS[log_dir] = (server, urlsplit(local_url).port)
    port = _TENSORBOARDS[log_dir][1]
    browser_url = f"{base_url.rstrip('/')}/proxy/{port}/"
    return HTML(f'<a href="{html.escape(browser_url, quote=True)}" target="_blank" rel="noopener">'
                'Open TensorBoard through Jupyter</a>')


def _values(value, shape=None):
    result = np.asarray(value)
    if result.dtype.kind not in "fiu" or not result.size or not np.isfinite(result).all():
        raise ValueError("Visualization arrays must contain finite real values")
    if shape is not None and result.shape != shape:
        raise ValueError(f"Expected shape {shape}, got {result.shape}")
    return result


def write_vtp(path, xyz, fields, *, lines=False):
    """Write connected curves or vertex sets using standard VTK PolyData XML."""
    xyz = _values(xyz)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("Point coordinates must have shape (N, 3)")
    count = len(xyz)
    checked = {name: _values(value).reshape(count, -1) for name, value in fields.items()}
    root = ET.Element("VTKFile", type="PolyData", version="0.1", byte_order="LittleEndian")
    poly = ET.SubElement(root, "PolyData")
    piece = ET.SubElement(poly, "Piece", NumberOfPoints=str(count),
                          NumberOfVerts="0" if lines else str(count),
                          NumberOfLines="1" if lines else "0", NumberOfStrips="0", NumberOfPolys="0")

    def array(parent, name, values, dtype="Float64"):
        values = np.asarray(values)
        node = ET.SubElement(parent, "DataArray", type=dtype, Name=name,
                             NumberOfComponents=str(values.shape[1] if values.ndim > 1 else 1),
                             format="ascii")
        node.text = " ".join(format(float(item), ".17g") if dtype == "Float64" else str(int(item))
                             for item in values.ravel())

    point_data = ET.SubElement(piece, "PointData")
    for name, values in checked.items():
        array(point_data, name, values)
    ET.SubElement(piece, "CellData")
    array(ET.SubElement(piece, "Points"), "Points", xyz)
    topology = ET.SubElement(piece, "Lines" if lines else "Verts")
    array(topology, "connectivity", np.arange(count), "Int64")
    array(topology, "offsets", [count] if lines else np.arange(1, count + 1), "Int64")
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def _archive(staging, name):
    with ZipFile(staging / name, "w", compression=ZIP_DEFLATED) as archive:
        for path in sorted(staging.iterdir()):
            if path.suffix != ".zip":
                archive.write(path, arcname=path.name)


def export_projectile(run_dir):
    """Export actual last training points and held-out trajectory comparisons."""
    run_dir = Path(run_dir).resolve(strict=True)
    with np.load(run_dir / "predictions.npz", allow_pickle=False) as saved:
        t = _values(saved["t"]).reshape(-1)
        prediction = _values(saved["prediction"], (len(t), 2))
        reference = _values(saved["reference"], prediction.shape)
    with np.load(run_dir / "training_points.npz", allow_pickle=False) as saved:
        interior_t = _values(saved["interior_t"]).reshape(-1)
        initial_t = _values(saved["initial_t"]).reshape(-1)
    target = run_dir / f"paraview-{uuid.uuid4().hex[:12]}"
    with staged_output(target) as staging:
        for name, points in (("interior", interior_t), ("initial", initial_t)):
            # Time is the PINN input domain, not a measured physical trajectory.
            write_vtp(staging / f"{name}_points.vtp", np.column_stack((points, np.zeros((len(points), 2)))),
                      {"time_s": points})
        xyz = np.column_stack((prediction, np.zeros(len(t))))
        errors = np.linalg.norm(prediction - reference, axis=1)
        in_domain = t <= 5
        write_vtp(staging / "prediction.vtp", xyz,
                  {"time_s": t, "in_training_interval": in_domain.astype(int), "position_error_m": errors}, lines=True)
        write_vtp(staging / "reference.vtp", np.column_stack((reference, np.zeros(len(t)))),
                  {"time_s": t}, lines=True)
        # Validator uses the same coordinates for reference/prediction arrays.
        write_vtp(staging / "validation.vtp", xyz[in_domain],
                  {"time_s": t[in_domain], "predicted_xy_m": prediction[in_domain],
                   "reference_xy_m": reference[in_domain], "position_error_m": errors[in_domain]}, lines=True)
        (staging / "README.txt").write_text(
            "Lab 2: projectile teaching views\n\n"
            "1. File > Open > prediction.vtp and reference.vtp > Apply. View from +Z.\n"
            "   Both are connected physical trajectories; use different solid colors.\n"
            "   Color prediction by time_s or position_error_m to inspect errors.\n"
            "2. Open validation.vtp. Spreadsheet View lists held-out reference and prediction\n"
            "   values for 0 <= t <= 5 s, separate from the training collocation points.\n"
            "3. In a separate view open initial_points.vtp and interior_points.vtp.\n"
            "   Their X coordinate is TIME IN SECONDS, not projectile position. Use Points.\n"
            "   These are the actual last optimizer batch, not invented training samples.\n"
            "4. Prediction beyond 5 s is extrapolation, not guaranteed by training.\n",
            encoding="utf-8")
        _archive(staging, "projectile_paraview.zip")
    return target / "projectile_paraview.zip"


def plot_projectile_training(run_dir):
    """Compare the physical trajectory, held-out errors and actual input points."""
    import matplotlib.pyplot as plt

    with np.load(Path(run_dir) / "predictions.npz", allow_pickle=False) as saved:
        t, prediction, reference = saved["t"].ravel(), saved["prediction"], saved["reference"]
    with np.load(Path(run_dir) / "training_points.npz", allow_pickle=False) as saved:
        interior, initial = saved["interior_t"].ravel(), saved["initial_t"].ravel()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), layout="constrained")
    axes[0].plot(reference[:, 0], reference[:, 1], "--", label="Analytical")
    axes[0].plot(prediction[t <= 5, 0], prediction[t <= 5, 1], label="PINN: 0–5 s")
    axes[0].plot(prediction[t > 5, 0], prediction[t > 5, 1], ":", label="PINN: extrapolation")
    axes[0].set(xlabel="x (m)", ylabel="y (m)", title="Physical trajectory")
    axes[0].legend()
    axes[1].plot(t, np.linalg.norm(prediction - reference, axis=1))
    axes[1].axvline(5, color="grey", linestyle="--")
    axes[1].set(xlabel="time (s)", ylabel="Position error (m)", title="Held-out error")
    axes[2].scatter(interior, np.ones_like(interior), s=9)
    axes[2].scatter(initial, np.zeros_like(initial), s=25)
    axes[2].set(xlabel="PINN input: time (s)", yticks=[0, 1],
                yticklabels=["Initial condition", "ODE residual"], ylim=(-.5, 1.5),
                title="Actual last training batch")
    return fig


def export_bar(run_dir):
    """Keep both interface points, both materials and physical heat flux."""
    run_dir = Path(run_dir).resolve(strict=True)
    with np.load(run_dir / "material_fields.npz", allow_pickle=False) as saved:
        data = {key: _values(saved[key]) for key in saved.files}
    x = data["x"].reshape(-1)
    material = data["material"].reshape(-1)
    conductivities = data["D1"].reshape(-1)
    if set(np.unique(material)) != {1, 2}:
        raise ValueError("Both material fields are required")
    for key in ("temperature", "reference", "heat_flux", "reference_heat_flux"):
        _values(data[key], (len(conductivities), len(x)))
    target = run_dir / f"paraview-{uuid.uuid4().hex[:12]}"
    with staged_output(target) as staging:
        entries = []
        for index, d1 in enumerate(conductivities):
            for number in (1, 2):
                mask = material == number
                filename = f"case_{index:02d}_material_{number}.vtp"
                write_vtp(staging / filename, np.column_stack((x[mask], np.zeros((mask.sum(), 2)))),
                          {"temperature": data["temperature"][index, mask],
                           "reference_temperature": data["reference"][index, mask],
                           "temperature_error": data["temperature"][index, mask] - data["reference"][index, mask],
                           "heat_flux": data["heat_flux"][index, mask],
                           "reference_heat_flux": data["reference_heat_flux"][index, mask],
                           "D1": np.full(mask.sum(), d1)}, lines=True)
                entries.append({"D1": float(d1), "material": number, "file": filename})
        manifest = {"cases": entries, "physical_heat_flux": "q = -D * dT/dx",
                    "temperature_range": [float(min(data["temperature"].min(), data["reference"].min())),
                                          float(max(data["temperature"].max(), data["reference"].max()))]}
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        (staging / "README.txt").write_text(
            "Lab 3: two-material temperature and heat flux\n\n"
            "1. Open both material_1.vtp and material_2.vtp for one case > Apply.\n"
            "2. Color both by temperature. Use the SAME fixed color range for both,\n"
            "   covering the temperature_range in manifest.json (normally about 0 to 100).\n"
            "3. Apply Plot Data to each curve; compare temperature against reference_temperature.\n"
            "   Repeat for heat_flux and reference_heat_flux. Heat flux is -D*dT/dx.\n"
            "4. Both curves include x=1. Any jump is a real network error; the exporter\n"
            "   does not join, average, or replace the two interface values.\n"
            "5. Change D1 using the notebook's saved-model inference cell, not by editing VTK arrays.\n",
            encoding="utf-8")
        _archive(staging, "heat_conduction_paraview.zip")
    return target / "heat_conduction_paraview.zip"


def plot_bar_fields(run_dir):
    """Use distinct material lines so interface jumps remain visible."""
    import matplotlib.pyplot as plt

    with np.load(Path(run_dir) / "material_fields.npz", allow_pickle=False) as saved:
        data = {key: saved[key] for key in saved.files}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), layout="constrained")
    for i, d1 in enumerate(data["D1"]):
        for material in (1, 2):
            mask = data["material"] == material
            x = data["x"][mask]
            label = f"D1={d1:g}" if material == 1 else None
            for ax, field, exact in ((axes[0], "temperature", "reference"),
                                     (axes[1], "temperature", "reference"),
                                     (axes[2], "heat_flux", "reference_heat_flux")):
                ax.plot(x, data[exact][i, mask], "--", color=f"C{i}", linewidth=2)
                ax.plot(x, data[field][i, mask], color=f"C{i}", label=label)
    axes[0].set(title="Temperature: both materials", xlim=(0, 2), ylabel="Temperature")
    axes[1].set(title="Left material (zoom)", xlim=(0, 1), ylabel="Temperature")
    left = data["material"] == 1
    low = min(data["temperature"][:, left].min(), data["reference"][:, left].min())
    high = max(data["temperature"][:, left].max(), data["reference"][:, left].max())
    padding = max((high - low) * .05, 1e-4)
    axes[1].set_ylim(low - padding, high + padding)
    axes[2].set(title="Physical heat flux", xlim=(0, 2), ylabel="q = -D dT/dx")
    for ax in axes:
        ax.set_xlabel("x")
        ax.axvline(1, color="grey", alpha=.4)
        ax.grid(alpha=.2)
        ax.legend()
    fig.suptitle("PINN: solid; analytical reference: dashed")
    return fig
