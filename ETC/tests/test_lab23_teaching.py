"""Real event logs, ParaView data and saved-model inference teaching contracts."""
import hashlib
import importlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
from zipfile import ZipFile

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "01_labs/02_projectile/source_code"))
bar = importlib.import_module("01_labs.03_heat_conduction.source_code.diffusion_bar")
projectile = importlib.import_module("projectile")
from ETC.runtime.lab_visualization import (export_bar, export_projectile, plot_bar_fields,
                                           plot_projectile_training, training_writer, write_vtp)

SMALL = {"steps": 3, "batch_size": 8, "learning_rate": .001, "num_layers": 1, "layer_size": 8}


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(min(2, previous))
    yield
    torch.set_num_threads(previous)


def checkpoint(tmp_path, parameterized=True):
    cfg = {**SMALL, "lab3_parameterized": parameterized, "lab3_dtype": "float32",
           "lab3_checkpoint_version": 1, "lab3_D1_range": [5., 25.] if parameterized else [10., 10.],
           "lab3_physics": {"D2": bar.D2, "TA": bar.TA, "TC": bar.TC}}
    torch.manual_seed(42)
    model = bar.CompositeBar(cfg, parameterized).float()
    physics = (bar.informer(bar.Diffusion("u_1", "D1"), "cpu"),
               bar.informer(bar.Diffusion("u_2", bar.D2), "cpu"),
               bar.informer(bar.DiffusionInterface(), "cpu"))
    bar.optimize_lab(model, physics, cfg, "cpu")
    path = tmp_path / "model.pt"
    torch.save({"config": cfg, "model_state_dict": model.state_dict()}, path)
    return path, model


def vtp_arrays(path):
    piece = ET.parse(path).find("./PolyData/Piece")
    arrays = {node.attrib["Name"]: np.fromstring(node.text, sep=" ").reshape(
        -1, int(node.attrib["NumberOfComponents"])) for node in piece.findall("./PointData/DataArray")}
    points = np.fromstring(piece.find("./Points/DataArray").text, sep=" ").reshape(-1, 3)
    return piece, arrays, points


def test_saved_model_infers_new_conductivity_without_optimizer(tmp_path, monkeypatch):
    path, model = checkpoint(tmp_path)
    original = path.read_bytes()
    wanted, _ = bar.material_fields(model, [7.5, 15., 22.5])

    def forbidden(*args, **kwargs):
        raise AssertionError("Inference must not train or create an optimizer")

    monkeypatch.setattr(bar, "optimize_lab", forbidden)
    monkeypatch.setattr(torch.optim, "Adam", forbidden)
    monkeypatch.setattr(torch.optim, "LBFGS", forbidden)
    output = tmp_path / "inference"
    metrics = bar.infer_checkpoint(path, output, [7.5, 15., 22.5])
    assert metrics["optimizer_steps"] == 0
    assert metrics["checkpoint_sha256"] == hashlib.sha256(original).hexdigest()
    assert path.read_bytes() == original
    with np.load(output / "material_fields.npz", allow_pickle=False) as saved:
        for name, values in wanted.items():
            np.testing.assert_array_equal(saved[name], values)
        assert saved["x"][200] == saved["x"][201] == 1
        assert saved["material"][200:202].tolist() == [1, 2]
        assert saved["temperature"].dtype == np.float32
    assert not (output / "model.pt").exists()
    assert not (output / "loss.csv").exists()
    with pytest.raises(FileExistsError):
        bar.infer_checkpoint(path, output, [10.])


@pytest.mark.parametrize("values", [[], [float("nan")], [4.9], [25.1], [True]])
def test_inference_rejects_unsupported_parameter_values(tmp_path, values):
    path, _ = checkpoint(tmp_path)
    with pytest.raises(ValueError, match="trained range"):
        bar.infer_checkpoint(path, tmp_path / "invalid", values)
    assert not (tmp_path / "invalid").exists()


def test_fixed_model_does_not_pretend_to_be_parameterized(tmp_path):
    path, _ = checkpoint(tmp_path, False)
    with pytest.raises(ValueError, match="trained range"):
        bar.infer_checkpoint(path, tmp_path / "invalid", [15.])


@pytest.mark.parametrize("corrupt", ["dtype", "physics", "range", "architecture", "nan"])
def test_checkpoint_contract_rejects_wrong_model(tmp_path, corrupt):
    path, _ = checkpoint(tmp_path)
    saved = torch.load(path, weights_only=True)
    if corrupt == "dtype":
        saved["model_state_dict"] = {k: v.double() for k, v in saved["model_state_dict"].items()}
    elif corrupt == "physics":
        saved["config"]["lab3_physics"]["D2"] = 100
    elif corrupt == "range":
        saved["config"]["lab3_D1_range"] = [0, 100]
    elif corrupt == "architecture":
        saved["config"]["layer_size"] = -1
    else:
        next(iter(saved["model_state_dict"].values())).fill_(float("nan"))
    torch.save(saved, path)
    with pytest.raises(ValueError):
        bar.load_checkpoint(path)


def test_bar_export_keeps_each_material_and_interface_error(tmp_path):
    path, model = checkpoint(tmp_path)
    output = tmp_path / "inference"
    bar.infer_checkpoint(path, output, [7.5, 15.])
    archive = export_bar(output)
    with np.load(output / "material_fields.npz") as saved:
        for i in range(2):
            for material in (1, 2):
                piece, arrays, points = vtp_arrays(archive.parent / f"case_{i:02d}_material_{material}.vtp")
                region = saved["material"] == material
                assert piece.attrib["NumberOfLines"] == "1"
                np.testing.assert_array_equal(arrays["temperature"][:, 0], saved["temperature"][i, region])
                np.testing.assert_array_equal(arrays["heat_flux"][:, 0], saved["heat_flux"][i, region])
                np.testing.assert_array_equal(points[:, 0], saved["x"][region])
    with ZipFile(archive) as bundle:
        assert len([name for name in bundle.namelist() if name.endswith(".vtp")]) == 4
        assert "same fixed color range" in bundle.read("README.txt").decode().lower()
    first = archive.read_bytes()
    assert export_bar(output) != archive
    assert archive.read_bytes() == first
    import matplotlib.pyplot as plt
    figure = plot_bar_fields(output)
    assert len(figure.axes) == 3
    figure.canvas.draw()
    plt.close(figure)


def test_projectile_captures_actual_batch_and_exports_validation(tmp_path):
    model = projectile.ProjectileModel(SMALL)
    physics = projectile.informer(projectile.ProjectileEquation(), "cpu",
                                  supplied_derivatives=("x__t__t", "y__t__t"))
    samples, rows = {}, []
    history = projectile.optimize_projectile(model, physics, SMALL, "cpu", record=rows.append,
                                             teaching_samples=samples)
    assert rows == history and samples["interior_t"].shape == (8, 1)
    assert (samples["interior_t"] >= 0).all() and (samples["interior_t"] <= 5).all()
    assert not samples["initial_t"].any()
    t = torch.linspace(0, 8, 401)[:, None]
    with torch.no_grad():
        prediction, reference = model(t).numpy(), projectile.analytical(t).numpy()
    np.savez(tmp_path / "predictions.npz", t=t.numpy(), prediction=prediction, reference=reference)
    np.savez(tmp_path / "training_points.npz", **samples)
    archive = export_projectile(tmp_path)
    piece, arrays, points = vtp_arrays(archive.parent / "interior_points.vtp")
    np.testing.assert_array_equal(points[:, 0], samples["interior_t"][:, 0])
    assert piece.attrib["NumberOfVerts"] == "8"
    _, validation, _ = vtp_arrays(archive.parent / "validation.vtp")
    assert validation["time_s"].max() <= 5
    region = t[:, 0].numpy() <= 5
    np.testing.assert_array_equal(validation["reference_xy_m"], reference[region])
    np.testing.assert_allclose(validation["position_error_m"][:, 0],
                               np.linalg.norm(prediction[region] - reference[region], axis=1))
    import matplotlib.pyplot as plt
    figure = plot_projectile_training(tmp_path)
    figure.canvas.draw()
    plt.close(figure)


def test_logging_and_point_capture_do_not_change_projectile_training(tmp_path):
    def run(record=None, capture=None):
        torch.manual_seed(42)
        model = projectile.ProjectileModel(SMALL)
        physics = projectile.informer(projectile.ProjectileEquation(), "cpu",
                                      supplied_derivatives=("x__t__t", "y__t__t"))
        history = projectile.optimize_projectile(model, physics, SMALL, "cpu", record=record,
                                                 teaching_samples=capture)
        return history, model.state_dict(), torch.get_rng_state()

    original_history, original_weights, original_rng = run()
    with training_writer(tmp_path / "recorded") as record:
        history, weights, rng = run(record, {})
    assert history == original_history
    assert torch.equal(rng, original_rng)
    assert all(torch.equal(value, original_weights[name]) for name, value in weights.items())


def test_tensorboard_events_match_actual_optimizer_calls(tmp_path):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    run = tmp_path / "run"
    rows = [{"step": 1, "loss": 2.5, "physics": .2, "phase": "adam"},
            {"step": 2, "loss": .5, "physics": .01, "phase": "lbfgs", "closure_evaluations": 7}]
    with training_writer(run) as record:
        for row in rows:
            record(row)
    events = EventAccumulator(str(tmp_path / "_tensorboard/run")).Reload()
    loss = events.Scalars("training/loss")
    assert [event.step for event in loss] == [1, 2]
    assert [event.value for event in loss] == [2.5, .5]
    assert events.Scalars("training/closure_evaluations")[0].value == 7
    with pytest.raises(FileExistsError):
        with training_writer(run):
            pass


@pytest.mark.parametrize("base", ["https://remote.example/", "//remote.example/", "relative/", "/a/../b/", "/?token=x"])
def test_tensorboard_link_rejects_nonlocal_base_url_before_starting(tmp_path, base):
    from ETC.runtime.lab_visualization import tensorboard_link
    with pytest.raises(ValueError, match="base_url"):
        tensorboard_link(tmp_path, base_url=base)


def test_tensorboard_link_reuses_listener_and_respects_jupyter_prefix(tmp_path, monkeypatch):
    from ETC.runtime import lab_visualization as visualization
    program = importlib.import_module("tensorboard.program")
    calls = []

    class Server:
        def configure(self, argv):
            calls.append(argv)

        def launch(self):
            return "http://127.0.0.1:6123/"

    monkeypatch.setattr(program, "TensorBoard", Server)
    first = visualization.tensorboard_link(tmp_path, base_url="/user/learner/").data
    second = visualization.tensorboard_link(tmp_path, base_url="/user/learner/").data
    assert first == second and 'href="/user/learner/proxy/6123/"' in first
    assert len(calls) == 1 and calls[0][-4:] == ["--host", "127.0.0.1", "--port", "0"]


def test_vtk_reader_accepts_both_curve_and_vertex_topologies(tmp_path):
    vtk = pytest.importorskip("vtk")
    for lines in (False, True):
        path = tmp_path / f"{lines}.vtp"
        write_vtp(path, np.column_stack((np.arange(4), np.zeros((4, 2)))), {"field": np.arange(4)}, lines=lines)
        reader = vtk.vtkXMLPolyDataReader()
        reader.SetFileName(str(path))
        reader.Update()
        data = reader.GetOutput()
        assert data.GetNumberOfPoints() == 4
        assert data.GetNumberOfLines() == int(lines)
        assert data.GetNumberOfVerts() == (0 if lines else 4)
