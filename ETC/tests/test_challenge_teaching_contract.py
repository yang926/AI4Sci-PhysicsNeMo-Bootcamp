"""Teaching examples must agree with the submitted contract, without training."""
import ast
import json
from pathlib import Path
import re
import sys
from types import SimpleNamespace

import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "02_challenges/04_neural_operators"))
from ETC.runtime.notebook import result_html
import operator_training


def notebook(topic):
    paths = list((ROOT / "02_challenges" / topic).glob("Challenge_*.ipynb"))
    assert len(paths) == 1
    return json.loads(paths[0].read_text())


def markdown(topic):
    return ["".join(cell["source"]) for cell in notebook(topic)["cells"]
            if cell["cell_type"] == "markdown"]


@pytest.mark.parametrize("topic,levels", [
    ("01_wave", (1, 2, 3)), ("02_fluid", (1, 2, 3)),
    ("03_climate", (1, 2)), ("04_neural_operators", (1, 2, 3)),
])
def test_submission_defaults_and_unfinished_guidance(topic, levels):
    text = "\n".join(markdown(topic))
    assert "All Levels start selected." in text
    assert "Ctrl/Cmd-click to remove an unfinished Level" in text
    assert "Select only finished Levels" in text
    assert "Levels from different attempts are not combined" in text
    nodes = [ast.parse("".join(cell["source"])) for cell in notebook(topic)["cells"]
             if cell["cell_type"] == "code"]
    calls = [node for tree in nodes for node in ast.walk(tree)
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
             and node.func.id == "show_submission_controls"]
    assert len(calls) == 1
    assert ast.literal_eval(next(k.value for k in calls[0].keywords if k.arg == "levels")) == levels


@pytest.mark.parametrize("topic,count", [("01_wave", 3), ("02_fluid", 3), ("03_climate", 2)])
def test_each_symbolic_level_states_its_contract(topic, count):
    exercises = [cell for cell in markdown(topic) if "### Code and exercise" in cell]
    assert len(exercises) == count
    for cell in exercises:
        assert "do not replace" in cell
        assert "form and sign shown" in cell
        assert "exact dictionary keys" in cell
        if topic == "03_climate":
            assert "student_solution" in cell
            assert "independent" in cell






def test_quick_checks_keep_gpu_and_wave_step_mapping():
    for topic in ("01_wave", "02_fluid", "03_climate", "04_neural_operators"):
        text = "\n".join(markdown(topic))
        assert "keep the GPU" in text
        assert "rerun the setup cell before a real run" in text
        if topic == "01_wave":
            assert "STEPS = {1: 2, 2: 2, 3: 2}" in text
        else:
            assert "STEPS = 2" in text


def test_climate_and_fluid_explain_unchanged_baseline_limits():
    climate = "\n".join(markdown("03_climate"))
    assert "default `gamma0=0` run is uncoupled" in climate
    assert "late-time errors are still included in its numerator" in climate
    assert "Your conditions, coefficients, and geometry" not in climate
    fluid = "\n".join(markdown("02_fluid"))
    assert "rest initial state has $Q=0$" in fluid
    assert "not just the inlet corner" in fluid
    assert "no configurable condition-loss weights" in fluid
    assert "$2d$ spatial PDE weighting remains active" in fluid


def test_per_time_table_is_visible_and_escapes_bad_metadata():
    result = result_html({
        "reference_over_time": {"rmse": 0.1, "relative_l2": 0.2,
                                "per_time": [{"time": 0.0, "rmse": 0.125, "relative_l2": 0.25},
                                             {"time": 6.28, "rmse": 0.0002, "relative_l2": 12.5}]},
        "training_seconds": 1.25,
        "training_seconds_scope": "<script>bad</script>",
    }, "/tmp/run")
    summary = result.split("<details>")[0]
    assert "Analytical comparison at each evaluation time" in summary
    assert "6.28" in summary and "12.5" in summary and "0.0002" in summary
    assert "Training-loop time: 1.25 seconds" in summary
    assert "<script>" not in result and "&lt;script&gt;" in summary


def test_per_time_missing_values_are_not_presented_as_zero():
    result = result_html({"reference_over_time": {"per_time": [
        {"time": "<img onerror=bad>", "rmse": None, "relative_l2": float("nan")}, None,
    ]}}, "/tmp/run")
    summary = result.split("<details>")[0]
    assert "<img" not in summary
    assert "—" in summary and "non-finite" in summary


@pytest.mark.parametrize("kind", ("cpu", "cuda"))
def test_training_clock_synchronizes_cuda_before_reading_time(monkeypatch, kind):
    events = []
    device = torch.device(kind)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda target: events.append(("sync", target)))
    def now():
        events.append(("clock", None))
        return 12.5
    monkeypatch.setattr(operator_training.time, "perf_counter", now)
    assert operator_training.training_clock(device) == 12.5
    assert events == ([("sync", device)] if kind == "cuda" else []) + [("clock", None)]




def test_pino_mismatch_guides_equation_and_runtime_checks():
    field = torch.ones(1, 1, 4, 4)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(field, field))
    stats = {"f_mean": 0., "u_mean": 0., "f_std": 1., "u_std": 1.}
    physics = SimpleNamespace(forward=lambda fields: {
        "reaction_diffusion": torch.ones_like(fields["u"])})
    with pytest.raises(RuntimeError, match="ReactionDiffusionPDE") as error:
        operator_training.evaluate(torch.nn.Identity(), loader, "cpu", stats, physics)
    assert "u - u_xx - u_yy - f" in str(error.value)
    assert "numerical precision" in str(error.value)
    assert "does not prove the student PDE is wrong" in str(error.value)
