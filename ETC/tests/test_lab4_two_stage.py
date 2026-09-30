"""Keep the derived six-hour workflow distinct from legacy or instructor runs."""
import ast
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import uuid

import pytest

ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = ROOT / "ETC/reference_labs/04_navier_stokes/Lab_4_Navier_Stokes.ipynb"


def source(cell_id):
    notebook = json.loads(NOTEBOOK.read_text())
    return "".join(next(cell for cell in notebook["cells"] if cell["id"] == cell_id)["source"])


@pytest.mark.parametrize("keys", [(), ("navier_stokes",), ("instructor_inference",),
                                 ("navier_stokes", "instructor_inference")])
def test_paraview_exports_only_completed_student_result(tmp_path, monkeypatch, capsys, keys):
    from ETC.runtime import paraview
    paths = {key: tmp_path / key for key in keys}
    exported = []
    validated = []

    def export(path):
        assert validated == [path]
        exported.append(path)
        return {"archive": path / "flow_paraview.zip"}

    monkeypatch.setattr(paraview, "export_paraview", export)
    ns = dict(RUN_COMPLETED={key: True for key in keys}, RUN_DIRS=paths,
              completed_output=lambda runs, done, key: runs[key],
              load_derived_flow=lambda path: validated.append(path), display=lambda value: None,
              FileLink=lambda value: value, os=os, Path=Path)
    exec(source("05f915f71865"), ns)
    expected = [paths["navier_stokes"]] if "navier_stokes" in keys else []
    assert validated == expected
    assert exported == expected
    output = capsys.readouterr().out
    assert "Instructor model" not in output
    if expected:
        assert "Student practice model" in output
    else:
        assert "No completed result" in output


def test_export_rejects_invalid_saved_result(tmp_path, monkeypatch):
    from ETC.runtime import paraview
    calls = []
    monkeypatch.setattr(paraview, "export_paraview", lambda path: calls.append(path))

    def reject(path):
        raise ValueError("Not derived-initial-condition six-hour predictions")

    ns = dict(RUN_COMPLETED={"navier_stokes": True},
              RUN_DIRS={"navier_stokes": tmp_path},
              completed_output=lambda runs, done, key: runs[key],
              load_derived_flow=reject)
    with pytest.raises(ValueError, match="derived-initial-condition"):
        exec(source("05f915f71865"), ns)
    assert calls == []


@pytest.mark.parametrize("failure", [None, "training", "saved_result"])
def test_six_hour_training_completion_and_failure_guards(tmp_path, failure, capsys):
    calls = []
    loaded = []
    ns = dict(RUN_COMPLETED={"navier_stokes": True}, RUN_DIRS={},
              OUTPUT_BASE=tmp_path / "outputs", LAB=tmp_path / "lab",
              ROOT=tmp_path, STEPS=3000, DEVICE="auto", uuid=uuid, sys=sys)

    def run(command, *, check, cwd):
        assert ns["RUN_COMPLETED"]["navier_stokes"] is False
        calls.append(command)
        assert check is True and cwd == tmp_path
        if failure == "training":
            raise subprocess.CalledProcessError(1, command)

    def load(path):
        assert ns["RUN_COMPLETED"]["navier_stokes"] is False
        assert path == ns["RUN_DIRS"]["navier_stokes"]
        loaded.append(path)
        if failure == "saved_result":
            raise ValueError("Not derived-initial-condition six-hour predictions")
        return {}

    ns.update(subprocess=SimpleNamespace(run=run), load_derived_flow=load,
              validate_settings=lambda device, steps: None,
              show_results=lambda *args, **kwargs: {})
    if failure is None:
        exec(source("91b766da0496"), ns)
        assert ns["RUN_COMPLETED"]["navier_stokes"] is True
        assert "not a validated weather forecast" in capsys.readouterr().out
    else:
        error = subprocess.CalledProcessError if failure == "training" else ValueError
        with pytest.raises(error):
            exec(source("91b766da0496"), ns)
        assert ns["RUN_COMPLETED"]["navier_stokes"] is False

    assert calls == [[sys.executable, "-u",
                      str(tmp_path / "lab/source_code/navier_stokes.py"),
                      "--device", "auto", "--steps", "3000", "--seed", "42",
                      "--recipe", "six_hour",
                      "--output-dir", str(ns["RUN_DIRS"]["navier_stokes"])]]
    assert loaded == ([] if failure == "training" else [ns["RUN_DIRS"]["navier_stokes"]])
    assert ns["RUN_DIRS"]["navier_stokes"].name.startswith("navier_stokes-")


def test_notebook_labels_derived_six_hour_scope_and_preserves_legacy_comparisons():
    notebook = json.loads(NOTEBOOK.read_text())
    text = "\n".join("".join(c["source"]) for c in notebook["cells"])
    assert "## Train your own model" in text
    assert "derived initial condition, not the unchanged ERA5-described field" in text
    assert "original array is preserved and shown separately" in text
    assert "Pressure is recalculated from the corrected wind" in text
    assert "66.9%" in text
    assert "t=0 to t=0.1" in text and "0 to 6 hours" in text
    assert "11 frames at 0.6-hour intervals" in text
    assert "independent numerical solution for large-scale flow" in text
    assert "Fine-scale errors and some damping remain" in text
    assert "not a validated weather forecast or a successful 60-hour forecast" in text
    assert "--recipe six_hour" in text
    assert "original 50,000-update settings remain available with `--recipe upstream`" in text
    assert "--recipe efficient" in text
    assert "Those two recipes keep their original data and losses" in text
    assert "derived_periodic_incompressible_initial_conditions" in text
    assert "stale raw-input or synthetic runs are rejected" in text
    assert "This cell trains your model from scratch" in text
    assert 'os.environ.get("AI4SCI_STEPS", "3000")' in text
    assert "not the output of this run or evidence of forecast accuracy" in text
    assert "one fixed speed scale across all times" in text
    assert "model's own prediction at 0 hours, not the supplied or corrected initial field" in text
    assert "heldout_after.initial_data_rmse" in text
    assert "Part A" not in text and "Part B" not in text
    assert "lab4_inference" not in text and "instructor_model.json" not in text
    assert not any(c["id"].startswith("lab4-instructor-") for c in notebook["cells"])
    # Challenge reference mode is unrelated to the removed instructor checkpoint.
    assert "USE_REFERENCE=True" in text
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))
