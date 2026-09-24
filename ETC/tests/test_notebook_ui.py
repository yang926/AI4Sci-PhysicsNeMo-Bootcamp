"""Check the notebook presentation without importing PhysicsNeMo or training."""

import json
from pathlib import Path
import sys
from types import SimpleNamespace
import uuid

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from ETC.runtime.notebook import (completed_output, load_results, require_artifacts,
                                  require_current_mode, result_html, validate_settings)


@pytest.mark.parametrize("device,steps,mode", [("auto", 2, False), ("cpu", 10, True), ("cuda", 1, None)])
def test_valid_controls(device, steps, mode):
    validate_settings(device, steps, mode)


@pytest.mark.parametrize("device,steps,mode", [("mps", 2, False), ("cpu", 0, False), ("cpu", True, False),
                                              ("cpu", 2.0, False), ("cpu", 2, "False")])
def test_invalid_controls(device, steps, mode):
    with pytest.raises(ValueError):
        validate_settings(device, steps, mode)


def test_read_results_checks_run_identity(tmp_path):
    metrics = {"steps": 2, "seed": 42, "reference": False}
    (tmp_path / "metrics.json").write_text(json.dumps(metrics))
    assert load_results(tmp_path, steps=2, seed=42, reference=False) == metrics
    for kwargs in ({"steps": 3}, {"seed": 7}, {"reference": True}):
        with pytest.raises(RuntimeError, match="rerun"):
            load_results(tmp_path, **kwargs)


@pytest.mark.parametrize("contents", [None, "not json", "[]"])
def test_unreadable_results_do_not_fall_back(tmp_path, contents):
    if contents is not None:
        (tmp_path / "metrics.json").write_text(contents)
    with pytest.raises(RuntimeError):
        load_results(tmp_path)


@pytest.mark.parametrize("recorded", [None, "False", 0])
def test_unknown_mode_is_not_student_mode(recorded):
    with pytest.raises(RuntimeError, match="mode"):
        require_current_mode({"reference": recorded}, False)


def test_pinn_mode_flag_is_supported_and_conflicts_are_rejected():
    require_current_mode({"reference_implementation": False}, False)
    require_current_mode({"reference_implementation": True, "reference": True}, True)
    for metrics in ({"reference_implementation": 1}, {"reference": False, "reference_implementation": True}):
        with pytest.raises(RuntimeError, match="mode"):
            require_current_mode(metrics, False)
    assert "Instructor reference" in result_html({"reference_implementation": True}, "/tmp/run")


def test_metrics_alone_are_not_a_completed_run(tmp_path):
    (tmp_path / "metrics.json").write_text('{}')
    with pytest.raises(RuntimeError, match="Incomplete"):
        require_artifacts(tmp_path)
    for name in ("loss.csv", "model.pt", "predictions.npz"):
        (tmp_path / name).touch()
    with pytest.raises(RuntimeError, match="empty"):
        require_artifacts(tmp_path)
    for name in ("loss.csv", "model.pt", "predictions.npz"):
        (tmp_path / name).write_bytes(b"test artifact; binary contents are checked by the training runner")
    require_artifacts(tmp_path)


def test_completed_output_keeps_subproblems_separate():
    paths = {"forward": Path("first"), "inverse": Path("second")}
    completed = {"forward": True, "inverse": False}
    assert completed_output(paths, completed, "forward") == Path("first")
    with pytest.raises(RuntimeError, match="not completed"):
        completed_output(paths, completed, "inverse")


def test_summary_escapes_content_and_marks_missing_evidence():
    html = result_html({"steps": 2, "reference": False,
                        "initial_test": {"<script>alert(1)</script>": 1.0},
                        "test": {"pde_rmse": float("nan")}, "reference_over_time": {}}, "<img onerror=bad>")
    assert "<script>" not in html and "<img" not in html
    assert "&lt;script&gt;" in html and "&lt;img" in html
    assert "non-finite" in html and "does not mean zero error" in html
    assert 'scope="col"' in html and 'scope="row"' in html
    assert "official points" in html


def test_lab_and_challenge_result_formats_are_supported():
    for before, after in (("heldout_before", "heldout_after"), ("initial_test", "test")):
        html = result_html({before: {"pde_rmse": 1}, after: {"pde_rmse": 0.1},
                            "reference_over_time": {"rmse": 0.2, "relative_l2": 0.3,
                                                     "per_field": {"temperature": {"rmse": 0.2}}}}, "/tmp/run")
        assert "pde_rmse" in html and "temperature" in html
        assert "Before" in html and "After" in html


def test_legacy_pinn_schema_has_a_labeled_before_after_table():
    html = result_html({"reference_implementation": True, "initial_loss": 4.0,
                        "final_loss": 1.0, "initial_reference_error": {"rmse": 2.0},
                        "final_reference_error": {"rmse": 0.5}, "boundary_rmse": 0.1}, "/tmp/run")
    assert "weighted_objective" in html and "analytical_slice.rmse" in html
    assert "boundary_rmse" in html and "No before/after summary" not in html
    assert "Instructor reference" in html


def test_each_challenge_uses_shared_safe_results():
    manifest = json.loads((ROOT / "ETC/course_materials/course_manifest.json").read_text())
    paths = [ROOT / course["notebook"] for course in manifest["course"]
             if Path(course["notebook"]).parts[0] == "02_challenges"]
    assert len(paths) == 4
    for path in paths:
        notebook = json.loads(path.read_text())
        sources = ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]
        setup = next(source for source in sources if "def show_mode():" in source)
        assert "from ETC.runtime.notebook import" in setup
        assert "validate_settings(DEVICE, STEPS, USE_REFERENCE)" in setup
        results = "\n".join(sources)
        assert "reference=USE_REFERENCE" in results
        assert "show_results(" in results


def test_operator_setup_finds_its_own_program_not_any_challenge(monkeypatch):
    notebook = json.loads((ROOT / "02_challenges/04_neural_operators/Challenge_4_Neural_Operators.ipynb").read_text())
    setup = next("".join(cell["source"]) for cell in notebook["cells"]
                 if cell["cell_type"] == "code" and "def show_mode():" in "".join(cell["source"]))
    monkeypatch.chdir(ROOT / "02_challenges/01_wave")
    namespace = {}
    exec(compile(setup, "<operator-setup>", "exec"), namespace)
    assert namespace["LESSON_DIR"] == ROOT / "02_challenges/04_neural_operators"


@pytest.mark.parametrize("course_id", ["lab1", "lab3"])
def test_lab_subproblem_plots_survive_other_runs_and_reject_failure(course_id, tmp_path):
    manifest = json.loads((ROOT / "ETC/course_materials/course_manifest.json").read_text())
    path = ROOT / next(course["notebook"] for course in manifest["course"] if course["id"] == course_id)
    notebook = json.loads(path.read_text())
    sources = ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]
    training = [source for source in sources if source.startswith("RUN_COMPLETED[")]
    inspections = [source.split("data = np.load", 1)[0] for source in sources
                   if source.startswith("OUTPUT = completed_output") and "plt." in source]
    namespace = {"RUN_DIRS": {}, "RUN_COMPLETED": {}, "OUTPUT_BASE": tmp_path,
                 "LAB": path.parent, "ROOT": ROOT, "sys": sys, "uuid": uuid,
                 "DEVICE": "cpu", "STEPS": 2, "validate_settings": validate_settings,
                 "show_results": lambda *args, **kwargs: {}, "completed_output": completed_output,
                 "subprocess": SimpleNamespace(run=lambda *args, **kwargs: None)}
    for source in training:
        exec(source, namespace)
    expected = list(namespace["RUN_DIRS"].values())
    assert len(expected) == len(inspections) >= 2
    for index in reversed(range(len(inspections))):
        exec(inspections[index], namespace)
        assert namespace["OUTPUT"] == expected[index]

    def fail(*args, **kwargs):
        raise RuntimeError("training failed")

    namespace["subprocess"] = SimpleNamespace(run=fail)
    with pytest.raises(RuntimeError, match="training failed"):
        exec(training[0], namespace)
    with pytest.raises(RuntimeError, match="not completed"):
        exec(inspections[0], namespace)
    exec(inspections[-1], namespace)
    assert namespace["OUTPUT"] == expected[-1]
