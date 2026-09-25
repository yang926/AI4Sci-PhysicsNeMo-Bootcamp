"""Exercise notebook orchestration without importing models or running training."""

import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]
NOTEBOOKS = (
    "02_challenges/01_wave/Challenge_1_Wave_Dynamics.ipynb",
    "02_challenges/02_fluid/Challenge_2_Fluid_Flow.ipynb",
    "02_challenges/03_climate/Challenge_3_Climate_Modeling.ipynb",
    "02_challenges/04_neural_operators/Challenge_4_Neural_Operators.ipynb",
)


def code_cells(relative):
    notebook = json.loads((ROOT / relative).read_text(encoding="utf-8"))
    return ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]


def training_cases():
    for relative in NOTEBOOKS:
        cells = code_cells(relative)
        level = 0
        for index, source in enumerate(cells):
            if '"--steps"' in source and "subprocess.run(command," in source:
                level += 1
                yield pytest.param(relative, level, source, cells[index + 1],
                                   id=f"{Path(relative).parent.name}-level{level}")


def execute(source, namespace):
    exec(compile(source, "<notebook-cell>", "exec"), namespace)


def mock_results(output, metrics):
    """UI-only fixture; real checkpoint contents are tested by execution validation."""
    (output / "metrics.json").write_text(json.dumps({"seed": 42, **metrics}), encoding="utf-8")
    for name in ("loss.csv", "model.pt", "predictions.npz"):
        (output / name).write_text("mock artifact for UI workflow tests", encoding="utf-8")
    from PIL import Image
    Image.new("RGB", (1, 1), color="white").save(output / "openfoam_comparison.png")


def setup_namespace(relative, monkeypatch, tmp_path, reference="1"):
    monkeypatch.chdir(ROOT)
    if reference is None:
        monkeypatch.delenv("AI4SCI_REFERENCE", raising=False)
    else:
        monkeypatch.setenv("AI4SCI_REFERENCE", reference)
    monkeypatch.setenv("AI4SCI_DEVICE", "cuda")
    monkeypatch.setenv("AI4SCI_STEPS", "2")
    monkeypatch.setenv("AI4SCI_OUTPUT_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("AI4SCI_DATA_DIR", str(tmp_path / "data"))

    def forbid_subprocess(*args, **kwargs):
        raise AssertionError("No real subprocess is allowed in notebook workflow tests")

    monkeypatch.setattr(subprocess, "run", forbid_subprocess)
    setup = next(source for source in code_cells(relative) if "def show_mode():" in source)
    namespace = {}
    execute(setup, namespace)
    return namespace


@pytest.mark.parametrize("relative", NOTEBOOKS)
@pytest.mark.parametrize("reference, expected", [(None, False), ("0", False), ("1", True)])
def test_setup_respects_automation_and_displays_actual_mode(relative, reference, expected,
                                                           monkeypatch, tmp_path, capsys):
    namespace = setup_namespace(relative, monkeypatch, tmp_path, reference)
    expected = False  # All student notebooks ignore an inherited demo environment.
    assert namespace["USE_REFERENCE"] is expected
    assert namespace["DEVICE"] == "cuda"
    assert namespace["STEPS"] == ({1: 2, 2: 2, 3: 2} if "/01_wave/" in relative else 2)
    assert namespace["OUTPUT_BASE"] == tmp_path / "runs"
    message = capsys.readouterr().out
    assert ("REFERENCE" if expected else "STUDENT") in message.upper()
    assert not (tmp_path / "runs").exists(), "Setup must not start training or create results"


@pytest.mark.parametrize("relative, level, training, inspection", list(training_cases()))
def test_retries_use_fresh_paths_switch_mode_and_hide_failed_results(
        relative, level, training, inspection, monkeypatch, tmp_path, capsys):
    namespace = setup_namespace(relative, monkeypatch, tmp_path)
    capsys.readouterr()
    calls = []

    def mock_training(command, *, cwd, check):
        assert check is True
        assert Path(cwd) == ROOT / Path(relative).parent
        output = Path(command[command.index("--output-dir") + 1])
        assert output.parent == tmp_path / "runs"
        output.mkdir(parents=True, exist_ok=False)
        metrics = {"steps": 2, "seed": 42, "initial_test": {}, "test": {},
                   "convergence_claim": False, "attempt": len(calls) + 1,
                   "reference": "--reference" in command}
        mock_results(output, metrics)
        calls.append((list(command), output, capsys.readouterr().out))
        return subprocess.CompletedProcess(command, 0)

    namespace["subprocess"] = SimpleNamespace(run=mock_training)
    for expected_reference in (True, False, True):
        # This assignment is the documented override; setup is not rerun.
        namespace["USE_REFERENCE"] = expected_reference
        execute(training, namespace)
        command, output, message = calls[-1]
        assert ("--reference" in command) is expected_reference
        assert command[command.index("--device") + 1] == "cuda"
        assert command[command.index("--steps") + 1] == "2"
        assert ("REFERENCE" if expected_reference else "STUDENT") in message.upper()
        execute(inspection, namespace)
        inspected = namespace.get(f"metrics_l{level}", namespace.get("metrics"))
        assert inspected["attempt"] == len(calls)
        capsys.readouterr()

    outputs = [output for _, output, _ in calls]
    assert len(set(outputs)) == 3
    for attempt, output in enumerate(outputs, 1):
        assert json.loads((output / "metrics.json").read_text())["attempt"] == attempt

    def failed_training(command, *, cwd, check):
        failed_output = Path(command[command.index("--output-dir") + 1])
        assert failed_output not in outputs
        # A partial artifact must not make a failed attempt inspectable.
        failed_output.mkdir(parents=True, exist_ok=False)
        (failed_output / "metrics.json").write_text('{"attempt": "incomplete"}', encoding="utf-8")
        raise subprocess.CalledProcessError(1, command)

    namespace["subprocess"] = SimpleNamespace(run=failed_training)
    with pytest.raises(subprocess.CalledProcessError):
        execute(training, namespace)
    with pytest.raises(RuntimeError, match="has not completed"):
        execute(inspection, namespace)
    for output in outputs:
        assert (output / "metrics.json").exists()


def test_operator_summary_skips_only_the_failed_levels(monkeypatch, tmp_path, capsys):
    relative = NOTEBOOKS[-1]
    namespace = setup_namespace(relative, monkeypatch, tmp_path)
    capsys.readouterr()
    for level in (1, 2, 3):
        output = tmp_path / f"previous-level{level}"
        output.mkdir()
        metrics = {"method": f"method-{level}", "steps": 2,
                   "test_relative_l2_before": 1.0, "test_relative_l2_after": 0.5,
                   "test": {"pde_rmse_fft": 0.1}, "reference": namespace["USE_REFERENCE"]}
        mock_results(output, metrics)
        namespace["RUN_DIRS"][level] = output
        namespace["RUN_COMPLETED"][level] = level != 2
    summary = next(source for source in code_cells(relative) if "for level in (1, 2, 3):" in source)
    execute(summary, namespace)
    output = capsys.readouterr().out
    assert "method-1" in output and "method-3" in output
    assert "method-2" not in output
    assert "Level 2" in output and "incomplete" in output.lower()


@pytest.mark.parametrize("relative", NOTEBOOKS)
def test_level_results_remain_isolated_after_other_levels_and_a_failure(
        relative, monkeypatch, tmp_path):
    namespace = setup_namespace(relative, monkeypatch, tmp_path)
    cells = code_cells(relative)
    levels = [(source, cells[index + 1]) for index, source in enumerate(cells)
              if '"--steps"' in source and "subprocess.run(command," in source]
    successful_outputs = {}

    def mock_training(command, *, cwd, check):
        level = int(Path(command[1]).stem.rsplit("_l", 1)[1])
        output = Path(command[command.index("--output-dir") + 1])
        output.mkdir(parents=True, exist_ok=False)
        metrics = {"steps": 2, "seed": 42, "initial_test": {}, "test": {},
                   "convergence_claim": False, "reference": "--reference" in command,
                   "actual_level": level}
        mock_results(output, metrics)
        successful_outputs[level] = output
        return subprocess.CompletedProcess(command, 0)

    namespace["subprocess"] = SimpleNamespace(run=mock_training)
    for training, _ in levels:
        execute(training, namespace)
    for level in (*range(len(levels), 0, -1), *range(1, len(levels) + 1)):
        execute(levels[level - 1][1], namespace)
        inspected = namespace.get(f"metrics_l{level}", namespace.get("metrics"))
        assert inspected["actual_level"] == level

    def failed_training(command, **kwargs):
        raise subprocess.CalledProcessError(1, command)

    namespace["subprocess"] = SimpleNamespace(run=failed_training)
    with pytest.raises(subprocess.CalledProcessError):
        execute(levels[1][0], namespace)
    assert namespace["RUN_DIRS"][2] != successful_outputs[2]
    assert namespace["RUN_COMPLETED"][2] is False
    # Inspecting another successful level must not make the failed level inspectable.
    execute(levels[0][1], namespace)
    with pytest.raises(RuntimeError, match="not completed"):
        execute(levels[1][1], namespace)
    for output in successful_outputs.values():
        assert (output / "metrics.json").exists()


def test_all_eleven_challenge_training_cells_are_covered():
    assert len(list(training_cases())) == 11


@pytest.mark.parametrize("current_reference", [False, True])
@pytest.mark.parametrize("invalid_kind", ["opposite", "missing", "string", "integer"])
def test_operator_inspection_and_summary_reject_other_or_unknown_modes(
        current_reference, invalid_kind, monkeypatch, tmp_path, capsys):
    relative = NOTEBOOKS[-1]
    namespace = setup_namespace(relative, monkeypatch, tmp_path)
    # Change mode in a new cell without rerunning setup; saved metadata remains authoritative.
    namespace["USE_REFERENCE"] = current_reference
    for level in (1, 2):
        output = tmp_path / f"recorded-level{level}"
        output.mkdir()
        metrics = {"method": f"method-{level}", "steps": 2, "initial_test": {},
                   "test_relative_l2_before": 1.0, "test_relative_l2_after": 0.5,
                   "test": {"pde_rmse_fft": 0.1}, "convergence_claim": False,
                   "reference": current_reference}
        if level == 1:
            if invalid_kind == "missing":
                del metrics["reference"]
            else:
                metrics["reference"] = {"opposite": not current_reference,
                                        "string": str(current_reference),
                                        "integer": int(current_reference)}[invalid_kind]
        mock_results(output, metrics)
        namespace["RUN_DIRS"][level] = output
        namespace["RUN_COMPLETED"][level] = True

    with pytest.raises(RuntimeError, match="mode"):
        namespace["inspect_level"](1)
    assert namespace["inspect_level"](2)["reference"] is current_reference
    capsys.readouterr()
    summary = next(source for source in code_cells(relative) if "for level in (1, 2, 3):" in source)
    execute(summary, namespace)
    output = capsys.readouterr().out
    assert "method-1" not in output
    assert "method-2" in output
    assert "mode" in output.lower()
