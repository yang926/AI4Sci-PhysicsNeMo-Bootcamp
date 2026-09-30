"""The convergence runner must reject good-looking u with a wrong source f."""
import copy
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from ETC.course_materials import run_validation as runner


def measured_metrics(mode):
    errors = {"solution_rmse": 1e-5, "pde_rmse": 1e-4, "boundary_max_abs": 0.0}
    if mode == "inverse":
        errors.update(source_rmse=.01, source_max_abs=.02)
    elif mode == "parameterized":
        errors["per_length"] = [{"length": length, **errors}
                                for length in (1., 1.25, 1.5, 1.75, 2.)]
    return {"heldout_before": {"solution_rmse": .1}, "heldout_after": errors,
            "accuracy": {"passed": True, "checks": []}}


@pytest.mark.parametrize("case_id,mode", list(runner.LAB1_CASE_MODES.items()))
def test_lab1_cases_require_improvement_and_absolute_accuracy(case_id, mode):
    assert case_id in runner.CONVERGENCE_CASES
    metrics = measured_metrics(mode)
    original = copy.deepcopy(metrics)
    result = runner.improvement(metrics, case_id, .99)
    assert result["passed"] and result["accuracy"]["passed"]
    assert result["accuracy"]["checks"]
    assert metrics == original
    metrics["heldout_before"]["solution_rmse"] = 1e-6
    assert not runner.improvement(metrics, case_id, .99)["passed"]


@pytest.mark.parametrize("name,value", [("source_rmse", .68), ("source_max_abs", .2),
                                        ("source_rmse", float("nan")),
                                        ("source_rmse", -.01), ("source_rmse", True)])
def test_inverse_does_not_trust_saved_pass_flag_or_solution_improvement(name, value):
    metrics = measured_metrics("inverse")
    metrics["heldout_after"][name] = value
    result = runner.improvement(metrics, "pinn_inverse", .99)
    assert result["actual_after_before_ratio"] < .99
    assert metrics["accuracy"]["passed"]  # Deliberately misleading saved claim.
    assert not result["accuracy"]["passed"]
    assert not result["passed"]


def test_parameterized_checks_each_length_not_only_pooled_error():
    metrics = measured_metrics("parameterized")
    metrics["heldout_after"]["per_length"][-1]["solution_rmse"] = .01
    result = runner.improvement(metrics, "pinn_parameterized", .99)
    assert result["actual_after_before_ratio"] < .99
    assert not result["passed"]
    assert any(not check["passed"] and check["length"] == 2.
               for check in result["accuracy"]["checks"])


@pytest.mark.parametrize("missing", [None, [], [{"length": 1.0}]])
def test_parameterized_cannot_omit_required_lengths(missing):
    metrics = measured_metrics("parameterized")
    metrics["heldout_after"]["per_length"] = missing
    with pytest.raises(ValueError, match="five evaluation lengths"):
        runner.improvement(metrics, "pinn_parameterized", .99)


@pytest.mark.parametrize("case_id,mode", list(runner.LAB1_CASE_MODES.items()))
def test_cli_recognizes_each_lab1_case_and_preserves_requested_budget(case_id, mode, tmp_path, monkeypatch):
    manifest = json.loads(runner.MANIFEST.read_text())
    requested_steps = 17
    observed_commands = []
    artifacts = measured_metrics(mode)
    # A short run may execute successfully but still fail its accuracy gate.
    if mode == "parameterized":
        artifacts["heldout_after"]["per_length"][-1]["pde_rmse"] = 1.
    else:
        artifacts["heldout_after"]["pde_rmse"] = 1.

    def execute(command, output, label, timeout):
        observed_commands.append(command)
        return {"label": label, "passed": True, "exit_code": 0}

    def validate(path, *, steps, device, expected_version, seed):
        assert steps == requested_steps
        return {"passed": True, "metrics": artifacts}

    monkeypatch.setattr(runner, "versions", lambda: {"packages": {"nvidia-physicsnemo": manifest["physicsnemo"]}})
    monkeypatch.setattr(runner, "source_snapshot", lambda output: {})
    monkeypatch.setattr(runner, "execute", execute)
    monkeypatch.setattr(runner, "validate_artifacts", validate)
    output = tmp_path / case_id
    status = runner.main(["--suite", "convergence", "--case", case_id,
                          "--convergence-steps", str(requested_steps), "--output-dir", str(output)])
    assert status == 1
    assert len(observed_commands) == 1
    command = observed_commands[0]
    assert command[command.index("--steps") + 1] == str(requested_steps)
    assert command[command.index("--mode") + 1] == mode
    report = json.loads((output / "report.json").read_text())
    assert not report["passed"]
    assert report["runs"][0]["steps"] == requested_steps
    assert not report["runs"][0]["improvement"]["accuracy"]["passed"]
