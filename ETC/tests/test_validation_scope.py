"""Validation budgets and labels must not redefine or certify student lessons."""
import json

import pytest

from ETC.course_materials import run_validation as runner


@pytest.mark.parametrize("budget", [20_000, 40_000, 50_000])
def test_existing_course_budgets_are_forwarded_without_shortening(budget, tmp_path, monkeypatch):
    commands = []
    manifest = json.loads(runner.MANIFEST.read_text())
    monkeypatch.setattr(runner, "versions", lambda: {
        "packages": {"nvidia-physicsnemo": manifest["physicsnemo"]}})
    monkeypatch.setattr(runner, "source_snapshot", lambda output: {})

    def execute(command, output, label, timeout):
        commands.append(command)
        return {"label": label, "passed": True, "exit_code": 0}

    monkeypatch.setattr(runner, "execute", execute)
    monkeypatch.setattr(runner, "validate_artifacts", lambda *a, **kw: {
        "passed": True, "metrics": {"initial_reference_error": {"relative_l2": 1.},
                                    "final_reference_error": {"relative_l2": .5}}})
    output = tmp_path / "validation"
    assert runner.main(["--suite", "convergence", "--case", "wave_l1",
                        "--convergence-steps", str(budget), "--output-dir", str(output)]) == 0
    assert len(commands) == 1
    assert commands[0][commands[0].index("--steps") + 1] == str(budget)
    report = json.loads((output / "report.json").read_text())
    assert report["passed"] and report["course_readiness_certified"] is False
    assert report["quality_check_scope"]["wave_l1"] == "relative_improvement_only_not_absolute_accuracy"
    assert report["runs"][0]["quality_check_scope"] == report["quality_check_scope"]["wave_l1"]


@pytest.mark.parametrize("option", ["--steps", "--convergence-steps"])
@pytest.mark.parametrize("budget", [1, 50_001])
def test_invalid_budget_stops_before_creating_output(option, budget, tmp_path):
    output = tmp_path / "never_created"
    with pytest.raises(SystemExit) as exc:
        runner.main([option, str(budget), "--output-dir", str(output)])
    assert exc.value.code == 2 and not output.exists()


@pytest.mark.parametrize("suite", ["convergence", "all"])
def test_regression_fixtures_are_not_reported_as_student_accuracy(suite):
    assert runner.quality_check_scope("navier_stokes", suite) == "synthetic_regression_only_not_original_data_accuracy"
    for level in (1, 2, 3):
        assert runner.quality_check_scope(f"operators_l{level}", suite) == "reduced_model_and_data_regression_not_full_lesson_accuracy"
    for case in ("wave_l2", "wave_l3", "fluid_l1", "fluid_l2", "fluid_l3", "climate_l1", "climate_l2"):
        assert runner.quality_check_scope(case, suite) == "no_convergence_check_available"


def test_original_data_execution_and_unit_tests_are_not_accuracy_checks():
    assert runner.quality_check_scope("navier_stokes", "smoke") == "execution_only"
    assert runner.quality_check_scope("navier_stokes", "unit") == "unit_tests_only"
