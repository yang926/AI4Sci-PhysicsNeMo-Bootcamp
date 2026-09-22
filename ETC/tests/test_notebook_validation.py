"""Execution validation includes result completeness, not just rendered plots."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ETC.course_materials import run_notebooks

MANIFEST = {"physicsnemo": "2.2.2", "runs": [
    {"course": "wave"}, {"course": "wave"}, {"course": "wave"}]}


def test_training_notebook_requires_every_level(tmp_path):
    with pytest.raises(AssertionError, match="expected 3 completed training runs, found 0"):
        run_notebooks.check_artifacts("wave", tmp_path, MANIFEST, 2, "cpu")


def test_reading_only_notebook_has_no_training_artifacts(tmp_path):
    assert run_notebooks.check_artifacts("start", tmp_path, MANIFEST, 2, "cpu") == []


def test_supplementary_reference_is_also_checked(tmp_path):
    with pytest.raises(AssertionError, match="expected 1 completed training runs"):
        run_notebooks.check_artifacts("wave_reference", tmp_path, MANIFEST, 2, "cpu")


def test_each_result_passes_through_the_real_artifact_contract(tmp_path, monkeypatch):
    for level in range(3):
        path = tmp_path / "training" / f"level{level}"
        path.mkdir(parents=True)
        (path / "metrics.json").write_text("{}")
    calls = []
    def validate(path, **settings):
        calls.append((path, settings))
        return {"passed": True}
    monkeypatch.setattr(run_notebooks, "validate_artifacts", validate)
    result = run_notebooks.check_artifacts("wave", tmp_path, MANIFEST, 2, "cpu")
    assert len(result) == len(calls) == 3
    assert all(settings == {"steps": 2, "device": "cpu", "expected_version": "2.2.2", "seed": 42}
               for _, settings in calls)


def test_invalid_selection_does_not_create_a_validation_directory(tmp_path, monkeypatch):
    output = tmp_path / "invalid"
    monkeypatch.setattr(sys, "argv", ["run_notebooks.py", "--output-dir", str(output), "--case", "typo"])
    with pytest.raises(SystemExit) as result:
        run_notebooks.main()
    assert result.value.code == 2
    assert not output.exists()


def test_runner_preserves_dangling_output_symlink(tmp_path, monkeypatch):
    target = tmp_path / "absent"
    output = tmp_path / "occupied"
    output.symlink_to(target, target_is_directory=True)
    monkeypatch.setattr(sys, "argv", ["run_notebooks.py", "--output-dir", str(output)])
    with pytest.raises(SystemExit) as result:
        run_notebooks.main()
    assert result.value.code == 2
    assert output.is_symlink() and not target.exists()
