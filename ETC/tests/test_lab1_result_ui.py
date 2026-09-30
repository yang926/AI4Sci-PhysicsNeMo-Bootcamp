"""Do not present successful execution as an accurate Lab 1 solution."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ETC.runtime.notebook import result_html


@pytest.mark.parametrize("passed,expected", [(True, "PASS"), (False, "NOT MET")])
def test_accuracy_result_is_visible_and_scoped(passed, expected):
    metrics = {"steps": 3000, "training_recipe": {},
               "accuracy": {"passed": passed, "scope": "401 points <not a global bound>"}}
    html = result_html(metrics, "local-run")
    assert f"Lesson accuracy checks: {expected}" in html
    assert "3000 optimizer calls" in html
    assert "&lt;not a global bound&gt;" in html
    assert ("does not yet meet" in html) is (not passed)


def test_older_runs_are_not_given_an_unmeasured_accuracy_pass():
    html = result_html({"steps": 2000}, "older-run")
    assert "Lesson accuracy checks" not in html
    assert "2000 training steps" in html
