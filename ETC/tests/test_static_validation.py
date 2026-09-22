"""Working notebook validation must preserve learner outputs and ignore checkpoints."""

import copy
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ETC.course_materials.validate_materials import ROOT, is_active, notebook_output_messages


def test_jupyter_checkpoints_are_not_course_sources():
    assert not is_active(ROOT / "01_labs/01_pinn/.ipynb_checkpoints/Lab-checkpoint.ipynb")
    assert not is_active(ROOT / ".ipynb_checkpoints/Start_Here-checkpoint.ipynb")
    assert is_active(ROOT / "Start_Here.ipynb")


def test_publication_validation_still_rejects_saved_outputs():
    notebook = {"cells": [{"cell_type": "code", "outputs": [{"output_type": "stream", "text": "previous run"}]}]}
    original = copy.deepcopy(notebook)
    errors, warnings = notebook_output_messages(notebook, "lesson.ipynb")
    assert len(errors) == 1 and "lesson.ipynb" in errors[0]
    assert warnings == []
    assert notebook == original


def test_working_checkout_mode_preserves_but_does_not_certify_outputs():
    notebook = {"cells": [{"cell_type": "code", "outputs": [{"output_type": "stream", "text": "previous run"}]}]}
    original = copy.deepcopy(notebook)
    errors, warnings = notebook_output_messages(notebook, "lesson.ipynb", allow_executed=True)
    assert errors == []
    assert len(warnings) == 1 and "lesson.ipynb" in warnings[0]
    assert "not validated" in warnings[0]
    assert notebook == original


def test_clean_notebooks_need_no_output_exception():
    notebook = {"cells": [{"cell_type": "markdown", "source": "lesson"}, {"cell_type": "code", "outputs": []}]}
    assert notebook_output_messages(notebook, "lesson.ipynb") == ([], [])
