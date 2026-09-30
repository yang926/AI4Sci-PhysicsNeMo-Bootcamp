"""Exercise placeholders must identify every unfinished selected Level clearly."""
import ast
import json
from pathlib import Path
import re

import pytest

from ETC.judge.catalog import CHALLENGES
from ETC.judge.expressions import SubmissionError
from ETC.runtime.submission import collect_submission, submission_html
from ETC.runtime.submission_widgets import SubmissionPanel
from ETC.tests.test_notebook_submission import FakeClient


ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("challenge", ["1", "2", "3", "4"])
def test_all_unfinished_selected_levels_report_file_line_and_function(challenge, tmp_path):
    spec = CHALLENGES[challenge]
    for filename in spec["files"]:
        (tmp_path / filename).write_text(
            (ROOT / "02_challenges" / spec["directory"] / filename).read_text()
        )
    levels = tuple(range(1, len(spec["files"]) + 1))
    with pytest.raises(SubmissionError) as error:
        collect_submission(challenge, tmp_path, levels=levels)
    message = str(error.value)
    assert "Nothing was submitted" in message
    assert "Save the .py files with Ctrl+S / Command+S" in message
    for filename in spec["files"]:
        assert re.search(re.escape(filename) + r":\d+ \([\w.]+\): replace ", message)
    if challenge == "1":
        for function in ("student_equations", "student_speed", "student_conditions"):
            assert message.count(f"({function}):") == 3
    if challenge == "4":
        assert "(ReactionDiffusionPDE.__init__):" in message


def wave_source(body):
    # Only exercises source collection, not equation correctness or training.
    return (
        "def student_equations(x, y, t, u, c=1.0):\n    return {}\n\n"
        "def student_conditions(x, y, t, u):\n    return {}\n\n"
        "def student_speed(x, y):\n" + body + "\n"
    )


@pytest.mark.parametrize("marker", [
    'raise NotImplementedError("TODO")',
    'raise UnfinishedExerciseError("TODO")',
    "pass",
])
def test_placeholder_below_return_is_explained_then_saved_replacement_is_collected(marker, tmp_path):
    path = tmp_path / "wave_l1.py"
    path.write_text(wave_source('    return {"c": 1}\n    ' + marker))
    with pytest.raises(SubmissionError) as error:
        collect_submission("1", tmp_path)
    assert "(student_speed): replace " in str(error.value)
    assert "below your return statement" in str(error.value)
    path.write_text(wave_source('    return {"c": 1}'))
    assert set(collect_submission("1", tmp_path)["sources"]) == {"wave_l1.py"}


def test_real_input_guards_are_not_reported_as_placeholders(tmp_path):
    source = wave_source('    if x is None:\n        raise ValueError("Missing x")\n    return {"c": 1}')
    (tmp_path / "wave_l1.py").write_text(source)
    assert 'raise ValueError' in collect_submission("1", tmp_path)["sources"]["wave_l1.py"]


@pytest.mark.parametrize("action", ["check_button", "submit_button"])
def test_widget_shows_all_placeholder_locations_without_sending(action, tmp_path, monkeypatch):
    # The client is a local fake: keep this guidance test independent of thread
    # scheduling and live transport; transport has its own integration tests.
    async def inline_call(function, *args, **kwargs):
        return function(*args, **kwargs)

    monkeypatch.setattr("ETC.runtime.submission_widgets.asyncio.to_thread", inline_call)
    for level in (1, 2, 3):
        (tmp_path / f"wave_l{level}.py").write_text(wave_source('    raise NotImplementedError("<script>secret</script>")'))
    client = FakeClient()
    panel = SubmissionPanel("1", tmp_path, levels=(1, 2, 3), client=client)
    try:
        getattr(panel, action).click()
        assert not client.sent
        for level in (1, 2, 3):
            assert f"wave_l{level}.py:" in panel.status.value
        assert "student_speed" in panel.status.value
        assert "<br>" in panel.status.value
        assert "<script>" not in panel.status.value and "secret" not in panel.status.value
    finally:
        panel.close()


@pytest.mark.parametrize("challenge", ["1", "2", "3", "4"])
def test_challenge_guidance_explains_replacing_and_saving_placeholders(challenge):
    directory = ROOT / "02_challenges" / CHALLENGES[challenge]["directory"]
    notebook = json.loads(next(directory.glob("Challenge_*.ipynb")).read_text())
    markdown = "\n".join("".join(cell["source"]) for cell in notebook["cells"]
                         if cell["cell_type"] == "markdown")
    assert "Replace each `raise NotImplementedError(...)`" in markdown
    assert "deleting it alone does not complete the exercise" in markdown
    assert "Ctrl+S / Command+S" in markdown
    assert "Check saved code" in markdown
    assert "search for **EDIT HERE**" in markdown
    assert "Replace each exercise placeholder" in submission_html(challenge, False)


@pytest.mark.parametrize("challenge", ["1", "2", "3", "4"])
def test_every_exercise_has_numbered_visible_edit_boundaries(challenge):
    spec = CHALLENGES[challenge]
    for filename in spec["files"]:
        path = ROOT / "02_challenges" / spec["directory"] / filename
        source = path.read_text()
        lines = source.splitlines()
        exercises = []
        for node in ast.parse(source).body:
            if isinstance(node, ast.FunctionDef) and (node.name.startswith("student_")
                                                     or node.name in ("build_datasets", "build_model")):
                exercises.append((node, node.name))
            elif isinstance(node, ast.ClassDef) and node.name == "ReactionDiffusionPDE":
                exercises.extend((method, "ReactionDiffusionPDE.__init__") for method in node.body
                                 if isinstance(method, ast.FunctionDef) and method.name == "__init__")
        for index, (node, name) in enumerate(exercises, 1):
            header = "\n".join(lines[node.lineno - 6:node.lineno - 1])
            footer = "\n".join(lines[node.end_lineno:node.end_lineno + 2])
            assert f"# EDIT HERE {index}/{len(exercises)}: {name}" in header
            assert f"# END EDIT HERE {index}/{len(exercises)}: {name}" in footer
            assert "############" in header and "############" in footer
