"""Student release contracts: no executable answer mode or answer fixtures."""
import ast
import importlib
import json
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
TOPICS = ("01_wave", "02_fluid", "03_climate", "04_neural_operators")
LESSONS = sorted((ROOT / "02_challenges").glob("*/*_l[123].py"))
ANSWER_NAMES = {
    "reference_equations", "reference_speed", "reference_conditions",
    "reference_geometry", "reference_parameters", "reference_solution",
    "reference_fno", "reference_afno", "reference_physics", "reference_datasets",
}


def test_all_eleven_student_levels_are_preserved():
    assert len(LESSONS) == 11


@pytest.mark.parametrize("path", LESSONS, ids=lambda path: path.stem)
def test_exercises_keep_edit_sections_without_adjacent_answers(path):
    source = path.read_text()
    tree = ast.parse(source)
    definitions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    assert not ({node.name for node in definitions} & ANSWER_NAMES)
    assert "--reference" not in source
    exercises = [node for node in definitions
                 if node.name.startswith("student_") or node.name in ("build_model", "build_datasets")]
    assert exercises
    for exercise in exercises:
        assert any(isinstance(node, ast.Raise) for node in ast.walk(exercise)), exercise.name
        assert f"EDIT HERE" in source[:source.index(f"def {exercise.name}(")]


@pytest.mark.parametrize("topic", TOPICS)
def test_notebook_has_no_answer_toggle_and_uses_unbuffered_python(topic):
    paths = list((ROOT / "02_challenges" / topic).glob("Challenge_*.ipynb"))
    assert len(paths) == 1
    notebook = json.loads(paths[0].read_text())
    code = "\n".join("".join(cell["source"]) for cell in notebook["cells"]
                     if cell["cell_type"] == "code")
    assert "USE_REFERENCE" not in code and '"--reference"' not in code
    tree = ast.parse(code)
    commands = [node for node in ast.walk(tree) if isinstance(node, ast.List) and node.elts
                and isinstance(node.elts[0], ast.Attribute)
                and isinstance(node.elts[0].value, ast.Name)
                and node.elts[0].value.id == "sys" and node.elts[0].attr == "executable"]
    assert commands
    for command in commands:
        assert isinstance(command.elts[1], ast.Constant) and command.elts[1].value == "-u"


def test_pin_and_operator_budgets_are_explicit_without_lr_change():
    configs = list((ROOT / "02_challenges").glob("*/conf/config*.yaml"))
    assert configs
    for path in configs:
        config = yaml.safe_load(path.read_text())
        expected = 3000 if "04_neural_operators" in path.parts else 5000
        assert config["training"]["steps"] == expected, path
        assert config["training"]["learning_rate"] == 0.001, path


def test_generic_geometry_helper_does_not_supply_exercise_answers():
    helper = importlib.import_module("02_challenges.02_fluid.fluid_geometry")
    assert not any(hasattr(helper, name) for name in ("SINGLE_BLOCK", "THREE_BLOCKS", "inlet_velocity"))
    blocks = [(-.8, -.2, -.2)]  # Arbitrary test obstacle, not an exercise solution.
    interior = helper.sample_interior(64, blocks, "cpu")
    assert helper.outside_blocks(interior, blocks).all()
    assert len(helper.wall_segments(blocks)) == 6


def test_submission_fixtures_are_intentionally_incorrect_not_answer_generators():
    from ETC.tests.test_judge import dummy_source
    from ETC.tests.test_judge_operators import dummy_operator_source
    pin = ast.parse(dummy_source("1", "wave_l1.py"))
    for function in pin.body:
        assert len(function.body) == 1 and isinstance(function.body[0], ast.Return)
        assert isinstance(function.body[0].value, ast.Dict) and not function.body[0].value.keys
    operator = dummy_operator_source(3)
    assert "return ()" in operator and "return 0" in operator
    assert "self.equations = {}" in operator
