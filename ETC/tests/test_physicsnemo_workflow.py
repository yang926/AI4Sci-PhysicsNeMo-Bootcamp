"""Keep the helper-to-API teaching path connected to the executable lessons."""

import ast
import json
from pathlib import Path
import re
import textwrap

import pytest


ROOT = Path(__file__).resolve().parents[2]
GUIDE = ROOT / "ETC/course_materials/PHYSICSNEMO_WORKFLOW.md"
TOPICS = (
    ("01_wave", "Challenge_1_Wave_Dynamics.ipynb", "wave"),
    ("02_fluid", "Challenge_2_Fluid_Flow.ipynb", "fluid"),
    ("03_climate", "Challenge_3_Climate_Modeling.ipynb", "climate"),
    ("04_neural_operators", "Challenge_4_Neural_Operators.ipynb", "operators"),
)


def source(cell):
    return "".join(cell["source"])


def notebook(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def function(tree, name):
    return next(node for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name == name)


def called_names(node):
    return {call.func.id if isinstance(call.func, ast.Name) else call.func.attr
            for call in ast.walk(node) if isinstance(call, ast.Call)
            and isinstance(call.func, (ast.Name, ast.Attribute))}


def test_walkthrough_matches_the_real_pinn_helper_connections():
    guide = GUIDE.read_text(encoding="utf-8")
    helper = ast.parse((ROOT / "ETC/runtime/pinn.py").read_text(encoding="utf-8"))
    for name, downstream in (
        ("create_model", "FullyConnected"),
        ("create_informer", "informer"),
        ("residuals", "forward"),
    ):
        assert name in guide
        assert downstream in called_names(function(helper, name))
    assert {"evaluate_fields", "gradient", "forward"} <= called_names(function(helper, "residuals"))
    for concept in ("PhysicsInformer", "FullyConnected", "PDE", "SymPy", "PyTorch",
                    "autodiff", "spectral", "u__t", "u__t__t"):
        assert concept in guide
    assert "ETC/runtime/pinn.py" in guide
    assert "ETC/runtime/labs.py" in guide


@pytest.mark.parametrize("folder,name,topic", TOPICS)
def test_each_challenge_explains_its_actual_workflow_before_training(folder, name, topic):
    cells = notebook(f"02_challenges/{folder}/{name}")["cells"]
    index = next(i for i, cell in enumerate(cells)
                 if cell["id"] == f"physicsnemo-workflow-{topic}")
    bridge = cells[index]
    assert bridge["cell_type"] == "markdown"
    text = source(bridge)
    assert "PHYSICSNEMO_WORKFLOW.md" in text
    assert "PhysicsNeMo" in text and "PyTorch" in text
    first_training = next(i for i, cell in enumerate(cells)
                          if cell["cell_type"] == "code"
                          and "subprocess.run(command," in source(cell)
                          and "RUN_COMPLETED[" in source(cell))
    assert index < first_training
    expected = ("build_model", "build_datasets", "build_physics", "PhysicsInformer", "spectral") \
        if topic == "operators" else ("create_model", "create_informer", "residuals", "PhysicsInformer")
    for name in expected:
        assert name in text


@pytest.mark.parametrize("path", (
    "README.md", "ETC/course_materials/README.md", "ETC/course_materials/INSTRUCTOR.md",
    "ETC/course_materials/course-plan.md", "Start_Here.ipynb", "00_Setup.ipynb",
    "01_Introduction.ipynb", "01_labs/01_pinn/Lab_1_PINN_Fundamentals.ipynb",
    "01_labs/02_projectile/Lab_2_Projectile_Motion.ipynb",
    "01_labs/03_heat_conduction/Lab_3_Heat_Conduction.ipynb",
    "ETC/reference_labs/04_navier_stokes/Lab_4_Navier_Stokes.ipynb",
))
def test_workflow_guide_is_reachable_from_course_entry_points_and_labs(path):
    if path.endswith(".ipynb"):
        text = "\n".join(source(cell) for cell in notebook(path)["cells"]
                         if cell["cell_type"] == "markdown")
    else:
        text = (ROOT / path).read_text(encoding="utf-8")
    assert "PHYSICSNEMO_WORKFLOW.md" in text


def test_workflow_python_excerpts_are_syntax_valid_without_executing_them():
    text = GUIDE.read_text(encoding="utf-8")
    snippets = re.findall(r"^[ \t]*```python[ \t]*\n(.*?)\n[ \t]*```", text, re.DOTALL | re.MULTILINE)
    assert snippets, "The workflow needs a concrete source-code walkthrough"
    for i, snippet in enumerate(snippets):
        compile(textwrap.dedent(snippet), f"PHYSICSNEMO_WORKFLOW.md:snippet-{i}", "exec")


def test_instructor_walkthrough_does_not_introduce_new_scoring_tasks():
    text = (ROOT / "ETC/course_materials/INSTRUCTOR.md").read_text(encoding="utf-8")
    assert "not as new unannounced scoring requirements" in text
    assert "not PhysicsNeMo API names" in text
    assert "coordinate" in text and "trainable parameters" in text
    assert "not an error against the analytical solution" in text
