"""Keep the Wave teaching trace aligned with code, without importing training code."""

import ast
import hashlib
import json
from pathlib import Path
import re

import pytest


ROOT = Path(__file__).resolve().parents[2]
WAVE = ROOT / "02_challenges/01_wave"
NOTEBOOK = WAVE / "Challenge_1_Wave_Dynamics.ipynb"


def notebook():
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def source(cell):
    return cell["source"] if isinstance(cell["source"], str) else "".join(cell["source"])


def syntax(path):
    return ast.parse(path.read_text(encoding="utf-8"))


def function(tree, name):
    return next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)


def calls(node):
    return {ast.unparse(item.func) for item in ast.walk(node) if isinstance(item, ast.Call)}


def expressions(node):
    return {ast.unparse(item) for item in ast.walk(node)}


def markdown(cell_id):
    cell = next(cell for cell in notebook()["cells"] if cell.get("id") == cell_id)
    assert cell["cell_type"] == "markdown"
    return source(cell)


def test_wave_explanation_changes_do_not_change_executable_notebook_cells():
    # The explanation-only revision starts at fbaa33d. Hash only code source and
    # IDs, not Markdown, output, or execution metadata; no git binary is needed.
    code = [{"id": cell.get("id"), "source": cell["source"]}
            for cell in notebook()["cells"] if cell["cell_type"] == "code"]
    digest = hashlib.sha256(json.dumps(code, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    assert len(code) == 8
    assert digest == "bad5706ba2e1f7e7a51e6c353b36bd227af30f5b6f7812900a86306b9e6420a7"


def test_wave_guides_precede_training_and_link_real_course_sources():
    cells = notebook()["cells"]
    first_training = next(i for i, cell in enumerate(cells)
                          if cell["cell_type"] == "code" and "subprocess.run" in source(cell))
    for cell_id in ("wave-domain-and-sampling", "physicsnemo-workflow-wave"):
        index = next(i for i, cell in enumerate(cells) if cell.get("id") == cell_id)
        assert index < first_training
        text = markdown(cell_id)
        targets = re.findall(r"\[[^\]]+\]\(([^)]+)\)", text)
        assert targets
        for target in targets:
            assert "://" not in target
            path = (WAVE / target.split("#", 1)[0]).resolve()
            assert path.is_relative_to(ROOT)
            assert path.is_file(), target


def test_wave_trace_names_student_functions_and_their_separate_consumers():
    text = markdown("physicsnemo-workflow-wave")
    assert "Where your three functions go" in text
    for name in ("student_speed", "student_equations", "student_conditions",
                 "WaveEquation2D", "self.equations", "create_informer", "PhysicsInformer",
                 "create_model", "FullyConnected", "conditions", "condition_tensor",
                 "loss_terms", "backward()", "optimizer.step()"):
        assert name in text
    assert "USE_REFERENCE" not in text
    assert "reference_equations" not in text


@pytest.mark.parametrize("level", (1, 2, 3))
def test_student_speed_and_equations_feed_pde_not_network_constructor(level):
    tree = syntax(WAVE / f"wave_l{level}.py")
    pde = next(node for node in tree.body if isinstance(node, ast.ClassDef)
               and node.name == "WaveEquation2D")
    assert any(ast.unparse(base) == "PDE" for base in pde.bases)
    initializer = function(pde, "__init__")
    nodes = expressions(initializer)
    assert "speed = student_speed(x, y)" in nodes
    assert "self.equations = student_equations(x, y, t, u, c * speed['c'])" in nodes
    main = function(tree, "main")
    main_nodes = expressions(main)
    assert "pde = WaveEquation2D()" in main_nodes
    assert "informer = create_informer(pde, args.device)" in main_nodes
    assert "model = create_model(3, 1, config, args.device)" in main_nodes


@pytest.mark.parametrize("level", (1, 2, 3))
def test_student_conditions_and_pde_residuals_join_the_training_loss(level):
    tree = syntax(WAVE / f"wave_l{level}.py")
    main = function(tree, "main")
    assert "exercise = conditions(student_conditions, wave=True)" in expressions(main)
    assert {"loss_terms", "record_step", "total.backward", "optimizer.step"} <= calls(main)
    losses = function(tree, "loss_terms")
    assert {"residuals", "condition_tensor", "boundary_residual", "gradient"} <= calls(losses)
    returned = next(node.value for node in ast.walk(losses) if isinstance(node, ast.Return))
    assert isinstance(returned, ast.Dict)
    assert {key.value for key in returned.keys} == {
        "pde", "initial_displacement", "initial_velocity", "boundary"}
    boundary = function(tree, "boundary_residual")
    assert "condition_tensor(exercise, 'boundary', coordinates, time, fields)" in expressions(boundary)


def test_shared_helpers_match_the_network_and_physics_trace():
    tree = syntax(ROOT / "ETC/runtime/pinn.py")
    network = function(tree, "create_model")
    assert "FullyConnected" in calls(network)
    assert not {"student_equations", "student_conditions", "student_speed"} & calls(network)
    informer = function(tree, "create_informer")
    assert "pde.equations" in expressions(informer)
    assert "informer(pde, device, supplied_derivatives=supplied)" in expressions(informer)
    residuals = function(tree, "residuals")
    assert {"evaluate_fields", "gradient", "informer.forward"} <= calls(residuals)
    assert "checked_total_loss" in calls(function(tree, "record_step"))
    assert "values.sum" in calls(function(tree, "checked_total_loss"))
    conditions_tree = syntax(ROOT / "ETC/runtime/exercises.py")
    assert "builder" in calls(function(conditions_tree, "conditions"))
    labs = syntax(ROOT / "ETC/runtime/labs.py")
    assert "PhysicsInformer" in calls(function(labs, "informer"))


def test_wave_domain_guide_points_to_sampling_not_just_condition_formulas():
    text = markdown("wave-domain-and-sampling")
    for name in ("spatial_sample", "sample_square", "sample_circle", "boundary=True",
                 "TIME_END", "loss_terms", "student_conditions"):
        assert name in text
    for level in (1, 2, 3):
        assert f"wave_l{level}.py" in text
    assert re.search(r"\bsquare\b", text, re.IGNORECASE)
    assert re.search(r"\b(disk|disc)\b", text, re.IGNORECASE)


@pytest.mark.parametrize("level,sampler,time_end", [
    (1, "sample_square", "2 * math.pi"),
    (2, "sample_square", "2 * math.pi"),
    (3, "sample_circle", "3.0"),
])
def test_domain_and_time_names_match_each_level(level, sampler, time_end):
    tree = syntax(WAVE / f"wave_l{level}.py")
    assert f"TIME_END = {time_end}" in expressions(tree)
    assert calls(function(tree, "spatial_sample")) == {sampler}
    losses = expressions(function(tree, "loss_terms"))
    assert "spatial_sample(count['interior'], device)" in losses
    assert "spatial_sample(count['initial'], device)" in losses
    assert "spatial_sample(count['boundary'], device, boundary=True)" in losses
    assert "sample_time(len(xy), TIME_END, device)" in losses


def test_square_and_disk_sampler_dimensions_match_the_teaching_guide():
    tree = syntax(ROOT / "ETC/runtime/pinn.py")
    square = function(tree, "sample_square")
    assert ast.unparse(square.args.defaults[-1]) == "math.pi"
    assert "torch.rand(count, 2, device=device) * length" in expressions(square)
    disk = function(tree, "sample_circle")
    assert "torch.ones_like(theta) if boundary else torch.sqrt(torch.rand_like(theta))" in expressions(disk)
    assert {"radius * torch.cos(theta)", "radius * torch.sin(theta)"} <= expressions(disk)
