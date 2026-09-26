"""Protect the short conceptual bridges without running training or importing CUDA."""

import ast
import json
from pathlib import Path
import re

import pytest


ROOT = Path(__file__).resolve().parents[2]
INTRO = "01_Introduction.ipynb"
LAB1 = "01_labs/01_pinn/Lab_1_PINN_Fundamentals.ipynb"
LAB3 = "01_labs/03_heat_conduction/Lab_3_Heat_Conduction.ipynb"
LAB4 = "01_labs/04_navier_stokes/Lab_4_Navier_Stokes.ipynb"


def notebook(relative):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def source(cell):
    value = cell["source"]
    return value if isinstance(value, str) else "".join(value)


def cell_text(relative, cell_id):
    return source(next(cell for cell in notebook(relative)["cells"] if cell["id"] == cell_id))


def syntax(relative):
    return ast.parse((ROOT / relative).read_text(encoding="utf-8"))


def test_introduction_separates_spatial_derivatives_from_optimizer_parameters():
    text = cell_text(INTRO, "8fd2608f0a62")
    for operation in ("PhysicsInformer", "loss.backward()", "optimizer.step()"):
        assert operation in text
    assert re.search(r"spatial (coordinate|derivative)", text, re.IGNORECASE)
    assert re.search(r"network parameters", text, re.IGNORECASE)
    assert re.search(r"coordinates[^.\n]*inputs", text, re.IGNORECASE)
    assert re.search(r"not[^.\n]*training target", text, re.IGNORECASE)
    assert all(cell["cell_type"] == "markdown" for cell in notebook(INTRO)["cells"])


def test_introduction_names_actual_lab_optimizers_and_lab1_loop():
    workflow = cell_text(INTRO, "3bb368ef0295")
    assert "L-BFGS in Lab 1" in workflow
    assert "Adam in Labs 2 and 4" in workflow
    assert "Adam followed by L-BFGS in Lab 3" in workflow
    bridge = cell_text(INTRO, "8fd2608f0a62")
    assert "optimize_lab" in bridge and "optimizer.step(closure)" in bridge
    assert "ETC/runtime/labs.py" not in bridge


def test_lab1_parameterized_description_matches_fixed_grid_and_evaluation_scope():
    text = next(source(cell) for cell in notebook(LAB1)["cells"]
                if "## Lab 1.2: Parameterized problems" in source(cell))
    assert "17 equally spaced lengths" in text and "32 midpoint positions" in text
    assert "five lengths are also in the training grid" in text
    assert "do not establish accuracy at new lengths" in text
    assert "Sample $l_i" not in text


def test_legacy_tutorial_and_setup_do_not_reintroduce_old_lab4_defaults():
    legacy = (ROOT / "ETC/legacy/tutorial/readme.md").read_text(encoding="utf-8")
    assert "SMOKE_DATA=False" not in legacy
    assert "no dataset-selection switch" in legacy
    assert "separate internal test fixture, not the student lesson" in legacy
    assert "CPU and 200 steps" not in legacy
    setup = (ROOT / "ETC/environment/SETUP.md").read_text(encoding="utf-8")
    rehearsal = setup.split("For an instructor's short CUDA rehearsal:", 1)[1].split("## Read Markdown", 1)[0]
    data_guide = setup.split("## Data, edits, and saved runs", 1)[1].split("## Short checks", 1)[0]
    for section in (rehearsal, data_guide):
        assert "3,000-update class preset" in section
        assert "`--recipe upstream`" in section and "50,000-update settings" in section
    assert "Lab 4 restores the original 50,000-update budget" not in setup


def test_course_guides_distinguish_climate_baseline_and_planned_gpu_capacity():
    for relative in ("ETC/course_materials/README.md", "ETC/course_materials/course-plan.md"):
        text = (ROOT / relative).read_text(encoding="utf-8")
        assert "uncoupled case (`gamma0=0`)" in text
        assert "nonzero exchange is a separate local experiment" in text
        assert "active atmosphere–ocean exchange" not in text
    plan = (ROOT / "ETC/course_materials/course-plan.md").read_text(encoding="utf-8")
    assert "one Brev L4 instance per participant" in plan
    assert "Availability and concurrent startup capacity still require confirmation" in plan
    assert "Participant training resources are undecided" not in plan


def test_course_guide_requires_complete_challenge_contracts_not_only_equations():
    guide = (ROOT / "ETC/course_materials/README.md").read_text(encoding="utf-8")
    for name in ("student_equations", "student_conditions", "student_speed",
                 "student_geometry", "student_parameters", "student_solution"):
        assert f"`{name}`" in guide
    assert "CHALLENGE_CONTRACTS.md" in guide
    assert "Complete all marked functions for the level" in guide


def test_lab1_bridge_precedes_training_and_names_real_program_functions():
    cells = notebook(LAB1)["cells"]
    bridge_index = next(i for i, cell in enumerate(cells) if cell["id"] == "b6d6718e1a8d")
    first_training_index = next(i for i, cell in enumerate(cells)
                                if cell["cell_type"] == "code" and "subprocess.run" in source(cell))
    assert bridge_index < first_training_index
    text = source(cells[bridge_index])
    for concept in ("BasicPINN.forward", "loss_terms", "PhysicsInformer", "optimize",
                    "loss.backward()", "optimizer.step(closure)"):
        assert concept in text
    assert re.search(r"\bsoft\b", text, re.IGNORECASE)
    assert re.search(r"parameter gradients", text, re.IGNORECASE)
    basics = syntax("01_labs/01_pinn/source_code/pinn_basics.py")
    model = next(node for node in basics.body if isinstance(node, ast.ClassDef) and node.name == "BasicPINN")
    assert any(isinstance(node, ast.FunctionDef) and node.name == "forward" for node in model.body)
    assert any(isinstance(node, ast.FunctionDef) and node.name == "loss_terms" for node in basics.body)
    assert any(isinstance(node, ast.FunctionDef) and node.name == "optimize_lab" for node in basics.body)
    assert "source_code/pinn_basics.py" in text


def test_heat_reduction_keeps_conductivity_and_diffusivity_distinct():
    text = cell_text(LAB3, "411b22b0a5c7")
    for concept in (r"\rho c_p T_t", "heat capacity", "conductivity", "T_t=0", "q=0",
                    "ordinary differential equation", "diffusivity", r"D=k/(\rho c_p)",
                    "heat flux", "temperature_jumps", "physical_flux_jumps"):
        assert concept in text
    assert "no time input" in text
    assert re.search(r"`D1`[^.\n]*`D2`[^.\n]*conductivity", text)


def test_lab4_original_data_and_result_scope_are_immediately_before_training():
    cells = notebook(LAB4)["cells"]
    training_index = next(i for i, cell in enumerate(cells)
                          if cell["cell_type"] == "code" and "subprocess.run" in source(cell))
    explanation = cells[training_index - 1]
    assert explanation["cell_type"] == "markdown"
    text = source(explanation)
    for concept in ("upstream_data_lat_legacy_normalization", "heldout_after.initial_data_rmse",
                    "heldout_after.pde_rmse", "initial_data", "loss.csv",
                    "weather_forecast_validated", "DATA_PROVENANCE.md"):
        assert concept in text
    full_text = "\n".join(source(cell) for cell in cells)
    assert "SMOKE_DATA" not in full_text
    assert "--smoke-data" not in full_text
    assert "Taylor" not in full_text
    assert "load_original_flow(OUTPUT)" in source(cells[training_index])
    assert re.search(r"no[^.\n]*future[- ]weather targets", text, re.IGNORECASE)
    assert re.search(r"weather_forecast_validated[^.\n]*false", text, re.IGNORECASE)
    code_strings = {node.value for node in ast.walk(syntax("01_labs/04_navier_stokes/source_code/navier_stokes.py"))
                    if isinstance(node, ast.Constant) and isinstance(node.value, str)}
    assert {"data_kind", "synthetic_reference_rmse", "heldout_after", "pde_rmse",
            "initial_data", "weather_forecast_validated"} <= code_strings


def test_lab4_transition_links_all_challenges_and_explains_student_mode():
    text = cell_text(LAB4, "17e8bef0605d")
    manifest = json.loads((ROOT / "ETC/course_materials/course_manifest.json").read_text(encoding="utf-8"))
    courses = {course["id"]: course for course in manifest["course"]}
    targets = re.findall(r"\[[^\]]+\]\(([^)]+)\)", text)
    linked_files = [(ROOT / LAB4).parent.joinpath(target.split("#", 1)[0]).resolve()
                    for target in targets]
    positions = []
    for course, level_count in (("wave", 3), ("fluid", 3), ("climate", 2), ("operators", 3)):
        levels = sorted(run["level"] for run in manifest["runs"] if run["course"] == course)
        assert levels == list(range(1, level_count + 1))
        positions.append(linked_files.index((ROOT / courses[course]["notebook"]).resolve()))
    assert positions == sorted(positions)
    for concept in ("student_equations", "Python file", "save", "PDE", "boundary-condition",
                    "USE_REFERENCE=False", "USE_REFERENCE=True"):
        assert concept in text
    assert re.search(r"trains? a new model", text)
    assert re.search(r"not[^.\n]*pretrained", text)


@pytest.mark.parametrize("relative,cell_id", [
    (INTRO, "8fd2608f0a62"), (LAB1, "b6d6718e1a8d"),
    (LAB4, "aa0b7cd3ddc6"), (LAB4, "17e8bef0605d"),
])
def test_teaching_bridges_link_to_existing_local_sources(relative, cell_id):
    text = cell_text(relative, cell_id)
    targets = re.findall(r"\[[^\]]+\]\(([^)]+)\)", text)
    assert targets
    for target in targets:
        assert "://" not in target, "These bridges should point to this course's actual files"
        path = (ROOT / relative).parent / target.split("#", 1)[0]
        assert path.resolve().is_relative_to(ROOT)
        assert path.is_file(), f"Missing source link in {relative}: {target}"
