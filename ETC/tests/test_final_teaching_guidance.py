"""Student instructions must describe the existing controls and scoring contract."""

import json
from pathlib import Path
import re

import pytest


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = json.loads((ROOT / "ETC/course_materials/course_manifest.json").read_text())
CHALLENGES = ("wave", "fluid", "climate", "operators")


def markdown(relative):
    notebook = json.loads((ROOT / relative).read_text(encoding="utf-8"))
    return "\n".join("".join(cell["source"]) for cell in notebook["cells"]
                     if cell["cell_type"] == "markdown")


@pytest.mark.parametrize("course", CHALLENGES)
def test_challenge_explains_offline_preflight_and_all_level_average(course):
    lesson = next(item for item in MANIFEST["course"] if item["id"] == course)
    levels = [item for item in MANIFEST["runs"] if item["course"] == course]
    text = markdown(lesson["notebook"])
    assert "Check saved code" in text
    assert "no training or judge connection" in text
    assert "submits nothing" in text
    assert "Ctrl/Cmd-click" in text
    assert "average" in text
    assert "omitted Levels worth zero" in text
    example = f"{100 / len(levels):.2f}".removesuffix(".00")
    assert f"Level 1 alone gives {example}/100" in text


def test_start_here_explains_mode_saving_preflight_and_nickname():
    text = markdown("Start_Here.ipynb")
    for instruction in ("Recommended background", "first and second derivatives",
                        "setup cell first", "Challenges run only your saved exercise functions", "Run All",
                        "save the file", "Check saved code", "Register nickname",
                        "Submit code", "separate attempts are not combined"):
        assert instruction in text


def test_event_bootstrap_example_opts_into_enrollment():
    text = (ROOT / "ETC/launchable/README.md").read_text()
    examples = re.findall(r"```bash\n(.*?)```", text, re.DOTALL)
    bootstrap = next(block for block in examples if "bootstrap_file=" in block)
    assert bootstrap.startswith("#!/bin/bash\n")
    assert 'python3 "$bootstrap_file" --launchable --enroll-event ai4science-korea-2026' in bootstrap
    assert "generic" in text and "does not opt in to event enrollment" in text


def test_manual_class_start_does_not_inherit_short_training_budget():
    text = (ROOT / "ETC/environment/SETUP.md").read_text()
    examples = re.findall(r"```bash\n(.*?)```", text, re.DOTALL)
    command = next(block for block in examples if "jupyter lab --no-browser" in block)
    assert "env -u AI4SCI_STEPS" in command
    assert "AI4SCI_STEPS=20" not in command


@pytest.mark.parametrize("relative", [
    "01_labs/02_projectile/Lab_2_Projectile_Motion.ipynb",
    "01_labs/03_heat_conduction/Lab_3_Heat_Conduction.ipynb",
])
def test_lab_configuration_instructions_explain_notebook_step_precedence(relative):
    text = markdown(relative)
    assert "edit `STEPS`" in text
    assert "overrides the YAML `steps` value" in text
