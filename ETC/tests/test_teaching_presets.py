"""Keep notebook budgets and executable course presets in agreement."""
import ast
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def notebook_source(relative):
    notebook = json.loads((ROOT / relative).read_text())
    return "\n".join("".join(cell["source"]) for cell in notebook["cells"]
                     if cell["cell_type"] == "code")


def test_wave_level_budgets_and_config():
    source = notebook_source("02_challenges/01_wave/Challenge_1_Wave_Dynamics.ipynb")
    tree = ast.parse(source)
    assignment = next(node for node in tree.body if isinstance(node, ast.Assign)
                      and any(isinstance(target, ast.Name) and target.id == "STEPS"
                              for target in node.targets))
    assert ast.literal_eval(assignment.value) == {1: 20000, 2: 10000, 3: 10000}
    config = yaml.safe_load((ROOT / "02_challenges/01_wave/conf/config_wave_l1.yaml").read_text())
    assert config["training"]["steps"] == 20000
    assert config["samples"] == {"interior": 512, "initial": 256, "boundary": 256}
    assert '"config_wave_l1.yaml"' in (ROOT / "02_challenges/01_wave/wave_l1.py").read_text()
    for level in (1, 2, 3):
        assert f"str(STEPS[{level}])" in source
        assert f"steps=STEPS[{level}]" in source


def test_operators_keep_full_models_and_common_budget():
    folder = ROOT / "02_challenges/04_neural_operators"
    for name in ("FNO", "AFNO", "PINO"):
        config = yaml.safe_load((folder / f"conf/config_{name}.yaml").read_text())
        assert config["training"]["steps"] == 3000
        assert config["data"]["grid_size"] == 64
    afno = yaml.safe_load((folder / "conf/config_AFNO.yaml").read_text())
    assert afno["model"]["patch_size"] == [8, 8]
    assert afno["model"]["embed_dim"] == 256
    source = notebook_source("02_challenges/04_neural_operators/Challenge_4_Neural_Operators.ipynb")
    assert 'os.environ.get("AI4SCI_STEPS", "3000")' in source


def test_lab4_measured_class_budget_keeps_upstream_option():
    folder = ROOT / "01_labs/04_navier_stokes/source_code/conf"
    classroom = yaml.safe_load((folder / "config.yaml").read_text())
    upstream = yaml.safe_load((folder / "upstream.yaml").read_text())
    assert classroom["steps"] == 3000
    assert upstream["steps"] == 50000
    for config in (classroom, upstream):
        assert config["layer_size"] == 256
        assert config["num_layers"] == 6


def test_fluid_and_climate_no_longer_default_to_execution_checks():
    for folder, notebook, configs in (
        ("02_fluid", "Challenge_2_Fluid_Flow.ipynb", ["config_chip_2d.yaml"]),
        ("03_climate", "Challenge_3_Climate_Modeling.ipynb", ["config_atmos.yaml", "config_coupled.yaml"]),
    ):
        source = notebook_source(f"02_challenges/{folder}/{notebook}")
        assert 'os.environ.get("AI4SCI_STEPS", "10000")' in source
        for name in configs:
            config = yaml.safe_load((ROOT / "02_challenges" / folder / "conf" / name).read_text())
            assert config["training"]["steps"] == 10000
