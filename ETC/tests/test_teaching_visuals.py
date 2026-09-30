"""Teaching pictures stay local, visible, and separate from executable lessons."""
import hashlib
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import nbformat
import pytest

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "ETC/assets/teaching"
# Full executable cell objects at 0341a99, before this illustration-only change.
CODE_HASHES = {
    "01_labs/01_pinn/Lab_1_PINN_Fundamentals.ipynb": "b0690a0b4f9cb97b134053e56d7ef88fbccc7eead5862d3ff49c16e6e2e41f25",
    "01_labs/02_projectile/Lab_2_Projectile_Motion.ipynb": "b098408cba8f1d09d663a4511f59d436a8d400a3b6e2fbae2fb1a6633ca117eb",
    "01_labs/03_heat_conduction/Lab_3_Heat_Conduction.ipynb": "10efa63b515b7e83b83c22ce63ecb0e46a0aff09fa6faf4cb1851485a99e2ac5",
    "01_labs/04_weather_forecasting/Lab_4_Weather_Forecasting.ipynb": "7d92dc28f9495b2a6625072685f786459de11810271b4a387858eef5c19316eb",
    "02_challenges/01_wave/Challenge_1_Wave_Dynamics.ipynb": "fb2d2e819a5a896cbc5ee3ae56093161d6e8fdf9b6f6231cd8c83e07a5560887",
    "02_challenges/02_fluid/Challenge_2_Fluid_Flow.ipynb": "74367d9abafbbfac64967df5dc075cefcbf2afe39c57ab31186d0c6cdf9372d6",
    "02_challenges/03_climate/Challenge_3_Climate_Modeling.ipynb": "c17b0b64c13997c65c24fc1a82243c892847cf60f6531baffe9bd3692204d2ba",
    "02_challenges/04_neural_operators/Challenge_4_Neural_Operators.ipynb": "3e764c52d8e983196087f2ee287302cd9deed60d7f619cfceaeb083014d029ee"
}
NS = "{http://www.w3.org/2000/svg}"


def text(cell):
    return "".join(cell["source"])


@pytest.mark.parametrize("relative,digest", CODE_HASHES.items())
def test_visual_changes_preserve_all_code_cells(relative, digest):
    notebook = json.loads((ROOT / relative).read_text())
    nbformat.validate(nbformat.from_dict(notebook))
    code = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert hashlib.sha256(json.dumps(code, sort_keys=True, ensure_ascii=False).encode()).hexdigest() == digest


@pytest.mark.parametrize("relative", CODE_HASHES)
def test_each_lesson_embeds_geometry_and_flow_before_execution(relative):
    path = ROOT / relative
    cells = json.loads(path.read_text())["cells"]
    execution = next(i for i, cell in enumerate(cells)
                     if cell["cell_type"] == "code" and
                     ("subprocess.run" in text(cell) or "run_script(" in text(cell)))
    embedded = []
    for i, cell in enumerate(cells):
        if cell["cell_type"] != "markdown":
            continue
        for alt, target in re.findall(r"!\[([^\]]*)\]\(([^)]+)\)", text(cell)):
            if "/assets/teaching/" not in target:
                continue
            assert i < execution
            assert len(alt) >= 30, "Include a meaningful text alternative"
            assert "://" not in target
            image = (path.parent / target).resolve()
            assert image.parent == ASSETS and image.is_file()
            embedded.append(image.name)
    assert len(embedded) >= 2
    assert any("domain" in name for name in embedded)
    assert any("flow" in name or "workflow" in name for name in embedded)


@pytest.mark.parametrize("path", sorted(ASSETS.glob("*.svg")), ids=lambda p: p.name)
def test_svg_assets_are_accessible_offline_vector_diagrams(path):
    xml = ET.parse(path).getroot()
    assert xml.tag == NS + "svg"
    assert xml.get("viewBox")
    assert xml.get("role") == "img"
    assert "".join(xml.find(NS + "title").itertext()).strip()
    assert "".join(xml.find(NS + "desc").itertext()).strip()
    labels = " ".join(xml.itertext())
    assert not re.search(r"[\uac00-\ud7a3]", labels)
    assert "reference_equations" not in labels
    assert "USE_REFERENCE" not in labels
    for node in xml.iter():
        assert node.tag not in {NS + "script", NS + "foreignObject", NS + "image"}
        for key, value in node.attrib.items():
            assert not key.lower().startswith("on")
            if key.endswith("href"):
                assert value.startswith("#")
            assert not re.search(r"url\(\s*['\"]?https?://", value)
    assert sum(node.tag in {NS + name for name in ("path", "rect", "circle", "line", "polyline")}
               for node in xml.iter()) >= 5


def test_wave_geometry_is_illustrated_not_only_described():
    notebook = json.loads((ROOT / "02_challenges/01_wave/Challenge_1_Wave_Dynamics.ipynb").read_text())
    cell = next(cell for cell in notebook["cells"] if cell.get("id") == "wave-domain-and-sampling")
    assert "challenge1-circle.svg" in text(cell)
    diagram = ET.parse(ASSETS / "challenge1-circle.svg").getroot()
    labels = " ".join(diagram.itertext())
    for term in ("sample_circle", "r cos θ", "r sin θ", "sqrt(rand_like(θ))",
                 "ones_like(θ)", "r = 1", "Uniform area"):
        assert term in labels
    assert diagram.find(".//" + NS + "circle") is not None
    domain = " ".join(ET.parse(ASSETS / "challenge1-domain.svg").getroot().itertext())
    assert "TIME_END = 2π" in domain and "TIME_END = 3" in domain


def test_weather_visual_is_inference_not_the_archived_pinn():
    labels = " ".join(ET.parse(ASSETS / "lab4-workflow.svg").getroot().itertext())
    assert "inference_mode" in labels and "AFNO" in labels
    assert "truth.npz" in labels and "fixed" in labels.lower()
    assert "no optimizer" in labels.lower()
    assert "Taylor" not in labels

