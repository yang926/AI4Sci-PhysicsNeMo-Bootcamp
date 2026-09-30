"""The active weather lesson is honest inference, not a renamed PINN exercise."""
import ast
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
LAB = ROOT / "01_labs/04_weather_forecasting"


def test_active_weather_lesson_is_implemented_and_separate_from_reference():
    manifest = json.loads((ROOT / "ETC/course_materials/course_manifest.json").read_text())
    course = next(x for x in manifest["course"] if x["id"] == "lab4")
    assert course["notebook"] == "01_labs/04_weather_forecasting/Lab_4_Weather_Forecasting.ipynb"
    assert manifest["inference_runs"][0]["course"] == "lab4"
    assert len(manifest["runs"]) == 17
    assert [r["id"] for r in manifest["reference_runs"]] == ["navier_stokes"]
    code = (LAB / "source_code/run_forecast.py").read_text()
    assert "AFNO.from_checkpoint" in code and "torch.inference_mode()" in code
    assert "torch.optim" not in code and ".backward(" not in code
    assert "--truth" not in code


def test_student_notebook_is_clean_english_and_teaches_verification():
    book = json.loads((LAB / "Lab_4_Weather_Forecasting.ipynb").read_text())
    text = "\n".join("".join(c["source"]) for c in book["cells"])
    assert not re.search(r"[\uac00-\ud7a3]", text)
    for concept in ("FourCastNet", "ERA5", "persistence", "48", "26", "pretrained"):
        assert concept in text
    assert "ETC/reference_labs/04_navier_stokes" in text
    for cell in book["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))
            assert not cell["outputs"] and cell["execution_count"] is None


def test_weather_cache_is_not_a_tracked_model_or_dataset():
    assert not (LAB / "model").exists()
    assert not (LAB / "data-20220901").exists()
    installer = (ROOT / "ETC/launchable/install.py").read_text()
    assert "prepare_weather_assets(course, prefix, Path.home())" in installer
    assert ".cache/ai4sci/weather" in installer
