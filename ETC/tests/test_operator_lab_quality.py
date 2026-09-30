"""Dataset integrity and physically interpretable Lab feedback; CPU only."""
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch
from torch.utils.data import TensorDataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "02_challenges/04_neural_operators"))
from operator_training import predict_field










def test_operator_prediction_rejects_broadcast_shape_and_nonfinite_fields():
    inputs = torch.zeros(2, 1, 4, 4)
    assert predict_field(torch.nn.Identity(), inputs) is inputs
    with pytest.raises(ValueError, match="same.*shape"):
        predict_field(lambda values: values.mean(dim=(-1, -2), keepdim=True), inputs)
    with pytest.raises(FloatingPointError, match="Non-finite"):
        predict_field(lambda values: values + float("nan"), inputs)




@pytest.fixture(scope="module")
def flow_module():
    path = ROOT / "ETC/reference_labs/04_navier_stokes/source_code/navier_stokes.py"
    spec = importlib.util.spec_from_file_location("lab_flow_quality", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pressure_gauge_alignment_removes_only_each_time_constant(flow_module):
    reference = torch.zeros(3, 5, 3)
    prediction = reference.clone()
    prediction[..., 2] = torch.tensor([-5., 0., 6.])[:, None]
    result = flow_module.solution_errors(prediction, reference)
    assert result["velocity_rmse"] == 0
    assert result["pressure_gauge_aligned_rmse"] == 0
    assert result["pressure_rmse"] > 0
    assert result["pressure_offset_rmse"] == result["pressure_rmse"]


def test_pressure_gauge_alignment_keeps_spatial_and_velocity_error(flow_module):
    reference = torch.zeros(5, 3)
    prediction = reference.clone()
    prediction[:, :2] = 2
    prediction[:, 2] = torch.arange(-2., 3.) + 7
    result = flow_module.solution_errors(prediction, reference)
    assert result["velocity_rmse"] == 2
    assert result["pressure_offset_rmse"] == 7
    assert result["pressure_gauge_aligned_rmse"] == pytest.approx(2 ** .5)
    with pytest.raises(ValueError, match="matching"):
        flow_module.solution_errors(torch.zeros(5, 2), reference)


def test_notebook_equation_and_feedback_language():
    notebook = json.loads((ROOT / "01_labs/01_pinn/Lab_1_PINN_Fundamentals.ipynb").read_text())
    markdown = "\n".join("".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "markdown")
    assert "(x_i, l_i) - f_{net}(x_i)" not in markdown
    notebook = json.loads((ROOT / "02_challenges/04_neural_operators/Challenge_4_Neural_Operators.ipynb").read_text())
    markdown = "\n".join("".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "markdown")
    assert "practice feedback" in markdown
    assert "official competition scores and ranking rules have not been defined" in markdown
    assert "Raw training losses are not directly comparable because the objectives differ" in markdown
