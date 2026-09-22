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
from operator_training import checked_datasets, predict_field, reference_datasets


def sample_splits():
    return [(torch.full((2, 1, 4, 4), float(index)),
             torch.full((2, 1, 4, 4), float(index + 10))) for index in range(3)]


def test_dataset_exercise_keeps_canonical_evaluation_objects():
    pairs = sample_splits()
    student_objects = []

    def builder(*provided):
        student_objects.extend(reference_datasets(*provided))
        return student_objects

    datasets = checked_datasets(pairs, builder)
    assert datasets[0] is student_objects[0]
    for index in (1, 2):
        assert datasets[index] is not student_objects[index]
        assert datasets[index].tensors[1] is pairs[index][1]
        student_objects[index].tensors[1].zero_()
        torch.testing.assert_close(datasets[index].tensors[1], pairs[index][1])
        assert datasets[index].tensors[1].mean() == index + 10


@pytest.mark.parametrize("kind", ["swapped", "changed_target", "changed_input", "reordered"])
def test_dataset_exercise_rejects_modified_splits(kind):
    pairs = sample_splits()
    pairs[2][0][1] += 1
    originals = [tuple(value.clone() for value in pair) for pair in pairs]

    def builder(train, val, test):
        if kind == "swapped":
            return reference_datasets(test, val, train)
        if kind == "changed_target":
            test[1].zero_()
        elif kind == "changed_input":
            test[0].mul_(2)
        elif kind == "reordered":
            test = tuple(value.flip(0) for value in test)
        return reference_datasets(train, val, test)

    with pytest.raises(ValueError, match="preserve the provided split"):
        checked_datasets(pairs, builder)
    for pair, original in zip(pairs, originals):
        for value, expected in zip(pair, original):
            torch.testing.assert_close(value, expected)


def test_dataset_exercise_rejects_custom_wrapper_and_missing_split():
    class Wrapper(TensorDataset):
        pass

    with pytest.raises(ValueError, match="without a custom dataset wrapper"):
        checked_datasets(sample_splits(), lambda *pairs: tuple(Wrapper(*pair) for pair in pairs))
    with pytest.raises(ValueError, match="train, validation and test"):
        checked_datasets(sample_splits(), lambda *pairs: reference_datasets(*pairs)[:2])


def test_operator_prediction_rejects_broadcast_shape_and_nonfinite_fields():
    inputs = torch.zeros(2, 1, 4, 4)
    assert predict_field(torch.nn.Identity(), inputs) is inputs
    with pytest.raises(ValueError, match="same.*shape"):
        predict_field(lambda values: values.mean(dim=(-1, -2), keepdim=True), inputs)
    with pytest.raises(FloatingPointError, match="Non-finite"):
        predict_field(lambda values: values + float("nan"), inputs)


def test_operator_run_records_canonical_local_feedback(tmp_path):
    import yaml
    from generate_data import generate_splits
    from operator_training import run

    data_dir = tmp_path / "data"
    generate_splits(data_dir, train_samples=3, val_samples=2, test_samples=2,
                    grid_size=8, max_mode=2)
    config = {"data": {"grid_size": 8}, "model": {},
              "training": {"steps": 1, "batch_size": 2, "learning_rate": .001, "cpu_threads": 2}}
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    metrics = run(1, lambda config, size: torch.nn.Conv2d(1, 1, 1),
                  argv=["--device", "cpu", "--data-dir", str(data_dir),
                        "--config", str(config_path), "--output-dir", str(tmp_path / "output")])
    assert metrics["assessment"] == {"kind": "local_practice_feedback", "official_score": None,
                                     "ranking_ready": False}
    assert metrics["dataset_exercise_checked"] is True
    assert metrics["evaluation_data"] == "canonical_local_validation_and_test_splits"
    assert metrics["evaluation_equations"] == "provided_reference_equations"
    assert metrics["split_samples"] == {"train": 3, "val": 2, "test": 2}
    assert metrics["test"]["pde_rmse_fft"] >= 0


@pytest.fixture(scope="module")
def flow_module():
    path = ROOT / "01_labs/04_navier_stokes/source_code/navier_stokes.py"
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
