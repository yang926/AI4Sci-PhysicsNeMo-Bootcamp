"""Local residual feedback is student-defined, never authoritative scoring."""
import math
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from ETC.runtime.pinn import create_evaluation_informer, reference_errors_over_time
from ETC.runtime import pinn
from ETC.runtime.notebook import result_html


class Field(torch.nn.Module):
    def __init__(self, function):
        super().__init__()
        self.function = function

    def forward(self, inputs):
        return self.function(inputs)


def test_local_evaluation_builds_student_pde_without_reference_dispatch():
    calls = []
    class StudentPDE:
        def __init__(self, *, coefficient):
            calls.append(coefficient)
    with patch("ETC.runtime.pinn.create_informer", side_effect=lambda pde, device: (pde, device)):
        pde, device = create_evaluation_informer(StudentPDE, "cpu", coefficient=7)
    assert isinstance(pde, StudentPDE) and device == "cpu" and calls == [7]


def test_local_evaluation_rejects_reference_mode_argument():
    with pytest.raises(ValueError, match="Reference mode"):
        create_evaluation_informer(object, "cpu", reference=True)


def test_reference_evaluation_includes_initial_and_final_times():
    xy = torch.tensor([[.4, .8], [1.1, 1.4]])
    truth = lambda xy, time: xy[:, :1] * (1 + time)
    model = Field(lambda inputs: truth(inputs[:, :2], inputs[:, 2:3]))
    result = reference_errors_over_time(model, xy, 4., ["u"], truth)
    assert result["rmse"] < 1e-7
    assert len(result["per_time"]) == 5
    assert result["per_time"][0]["time"] == 0
    assert result["per_time"][-1]["time"] == 4.


def test_zero_prediction_does_not_pass_a_decaying_toy_reference():
    xy = torch.tensor([[.4, .8], [1.1, 1.4]])
    truth = lambda xy, time: 2*torch.exp(-3*time)
    result = reference_errors_over_time(Field(lambda inputs: torch.zeros_like(inputs[:, :1])),
                                       xy, 5., ["toy"], truth)
    assert result["relative_l2"] == pytest.approx(1.)
    assert result["rmse"] > .1 and result["per_time"][-1]["rmse"] < 1e-5


def test_missing_reference_is_not_reported_as_zero_error():
    result = reference_errors_over_time(Field(lambda inputs: inputs[:, :1]), torch.ones(3, 2),
                                       1., ["u"], lambda xy, time: None)
    assert result == {}


@pytest.mark.parametrize("fractions", [(), (-.1,), (1.1,), (float("nan"),)])
def test_reference_rejects_invalid_time_samples(fractions):
    with pytest.raises(ValueError, match="time_fractions"):
        reference_errors_over_time(Field(lambda inputs: inputs[:, :1]), torch.ones(3, 2),
                                   1., ["u"], lambda xy, time: xy[:, :1], fractions)


@pytest.mark.parametrize("include_temperature_scale", (False, True))
def test_temperature_table_keeps_absolute_error_and_optional_scale_columns(include_temperature_scale):
    row = {"time": .5, "rmse": .0002, "relative_l2": .125}
    if include_temperature_scale:
        row.update({"reference_max": .0019, "initial_scale_rmse": .0004})
    result = result_html({
        "comparison_source": "student_solution",
        "reference_over_time": {"per_time": [row]},
    }, "/tmp/climate-run").split("<details>")[0]
    assert "Comparison with your student_solution (not independently checked)" in result
    assert '<th scope="col">RMSE</th>' in result
    assert '<th scope="col">Relative L2</th>' in result
    assert '<td>0.0002</td><td>0.125</td>' in result
    if include_temperature_scale:
        assert '<th scope="col">Expression max |T|</th>' in result
        assert '<th scope="col">RMSE / initial RMS</th>' in result
        assert '<td>0.0019</td><td>0.0004</td>' in result
    else:
        assert "Expression max |T|" not in result
        assert "RMSE / initial RMS" not in result


@pytest.mark.parametrize("preview_time", (None, math.pi))
def test_temperature_preview_titles_identify_the_selected_time(preview_time, tmp_path, monkeypatch):
    import matplotlib.pyplot as plt

    subplots = plt.subplots
    captured = []
    def capture(*args, **kwargs):
        figure, axes = subplots(*args, **kwargs)
        captured.append(axes)
        return figure, axes
    monkeypatch.setattr(plt, "subplots", capture)
    xy = torch.tensor([[.2, .3], [.5, .7], [.8, .4]])
    metrics = {"comparison_source": "student_solution"}
    if preview_time is not None:
        metrics["preview_time"] = preview_time
    args = SimpleNamespace(output_dir=tmp_path / "run", seed=42, steps=1,
                           device="cpu", reference=False)
    pinn.save_results(args, {}, Field(lambda inputs: inputs[:, :1]),
                      [{"step": 1, "phase": "training", "total": 1.}],
                      xy, None, ["T"], metrics, reference=xy[:, :1])
    suffix = "" if preview_time is None else ", t=3.142"
    assert [ax.get_title() for ax in captured[0][0]] == [
        title + suffix for title in ("Student expression T", "Prediction T", "Absolute difference T")]
    assert (args.output_dir / "preview.png").is_file()
