"""Local residual feedback is student-defined, never authoritative scoring."""
from unittest.mock import patch

import pytest
import torch

from ETC.runtime.pinn import create_evaluation_informer, reference_errors_over_time


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
