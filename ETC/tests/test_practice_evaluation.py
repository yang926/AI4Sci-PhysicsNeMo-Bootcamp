"""Local feedback must test the stated problem, not the learner's chosen PDE."""
import importlib
import math
from pathlib import Path
import sys
from unittest.mock import patch

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ETC.runtime.pinn import (create_informer, create_evaluation_informer,
                              reference_errors_over_time, residuals)

wave = importlib.import_module("02_challenges.01_wave.wave_l1")


class Field(torch.nn.Module):
    def __init__(self, function):
        super().__init__()
        self.function = function

    def forward(self, inputs):
        return self.function(inputs)


def test_incorrect_student_pde_cannot_lower_canonical_residual():
    def wrong_equation(x, y, t, u, c=1.):
        return {"wave": u.diff(t, 2)}

    xy = torch.tensor([[.2, .3], [.5, .7]])
    time = torch.ones(2, 1)
    model = Field(lambda a: .5 * a[:, :2].square().sum(1, keepdim=True)
                  + 0 * a[:, 2:3].pow(3))
    with patch.object(wave, "student_equations", wrong_equation):
        student = create_informer(wave.WaveEquation2D(), "cpu")
        canonical = create_evaluation_informer(wave.WaveEquation2D, "cpu")
    torch.testing.assert_close(residuals(model, student, xy, time, ["u"])["wave"],
                               torch.zeros(2, 1))
    torch.testing.assert_close(residuals(model, canonical, xy, time, ["u"])["wave"],
                               torch.full((2, 1), -2.))


def test_evaluation_cannot_select_student_equations():
    with pytest.raises(ValueError, match="always uses"):
        create_evaluation_informer(wave.WaveEquation2D, "cpu", reference=False)


def test_reference_evaluation_includes_initial_and_final_times():
    xy = torch.tensor([[.4, .8], [1.1, 1.4]])
    model = Field(lambda a: wave.exact_reference(a[:, :2], a[:, 2:3]))
    result = reference_errors_over_time(model, xy, 2 * math.pi, ["u"], wave.exact_reference)
    assert result["rmse"] < 1e-7
    assert len(result["per_time"]) == 5
    assert result["per_time"][0]["time"] == 0
    assert result["per_time"][-1]["time"] == 2 * math.pi


def test_zero_prediction_does_not_pass_decayed_temperature_reference():
    xy = torch.tensor([[.4, .8], [1.1, 1.4]])
    def heat_reference(xy, time):
        return torch.sin(xy[:, :1]) * torch.sin(xy[:, 1:2]) * torch.exp(-2 * time)
    model = Field(lambda a: torch.zeros_like(a[:, :1]))
    result = reference_errors_over_time(model, xy, 2 * math.pi, ["T"], heat_reference)
    assert result["relative_l2"] == pytest.approx(1.)
    assert result["rmse"] > .1
    assert result["per_time"][-1]["rmse"] < 1e-5


def test_missing_reference_is_not_reported_as_zero_error():
    xy = torch.ones(3, 2)
    result = reference_errors_over_time(Field(lambda a: a[:, :1]), xy, 1., ["u"],
                                       lambda xy, time: None)
    assert result == {}


@pytest.mark.parametrize("fractions", [(), (-.1,), (1.1,), (float("nan"),)])
def test_reference_rejects_invalid_time_samples(fractions):
    with pytest.raises(ValueError, match="time_fractions"):
        reference_errors_over_time(Field(lambda a: a[:, :1]), torch.ones(3, 2),
                                   1., ["u"], lambda xy, time: xy[:, :1], fractions)
