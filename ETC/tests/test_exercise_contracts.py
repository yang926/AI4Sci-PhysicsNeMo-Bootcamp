"""Neutral learner contracts without canonical exercise answers."""
import pytest
import sympy as sp
import torch

from ETC.judge.contracts import source_nodes
from ETC.judge.expressions import SubmissionError
from ETC.runtime.exercises import block_geometry, condition_tensor, conditions
from ETC.runtime.submission import collect_submission
from ETC.tests.test_judge import dummy_source


def test_old_pde_only_submission_requires_other_exercises():
    source = 'def student_equations(x,y,t,u,c=1.0):\n    return {}\n'
    with pytest.raises(SubmissionError, match="student_conditions"):
        source_nodes(source, "1")


def test_collection_only_extracts_signatures_without_executing_source(tmp_path):
    marker = tmp_path / "MUST_NOT_EXIST"
    source = f'open({str(marker)!r}, "w").write("bad")\n' + dummy_source("1", "wave_l1.py")
    (tmp_path / "wave_l1.py").write_text(source)
    payload = collect_submission("1", tmp_path)
    assert "open(" not in payload["sources"]["wave_l1.py"]
    assert not marker.exists()


@pytest.mark.parametrize("blocks", [
    (), ((-3, 0, .2),), ((-.8, -.2, .5),),
    ((-.8, .1, -.1), (0, .7, .2)), ((-.8, "zero", .2),),
])
def test_invalid_geometry_cannot_enter_sampling_loop(blocks):
    with pytest.raises(ValueError):
        block_geometry(lambda: {"blocks": blocks})


def test_generic_condition_tensor_keeps_autograd():
    x, y, t = sp.symbols("x y t")
    u = sp.Function("u")(x, y, t)
    xy = torch.tensor([[.2, .4], [.7, -.1]], requires_grad=True)
    value = xy[:, :1].square()
    expression = {"toy_boundary": 3*u + x}
    actual = condition_tensor(expression, "toy_boundary", xy, torch.ones(2, 1), {"u": value})
    torch.testing.assert_close(actual, 3*value + xy[:, :1])
    actual.sum().backward()
    assert xy.grad is not None and xy.grad.abs().sum() > 0


@pytest.mark.parametrize("builder", [lambda x,y,t: {}, lambda x,y,t: {"bad": float("inf")}])
def test_bad_condition_builders_fail_without_training(builder):
    with pytest.raises(ValueError):
        conditions(builder)
