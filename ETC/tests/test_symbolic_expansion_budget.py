"""Reciprocal denominators must share the bounded symbolic expansion budget."""
import pytest
import sympy as sp

from ETC.judge.expressions import build
from ETC.runtime.symbolic_checks import equivalent_expression


x, y = sp.symbols("x y")


def speed_expression(body):
    source = "def student_speed(x, y):\n" + body
    student = build(source, {"x": None, "y": None}, name="student_speed", data=True)
    return student(x, y)["c"]


@pytest.mark.parametrize("summands", [
    "sin(x) + sin(y) + sin(x*y) + sin(x*x) + sin(y*y) + sin(x*y*y)",
    " + ".join(f"sin(exp({index}*x))" for index in range(1, 21)),
])
@pytest.mark.parametrize("result", [
    "(q**-2)**2",
    "sin((q**-2)**2)",
    "exp((q**-2)**2)",
])
def test_oversized_reciprocals_fail_before_symbolic_rewrite(summands, result, monkeypatch):
    # Both reported bypasses pass the submission parser. The larger case must
    # never be expanded, including when its denominator is nested in a function.
    expression = speed_expression(f'    q = {summands}\n    return {{"c": {result}}}\n')

    def forbidden(*args, **kwargs):
        pytest.fail("Oversized denominators must be rejected before rewrite or expansion")

    monkeypatch.setattr(sp.Expr, "rewrite", forbidden)
    monkeypatch.setattr(sp, "expand", forbidden)
    with pytest.raises(ValueError, match="comparison budget"):
        equivalent_expression(expression, sp.Integer(1))


@pytest.mark.parametrize("actual,expected", [
    ("(x+y)**-2", "(x*x + 2*x*y + y*y)**-1"),
    ("(sin(x)+cos(x))**-2", "(1 + sin(2*x))**-1"),
    ("(sin(x)**-2)**2", "(sin(x)*sin(x)*sin(x)*sin(x))**-1"),
])
def test_bounded_reciprocal_identities_remain_accepted(actual, expected):
    actual_expression = speed_expression(f'    return {{"c": {actual}}}\n')
    expected_expression = speed_expression(f'    return {{"c": {expected}}}\n')
    assert equivalent_expression(actual_expression, expected_expression)
    assert equivalent_expression(expected_expression, actual_expression)
