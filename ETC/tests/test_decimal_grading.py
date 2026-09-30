"""Regression cases for exact written decimal coefficients in setup grading."""
import pytest
import sympy as sp

from ETC.judge.catalog import rules
from ETC.judge.expressions import SubmissionError, build












@pytest.mark.parametrize("literal,expected", [
    (".3", sp.Rational(3, 10)), ("3e-1", sp.Rational(3, 10)),
    ("0.3_0", sp.Rational(3, 10)), ("1.", sp.Integer(1)),
    ("1e-20", sp.Rational(1, 10**20)), ("0xA", sp.Integer(10)),
    ("10000.0", sp.Integer(10000)),
])
def test_supported_literal_spelling(literal, expected):
    source = 'def student_speed(x,y):\n    return {"c": ' + literal + '}\n'
    function = build(source, {"x": None, "y": None}, name="student_speed",
                     data=True, exact_numbers=True)
    assert function(*sp.symbols("x y"))["c"] == expected


@pytest.mark.parametrize("literal", [
    "1e-999999999", "0e999999999",
    "10000.0000000000000000000000000000000001",
    "0." + "1"*129,
])
def test_exact_numeric_limits_reject_before_large_rational_construction(literal, monkeypatch):
    source = 'def student_speed(x,y):\n    return {"c": ' + literal + '}\n'
    function = build(source, {"x": None, "y": None}, name="student_speed",
                     data=True, exact_numbers=True)
    x, y = sp.symbols("x y")
    monkeypatch.setattr(sp, "Rational", lambda *a, **kw: pytest.fail("must reject before Rational"))
    with pytest.raises(SubmissionError):
        function(x, y)


def test_exact_arithmetic_does_not_drop_small_addends():
    source = 'def student_speed(x,y):\n    return {"c": .3 + .6 - .9 + 1e-20}\n'
    function = build(source, {"x": None, "y": None}, name="student_speed",
                     data=True, exact_numbers=True)
    assert function(*sp.symbols("x y"))["c"] == sp.Rational(1, 10**20)
