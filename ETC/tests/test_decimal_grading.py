"""Regression cases for exact written decimal coefficients in setup grading."""
import pytest
import sympy as sp

from ETC.judge.catalog import rules
from ETC.judge.contracts import check_setup
from ETC.judge.evaluate import evaluate, load_lesson
from ETC.judge.expressions import SubmissionError, build
from ETC.tests.test_judge import answer


def expanded_gaussian_source():
    return answer("1", "wave_l3.py").replace(
        "((x - .3)**2 + y**2)", "(x**2 - .6*x + .09 + y**2)"
    ).replace(
        "((x + .3)**2 + y**2)", "(x**2 + .6*x + .09 + y**2)"
    )


@pytest.mark.parametrize("variant", ["original", "expanded", "fractions", "scientific"])
def test_wave_three_decimal_algebra_is_accepted(variant):
    source = answer("1", "wave_l3.py")
    if variant == "expanded":
        source = expanded_gaussian_source()
    elif variant == "fractions":
        source = source.replace(".3", "(3/10)")
    elif variant == "scientific":
        source = source.replace(".3", "3e-1")
    _, checks = check_setup(load_lesson("1", "wave_l3.py"), source, "1")
    assert all(checks.values())


def test_expanded_gaussian_receives_full_level_credit_after_real_cpu_run(tmp_path):
    result = evaluate("1", {"wave_l3.py": expanded_gaussian_source()},
                      rules(2), tmp_path, "cpu")
    level = result["levels"]["wave_l3.py"]
    assert level["score"] == 100
    assert level["status"] == "evaluated"
    assert all(level["components"].values())
    assert list(tmp_path.rglob("metrics.json"))


@pytest.mark.parametrize("coefficient", ["1.00000000000000000000001", "1 + 1e-20"])
def test_literal_precision_is_not_lost_before_comparison(coefficient):
    source = answer("1", "wave_l1.py").replace(
        'return {"c": 1}', 'return {"c": ' + coefficient + '}')
    _, checks = check_setup(load_lesson("1", "wave_l1.py"), source, "1")
    assert checks["parameter.wave_speed"] is False


@pytest.mark.parametrize("offset", ["1e-12*x", "1e-20*x", "1e-20"])
def test_small_real_errors_in_expanded_gaussian_are_rejected(offset):
    source = expanded_gaussian_source().replace(
        '"initial_u": ', '"initial_u": ' + offset + ' + ')
    _, checks = check_setup(load_lesson("1", "wave_l3.py"), source, "1")
    assert checks["condition.initial_u"] is False


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
