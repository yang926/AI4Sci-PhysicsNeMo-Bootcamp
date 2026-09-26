"""Independent identities, counterexamples and resource limits for grading."""
import pytest
import sympy as sp

from ETC.judge.contracts import check_setup
from ETC.judge.evaluate import load_lesson
from ETC.judge.expressions import SubmissionError, build
from ETC.runtime.symbolic_checks import equivalent_expression
from ETC.tests.test_judge import answer


x, y, t = sp.symbols("x y t")


@pytest.mark.parametrize("actual,expected", [
    (1 + sp.Float(.25)*(sp.sin(x+y) + sp.sin(x-y)), 1 + sp.Float(.5)*sp.sin(x)*sp.cos(y)),
    ((sp.cos(x-y)-sp.cos(x+y))/2, sp.sin(x)*sp.sin(y)),
    (sp.sin(x)**2 + sp.cos(x)**2, sp.Integer(1)),
    (sp.sin(2*x), 2*sp.sin(x)*sp.cos(x)),
    (sp.exp(x+y), sp.exp(x)*sp.exp(y)),
    ((x+y)**2, x*x + 2*x*y + y*y),
    (sp.Integer(1), 1),
    (sp.Integer(0), 0.0),
])
def test_exact_alternate_forms(actual, expected):
    assert equivalent_expression(actual, expected)
    assert equivalent_expression(expected, actual)


@pytest.mark.parametrize("offset", [sp.Rational(1, 10**12), sp.Float("1e-20")])
def test_near_equal_is_not_equal(offset):
    expected = 1 + sp.sin(x)*sp.cos(y)/2
    alternate = 1 + (sp.sin(x+y)+sp.sin(x-y))/4
    assert not equivalent_expression(alternate + offset*x, expected)


def test_fields_and_derivatives_remain_independent_atoms():
    u = sp.Function("u")(x, y, t)
    robin = u + sp.Rational(1, 2)*(x*u.diff(x) + y*u.diff(y))
    assert equivalent_expression(robin, (2*u + x*u.diff(x) + y*u.diff(y))/2)
    assert not equivalent_expression(robin, u)
    assert not equivalent_expression(u.diff(x), u.diff(y))


def test_wave_two_alternate_speed_and_initial_condition_receive_credit():
    source = answer("1", "wave_l2.py").replace(
        '1 + .5 * sin(x) * cos(y)', '1 + .25*(sin(x+y) + sin(x-y))'
    ).replace('sin(x) * sin(y)', '(cos(x-y)-cos(x+y))/2')
    _, checks = check_setup(load_lesson("1", "wave_l2.py"), source, "1")
    assert checks["parameter.wave_speed"]
    assert checks["condition.initial_u"]
    assert all(checks.values())


@pytest.mark.parametrize("filename", ["climate_l1.py", "climate_l2.py"])
def test_climate_condition_and_solution_product_to_sum_identity(filename):
    source = answer("3", filename).replace('sin(x)*sin(y)', '(cos(x-y)-cos(x+y))/2')
    _, checks = check_setup(load_lesson("3", filename), source, "3")
    assert all(checks.values())
    assert any(name.startswith("solution.") for name in checks)


def test_wrong_speed_close_to_reference_does_not_get_credit():
    source = answer("1", "wave_l2.py").replace(
        '1 + .5 * sin(x) * cos(y)', '1 + .25*(sin(x+y) + sin(x-y)) + 1e-12*x')
    _, checks = check_setup(load_lesson("1", "wave_l2.py"), source, "1")
    assert checks["parameter.wave_speed"] is False


@pytest.mark.parametrize("expression", [sp.nan, sp.oo, "sin(x)", sp.log(x), sp.Pow(x, 100, evaluate=False)])
def test_nonfinite_and_unsupported_expressions_fail_before_expansion(expression, monkeypatch):
    monkeypatch.setattr(sp, "expand", lambda *a, **kw: pytest.fail("Expansion must not run"))
    with pytest.raises(ValueError):
        equivalent_expression(expression, sp.Integer(1))


def test_trigonometric_expansion_is_bounded_before_rewrite(monkeypatch):
    # Only 27 input nodes, but Euler expansion would produce 2**13 terms.
    expression = sp.Mul(*(sp.sin(z) for z in sp.symbols("z:13")), evaluate=False)
    monkeypatch.setattr(sp, "expand", lambda *a, **kw: pytest.fail("Expansion must not run"))
    with pytest.raises(ValueError, match="budget"):
        equivalent_expression(expression, sp.Integer(1))


def test_nested_functions_are_not_hidden_from_resource_bounds(monkeypatch):
    expression = x
    for _ in range(15):
        expression = sp.sin(expression, evaluate=False)
    monkeypatch.setattr(sp, "expand", lambda *a, **kw: pytest.fail("Expansion must not run"))
    with pytest.raises(ValueError, match="budget"):
        equivalent_expression(expression, x)


def test_original_submission_ast_limit_still_runs_before_interpretation():
    source = 'def student_speed(x,y):\n' + '    a = x\n'*110 + '    return {"c": x}\n'
    with pytest.raises(SubmissionError, match="400 syntax nodes"):
        build(source, {"x": None, "y": None}, name="student_speed", data=True)


def test_no_numeric_sampling_or_heuristic_simplifier(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Do not award points using heuristic or sampled equivalence")
    monkeypatch.setattr(sp, "simplify", forbidden)
    monkeypatch.setattr(sp, "trigsimp", forbidden)
    monkeypatch.setattr(sp.Expr, "equals", forbidden)
    monkeypatch.setattr(sp.Expr, "evalf", forbidden)
    assert equivalent_expression((sp.cos(x-y)-sp.cos(x+y))/2, sp.sin(x)*sp.sin(y))
