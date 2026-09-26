"""Small, bounded exact proofs for exercise-answer comparisons.

This is not a general simplifier. Polynomial expansion and Euler's sin/cos
identities cover alternate forms of the course answers. No sampling, tolerance,
string evaluation, or heuristic ``simplify`` is used to award credit.
"""
from functools import lru_cache

import sympy as sp
from sympy.core.function import AppliedUndef

MAX_INPUT_NODES = 512
MAX_DEPTH = 24
MAX_TERMS = 256
MAX_EXPANDED_NODES = 8192


def _validate(expression):
    """Reject oversized/unsupported trees before any symbolic transformation."""
    if not isinstance(expression, sp.Expr):
        raise ValueError("Answer comparisons require symbolic expressions.")
    pending = [(expression, 0)]
    count = 0
    while pending:
        node, depth = pending.pop()
        count += 1
        if count > MAX_INPUT_NODES or depth > MAX_DEPTH:
            raise ValueError("Answer is too complex for bounded equivalence checking.")
        if isinstance(node, sp.Number):
            if node.is_finite is not True or abs(node) > 10**8:
                raise ValueError("Answer coefficients must be finite and bounded.")
        elif node == sp.I or isinstance(node, sp.Symbol):
            pass
        elif isinstance(node, (AppliedUndef, sp.Derivative)):
            # Fields and their derivatives are independent algebraic atoms.
            # The submission interpreter has already validated derivative order.
            if isinstance(node, sp.Derivative):
                if sum(order for _, order in node.variable_count) > 2:
                    raise ValueError("Only first and second field derivatives are supported.")
                pending.append((node.expr, depth + 1))
            else:
                pending.extend((arg, depth + 1) for arg in node.args)
        elif node.is_Add or node.is_Mul or node.func in (sp.sin, sp.cos, sp.exp):
            pending.extend((arg, depth + 1) for arg in node.args)
        elif node.is_Pow:
            if not node.exp.is_Integer or not -4 <= node.exp <= 4:
                raise ValueError("Answer powers exceed the bounded comparison language.")
            pending.append((node.base, depth + 1))
        else:
            raise ValueError("Unsupported expression in bounded answer comparison.")


def _check_expansion_cost(expression):
    """Upper bounds for distributive terms and nodes after the fixed rewrite.

    Functions do not count as polynomial terms, but their arguments can expand.
    Count those too: sin(sin(...)) must not evade a limit on outer terms alone.
    """
    @lru_cache(maxsize=MAX_INPUT_NODES)
    def cost(node):
        if node.is_Atom or isinstance(node, (AppliedUndef, sp.Derivative)):
            return 1, 1
        parts = [cost(arg) for arg in node.args if isinstance(arg, sp.Expr)]
        if node.is_Add:
            terms, width = sum(t for t, _ in parts), max(w for _, w in parts)
        elif node.is_Mul:
            terms, width = 1, 1
            for term_count, term_width in parts:
                terms *= term_count
                width += term_width
                if terms > MAX_TERMS or terms * width > MAX_EXPANDED_NODES:
                    raise ValueError("Answer expansion exceeds the comparison budget.")
        elif node.is_Pow:
            base_terms, base_width = cost(node.base)
            power = int(node.exp)
            # expand() also distributes powers inside reciprocal denominators.
            # Count those terms even though the outer reciprocal is one term;
            # treating a negative power as an atom would hide that expansion.
            terms = base_terms ** abs(power)
            width = 1 + abs(power) * base_terms * base_width
        elif node.func in (sp.sin, sp.cos, sp.exp):
            arg_terms, arg_width = parts[0]
            # exp(a+b) becomes exp(a)*exp(b). sin/cos introduce two such
            # exponential terms, including their exact scalar coefficients.
            terms = 2 if node.func in (sp.sin, sp.cos) else 1
            width = 5 + arg_terms * (arg_width + 2)
        else:
            raise ValueError("Unsupported expression in bounded answer comparison.")
        if terms > MAX_TERMS or terms * width > MAX_EXPANDED_NODES:
            raise ValueError("Answer expansion exceeds the comparison budget.")
        return terms, width

    cost(expression)


def equivalent_expression(actual, expected):
    """Prove equality within a bounded identity set; never accept approximate.

    Unsupported or excessive expressions raise ValueError, rather than run an
    unbounded CAS search. Valid expressions not proved equal return False.
    Existing AST limits are applied before this helper by the submission parser.
    """
    # Trusted reference builders may return plain 0/1. Convert numeric objects
    # only; never feed submitted strings into SymPy's string parser.
    actual = sp.sympify(actual) if type(actual) in (int, float) else actual
    expected = sp.sympify(expected) if type(expected) in (int, float) else expected
    for expression in (actual, expected):
        _validate(expression)
        _check_expansion_cost(expression)

    # Preserve exact values of floating-point literals during transformations.
    # Rational(Float) is exact; unlike nsimplify it does not guess a nearby value.
    floats = actual.atoms(sp.Float) | expected.atoms(sp.Float)
    replacements = {number: sp.Rational(number) for number in floats}
    atoms = actual.atoms(AppliedUndef, sp.Derivative) | expected.atoms(AppliedUndef, sp.Derivative)
    replacements.update({atom: sp.Dummy() for atom in atoms})

    def normal_form(expression):
        value = expression.xreplace(replacements).rewrite(sp.exp)
        # No power-base/log/complex expansion: those require extra assumptions.
        return sp.expand(value, power_base=False, log=False, complex=False,
                         func=False, trig=False)

    return normal_form(actual) == normal_form(expected)
