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
MAX_NUMBER_BITS = 4096


def _validate(expression, *, max_nodes=MAX_INPUT_NODES):
    """Reject oversized/unsupported trees before any symbolic transformation."""
    if not isinstance(expression, sp.Expr):
        raise ValueError("Answer comparisons require symbolic expressions.")
    pending = [(expression, 0)]
    count = 0
    while pending:
        node, depth = pending.pop()
        count += 1
        if count > max_nodes or depth > MAX_DEPTH:
            raise ValueError("Answer is too complex for bounded equivalence checking.")
        if isinstance(node, sp.Number):
            if node.is_finite is not True or abs(node) > 10**8:
                raise ValueError("Answer coefficients must be finite and bounded.")
            # Small magnitude does not imply a small representation: repeated
            # reciprocal squaring can otherwise create enormous denominators.
            if isinstance(node, sp.Rational) and max(int(node.p).bit_length(), int(node.q).bit_length()) > MAX_NUMBER_BITS:
                raise ValueError("Answer coefficient representation exceeds the numeric budget.")
            if isinstance(node, sp.Float) and node._prec > MAX_NUMBER_BITS:
                raise ValueError("Answer coefficient precision exceeds the numeric budget.")
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
            # SymPy would fold nested integer powers on evaluation. Reject the
            # accumulated exponent on the unevaluated tree, before that work.
            base, power = node.base, int(node.exp)
            while base.is_Pow and base.exp.is_Integer:
                power *= int(base.exp)
                if abs(power) > 4:
                    raise ValueError("Answer powers exceed the bounded comparison language.")
                base = base.base
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
            base, power = node.base, int(node.exp)
            while base.is_Pow and base.exp.is_Integer:
                power *= int(base.exp)
                base = base.base
            base_terms, base_width = cost(base)
            # expand() also distributes powers inside reciprocal denominators.
            # Count those terms even though the outer reciprocal is one term;
            # treating a negative power as an atom would hide that expansion.
            terms = base_terms ** abs(power)
            # Width is per distributive term; the term count above already
            # includes every base term. Counting it twice rejects the ordinary
            # variable-speed Wave PDE after its supplied c(x,y) is substituted.
            width = 1 + abs(power) * base_width
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


def validate_expression(expression):
    """Bound a submitted expression *before* evaluation, rewrite or expansion.

    The restricted interpreter also calls this on unevaluated arithmetic trees,
    so rejected intermediates cannot hide inside an otherwise small final answer.
    This only checks cost and syntax; it does not test whether an answer is right.
    """
    expression = sp.sympify(expression) if type(expression) in (int, float) else expression
    _validate(expression)
    _check_expansion_cost(expression)
    return expression


def equivalent_expression(actual, expected):
    """Prove equality within a bounded identity set; never accept approximate.

    Unsupported or excessive expressions raise ValueError, rather than run an
    unbounded CAS search. Valid expressions not proved equal return False.
    Existing AST limits are applied before this helper by the submission parser.
    """
    # Trusted reference builders may return plain 0/1. Convert numeric objects
    # only; never feed submitted strings into SymPy's string parser.
    actual = validate_expression(actual)
    expected = validate_expression(expected)

    # Preserve exact values of floating-point literals during transformations.
    # Rational(Float) is exact; unlike nsimplify it does not guess a nearby value.
    floats = actual.atoms(sp.Float) | expected.atoms(sp.Float)
    replacements = {number: sp.Rational(number) for number in floats}
    atoms = actual.atoms(AppliedUndef, sp.Derivative) | expected.atoms(AppliedUndef, sp.Derivative)
    replacements.update({atom: sp.Dummy() for atom in atoms})

    def normal_form(expression):
        value = expression.xreplace(replacements).rewrite([sp.sin, sp.cos], sp.exp)
        # Check the actual rewritten tree too. The input estimate bounds the
        # rewrite; this second gate must still run before distributive expansion.
        _validate(value, max_nodes=MAX_EXPANDED_NODES)
        _check_expansion_cost(value)
        # No power-base/log/complex expansion: those require extra assumptions.
        result = sp.expand(value, power_base=False, log=False, complex=False,
                           func=False, trig=False)
        _validate(result, max_nodes=MAX_EXPANDED_NODES)
        return result

    return normal_form(actual) == normal_form(expected)
