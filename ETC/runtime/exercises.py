"""Small data/expression contracts shared by learner runs and the trusted judge.

No submitted expression is compiled or lambdified. Tensor evaluation walks a
bounded SymPy tree and keeps the model's autograd graph for boundary losses.
"""
import math

import sympy as sp
import torch


def conditions(builder, *, wave=False):
    x, y, t = sp.symbols("x y t")
    values = builder(x, y, t, sp.Function("u")(x, y, t)) if wave else builder(x, y, t)
    if not isinstance(values, dict) or not values or len(values) > 16:
        raise ValueError("student_conditions must return a dictionary of condition expressions")
    result = {}
    for name, value in values.items():
        if type(value) in (int, float):
            value = sp.Number(value)
        if not isinstance(value, sp.Expr) or not isinstance(name, str):
            raise ValueError("Each condition needs a name and a symbolic expression")
        if value.free_symbols - {x, y, t} or sp.count_ops(value) > 180:
            raise ValueError("Conditions may use only x, y, t and the supplied field")
        if value.has(sp.oo, -sp.oo, sp.zoo, sp.nan):
            raise ValueError("Condition expressions must be finite")
        result[name] = value
    return result


def condition_tensor(expressions, name, coordinates, time=None, fields=None):
    """Evaluate one named target/residual as an N x 1 tensor on the input device."""
    if name not in expressions:
        raise ValueError(f"student_conditions is missing {name!r}")
    x, y, t = sp.symbols("x y t")
    fields = {} if fields is None else fields
    zero = torch.zeros_like(coordinates[:, :1])
    cache = {x: coordinates[:, :1], y: coordinates[:, 1:2], t: zero if time is None else time}
    budget = [400]

    def evaluate(node):
        budget[0] -= 1
        if budget[0] < 0:
            raise ValueError("Condition expression is too complex")
        if node in cache:
            return cache[node]
        if node.is_Number:
            value = float(node)
            if not math.isfinite(value) or abs(value) > 1e8:
                raise ValueError("Condition coefficients must be finite and bounded")
            result = zero + value
        elif node.is_Add:
            result = sum(evaluate(arg) for arg in node.args)
        elif node.is_Mul:
            result = torch.ones_like(zero)
            for arg in node.args:
                result = result * evaluate(arg)
        elif node.is_Pow and node.exp.is_Integer and -2 <= int(node.exp) <= 2:
            result = evaluate(node.base) ** int(node.exp)
        elif node.func in (sp.sin, sp.cos, sp.exp):
            result = {sp.sin: torch.sin, sp.cos: torch.cos, sp.exp: torch.exp}[node.func](evaluate(node.args[0]))
        elif isinstance(node, sp.Derivative):
            from ETC.runtime.pinn import gradient
            result = evaluate(node.expr)
            if sum(order for _, order in node.variable_count) > 2:
                raise ValueError("At most second derivatives are supported")
            for variable, order in node.variable_count:
                for _ in range(order):
                    if variable == t and time is not None:
                        result = gradient(result, time)
                    elif variable in (x, y):
                        result = gradient(result, coordinates)[:, :1 if variable == x else 2]
                        if variable == y:
                            result = result[:, 1:2]
                    else:
                        raise ValueError("Unsupported condition derivative")
        elif node.func.__name__ in fields and node.args == (x, y, t):
            result = fields[node.func.__name__]
        else:
            raise ValueError(f"Unsupported condition expression: {node}")
        cache[node] = result
        return result

    result = evaluate(expressions[name])
    if result.shape != zero.shape or not torch.isfinite(result).all():
        raise ValueError(f"Condition {name} must produce finite values of shape {tuple(zero.shape)}")
    return result


def block_geometry(builder):
    """Channel bounds stay fixed; learners construct one or three chip cutouts."""
    result = builder()
    if not isinstance(result, dict) or set(result) != {"blocks"}:
        raise ValueError("student_geometry must return {'blocks': ((xmin, xmax, top), ...)}")
    blocks = result["blocks"]
    if not isinstance(blocks, (list, tuple)) or not 1 <= len(blocks) <= 3:
        raise ValueError("Return one to three rectangular chip cutouts")
    parsed = []
    for block in blocks:
        if not isinstance(block, (list, tuple)) or len(block) != 3:
            raise ValueError("Each chip is (xmin, xmax, top); its bottom is y=-0.5")
        if any(isinstance(value, (str, bool)) or not isinstance(value, (int, float, sp.Number)) for value in block):
            raise ValueError("Chip coordinates must be finite numbers")
        xmin, xmax, top = map(float, block)
        if not all(math.isfinite(v) for v in (xmin, xmax, top)) or not (-2.5 < xmin < xmax < 2.5 and -.5 < top < .5):
            raise ValueError("Chip cutouts must lie inside the fixed channel")
        parsed.append((xmin, xmax, top))
    parsed.sort()
    if any(left[1] >= right[0] for left, right in zip(parsed, parsed[1:])):
        raise ValueError("Chip cutouts must not touch or overlap")
    return parsed


def physical_parameters(builder, expected):
    values = builder()
    if not isinstance(values, dict) or set(values) != set(expected):
        raise ValueError("student_parameters must return exactly: " + ", ".join(expected))
    result = {}
    for name, value in values.items():
        if isinstance(value, (bool, str)) or not isinstance(value, (int, float, sp.Number)):
            raise ValueError("Physics parameters must be finite numbers")
        value = float(value)
        if not math.isfinite(value) or abs(value) > 10000:
            raise ValueError("Physics parameters must be finite and bounded")
        if name.startswith("kappa") and value <= 0 or name == "gamma0" and value < 0:
            raise ValueError("Diffusivities must be positive and heat exchange nonnegative")
        result[name] = value
    return result


def analytic_expression_checks(builder, reference, parameters):
    """Check learner-derived baseline solutions without using them as truth."""
    x, y, t = sp.symbols("x y t")
    params = {name: sp.Symbol(name) for name in parameters}
    actual, expected = builder(x, y, t, params), reference(x, y, t, params)
    if not isinstance(actual, dict) or set(actual) != set(expected):
        raise ValueError("student_solution must return exactly: " + ", ".join(expected))
    if any(not isinstance(value, sp.Expr) or sp.count_ops(value) > 180 for value in actual.values()):
        raise ValueError("Exact solutions must be bounded symbolic expressions")
    return {name: sp.expand(actual[name] - expected[name]) == 0 for name in expected}
