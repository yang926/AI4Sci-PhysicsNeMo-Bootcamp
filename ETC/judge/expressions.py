"""Interpret a small equation language, never import/exec submitted Python.

Full lesson files may be uploaded. Only named exercise functions are extracted. All
other source, imports, decorators and default expressions are NOT executed.
This deliberately excludes arbitrary Python, even inside that function.
"""
import ast
from decimal import Decimal, InvalidOperation
import math
import operator

from ETC.runtime.symbolic_checks import validate_expression


class SubmissionError(ValueError):
    pass


def parse_source(source):
    if not isinstance(source, str) or len(source.encode()) > 65536:
        raise SubmissionError("Each file must contain at most 64 KiB of Python text.")
    try:
        return ast.parse(source)
    except (SyntaxError, ValueError, RecursionError) as exc:
        raise SubmissionError("Python syntax error. Save a valid .py file before submitting.") from exc


def validate_function(node, parameters=None):
    if node.decorator_list or node.args.vararg or node.args.kwarg or node.args.kwonlyargs or node.args.posonlyargs:
        raise SubmissionError("Keep the original function signature without decorators or extra arguments.")
    if sum(1 for _ in ast.walk(node)) > 400:
        raise SubmissionError("Exercise function is too large (maximum 400 syntax nodes).")
    if parameters is not None and [arg.arg for arg in node.args.args] != list(parameters):
        raise SubmissionError("Keep the original arguments: " + ", ".join(parameters))
    return node


def extract(source, name="student_equations", parameters=None):
    tree = parse_source(source)
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name]
    if len(functions) != 1:
        raise SubmissionError(f"Include exactly one top-level {name} function.")
    node = functions[0]
    return validate_function(node, parameters)


def build(source, parameters, *, name="student_equations", data=False, exact_numbers=False):
    """Return a callable interpreted by trusted code, not a compiled function."""
    node = extract(source, name, parameters)
    names = [arg.arg for arg in node.args.args]
    if names != list(parameters):
        raise SubmissionError("Keep the original arguments: " + ", ".join(parameters))

    def equations(*args, **kwargs):
        env = dict(parameters)
        env.update(zip(names, args))
        env.update(kwargs)
        return Interpreter(env, data=data, exact_numbers=exact_numbers, source=source).run(node.body)
    return equations


class Interpreter:
    def __init__(self, environment, *, data=False, exact_numbers=False, source=None):
        import sympy
        self.sp = sympy
        self.exact_numbers = exact_numbers
        self.source = source
        def convert(value):
            if type(value) in (int, float):
                if exact_numbers:
                    return self.decimal_number(str(value))
                return sympy.sympify(value)
            if isinstance(value, dict):
                return {key: convert(part) for key, part in value.items()}
            return value
        self.env = {key: convert(value) for key, value in environment.items()}
        self.budget = 400
        self.data = data

    def decimal_number(self, literal):
        """Exact decimal coefficients, before arithmetic can round them.

        Decimal reads only numeric literal text, never Python/SymPy expressions.
        Bound digits and exponent before constructing potentially large integers.
        """
        if not isinstance(literal, str) or len(literal) > 128:
            raise SubmissionError("Numeric literal is too long.")
        try:
            number = Decimal(literal.replace("_", ""))
        except InvalidOperation as exc:
            raise SubmissionError("Invalid decimal coefficient.") from exc
        if (not number.is_finite() or number.copy_abs() > 10000
                or abs(number.as_tuple().exponent) > 400):
            raise SubmissionError("Decimal coefficient exceeds the supported numeric bounds.")
        numerator, denominator = number.as_integer_ratio()
        return self.sp.Rational(numerator, denominator)

    def checked(self, value):
        if isinstance(value, self.sp.Expr):
            try:
                validate_expression(value)
            except ValueError as exc:
                raise SubmissionError(str(exc)) from exc
            if self.sp.count_ops(value) > 180:
                raise SubmissionError("Expression is too complex.")
        return value

    def run(self, statements):
        for index, statement in enumerate(statements):
            if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant) and isinstance(statement.value.value, str):
                continue  # Function docstring, not executable source.
            if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
                self.assign(statement.targets[0], self.value(statement.value))
            elif isinstance(statement, ast.Return) and index == len(statements) - 1:
                result = self.value(statement.value)
                if not isinstance(result, dict) or not result or len(result) > (16 if self.data else 4):
                    raise SubmissionError("Return a dictionary of named exercise components.")
                if not self.data and any(not isinstance(v, self.sp.Expr) for v in result.values()):
                    raise SubmissionError("Each residual must be a mathematical expression.")
                for value in result.values():
                    self.checked(value)
                return result
            elif isinstance(statement, (ast.Raise, ast.Pass)):
                raise SubmissionError("Exercise is unfinished. Complete all required student functions first.")
            else:
                raise SubmissionError("Use local assignments and one final return; loops, imports and control flow are not accepted.")
        raise SubmissionError("Return the equation dictionary from student_equations.")

    def assign(self, target, value):
        if isinstance(target, ast.Name) and not target.id.startswith("_"):
            self.env[target.id] = value
        elif isinstance(target, (ast.Tuple, ast.List)) and isinstance(value, tuple) and len(target.elts) == len(value):
            for item, part in zip(target.elts, value):
                self.assign(item, part)
        else:
            raise SubmissionError("Assign only local names, not object attributes or dictionary entries.")

    def value(self, node):
        self.budget -= 1
        if self.budget < 0:
            raise SubmissionError("Expression operation limit reached.")
        sp = self.sp
        if isinstance(node, ast.Constant):
            if type(node.value) in (int, float) and abs(node.value) <= 10000 and math.isfinite(node.value):
                if self.exact_numbers:
                    if type(node.value) is int:
                        return sp.Integer(node.value)
                    literal = ast.get_source_segment(self.source, node) if self.source is not None else str(node.value)
                    return self.decimal_number(literal)
                return sp.sympify(node.value)  # Numeric objects only, never a string parser.
            if isinstance(node.value, str) and len(node.value) <= 40:
                return node.value
        elif isinstance(node, ast.Name) and node.id in self.env:
            return self.checked(self.env[node.id])
        elif isinstance(node, (ast.Tuple, ast.List)) and len(node.elts) <= 16:
            return tuple(self.value(item) for item in node.elts)
        elif isinstance(node, ast.Dict) and len(node.keys) <= (16 if self.data else 4):
            pairs = [(self.value(key), self.value(value)) for key, value in zip(node.keys, node.values)]
            if any(not isinstance(key, str) for key, _ in pairs) or len({key for key, _ in pairs}) != len(pairs):
                raise SubmissionError("Residual keys must be distinct strings.")
            return dict(pairs)
        elif isinstance(node, ast.Subscript):
            container, key = self.value(node.value), self.value(node.slice)
            if isinstance(container, dict) and isinstance(key, str) and key in container:
                return container[key]
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = self.value(node.operand)
            if isinstance(value, sp.Expr):
                return self.checked(value if isinstance(node.op, ast.UAdd) else -value)
        elif isinstance(node, ast.BinOp):
            left, right = self.value(node.left), self.value(node.right)
            if not isinstance(left, sp.Expr) or not isinstance(right, sp.Expr):
                raise SubmissionError("Arithmetic accepts only mathematical expressions.")
            operations = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}
            if isinstance(node.op, ast.Pow):
                if not right.is_Integer or not -2 <= int(right) <= 2:
                    raise SubmissionError("Use integer powers between -2 and 2.")
                self.checked(sp.Pow(left, right, evaluate=False))
                return self.checked(left ** right)
            if type(node.op) in operations:
                if isinstance(node.op, ast.Div) and not (right.is_Number or isinstance(right, sp.Symbol)):
                    raise SubmissionError("Divide only by a number or a supplied parameter.")
                if isinstance(node.op, (ast.Add, ast.Sub)):
                    candidate = sp.Add(left, right if isinstance(node.op, ast.Add) else -right, evaluate=False)
                else:
                    factor = sp.Pow(right, -1, evaluate=False) if isinstance(node.op, ast.Div) else right
                    candidate = sp.Mul(left, factor, evaluate=False)
                self.checked(candidate)
                return self.checked(operations[type(node.op)](left, right))
        elif isinstance(node, ast.Call) and not node.keywords:
            if isinstance(node.func, ast.Name) and node.func.id in {"sin", "cos", "exp"} and len(node.args) == 1:
                arg = self.value(node.args[0])
                if isinstance(arg, sp.Expr):
                    self.checked(getattr(sp, node.func.id)(arg, evaluate=False))
                    return self.checked(getattr(sp, node.func.id)(arg))
            if isinstance(node.func, ast.Attribute) and node.func.attr == "diff" and 1 <= len(node.args) <= 2:
                expression = self.value(node.func.value)
                coordinate = self.value(node.args[0])
                order = self.value(node.args[1]) if len(node.args) == 2 else sp.Integer(1)
                if (isinstance(expression, sp.Expr) and coordinate in [self.env.get(k) for k in ("x", "y", "t")]
                        and isinstance(coordinate, sp.Symbol) and order in (1, 2)):
                    # Limit derivative nesting before symbolic work starts.
                    if any(sum(n for _, n in d.variable_count) + int(order) > 2 for d in expression.atoms(sp.Derivative)):
                        raise SubmissionError("Only derivatives up to second order are supported.")
                    # Product/chain rules can grow even a bounded input tree.
                    # Preflight a conservative work estimate before diff runs.
                    complexity = max(1, int(sp.count_ops(expression)))
                    if complexity * (complexity + 1) ** int(order) > 65536:
                        raise SubmissionError("Derivative expansion exceeds the operation budget.")
                    return self.checked(expression.diff(coordinate, int(order)))
        raise SubmissionError("Unsupported expression. Use supplied variables, + - * / **, .diff(), sin(), cos() and exp().")
