"""Interpret a small equation language, never import/exec submitted Python.

Full lesson files may be uploaded. Only student_equations is extracted. All
other source, imports, decorators and default expressions are NOT executed.
This deliberately excludes arbitrary Python, even inside that function.
"""
import ast
import math
import operator


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


def build(source, parameters):
    """Return a callable interpreted by trusted code, not a compiled function."""
    node = extract(source)
    names = [arg.arg for arg in node.args.args]
    if names != list(parameters):
        raise SubmissionError("Keep the original arguments: " + ", ".join(parameters))

    def equations(*args, **kwargs):
        env = dict(parameters)
        env.update(zip(names, args))
        env.update(kwargs)
        return Interpreter(env).run(node.body)
    return equations


class Interpreter:
    def __init__(self, environment):
        import sympy
        self.sp = sympy
        def convert(value):
            if type(value) in (int, float):
                return sympy.sympify(value)
            if isinstance(value, dict):
                return {key: convert(part) for key, part in value.items()}
            return value
        self.env = {key: convert(value) for key, value in environment.items()}
        self.budget = 400

    def checked(self, value):
        if isinstance(value, self.sp.Expr):
            if any(abs(number) > 10 ** 8 for number in value.atoms(self.sp.Number) if number.is_finite):
                raise SubmissionError("Numeric coefficient is too large.")
            def terms(expr):
                if expr.is_Add:
                    count = sum(terms(arg) for arg in expr.args)
                elif expr.is_Mul:
                    count = math.prod(terms(arg) for arg in expr.args)
                elif expr.is_Pow and expr.exp.is_Integer and expr.exp > 0:
                    if expr.exp > 4:
                        raise SubmissionError("Expression power is too large.")
                    count = terms(expr.base) ** int(expr.exp)
                else:
                    count = 1
                if count > 256:
                    raise SubmissionError("Expanded expression is too large.")
                return count
            terms(value)
            if self.sp.count_ops(value) > 180 or len(str(value)) > 12000:
                raise SubmissionError("Expression is too complex.")
            if value.has(self.sp.zoo, self.sp.oo, -self.sp.oo, self.sp.nan):
                raise SubmissionError("Equations must contain finite expressions.")
        return value

    def run(self, statements):
        for index, statement in enumerate(statements):
            if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant) and isinstance(statement.value.value, str):
                continue  # Function docstring, not executable source.
            if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
                self.assign(statement.targets[0], self.value(statement.value))
            elif isinstance(statement, ast.Return) and index == len(statements) - 1:
                result = self.value(statement.value)
                if not isinstance(result, dict) or not result or len(result) > 4:
                    raise SubmissionError("Return a dictionary of named PDE residuals.")
                if any(not isinstance(v, self.sp.Expr) for v in result.values()):
                    raise SubmissionError("Each residual must be a mathematical expression.")
                return result
            elif isinstance(statement, (ast.Raise, ast.Pass)):
                raise SubmissionError("Exercise is unfinished. Complete student_equations first.")
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
            if type(node.value) in (int, float) and math.isfinite(node.value) and abs(node.value) <= 10000:
                return sp.sympify(node.value)  # Numeric objects only, never a string parser.
            if isinstance(node.value, str) and len(node.value) <= 40:
                return node.value
        elif isinstance(node, ast.Name) and node.id in self.env:
            return self.env[node.id]
        elif isinstance(node, (ast.Tuple, ast.List)):
            return tuple(self.value(item) for item in node.elts)
        elif isinstance(node, ast.Dict) and len(node.keys) <= 4:
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
                return self.checked(left ** right)
            if type(node.op) in operations:
                if isinstance(node.op, ast.Div) and not (right.is_Number or isinstance(right, sp.Symbol)):
                    raise SubmissionError("Divide only by a number or a supplied parameter.")
                return self.checked(operations[type(node.op)](left, right))
        elif isinstance(node, ast.Call) and not node.keywords:
            if isinstance(node.func, ast.Name) and node.func.id in {"sin", "cos"} and len(node.args) == 1:
                arg = self.value(node.args[0])
                if isinstance(arg, sp.Expr):
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
                    return self.checked(expression.diff(coordinate, int(order)))
        raise SubmissionError("Unsupported expression. Use supplied variables, + - * / **, .diff(), sin() and cos().")
