"""Restricted interpretation of the existing FNO/AFNO/PINO exercises.

Uploads are never imported, compiled or exec'd. Model calls create inert
descriptions first; only the exact server-owned architecture is constructed.
"""
import ast
from dataclasses import dataclass
import math
import operator

from .expressions import Interpreter, SubmissionError, extract, parse_source, validate_function


class GuardError(SubmissionError):
    """An explicit, supported participant input-validation guard fired."""


@dataclass
class ModelSpec:
    kind: str
    keywords: dict


def same_settings(actual, expected):
    """Do not confuse True or floating-point channel counts with integers."""
    if isinstance(expected, dict):
        return isinstance(actual, dict) and set(actual) == set(expected) and all(same_settings(actual[k], v) for k, v in expected.items())
    if isinstance(expected, (tuple, list)):
        return isinstance(actual, (tuple, list)) and len(actual) == len(expected) and all(same_settings(a, b) for a, b in zip(actual, expected))
    if type(expected) is float:
        return type(actual) in (int, float) and actual == expected
    return type(actual) is type(expected) and actual == expected


def source_nodes(source, level):
    nodes = [extract(source, "build_datasets", ("train_pairs", "val_pairs", "test_pairs")),
             extract(source, "build_model", ("model_config", "grid_size"))]
    if level == 3:
        classes = [n for n in parse_source(source).body if isinstance(n, ast.ClassDef) and n.name == "ReactionDiffusionPDE"]
        if len(classes) != 1 or classes[0].decorator_list:
            raise SubmissionError("Include one undecorated ReactionDiffusionPDE class for PINO.")
        initializers = [n for n in classes[0].body if isinstance(n, ast.FunctionDef) and n.name == "__init__"]
        if len(initializers) != 1:
            raise SubmissionError("Include ReactionDiffusionPDE.__init__(self).")
        validate_function(initializers[0], ("self",))
        nodes.append(classes[0])
    return nodes


class FactoryInterpreter:
    """Bounded assignments/returns plus model/data constructors and guards."""
    def __init__(self, environment):
        self.env = dict(environment)
        self.budget = 1000

    def run(self, statements):
        for index, statement in enumerate(statements):
            if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant) and isinstance(statement.value.value, str):
                continue
            if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
                self.assign(statement.targets[0], self.value(statement.value))
            elif isinstance(statement, ast.Return) and index == len(statements) - 1:
                return self.value(statement.value)
            elif isinstance(statement, ast.Assert):
                if not self.truth(self.value(statement.test)):
                    raise GuardError("Model input check failed.")
            elif isinstance(statement, ast.If) and not statement.orelse and len(statement.body) == 1 and isinstance(statement.body[0], ast.Raise):
                if self.truth(self.value(statement.test)):
                    raise GuardError("Model input check failed; verify both AFNO patch dimensions.")
            elif isinstance(statement, (ast.Raise, ast.Pass)):
                raise SubmissionError("Exercise is unfinished. Complete this factory function first.")
            else:
                raise SubmissionError("Use assignments, a final return, and optional assert/if-raise input checks. Arbitrary Python is not supported.")
        raise SubmissionError("Return the datasets or model from this function.")

    def assign(self, target, value):
        if isinstance(target, ast.Name) and not target.id.startswith("_"):
            if target.id in {"TensorDataset", "FNO", "AFNO", "tuple", "list", "any", "all", "len"}:
                raise SubmissionError("Do not reassign constructor or input-check names.")
            self.env[target.id] = value
        elif isinstance(target, (ast.Tuple, ast.List)) and isinstance(value, (tuple, list)) and len(target.elts) == len(value):
            for item, part in zip(target.elts, value):
                self.assign(item, part)
        else:
            raise SubmissionError("Assign local names only; do not mutate inputs or objects.")

    def sequence(self, value):
        if not isinstance(value, (tuple, list)) or len(value) > 32:
            raise SubmissionError("Use a tuple/list of at most 32 items.")
        return value

    def truth(self, value):
        if type(value) not in (bool, int, float):
            raise SubmissionError("Input checks must use scalar boolean/numeric conditions.")
        return bool(value)

    def value(self, node):
        self.budget -= 1
        if self.budget < 0:
            raise SubmissionError("Factory expression limit reached.")
        if isinstance(node, ast.Constant):
            value = node.value
            if type(value) in (int, float) and math.isfinite(value) and abs(value) <= 10000:
                return value
            if type(value) is bool or value is None or isinstance(value, str) and len(value) <= 200:
                return value
        elif isinstance(node, ast.Name) and node.id in self.env:
            return self.env[node.id]
        elif isinstance(node, (ast.Tuple, ast.List)) and len(node.elts) <= 32:
            values = [self.value(n) for n in node.elts]
            return tuple(values) if isinstance(node, ast.Tuple) else values
        elif isinstance(node, ast.Dict) and len(node.keys) <= 32:
            pairs = [(self.value(k), self.value(v)) for k, v in zip(node.keys, node.values)]
            if all(isinstance(k, str) for k, _ in pairs) and len({k for k, _ in pairs}) == len(pairs):
                return dict(pairs)
        elif isinstance(node, ast.Subscript):
            container, key = self.value(node.value), self.value(node.slice)
            if isinstance(container, dict) and isinstance(key, str) and key in container:
                return container[key]
            if isinstance(container, (tuple, list)) and type(key) is int and -len(container) <= key < len(container):
                return container[key]
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            return not self.truth(self.value(node.operand))
        elif isinstance(node, ast.BoolOp):
            for part in node.values:
                value = self.truth(self.value(part))
                if isinstance(node.op, ast.And) and not value:
                    return False
                if isinstance(node.op, ast.Or) and value:
                    return True
            return isinstance(node.op, ast.And)
        elif isinstance(node, ast.BinOp):
            left, right = self.value(node.left), self.value(node.right)
            operations = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
                          ast.Mod: operator.mod, ast.FloorDiv: operator.floordiv}
            if type(left) in (int, float) and type(right) in (int, float) and type(node.op) in operations:
                if isinstance(node.op, (ast.Mod, ast.FloorDiv)) and right == 0:
                    raise SubmissionError("Cannot divide by zero in a model setting.")
                value = operations[type(node.op)](left, right)
                if math.isfinite(value) and abs(value) <= 10000:
                    return value
        elif isinstance(node, ast.Compare) and len(node.ops) == 1:
            left, right = self.value(node.left), self.value(node.comparators[0])
            operations = {ast.Eq: operator.eq, ast.NotEq: operator.ne, ast.Lt: operator.lt,
                          ast.LtE: operator.le, ast.Gt: operator.gt, ast.GtE: operator.ge}
            if type(left) in (int, float, str, bool) and type(right) is type(left) and type(node.ops[0]) in operations:
                return operations[type(node.ops[0])](left, right)
        elif isinstance(node, (ast.GeneratorExp, ast.ListComp)) and len(node.generators) == 1:
            gen = node.generators[0]
            if isinstance(gen.target, ast.Name) and not gen.ifs and not gen.is_async:
                original = dict(self.env)
                values = []
                try:
                    for part in self.sequence(self.value(gen.iter)):
                        self.assign(gen.target, part)
                        values.append(self.value(node.elt))
                finally:
                    self.env = original
                return tuple(values)
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute) and node.func.attr == "get" and not node.keywords and 1 <= len(node.args) <= 2:
                mapping = self.value(node.func.value)
                args = [self.value(arg) for arg in node.args]
                if isinstance(mapping, dict) and isinstance(args[0], str):
                    return mapping.get(*args)
            if isinstance(node.func, ast.Name):
                args, kwargs = [], {}
                for arg in node.args:
                    if isinstance(arg, ast.Starred):
                        args.extend(self.sequence(self.value(arg.value)))
                    else:
                        args.append(self.value(arg))
                for item in node.keywords:
                    value = self.value(item.value)
                    additions = value if item.arg is None else {item.arg: value}
                    if not isinstance(additions, dict) or set(additions) & set(kwargs):
                        raise SubmissionError("Model keywords must be distinct; expand only model_config.")
                    kwargs.update(additions)
                name = node.func.id
                if name in {"tuple", "list", "any", "all", "len"} and len(args) == 1 and not kwargs:
                    values = self.sequence(args[0])
                    if name in {"any", "all"}:
                        values = [self.truth(value) for value in values]
                    return {"tuple": tuple, "list": list, "any": any, "all": all, "len": len}[name](values)
                if name == "TensorDataset" and len(args) == 2 and not kwargs:
                    import torch
                    from torch.utils.data import TensorDataset
                    if all(isinstance(value, torch.Tensor) for value in args):
                        if len(args[0]) != len(args[1]):
                            raise SubmissionError("Dataset tensors must have matching sample counts.")
                        return TensorDataset(*args)
                if name in {"FNO", "AFNO"} and not args:
                    return ModelSpec(name, kwargs)
        raise SubmissionError("Unsupported factory expression. Use supplied pairs/config, TensorDataset, FNO/AFNO, and bounded input checks.")


class OperatorPDEInterpreter(Interpreter):
    def value(self, node):
        if isinstance(node, ast.Call) and not node.keywords:
            if isinstance(node.func, ast.Name) and node.func.id == "Symbol" and len(node.args) == 1:
                name = super().value(node.args[0])
                if name in {"x", "y"}:
                    return self.sp.Symbol(name)
            if (isinstance(node.func, ast.Call) and isinstance(node.func.func, ast.Name)
                    and node.func.func.id == "Function" and len(node.func.args) == 1 and not node.func.keywords):
                name = super().value(node.func.args[0])
                args = [super(OperatorPDEInterpreter, self).value(arg) for arg in node.args]
                if name in {"u", "f"} and args == [self.sp.Symbol("x"), self.sp.Symbol("y")]:
                    return self.sp.Function(name)(*args)
        return super().value(node)

    def equations(self, node):
        transformed, dimensions = [], 0
        for statement in node.body:
            if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
                target = statement.targets[0]
                if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == "self":
                    if target.attr == "dim" and isinstance(statement.value, ast.Constant) and statement.value.value == 2:
                        dimensions += 1
                        continue
                    if target.attr == "equations":
                        transformed.append(ast.Return(value=statement.value))
                        continue
                    raise SubmissionError("Set only self.dim = 2 and self.equations in the PDE class.")
            transformed.append(statement)
        if dimensions != 1:
            raise SubmissionError("Set self.dim = 2 once in ReactionDiffusionPDE.")
        return self.run(transformed)


def check_operator(source, level, config, training):
    import sympy as sp
    import torch
    datasets_node, model_node, *pde_nodes = source_nodes(source, level)

    def datasets(*pairs):
        return FactoryInterpreter(dict(zip(("train_pairs", "val_pairs", "test_pairs"), pairs))).run(datasets_node.body)

    def description(model_config, grid_size):
        return FactoryInterpreter({"model_config": model_config, "grid_size": grid_size}).run(model_node.body)

    checks, messages = {}, {}
    for name, operation in (
        ("data_splits", lambda: training.checked_datasets([
            (torch.full((n, 1, 4, 4), float(i)), torch.full((n, 1, 4, 4), float(i + 10)))
            for i, n in enumerate((3, 2, 1))], datasets)),
        ("model", lambda: description(config["model"], config["data"]["grid_size"]))):
        try:
            result = operation()
            if name == "model":
                common = {"in_channels": 1, "out_channels": 1, **config["model"]}
                expected = ModelSpec("AFNO", {"inp_shape": [config["data"]["grid_size"]] * 2, **common}) if level == 2 else ModelSpec("FNO", {"dimension": 2, **common})
                checks[name] = isinstance(result, ModelSpec) and result.kind == expected.kind and same_settings(result.keywords, expected.keywords)
                if not checks[name]:
                    messages[name] = "Use the specified model, channels, grid and every supplied model_config setting."
            else:
                checks[name] = True
        except (SubmissionError, ValueError, TypeError, KeyError) as exc:
            checks[name], messages[name] = False, str(exc)
    if level == 2:
        checks["patch_compatibility"] = True
        for patch in ([2, 4], [4, 2]):
            try:
                description({**config["model"], "patch_size": patch}, 6)
            except GuardError:
                continue
            except (SubmissionError, ValueError, TypeError, KeyError):
                pass
            checks["patch_compatibility"] = False
        if not checks["patch_compatibility"]:
            messages["patch_compatibility"] = "Explicitly reject grids not divisible by either patch dimension; do not crop."
    equations = None
    if level == 3:
        try:
            initializer = next(n for n in pde_nodes[0].body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
            equations = OperatorPDEInterpreter({}).equations(initializer)
            x, y = sp.symbols("x y")
            u, f = sp.Function("u")(x, y), sp.Function("f")(x, y)
            checks["physics"] = set(equations) == {"reaction_diffusion"} and sp.expand(equations["reaction_diffusion"] - (u - u.diff(x, 2) - u.diff(y, 2) - f)) == 0
            if not checks["physics"]:
                messages["physics"] = "The residual must be u - u_xx - u_yy - f on the periodic unit square."
        except (SubmissionError, ValueError, TypeError, KeyError) as exc:
            checks["physics"], messages["physics"] = False, str(exc)

    def model(model_config, grid_size):
        # The verified description is evaluated again, then a trusted constructor
        # creates the frozen architecture. Never construct an arbitrary upload.
        factory = training.reference_afno if level == 2 else training.reference_fno
        spec = description(model_config, grid_size)
        if not checks["model"] or not isinstance(spec, ModelSpec):
            raise SubmissionError("Model check did not pass.")
        return factory(model_config, grid_size)

    def physics(device):
        from physicsnemo.sym.eq.pde import PDE
        from physicsnemo.sym.eq.phy_informer import PhysicsInformer
        class SubmittedPDE(PDE):
            def __init__(self):
                self.dim = 2
                self.equations = equations
        return training.BatchedPhysicsInformer(PhysicsInformer(
            required_outputs=["reaction_diffusion"], equations=SubmittedPDE(),
            grad_method="spectral", bounds=[1.0, 1.0], device=str(device)))
    return {"checks": checks, "messages": messages, "datasets": datasets,
            "model": model, "physics": physics if level == 3 else None}
