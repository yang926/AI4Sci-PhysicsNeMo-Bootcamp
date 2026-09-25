"""Version 3: submit the original problem setup, not just the PDE residual.

Only bounded symbolic/data functions are interpreted. The web process checks
signatures; the isolated evaluator checks values before running trusted code.
"""
import inspect

from .expressions import SubmissionError, build, extract

CONTRACT_VERSION = 3


def function_names(challenge):
    names = ["student_equations", "student_conditions"]
    if str(challenge) == "1":
        names.append("student_speed")
    if str(challenge) == "2":
        names.append("student_geometry")
    if str(challenge) == "3":
        names.extend(["student_parameters", "student_solution"])
    return names


def source_nodes(source, challenge):
    nodes = []
    for name in function_names(challenge):
        try:
            nodes.append(extract(source, name))
        except SubmissionError as exc:
            raise SubmissionError(
                f"Submission format v3 requires {name}. Update the course and complete all "
                "exercise functions; old PDE-only submissions cannot be reused unchanged. " + str(exc)
            ) from exc
    return nodes


def check_setup(module, source, challenge):
    """Return interpreted setup functions and independently checked components."""
    import sympy as sp
    from ETC.runtime.exercises import block_geometry, conditions, physical_parameters, analytic_expression_checks

    source_nodes(source, challenge)
    functions, checks = {}, {}
    for name in function_names(challenge)[1:]:
        reference = getattr(module, name.replace("student_", "reference_", 1))
        parameters = {name: p.default for name, p in inspect.signature(reference).parameters.items()}
        function = build(source, parameters, name=name, data=True)
        functions[name] = function
        try:
            if name == "student_conditions":
                actual = conditions(function, wave=str(challenge) == "1")
                expected = conditions(reference, wave=str(challenge) == "1")
                if set(actual) != set(expected):
                    raise SubmissionError("student_conditions must return exactly: " + ", ".join(expected))
                for key in expected:
                    checks["condition." + key] = sp.expand(actual[key] - expected[key]) == 0
            elif name == "student_speed":
                x, y = sp.symbols("x y")
                actual, expected = function(x, y), reference(x, y)
                if not isinstance(actual, dict) or set(actual) != {"c"} or not isinstance(actual["c"], sp.Expr):
                    raise SubmissionError("student_speed must return {'c': speed_expression}")
                checks["parameter.wave_speed"] = sp.expand(actual["c"] - expected["c"]) == 0
            elif name == "student_solution":
                checks.update({"solution." + key: value for key, value in
                    analytic_expression_checks(function, reference, module.DEFAULT_PHYSICS).items()})
            elif name == "student_geometry":
                actual, expected = block_geometry(function), block_geometry(reference)
                checks["geometry.block_count"] = len(actual) == len(expected)
                for index, block in enumerate(expected):
                    checks[f"geometry.block_{index + 1}"] = (
                        index < len(actual) and all(abs(a - b) <= 1e-12 for a, b in zip(actual[index], block)))
            else:
                actual = physical_parameters(function, module.DEFAULT_PHYSICS)
                expected = physical_parameters(reference, module.DEFAULT_PHYSICS)
                checks.update({"parameter." + key: abs(actual[key] - expected[key]) <= 1e-12 for key in expected})
        except (ValueError, TypeError, KeyError) as exc:
            raise SubmissionError(f"{name}: {exc}") from exc
    return functions, checks
