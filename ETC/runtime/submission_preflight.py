"""Offline format checks without a distributed answer key or code execution."""
import ast
import math

from ETC.judge.catalog import CHALLENGES
from ETC.judge.contracts import source_nodes as problem_source_nodes
from ETC.judge.expressions import SubmissionError


def check_saved_submission(payload, timeout=60):
    """Check bounded syntax and unfinished placeholders, never correctness.

    timeout remains accepted for notebook compatibility. This check only parses
    bounded text; it does not import or execute participant code.
    Mathematical correctness and points belong to the separate judge server.
    """
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or not 0 < timeout <= 60:
        raise ValueError("Code-check timeout must be greater than zero and at most 60 seconds.")
    if not isinstance(payload, dict):
        raise SubmissionError("Expected a saved-code submission.")
    challenge = str(payload.get("challenge", ""))
    sources = payload.get("sources")
    if challenge not in CHALLENGES or not isinstance(sources, dict) or not sources:
        raise SubmissionError("Choose a supported Challenge and at least one Level.")
    filenames = CHALLENGES[challenge]["files"]
    if set(sources) - set(filenames):
        raise SubmissionError("Unknown exercise filename.")
    from .submission import _exercise_placeholders
    levels = {}
    for level, filename in enumerate(filenames, 1):
        if filename not in sources:
            levels[filename] = {"status": "not_submitted"}
            continue
        try:
            if challenge == "4":
                from ETC.judge.operators import source_nodes
                nodes = source_nodes(sources[filename], level)
            else:
                nodes = problem_source_nodes(sources[filename], challenge)
            for node in nodes:
                if list(_exercise_placeholders(node)):
                    raise SubmissionError("Complete the exercise placeholders before submitting.")
                # A return is not proof of a correct answer; it only catches
                # accidentally empty functions. PINO has a class assignment.
                if isinstance(node, ast.FunctionDef) and not any(
                    isinstance(part, ast.Return) and part.value is not None
                    for part in ast.walk(node)
                ):
                    raise SubmissionError(f"{node.name}: return the required exercise components.")
            levels[filename] = {"status": "format_checked", "messages": [
                "Required definitions found; no unfinished placeholders. Correctness has not been checked."]}
        except SubmissionError as exc:
            levels[filename] = {"status": "invalid", "message": str(exc)}
    return {"scope": "format_only", "correctness_checked": False, "levels": levels}
