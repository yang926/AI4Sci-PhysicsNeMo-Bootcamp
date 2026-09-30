"""Bounded, job-bound implementation checkpoints written by the trusted runner.

The worker creates a fresh private (0700) directory and a random nonce for every
attempt. Student source is interpreted, never executed, and has no filesystem
operations. These files are internal state, not an upload format or an API.
"""
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

from .catalog import CHALLENGES, quality_metrics

MAX_RESULT_BYTES = 1024 * 1024


def atomic_json(path, value):
    path = Path(path)
    data = json.dumps(value, allow_nan=False).encode()
    if len(data) > MAX_RESULT_BYTES:
        raise ValueError("Evaluator result is too large")
    descriptor, temporary = tempfile.mkstemp(prefix=".assessment-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def request_digest(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()


def write_checkpoint(path, payload, result):
    if not isinstance(payload.get("run_nonce"), str) or len(payload["run_nonce"]) != 64:
        raise ValueError("A worker-issued checkpoint nonce is required")
    atomic_json(path, {"version": 1, "request_digest": request_digest(payload), "result": result})


def bounded_json(path):
    # Do not follow a symlink, read a stale artifact from another attempt, or
    # allocate an unbounded result before validation.
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_RESULT_BYTES:
        raise ValueError("Missing or invalid evaluator result")
    with path.open("rb") as stream:
        data = stream.read(MAX_RESULT_BYTES + 1)
    if len(data) > MAX_RESULT_BYTES:
        raise ValueError("Evaluator result is too large")
    return json.loads(data)


def validate_result(result, payload):
    """Fail closed on partial assessments and inconsistent claimed scores."""
    if not isinstance(result, dict):
        raise ValueError("Invalid evaluator result")
    settings, challenge = payload["settings"], payload["challenge"]
    for key, expected in (("kind", "pilot_not_official"), ("challenge", challenge),
                          ("rubric", settings["rubric"]), ("steps", settings["steps"]),
                          ("seed", settings["seed"]), ("implementation_assessment_complete", True)):
        if type(result.get(key)) is not type(expected) or result[key] != expected:
            raise ValueError("Evaluator result does not match this attempt")
    levels = result.get("levels")
    if not isinstance(levels, dict) or set(levels) != set(CHALLENGES[challenge]["files"]):
        raise ValueError("Every Level must have an implementation assessment")
    feedback_states = {"pending", "running", "completed", "failed", "time_limit", "not_requested", "skipped", "not_submitted"}
    if result.get("feedback_status") not in feedback_states:
        raise ValueError("Invalid numerical feedback status")
    if "elapsed_seconds" in result:
        seconds = result["elapsed_seconds"]
        if (type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds < 0
                or result["feedback_status"] == "not_requested"):
            raise ValueError("Invalid worker elapsed time")
    for filename, level in levels.items():
        if not isinstance(level, dict) or level.get("feedback_status") not in feedback_states:
            raise ValueError("Invalid Level assessment")
        status = level.get("status")
        if type(level.get("score")) not in (int, float) or not math.isfinite(level["score"]):
            raise ValueError("Invalid Level score")
        if "feedback_seconds" in level:
            seconds = level["feedback_seconds"]
            if (type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds < 0
                    or level["feedback_status"] not in {"completed", "failed", "time_limit"}
                    or status not in {"implementation_checked", "evaluated"}):
                raise ValueError("Invalid numerical feedback duration")
        elif level["feedback_status"] == "completed":
            raise ValueError("Completed numerical feedback is missing its duration")
        if filename not in payload["sources"]:
            if status != "not_submitted" or level.get("score") != 0:
                raise ValueError("Omitted Levels must receive zero")
            continue
        if status == "invalid":
            if level.get("score") != 0 or "components" in level:
                raise ValueError("Invalid submissions must receive zero")
            continue
        if status not in {"implementation_checked", "incorrect_implementation", "evaluated"}:
            raise ValueError("Unfinished implementation assessment")
        checks = level.get("components")
        if (not isinstance(checks, dict) or not 1 <= len(checks) <= 128
                or any(not isinstance(k, str) or len(k) > 200 or type(v) is not bool for k, v in checks.items())):
            raise ValueError("Invalid implementation components")
        correct = all(checks.values())
        if (status == "incorrect_implementation") == correct:
            raise ValueError("Implementation status and checks disagree")
        points = settings["implementation_points"] * sum(checks.values()) / len(checks)
        quality = level.get("quality_points")
        if (type(quality) not in (int, float) or not math.isfinite(quality)
                or not 0 <= quality <= settings["quality_points"]):
            raise ValueError("Invalid numerical quality points")
        if (quality != 0 or status == "evaluated") and (not correct or level["feedback_status"] != "completed"):
            raise ValueError("Unverified numerical feedback cannot receive credit")
        if (type(level.get("implementation_points")) not in (int, float)
                or level["implementation_points"] != points
                or level.get("score") != round(points + quality, 6)):
            raise ValueError("Implementation score and checks disagree")
        errors = level.get("evaluation_errors")
        if not isinstance(errors, dict) or any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in errors.values()):
            raise ValueError("Invalid numerical errors")
        if status == "evaluated" and set(errors) != set(quality_metrics(challenge, filename)):
            raise ValueError("Completed feedback is missing required numerical errors")
        if status != "evaluated" and (errors or level["feedback_status"] == "completed"):
            raise ValueError("Unrun feedback cannot contain numerical errors")
    score = result.get("score")
    expected = round(sum(level["score"] for level in levels.values()) / len(levels), 2)
    if type(score) not in (int, float) or not math.isfinite(score) or score != expected:
        raise ValueError("Attempt score and Level assessments disagree")
    return result


def recover_checkpoint(path, payload, reason):
    if reason not in {"time_limit", "failed"}:
        raise ValueError("Invalid numerical feedback failure")
    envelope = bounded_json(path)
    if (not isinstance(envelope, dict) or set(envelope) != {"version", "request_digest", "result"}
            or envelope["version"] != 1 or envelope["request_digest"] != request_digest(payload)):
        raise ValueError("Checkpoint belongs to a different attempt")
    result = copy.deepcopy(validate_result(envelope["result"], payload))
    result["feedback_status"] = reason
    for level in result["levels"].values():
        if level["feedback_status"] in {"pending", "running"}:
            level["feedback_status"] = reason
            level["message"] = ("Implementation points are preserved. Numerical feedback exceeded the time limit."
                                if reason == "time_limit" else
                                "Implementation points are preserved. Numerical feedback failed; ask the instructor to inspect the private runner log.")
    return validate_result(result, payload)
