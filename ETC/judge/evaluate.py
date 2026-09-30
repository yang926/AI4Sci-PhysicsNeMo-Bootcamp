"""Trusted fixed-budget runner. Submitted files are parsed, never imported."""
import argparse
import importlib.util
import importlib
import inspect
import json
import math
from pathlib import Path
import sys
import time
import traceback

from .catalog import CHALLENGES, lesson_path, quality_metrics
from .expressions import SubmissionError, build


def load_lesson(challenge, filename):
    path = lesson_path(challenge, filename)
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("judge_" + path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if challenge != "4" and not hasattr(module, "reference_equations"):
        raise RuntimeError(
            "This student checkout has no instructor answer bundle. Submit from the notebook "
            "to the separately deployed event judge; do not run a scoring worker from student main."
        )
    return module


def check_operator_level(source, filename, settings):
    """Check factories using tiny CPU fixtures; do not construct/train a model."""
    import torch
    import yaml
    from .operators import check_operator
    lesson_dir = lesson_path("4", filename).parent
    if str(lesson_dir) not in sys.path:
        sys.path.insert(0, str(lesson_dir))
    training = importlib.import_module("operator_training")
    level = int(filename.removesuffix(".py").rsplit("_l", 1)[1])
    config_file = lesson_dir / "conf" / f"config_{['FNO', 'AFNO', 'PINO'][level - 1]}.yaml"
    config = yaml.safe_load(config_file.read_text())
    if config["data"]["grid_size"] != settings["operator_data"]["grid_size"]:
        raise RuntimeError("Frozen operator dataset and course configuration have different grid sizes")
    torch.set_num_threads(2)
    exercise = check_operator(source, level, config, training)
    return exercise, training, level, config_file


def run_operator(prepared, filename, settings, directory, device):
    exercise, training, level, config_file = prepared
    generator = importlib.import_module("generate_data")
    metrics = {}
    if all(exercise["checks"].values()):
        data_dir = directory / "operator_data"
        if not data_dir.exists():
            generator.generate_splits(data_dir, **settings["operator_data"])
        metrics = training.run(level, exercise["model"], dataset_builder=exercise["datasets"],
            physics_builder=exercise["physics"], argv=["--config", str(config_file), "--data-dir", str(data_dir),
                "--device", device, "--steps", str(settings["steps"]), "--seed", str(settings["seed"]),
                "--output-dir", str(directory / filename.removesuffix(".py"))])
        if metrics.get("reference") is not False or metrics.get("dataset_exercise_checked") is not True:
            raise RuntimeError("Operator runner did not evaluate the submitted exercise")
    return exercise["checks"], metrics, exercise["messages"]


def prepare_operator(source, filename, settings, directory, device):
    return run_operator(check_operator_level(source, filename, settings), filename,
                        settings, directory, device)


def check_equations(module, source, challenge):
    import sympy as sp
    signature = inspect.signature(module.reference_equations)
    parameters = {name: p.default for name, p in signature.parameters.items()}
    student = build(source, parameters)
    x, y, t = sp.symbols("x y t")
    coords = (x, y, t) if module.TIME_END is not None else (x, y)
    fields = {name: sp.Function(name)(*coords) for name in module.FIELD_NAMES}
    if challenge == "1":
        args = (x, y, t, fields["u"], sp.Symbol("c"))
    elif challenge == "2":
        args = (x, y, t, fields["u"], fields["v"], fields["p"], sp.Symbol("nu"), sp.Symbol("rho"))
    else:
        # Nonzero symbolic coefficients catch missing terms hidden by default
        # zero advection/source/relaxation parameters in the training example.
        params = {key: sp.Symbol(key) for key in module.DEFAULT_PHYSICS}
        args = (x, y, t, fields, params)
    actual = student(*args)
    expected = module.reference_equations(*args)
    if set(actual) != set(expected):
        raise SubmissionError("Return exactly these residual names: " + ", ".join(expected))
    from ETC.runtime.symbolic_checks import equivalent_expression
    try:
        checks = {key: equivalent_expression(actual[key], expected[key]) for key in expected}
    except ValueError as exc:
        raise SubmissionError(str(exc)) from exc
    return student, checks


def level_points(checks, metrics, names, settings):
    implementation = settings["implementation_points"] * sum(checks.values()) / len(checks)
    errors = {}
    quality = 0.0
    if all(checks.values()):
        for name in names:
            value = metrics
            for part in name.split("."):
                if not isinstance(value, dict) or part not in value:
                    raise RuntimeError("Trusted evaluator did not produce " + name)
                value = value[part]
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise RuntimeError("Trusted evaluator produced an invalid metric: " + name)
            errors[name] = value
        if settings["quality_points"]:
            scale = settings["quality_error_scale"]
            quality = settings["quality_points"] * sum(1 / (1 + (v / scale) ** 2) for v in errors.values()) / len(errors)
    return {"score": round(implementation + quality, 6), "implementation_points": implementation,
            "quality_points": quality, "components": checks, "evaluation_errors": errors}


def implementation_points(checks, settings):
    """Award only checked implementation credit, independent of GPU feedback."""
    if not checks or any(type(value) is not bool for value in checks.values()):
        raise RuntimeError("Trusted evaluator produced invalid implementation checks")
    points = settings["implementation_points"] * sum(checks.values()) / len(checks)
    return {"score": round(points, 6), "implementation_points": points,
            "quality_points": 0, "components": checks, "evaluation_errors": {}}


def assessment(challenge, sources, settings):
    """Finish every submitted Level's checks before starting any training."""
    if challenge not in CHALLENGES or not isinstance(sources, dict) or not sources:
        raise SubmissionError("Provide one known Challenge and at least one Level")
    if set(sources) - set(CHALLENGES[challenge]["files"]):
        raise SubmissionError("Submission contains an unknown Level filename")
    results, prepared = {}, {}
    for filename in CHALLENGES[challenge]["files"]:
        if filename not in sources:
            results[filename] = {"status": "not_submitted", "score": 0,
                                 "feedback_status": "not_submitted"}
            continue
        try:
            if challenge == "4":
                item = check_operator_level(sources[filename], filename, settings)
                checks, messages = item[0]["checks"], item[0]["messages"]
            else:
                module = load_lesson(challenge, filename)
                from .contracts import check_setup
                setup_functions, setup_checks = check_setup(module, sources[filename], challenge)
                student, checks = check_equations(module, sources[filename], challenge)
                checks = {**checks, **setup_checks}
                messages = {key: "This component does not match the stated problem. Keep the required residual sign, coefficient symbols and key names."
                            for key, passed in checks.items() if not passed}
                item = module, student, setup_functions
        except SubmissionError as exc:
            results[filename] = {"status": "invalid", "score": 0,
                                 "feedback_status": "skipped", "message": str(exc)[:2000]}
            continue
        result = implementation_points(checks, settings)
        result.update(status="implementation_checked" if all(checks.values()) else "incorrect_implementation",
                      feedback_status="pending" if all(checks.values()) else "skipped", messages=messages)
        if not all(checks.values()):
            result["message"] = "Review failed PDE, condition, geometry or parameter components. Training is skipped until the complete stated problem matches."
        else:
            prepared[filename] = item
        results[filename] = result
    return results, prepared


def summarize(challenge, results, settings, feedback_status):
    score = round(sum(level["score"] for level in results.values()) / len(results), 2)
    return {"kind": "pilot_not_official", "challenge": challenge, "score": score,
            "levels": results, "rubric": settings["rubric"], "steps": settings["steps"],
            "seed": settings["seed"], "feedback_status": feedback_status,
            "implementation_assessment_complete": True}


def preflight(challenge, sources, settings):
    """Same implementation checks as the grader; no training or CUDA work."""
    results, _ = assessment(challenge, sources, settings)
    for level in results.values():
        if level["feedback_status"] == "pending":
            level["feedback_status"] = "not_requested"
    return summarize(challenge, results, settings, "not_requested")


def run_pinn(prepared, challenge, filename, settings, directory, device):
    # Only trusted course main() runs. Uploaded Python is never executed.
    module, student, setup_functions = prepared
    module.student_equations = student
    for name, function in setup_functions.items():
        setattr(module, name, function)
    output = directory / filename.removesuffix(".py")
    previous = sys.argv
    sys.argv = [str(lesson_path(challenge, filename)), "--steps", str(settings["steps"]),
                "--seed", str(settings["seed"]), "--device", device, "--output-dir", str(output)]
    try:
        module.main()
    finally:
        sys.argv = previous
    metrics = json.loads((output / "metrics.json").read_text())
    if metrics.get("reference_implementation") is not False:
        raise RuntimeError("Trusted runner unexpectedly selected reference mode")
    return metrics


def evaluate(challenge, sources, settings, directory, device, checkpoint=None):
    """Implementation credit survives failure of the separate numerical run."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    results, prepared = assessment(challenge, sources, settings)
    def save(status):
        result = summarize(challenge, results, settings, status)
        if checkpoint is not None:
            checkpoint(result)
        return result
    save("pending")
    failed = False
    for filename, item in prepared.items():
        results[filename]["feedback_status"] = "running"
        save("running")
        feedback_started = time.monotonic()
        try:
            metrics = (run_operator(item, filename, settings, directory, device)[1] if challenge == "4"
                       else run_pinn(item, challenge, filename, settings, directory, device))
            points = level_points(results[filename]["components"], metrics, quality_metrics(challenge, filename), settings)
            results[filename].update(points, status="evaluated", feedback_status="completed")
        except Exception:
            # Keep details only in the private worker log, never leak paths or
            # uploaded expressions through public error messages.
            traceback.print_exc()
            failed = True
            results[filename].update(feedback_status="failed", message=
                "Implementation points are preserved. Numerical feedback failed; ask the instructor to inspect the private runner log.")
        finally:
            # Wall time for training, numerical evaluation and saved artifacts,
            # not GPU kernel time or optimization alone. No student metric is used.
            results[filename]["feedback_seconds"] = round(time.monotonic() - feedback_started, 6)
        save("running")
    return save("failed" if failed else "completed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runs", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    payload = json.loads(args.input.read_text())
    # The runner interprets a bounded language, not arbitrary Python. Limits
    # provide extra protection against expensive symbolic/training work.
    import resource
    limit = payload["settings"]["timeout_seconds"]
    resource.setrlimit(resource.RLIMIT_CPU, (limit, limit))
    resource.setrlimit(resource.RLIMIT_FSIZE, (128 * 1024 ** 2, 128 * 1024 ** 2))
    if args.preflight_only:
        result = preflight(payload["challenge"], payload["sources"], payload["settings"])
    else:
        if args.runs is None or args.device is None:
            parser.error("--runs and --device are required for numerical feedback")
        callback = None
        if args.checkpoint is not None:
            from .result_checkpoint import write_checkpoint
            callback = lambda result: write_checkpoint(args.checkpoint, payload, result)
        result = evaluate(payload["challenge"], payload["sources"], payload["settings"], args.runs, args.device, callback)
    from .result_checkpoint import atomic_json
    atomic_json(args.output, result)


if __name__ == "__main__":
    main()
