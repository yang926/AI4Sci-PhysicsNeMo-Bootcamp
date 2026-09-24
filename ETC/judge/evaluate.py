"""Trusted fixed-budget runner. Submitted files are parsed, never imported."""
import argparse
import importlib.util
import importlib
import inspect
import json
import math
from pathlib import Path
import sys

from .catalog import CHALLENGES, lesson_path, quality_metrics
from .expressions import SubmissionError, build


def load_lesson(challenge, filename):
    path = lesson_path(challenge, filename)
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("judge_" + path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare_operator(source, filename, settings, directory, device):
    import torch
    import yaml
    from .operators import check_operator
    lesson_dir = lesson_path("4", filename).parent
    if str(lesson_dir) not in sys.path:
        sys.path.insert(0, str(lesson_dir))
    training = importlib.import_module("operator_training")
    generator = importlib.import_module("generate_data")
    level = int(filename.removesuffix(".py").rsplit("_l", 1)[1])
    config_file = lesson_dir / "conf" / f"config_{['FNO', 'AFNO', 'PINO'][level - 1]}.yaml"
    config = yaml.safe_load(config_file.read_text())
    if config["data"]["grid_size"] != settings["operator_data"]["grid_size"]:
        raise RuntimeError("Frozen operator dataset and course configuration have different grid sizes")
    torch.set_num_threads(2)
    exercise = check_operator(source, level, config, training)
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
    checks = {key: sp.expand(actual[key] - expected[key]) == 0 for key in expected}
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
        scale = settings["quality_error_scale"]
        quality = settings["quality_points"] * sum(1 / (1 + (v / scale) ** 2) for v in errors.values()) / len(errors)
    return {"score": round(implementation + quality, 6), "implementation_points": implementation,
            "quality_points": quality, "components": checks, "evaluation_errors": errors}


def evaluate(challenge, sources, settings, directory, device):
    """A coherent submission snapshot; omitted Levels count as zero."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    results = {}
    for filename in CHALLENGES[challenge]["files"]:
        if filename not in sources:
            results[filename] = {"status": "not_submitted", "score": 0}
            continue
        if challenge == "4":
            try:
                checks, metrics, messages = prepare_operator(sources[filename], filename, settings, directory, device)
            except SubmissionError as exc:
                results[filename] = {"status": "invalid", "score": 0, "message": str(exc)}
                continue
            result = level_points(checks, metrics, quality_metrics(challenge, filename), settings)
            result.update(status="evaluated" if all(checks.values()) else "incorrect_implementation", messages=messages)
            results[filename] = result
            continue
        module = load_lesson(challenge, filename)
        try:
            student, checks = check_equations(module, sources[filename], challenge)
        except SubmissionError as exc:
            results[filename] = {"status": "invalid", "score": 0, "message": str(exc)}
            continue
        metrics = {}
        if all(checks.values()):
            # Only trusted course main() is called. No code from the uploaded
            # module, defaults, annotations, or imports is executed.
            module.student_equations = student
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
        result = level_points(checks, metrics, quality_metrics(challenge, filename), settings)
        result["status"] = "evaluated" if all(checks.values()) else "incorrect_equations"
        if not all(checks.values()):
            result["message"] = "Review failed residual components. Training is skipped until all match."
        results[filename] = result
    score = round(sum(level["score"] for level in results.values()) / len(results), 2)
    return {"kind": "pilot_not_official", "challenge": challenge, "score": score,
            "levels": results, "rubric": settings["rubric"], "steps": settings["steps"], "seed": settings["seed"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text())
    # The runner interprets a bounded language, not arbitrary Python. Limits
    # provide extra protection against expensive symbolic/training work.
    import resource
    limit = payload["settings"]["timeout_seconds"]
    resource.setrlimit(resource.RLIMIT_CPU, (limit, limit))
    resource.setrlimit(resource.RLIMIT_FSIZE, (128 * 1024 ** 2, 128 * 1024 ** 2))
    result = evaluate(payload["challenge"], payload["sources"], payload["settings"], args.runs, args.device)
    args.output.write_text(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
