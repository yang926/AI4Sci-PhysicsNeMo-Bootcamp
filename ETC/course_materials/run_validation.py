#!/usr/bin/env python3
"""Run reproducible PhysicsNeMo course validation without downloads or remote access.

unit: pytest tests; smoke: all 18 actual lesson runs and artifact checks;
convergence: analytical cases with required held-out improvement; all seven
Lab modes also require their absolute physical lesson-accuracy limits.
A smoke pass never certifies convergence. These checks establish only their
recorded criteria, not full physical or numerical convergence.
Request each lesson's full training budget explicitly; the default 500-step
budget is not silently increased for any case. Navier-Stokes checks cover only
the synthetic Taylor-Green fixture, not weather forecast accuracy. The smoke
suite runs Lab 4 with the original data_lat.npy, just like the student notebook;
only the convergence regression explicitly selects --smoke-data.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "ETC/course_materials/course_manifest.json"
LAB1_CASE_MODES = {"pinn_forward": "forward", "pinn_parameterized": "parameterized", "pinn_inverse": "inverse"}
LAB_CASES = (*LAB1_CASE_MODES, "projectile", "diffusion", "diffusion_parameterized", "navier_stokes")
CONVERGENCE_CASES = (*LAB_CASES, "wave_l1", "operators_l1", "operators_l2", "operators_l3")
REQUIRED_ARTIFACTS = ("metrics.json", "loss.csv", "model.pt", "predictions.npz", "preview.png")


def now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_snapshot(output):
    """Hash source and teaching documents, including local uncommitted migrations."""
    try:
        listed = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
                                cwd=ROOT, capture_output=True, check=True).stdout.decode().split("\0")
        paths = [ROOT / name for name in listed if name]
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                              check=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        paths = [path for folder in ("01_labs", "02_challenges", "ETC")
                 for path in (ROOT / folder).rglob("*") if path.is_file()]
        head = None
    hashes = {}
    suffixes = {".py", ".yaml", ".yml", ".json", ".ipynb", ".md", ".toml"}
    for path in sorted(set(paths)):
        if not path.is_file() or output == path or output in path.parents or any(p.startswith("._") for p in path.parts):
            continue
        if any(part in {".git", "__pycache__", ".venv", "venv", "outputs", "runs", "validation-runs", ".ipynb_checkpoints", ".pytest_cache"}
               for part in path.relative_to(ROOT).parts):
            continue
        if path.suffix.lower() in suffixes or path.name.startswith(("Dockerfile", "requirements")):
            hashes[str(path.relative_to(ROOT))] = sha256(path)
    payload = {"git_head": head, "source_sha256": hashes}
    (output / "source_hashes.json").write_text(json.dumps(payload, indent=2) + "\n")
    return {"git_head": head, "source_file_count": len(hashes),
            "source_hashes_file": "source_hashes.json", "source_hashes_sha256": sha256(output / "source_hashes.json")}


def versions():
    result = {"python": sys.version, "executable": sys.executable,
              "platform": platform.platform(), "packages": {}}
    for package in ("nvidia-physicsnemo", "torch", "numpy", "h5py", "PyYAML", "matplotlib", "pytest"):
        try:
            result["packages"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result["packages"][package] = None
    return result


def execute(command, output, label, timeout):
    """Record the exact argv, log, process exit and timing for every command."""
    log = output / f"{label}.log"
    started, clock = now(), time.perf_counter()
    environment = dict(os.environ)
    environment.update(OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", MPLBACKEND="Agg",
                       PYTHONUNBUFFERED="1")
    record = {"label": label, "command": list(map(str, command)), "cwd": str(ROOT),
              "started_at": started, "log": log.name}
    with log.open("w") as stream:
        try:
            completed = subprocess.run(record["command"], cwd=ROOT, stdout=stream,
                                       stderr=subprocess.STDOUT, timeout=timeout, env=environment)
            record["exit_code"] = completed.returncode
        except subprocess.TimeoutExpired:
            record.update(exit_code=124, error=f"Command exceeded timeout of {timeout} seconds")
            stream.write(f"\nVALIDATOR: {record['error']}\n")
        except OSError as error:
            record.update(exit_code=127, error=str(error))
            stream.write(f"\nVALIDATOR: {error}\n")
    record.update(finished_at=now(), elapsed_seconds=round(time.perf_counter() - clock, 3),
                  log_sha256=sha256(log), passed=record["exit_code"] == 0)
    (output / f"{label}.command.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"label": label, "exit_code": record["exit_code"],
                      "elapsed_seconds": record["elapsed_seconds"]}), flush=True)
    return record


def finite_json(value, path="metrics"):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"Non-finite scalar: {path}")
    if isinstance(value, dict):
        for name, item in value.items():
            finite_json(item, f"{path}.{name}")
    if isinstance(value, list):
        for index, item in enumerate(value):
            finite_json(item, f"{path}[{index}]")


def validate_optimizer_history(rows, recipe, steps):
    """Check the complete call history promised by the new Lab recipes.

    Older/challenge artifacts may deliberately log only initial/final losses;
    retain that contract when no per-call recipe is declared.
    """
    split_calls = (recipe.get("optimizer") == "Adam then full-batch L-BFGS"
                   or "adam_step_calls" in recipe or "lbfgs_step_calls" in recipe)
    bar_calls = (recipe.get("step_unit") == "optimizer_calls"
                 and (recipe.get("optimizer") == "adam_then_lbfgs" or "adam_steps" in recipe))
    adam_only = (recipe.get("name") == "adam_cosine_fp32_v1" or "optimizer_step_calls" in recipe)
    if not (split_calls or bar_calls or adam_only):
        return
    if len(rows) != steps:
        raise ValueError("Per-call Lab loss.csv must contain every requested optimizer step")
    for expected, row in enumerate(rows, 1):
        try:
            actual = float(row.get("step", ""))
        except (TypeError, ValueError):
            raise ValueError("Per-call Lab steps must be the exact sequence 1..steps") from None
        if not math.isfinite(actual) or actual != expected:
            raise ValueError("Per-call Lab steps must be the exact sequence 1..steps")

    def count(name):
        value = recipe.get(name)
        if type(value) is not int or value < 0:
            raise ValueError(f"training_recipe.{name} must be a nonnegative integer call count")
        return value

    if adam_only and count("optimizer_step_calls") != len(rows):
        raise ValueError("training_recipe.optimizer_step_calls does not match loss.csv")
    if split_calls or bar_calls:
        adam = count("adam_step_calls" if split_calls else "adam_steps")
        lbfgs = count("lbfgs_step_calls") if split_calls else steps - adam
        if adam > steps or lbfgs < 0 or adam + lbfgs != steps:
            raise ValueError("Adam/L-BFGS call counts do not sum to the requested steps")
        expected_phases = ["adam"] * adam + ["lbfgs"] * lbfgs
        if [row.get("phase") for row in rows] != expected_phases:
            raise ValueError("loss.csv phases do not match the recorded Adam/L-BFGS call counts")
        closures = []
        for row in rows:
            try:
                value = float(row.get("closure_evaluations", ""))
            except (TypeError, ValueError):
                raise ValueError("closure_evaluations must be positive integer counts") from None
            if not math.isfinite(value) or value < 1 or not value.is_integer():
                raise ValueError("closure_evaluations must be positive integer counts")
            if row["phase"] == "adam" and value != 1:
                raise ValueError("Each Adam call must record exactly one closure evaluation")
            closures.append(int(value))
        if count("closure_evaluations") != sum(closures):
            raise ValueError("training_recipe.closure_evaluations does not match the loss.csv sum")


def validate_artifacts(path, *, steps, device, expected_version, seed):
    import numpy as np
    import torch
    from PIL import Image

    for filename in REQUIRED_ARTIFACTS:
        artifact = path / filename
        if artifact.is_symlink():
            raise ValueError(f"Required artifact must be produced in this run, not a symbolic link: {artifact}")
        if not artifact.is_file() or artifact.stat().st_size == 0:
            raise ValueError(f"Missing or empty required artifact: {artifact}")
    metrics = json.loads((path / "metrics.json").read_text())
    finite_json(metrics)
    recipe = metrics.get("training_recipe", {})
    if not isinstance(recipe, dict):
        raise ValueError("training_recipe metadata must be a dictionary")
    recorded_dtype = recipe.get("dtype")
    if recorded_dtype is not None and recorded_dtype not in ("float32", "float64"):
        raise ValueError("training_recipe dtype must be float32 or float64")
    require_fp32 = recorded_dtype == "float32"
    for name in ("initial_loss", "final_loss"):
        if not isinstance(metrics.get(name), (float, int)) or isinstance(metrics[name], bool):
            raise ValueError(f"Missing numeric metric: {name}")
        if metrics[name] < 0:
            raise ValueError(f"Negative loss metric: {name}")
    if metrics.get("steps") != steps or metrics.get("device", "").split(":")[0] != device:
        raise ValueError("Recorded steps/device do not match the requested run")
    if metrics.get("physicsnemo_version") != expected_version:
        raise ValueError("Recorded PhysicsNeMo version does not match the course manifest")
    installed_version = importlib.metadata.version("nvidia-physicsnemo")
    if installed_version != expected_version:
        raise ValueError("Installed PhysicsNeMo version does not match the course manifest")
    if metrics.get("torch_version") != str(torch.__version__):
        raise ValueError("Recorded PyTorch version does not match the validator's interpreter")
    if metrics.get("seed") != seed or isinstance(metrics.get("seed"), bool):
        raise ValueError("Recorded seed does not match the requested seed")
    with (path / "loss.csv").open(newline="") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        loss_columns = [name for name in (reader.fieldnames or []) if name in {"loss", "total"}]
    if len(rows) < min(2, steps) or not loss_columns:
        raise ValueError("loss.csv must contain initial/final or per-step numeric losses")
    if "step" not in rows[0]:
        raise ValueError("loss.csv is missing the step column")
    validate_optimizer_history(rows, recipe, steps)
    observed_steps = []
    for row in rows:
        observed_steps.append(int(float(row["step"])))
        for name, value in row.items():
            if value is None or value == "":
                continue
            try:
                numeric = float(value)
            except ValueError:
                if name in loss_columns or name == "step":
                    raise ValueError(f"Nonnumeric {name} in loss.csv") from None
                continue  # explanatory phase/category labels are allowed
            if not math.isfinite(numeric):
                raise ValueError(f"Non-finite {name} in loss.csv")
    if max(observed_steps) != steps:
        raise ValueError("loss.csv does not reach the requested final step")
    array_shapes = {}
    numeric_arrays = 0
    with np.load(path / "predictions.npz", allow_pickle=False) as arrays:
        for name in arrays.files:
            array = arrays[name]
            if array.size == 0:
                raise ValueError(f"Empty prediction array: {name}")
            if array.dtype.kind in "biufc":
                numeric_arrays += 1
                if not np.isfinite(array).all():
                    raise ValueError(f"Non-finite prediction array: {name}")
                if require_fp32 and array.dtype.kind == "f" and array.dtype != np.float32:
                    raise ValueError(f"FP32 recipe has non-FP32 prediction array: {name}={array.dtype}")
            elif array.dtype.kind not in "US":
                raise ValueError(f"Unexpected prediction array dtype: {name}={array.dtype}")
            array_shapes[name] = list(array.shape)
    if numeric_arrays == 0:
        raise ValueError("predictions.npz contains no numeric arrays")
    # These checkpoints were generated by the current run; weights_only still
    # avoids arbitrary pickle object execution during automated validation.
    checkpoint = torch.load(path / "model.pt", map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint must contain a model state and provenance dictionary")
    checkpoint_metrics = checkpoint.get("metrics", {})
    if not isinstance(checkpoint_metrics, dict):
        raise ValueError("Checkpoint metrics metadata must be a dictionary")
    checkpoint_version = checkpoint.get("physicsnemo_version", checkpoint_metrics.get("physicsnemo_version"))
    checkpoint_seed = checkpoint.get("seed", checkpoint_metrics.get("seed"))
    if checkpoint_version != installed_version:
        raise ValueError("Checkpoint PhysicsNeMo version does not match the actual installed version")
    if checkpoint_seed != seed or isinstance(checkpoint_seed, bool):
        raise ValueError("Checkpoint seed does not match the requested seed")
    checkpoint_steps = checkpoint.get("steps", checkpoint_metrics.get("steps"))
    if checkpoint_steps != steps or isinstance(checkpoint_steps, bool):
        raise ValueError("Checkpoint steps do not match the requested run")
    checkpoint_device = checkpoint.get("device", checkpoint_metrics.get("device", ""))
    if not isinstance(checkpoint_device, str) or checkpoint_device.split(":")[0] != device:
        raise ValueError("Checkpoint device does not match the requested run")
    state = checkpoint.get("model_state_dict", checkpoint.get("state_dict"))
    if not isinstance(state, dict) or not state:
        raise ValueError("Checkpoint has no model state dictionary")
    if require_fp32:
        config = checkpoint.get("config", {})
        if not isinstance(config, dict):
            raise ValueError("FP32 checkpoint config metadata must be a dictionary")
        declared = [value for key, value in config.items() if key.endswith("_dtype") or key == "dtype"]
        if declared and any(value != "float32" for value in declared):
            raise ValueError("Checkpoint dtype metadata does not match the FP32 recipe")
    tensor_count = 0
    for name, tensor in state.items():
        if torch.is_tensor(tensor):
            tensor_count += 1
            if not torch.isfinite(tensor).all():
                raise ValueError(f"Non-finite checkpoint tensor: {name}")
            if require_fp32 and tensor.is_floating_point() and tensor.dtype != torch.float32:
                raise ValueError(f"FP32 recipe has non-FP32 checkpoint tensor: {name}={tensor.dtype}")
    if not tensor_count:
        raise ValueError("Checkpoint state contains no tensors")
    with Image.open(path / "preview.png") as image:
        size = image.size
        image.verify()
    if min(size) < 1:
        raise ValueError("Empty PNG dimensions")
    return {"passed": True, "metrics": metrics, "array_shapes": array_shapes,
            "checkpoint_tensors": tensor_count, "checkpoint_physicsnemo_version": checkpoint_version,
            "checkpoint_seed": checkpoint_seed, "preview_size": list(size),
            "sha256": {name: sha256(path / name) for name in REQUIRED_ARTIFACTS}}


def lesson_accuracy(metrics, case_id):
    """Recompute every Lab gate from errors; stored accuracy flags are advisory."""
    final = metrics["heldout_after"]
    if not isinstance(final, dict):
        raise ValueError("heldout_after must contain measured lesson errors")
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    if case_id in LAB1_CASE_MODES:
        mode = LAB1_CASE_MODES[case_id]
        if mode == "parameterized":
            records = final.get("per_length")
            if (not isinstance(records, list) or len(records) != 5
                    or any(not isinstance(row, dict) or type(row.get("length")) not in (int, float)
                           or row["length"] != length
                           for row, length in zip(records, (1., 1.25, 1.5, 1.75, 2.)))):
                raise ValueError("Lab 1 parameterized accuracy requires all five evaluation lengths")
        lesson = importlib.import_module("01_labs.01_pinn.source_code.pinn_basics")
        accuracy = lesson.accuracy_checks(final, mode)
    elif case_id == "projectile":
        source = str(ROOT / "01_labs/02_projectile/source_code")
        if source not in sys.path:
            sys.path.insert(0, source)
        lesson = importlib.import_module("01_labs.02_projectile.source_code.projectile")
        accuracy = lesson.accuracy_checks(final)
    elif case_id in ("diffusion", "diffusion_parameterized"):
        lesson = importlib.import_module("01_labs.03_heat_conduction.source_code.diffusion_bar")
        accuracy = lesson.accuracy_checks(final, parameterized=case_id == "diffusion_parameterized")
    elif case_id == "navier_stokes":
        if metrics.get("data_kind") != "synthetic_taylor_green":
            raise ValueError("Navier-Stokes convergence requires the synthetic_taylor_green fixture")
        lesson = importlib.import_module("01_labs.04_navier_stokes.source_code.navier_stokes")
        accuracy = lesson.accuracy_checks(final)
    else:
        raise ValueError(f"No absolute lesson-accuracy checks for {case_id}")
    for check in accuracy["checks"]:
        value, limit = check["value"], check["limit"]
        check["passed"] = (bool(check["passed"]) and type(value) in (float, int)
                           and math.isfinite(value) and type(limit) in (float, int)
                           and math.isfinite(limit) and 0 <= value <= limit)
    accuracy["passed"] = bool(accuracy["checks"]) and all(check["passed"] for check in accuracy["checks"])
    return accuracy


def improvement(metrics, case_id, ratio):
    if case_id.startswith("operators_"):
        before, after = metrics["test_relative_l2_before"], metrics["test_relative_l2_after"]
        metric = "test_relative_l2"
    elif case_id == "wave_l1":
        before = metrics["initial_reference_error"]["relative_l2"]
        after = metrics["final_reference_error"]["relative_l2"]
        metric = "analytical_reference_relative_l2"
    elif case_id == "navier_stokes":
        before = metrics["heldout_before"]["synthetic_velocity_rmse"]
        after = metrics["heldout_after"]["synthetic_velocity_rmse"]
        metric = "heldout_synthetic_velocity_rmse"
    else:
        before = metrics["heldout_before"]["solution_rmse"]
        after = metrics["heldout_after"]["solution_rmse"]
        metric = "heldout_solution_rmse"
    if not all(type(value) in (float, int) and math.isfinite(value) for value in (before, after)):
        raise ValueError("Held-out improvement metrics must be finite numbers")
    passed = before > 0 and after >= 0 and after <= ratio * before
    result = {"metric": metric, "before": before, "after": after, "required_after_before_ratio": ratio,
              "actual_after_before_ratio": after / before if before > 0 else None,
              "passed": passed, "scope": "Held-out improvement only; full convergence is not certified."}
    if case_id in LAB_CASES:
        accuracy = lesson_accuracy(metrics, case_id)
        result.update(accuracy=accuracy, passed=passed and accuracy["passed"],
                      scope="Held-out improvement and absolute lesson-accuracy limits for this Lab. "
                            + accuracy["scope"] + " Not general convergence certification.")
    return result


def prepare_operator_data(output, suite, seed, timeout):
    import yaml

    data_dir = output / f"{suite}_operator_data"
    sizes = (24, 8, 8) if suite == "smoke" else (128, 32, 32)
    command = [sys.executable, str(ROOT / "02_challenges/04_neural_operators/generate_data.py"),
               "--output-dir", str(data_dir), "--grid-size", "16", "--max-mode", "3",
               "--train-samples", str(sizes[0]), "--val-samples", str(sizes[1]),
               "--test-samples", str(sizes[2]), "--seed", str(seed)]
    record = execute(command, output, f"{suite}_generate_operators", timeout)
    configs = {}
    for level in (1, 2, 3):
        model = ({"latent_channels": 16, "num_fno_layers": 2, "num_fno_modes": 4,
                  "padding": 0, "coord_features": False, "decoder_layers": 1, "decoder_layer_size": 16}
                 if level != 2 else {"patch_size": [4, 4], "embed_dim": 32, "depth": 2, "num_blocks": 4})
        config = {"data": {"grid_size": 16}, "model": model,
                  "training": {"steps": 20, "batch_size": 8, "learning_rate": 0.001,
                               "cpu_threads": 2, "physics_weight": 1.0}}
        path = output / f"{suite}_operators_l{level}.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False))
        configs[level] = path
    record["generated_config_sha256"] = {path.name: sha256(path) for path in configs.values()}
    return data_dir, configs, record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("unit", "smoke", "convergence", "all"), default="smoke")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output-dir", type=Path, required=True, help="Fresh directory; existing results are preserved")
    parser.add_argument("--steps", type=int, default=20, help="Smoke training steps per case")
    parser.add_argument("--convergence-steps", type=int, default=500,
                        help="Training steps per convergence case, unchanged across all Labs and cases (default: 500; request each lesson's full budget explicitly)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--case", action="append", dest="cases", help="Repeat to select manifest case IDs")
    parser.add_argument("--improvement-ratio", type=float, default=0.99,
                        help="Improvement requires after <= ratio * before; default at least 1 percent")
    parser.add_argument("--timeout", type=int, default=1200, help="Maximum seconds for each subprocess")
    args = parser.parse_args(argv)
    if not 2 <= args.steps <= 10000 or not 2 <= args.convergence_steps <= 10000:
        parser.error("steps and convergence-steps must be between 2 and 10000")
    if not 0 < args.improvement_ratio < 1 or args.timeout < 1:
        parser.error("improvement-ratio must be between 0 and 1 and timeout must be positive")
    manifest = json.loads(MANIFEST.read_text())
    runs = {case["id"]: case for case in manifest["runs"]}
    if len(runs) != len(manifest["runs"]):
        parser.error("Manifest has duplicate case IDs")
    selected = list(dict.fromkeys(args.cases)) if args.cases else list(runs)
    unknown = set(selected) - runs.keys()
    if unknown:
        parser.error(f"Unknown case IDs: {sorted(unknown)}")
    if args.suite == "convergence" and args.cases and set(selected) - set(CONVERGENCE_CASES):
        parser.error(f"Convergence supports only {', '.join(CONVERGENCE_CASES)}")
    output = args.output_dir.absolute()
    if output.exists() or output.is_symlink():
        parser.error("--output-dir must be fresh; previous results are preserved")
    output.mkdir(parents=True)
    report = {"started_at": now(), "suite": args.suite, "device": args.device,
              "seed": args.seed, "scope": {"smoke": "Execution, finite artifacts and checkpoint checks only",
              "convergence": "Held-out improvement plus absolute lesson-accuracy limits for all seven Lab modes; Navier-Stokes synthetic fixture only; not full convergence certification",
              "unit": "All repository tests; --case filters training cases only"},
              "environment": versions(), "source": source_snapshot(output), "runs": [], "passed": False}
    report_path = output / "report.json"

    def save_report():
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    try:
        if report["environment"]["packages"]["nvidia-physicsnemo"] != manifest["physicsnemo"]:
            raise RuntimeError(f"Install nvidia-physicsnemo=={manifest['physicsnemo']} in this interpreter before validation")
        import torch
        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
        if args.suite in ("unit", "all"):
            unit = execute([sys.executable, "-m", "pytest", "ETC/tests", "-q", "--tb=short",
                            f"--junitxml={output / 'unit.xml'}"], output, "unit", args.timeout)
            unit["suite"] = "unit"
            report["runs"].append(unit)
            save_report()
        for suite in ("smoke", "convergence"):
            if args.suite not in (suite, "all"):
                continue
            cases = selected if suite == "smoke" else [name for name in selected if name in CONVERGENCE_CASES]
            if not cases:
                raise ValueError(f"No eligible {suite} cases selected")
            steps = args.steps if suite == "smoke" else args.convergence_steps
            data_dir, configs, generation = None, {}, None
            if any(name.startswith("operators_") for name in cases):
                data_dir, configs, generation = prepare_operator_data(output, suite, args.seed, args.timeout)
                generation["suite"] = suite
                report["runs"].append(generation)
                save_report()
            for case_id in cases:
                case = runs[case_id]
                label = f"{suite}_{case_id}"
                destination = output / label
                command = [sys.executable, str(ROOT / case["script"]), *case.get("args", []),
                           "--device", args.device, "--steps", str(steps), "--seed", str(args.seed),
                           "--output-dir", str(destination)]
                if case_id == "navier_stokes" and suite == "convergence":
                    # A separate analytic regression, never the student lesson default.
                    command.append("--smoke-data")
                if case_id.startswith("operators_"):
                    command.extend(["--data-dir", str(data_dir), "--config", str(configs[case["level"]])])
                record = execute(command, output, label, args.timeout)
                record.update(case=case_id, suite=suite, steps=steps,
                              script_sha256=sha256(ROOT / case["script"]))
                if case_id == "navier_stokes":
                    record["data_scope"] = ("separate synthetic Taylor-Green regression" if suite == "convergence"
                                             else "original data_lat.npy student lesson; no future ground truth")
                if record["passed"]:
                    try:
                        artifacts = validate_artifacts(destination, steps=steps, device=args.device,
                                                       expected_version=manifest["physicsnemo"], seed=args.seed)
                        record["artifacts"] = artifacts
                        if suite == "convergence":
                            record["improvement"] = improvement(artifacts["metrics"], case_id, args.improvement_ratio)
                            record["passed"] = record["improvement"]["passed"]
                            if not record["passed"]:
                                record["error"] = "Held-out results did not meet the recorded improvement or lesson-accuracy criteria"
                    except Exception as error:
                        record.update(passed=False, artifact_error=f"{type(error).__name__}: {error}")
                report["runs"].append(record)
                save_report()
        report["passed"] = bool(report["runs"]) and all(run["passed"] for run in report["runs"])
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
    report["finished_at"] = now()
    save_report()
    print(json.dumps({"report": str(report_path), "passed": report["passed"],
                      "failed": [run["label"] for run in report["runs"] if not run["passed"]],
                      "error": report.get("error")}), flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
