#!/usr/bin/env python3
"""Run reproducible PhysicsNeMo course validation without downloads or remote access.

unit: pytest tests; smoke: all 18 actual lesson runs and artifact checks;
convergence: a separate analytical subset with required held-out improvement.
A smoke pass never certifies convergence. Even the improvement suite establishes
only its recorded metric criterion, not full physical or numerical convergence.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
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
CONVERGENCE_CASES = ("projectile", "diffusion", "wave_l1", "operators_l1", "operators_l2", "operators_l3")
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
    tensor_count = 0
    for name, tensor in state.items():
        if torch.is_tensor(tensor):
            tensor_count += 1
            if not torch.isfinite(tensor).all():
                raise ValueError(f"Non-finite checkpoint tensor: {name}")
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


def improvement(metrics, case_id, ratio):
    if case_id.startswith("operators_"):
        before, after = metrics["test_relative_l2_before"], metrics["test_relative_l2_after"]
        metric = "test_relative_l2"
    elif case_id == "wave_l1":
        before = metrics["initial_reference_error"]["relative_l2"]
        after = metrics["final_reference_error"]["relative_l2"]
        metric = "analytical_reference_relative_l2"
    else:
        before = metrics["heldout_before"]["solution_rmse"]
        after = metrics["heldout_after"]["solution_rmse"]
        metric = "heldout_solution_rmse"
    if not all(isinstance(value, (float, int)) and math.isfinite(value) for value in (before, after)):
        raise ValueError("Held-out improvement metrics must be finite numbers")
    passed = before > 0 and after >= 0 and after <= ratio * before
    return {"metric": metric, "before": before, "after": after, "required_after_before_ratio": ratio,
            "actual_after_before_ratio": after / before if before > 0 else None,
            "passed": passed, "scope": "Held-out improvement only; full convergence is not certified."}


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
    parser.add_argument("--convergence-steps", type=int, default=500, help="Separate longer improvement runs")
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
              "convergence": "Separate held-out improvement criterion; not full convergence certification",
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
                if case_id.startswith("operators_"):
                    command.extend(["--data-dir", str(data_dir), "--config", str(configs[case["level"]])])
                record = execute(command, output, label, args.timeout)
                record.update(case=case_id, suite=suite, steps=steps,
                              script_sha256=sha256(ROOT / case["script"]))
                if record["passed"]:
                    try:
                        artifacts = validate_artifacts(destination, steps=steps, device=args.device,
                                                       expected_version=manifest["physicsnemo"], seed=args.seed)
                        record["artifacts"] = artifacts
                        if suite == "convergence":
                            record["improvement"] = improvement(artifacts["metrics"], case_id, args.improvement_ratio)
                            record["passed"] = record["improvement"]["passed"]
                            if not record["passed"]:
                                record["error"] = "Held-out metric did not meet the recorded improvement criterion"
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
