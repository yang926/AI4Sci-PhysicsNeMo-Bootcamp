"""Organizer-only full-dataset operator timing checkpoints; no lesson edits.

Budget selection uses validation data only. The unchanged lesson runner still
reports the independent test split after its final optimizer update.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--level", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoints", type=int, nargs="+", default=[200, 1000, 3000])
    parser.add_argument("--config", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    wall_start = time.perf_counter()
    if not args.checkpoints or min(args.checkpoints) < 1:
        parser.error("Checkpoints must be positive")
    checkpoints_dir = args.output_dir.with_name(args.output_dir.name + "-checkpoints")
    if args.output_dir.exists() or checkpoints_dir.exists():
        parser.error("Use fresh output and checkpoint directories")
    source = Path(__file__).resolve().parents[2] / "02_challenges/04_neural_operators/operator_training.py"
    sys.path.insert(0, str(source.parent))
    spec = importlib.util.spec_from_file_location("operator_benchmark_training", source)
    lesson = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lesson)
    checkpoints_dir.mkdir(parents=True)
    state = {"step": 0, "paused": 0., "loaders": [], "rows": []}
    original_evaluate, original_loader, original_adam = lesson.evaluate, lesson.DataLoader, torch.optim.Adam

    def synchronize():
        if args.device == "cuda":
            torch.cuda.synchronize()

    def loader(*values, **kwargs):
        result = original_loader(*values, **kwargs)
        state["loaders"].append(result)
        return result

    def evaluate(model, loader, device, stats, physics=None, **kwargs):
        result = original_evaluate(model, loader, device, stats, physics, **kwargs)
        if "start" not in state:
            state.update(model=model, device=device, stats=stats, physics=physics)
            synchronize()
            state["start"] = time.perf_counter()
        return result

    def post_step(optimizer, values, kwargs):
        state["step"] += 1
        step = state["step"]
        if step not in args.checkpoints:
            return
        synchronize()
        started = time.perf_counter()
        cpu_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all() if args.device == "cuda" else None
        validation, _ = original_evaluate(state["model"], state["loaders"][1], state["device"],
                                          state["stats"], state["physics"], collect_predictions=False)
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
        row = {"step": step, "training_seconds": started - state["start"] - state["paused"],
               "validation": validation,
               "validation_pde_rmse_over_training_f_std": validation["pde_rmse_fft"] / state["stats"]["f_std"]}
        torch.save({"model_state_dict": state["model"].state_dict(), "step": step,
                    "normalization": state["stats"], "seed": args.seed},
                   checkpoints_dir / f"step-{step}.pt")
        synchronize()
        state["paused"] += time.perf_counter() - started
        state["rows"].append(row)
        print("BENCHMARK " + json.dumps(row), flush=True)
        (checkpoints_dir / "progress.json").write_text(json.dumps(state["rows"], indent=2) + "\n")

    def optimizer(*values, **kwargs):
        result = original_adam(*values, **kwargs)
        result.register_step_post_hook(post_step)
        return result

    argv = ["--reference", "--device", args.device, "--seed", str(args.seed),
            "--data-dir", str(args.data_dir), "--output-dir", str(args.output_dir),
            "--steps", str(max(args.checkpoints))]
    if args.config:
        argv += ["--config", str(args.config)]
    with patch.object(lesson, "DataLoader", loader), patch.object(lesson, "evaluate", evaluate), \
            patch.object(torch.optim, "Adam", optimizer):
        metrics = lesson.run(args.level, lesson.reference_fno, argv=argv)
    report = {"level": args.level, "seed": args.seed, "dtype": "float32",
              "gpu": torch.cuda.get_device_name() if args.device == "cuda" else "cpu",
              "training_settings": "unchanged lesson Adam schedule and model configuration",
              "split_samples": metrics["split_samples"], "checkpoints": state["rows"],
              "independent_test_at_final_step": metrics["test"],
              "test_pde_rmse_over_training_f_std": metrics["test"]["pde_rmse_fft"] / metrics["normalization_train_only"]["f_std"],
              "wall_seconds_including_intermediate_checks": time.perf_counter() - wall_start,
              "timing_scope": "training only, excluding data preparation and intermediate validation/serialization"}
    report["config"] = torch.load(args.output_dir / "model.pt", map_location="cpu", weights_only=True)["config"]
    report["dataset_manifest"] = json.loads((args.data_dir / "manifest.json").read_text())
    report["operator_training_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    (args.output_dir / "benchmark.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
