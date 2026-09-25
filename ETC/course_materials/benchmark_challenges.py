"""Organizer-only PINN timing/accuracy checkpoints, without changing lesson files.

Use a separate checkout and output directory. Each measured checkpoint uses
the provided conditions, fixed independent samples, and (when available) the
analytical solution. Training continues with its original RNG state. Timings
exclude checkpoint evaluation; they are not predictions for other GPUs.
"""
from __future__ import annotations

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
    parser.add_argument("lesson", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoints", type=int, nargs="+", default=[200, 1000, 3000, 10000])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--cosine", action="store_true", help="Experimental schedule, final LR = 1%% of initial")
    parser.add_argument("--samples-scale", type=int, default=1)
    args = parser.parse_args()
    if not args.checkpoints or min(args.checkpoints) < 1 or args.samples_scale < 1:
        parser.error("Checkpoints must be positive")
    if args.output_dir.exists():
        parser.error("Use a new output directory")
    lesson = args.lesson.resolve()
    root = Path(__file__).resolve().parents[2]
    sources = [lesson, *sorted((root / "ETC/runtime").glob("*.py"))]
    source_hashes = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in sources}
    spec = importlib.util.spec_from_file_location("benchmark_lesson", lesson)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(lesson.parent))
    spec.loader.exec_module(module)
    state = {"step": 0, "paused": 0.0}
    original_model = module.create_model
    original_eval = module.create_evaluation_informer
    original_parse = module.parse_args
    original_adam = torch.optim.Adam
    rows = []

    def model(*values, **kwargs):
        state["model"] = original_model(*values, **kwargs)
        return state["model"]

    def evaluation(*values, **kwargs):
        state["informer"] = original_eval(*values, **kwargs)
        return state["informer"]

    def parse(*values, **kwargs):
        parsed, config = original_parse(*values, **kwargs)
        config["samples"] = {key: value * args.samples_scale for key, value in config["samples"].items()}
        state["config"] = config
        return parsed, config

    def synchronize():
        if args.device == "cuda":
            torch.cuda.synchronize()

    def post_step(optimizer, values, kwargs):
        state["step"] += 1
        step = state["step"]
        if step in args.checkpoints:
            synchronize()
            start = time.perf_counter()
            losses = module.heldout_losses(module.loss_terms, state["model"], state["informer"],
                                          state["config"], args.device, args.seed)
            row = {"step": step, "training_seconds": start - state["start"] - state["paused"],
                   "heldout_rmse": {key: value.detach().sqrt().item() for key, value in losses.items()}}
            if hasattr(module, "exact_reference"):
                xy, _ = module.evaluation_grid(args.device, module.TIME_END,
                                                circle=False)
                reference = module.exact_reference
                if hasattr(module, "DEFAULT_PHYSICS"):
                    params = {**module.DEFAULT_PHYSICS, **state["config"].get("physics", {})}
                    reference = lambda points, times: module.exact_reference(points, times, params)
                row["reference_over_time"] = module.reference_errors_over_time(
                    state["model"], xy, module.TIME_END, module.FIELD_NAMES, reference)
            synchronize()
            state["paused"] += time.perf_counter() - start
            rows.append(row)
            print("BENCHMARK " + json.dumps(row), flush=True)
        if "scheduler" in state:
            state["scheduler"].step()

    def optimizer(*values, **kwargs):
        result = original_adam(*values, **kwargs)
        if args.cosine:
            state["scheduler"] = torch.optim.lr_scheduler.CosineAnnealingLR(
                result, T_max=max(args.checkpoints), eta_min=kwargs["lr"] * .01)
        result.register_step_post_hook(post_step)
        synchronize()
        state["start"] = time.perf_counter()
        return result

    sys.argv = [str(lesson), "--reference", "--device", args.device, "--steps", str(max(args.checkpoints)),
                "--seed", str(args.seed), "--output-dir", str(args.output_dir)]
    if args.config:
        sys.argv += ["--config", str(args.config)]
    with patch.object(module, "create_model", model), patch.object(module, "create_evaluation_informer", evaluation), \
            patch.object(module, "parse_args", parse), patch.object(torch.optim, "Adam", optimizer):
        module.main()
    report = {"lesson": str(lesson), "seed": args.seed, "dtype": "float32",
              "device": torch.cuda.get_device_name() if args.device == "cuda" else "cpu",
              "schedule": "cosine_1_percent" if args.cosine else "constant",
              "config": state["config"], "checkpoints": rows, "source_sha256": source_hashes,
              "scope": "Fixed sampled residuals/conditions, not proof of convergence or future truth"}
    (args.output_dir / "benchmark.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
