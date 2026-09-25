"""Organizer experiment: fixed-point FP32 L-BFGS, never a student default by itself."""
from __future__ import annotations
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time

import torch
import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("lesson", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--samples-scale", type=int, default=4)
    parser.add_argument("--checkpoints", type=int, nargs="+", default=[200, 500, 1000])
    args = parser.parse_args()
    if args.output_dir.exists() or min(args.checkpoints) < 1 or args.samples_scale < 1:
        parser.error("Use positive settings and a fresh output directory")
    lesson = args.lesson.resolve()
    sys.path.insert(0, str(lesson.parent))
    spec = importlib.util.spec_from_file_location("benchmark_lesson", lesson)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    config = yaml.safe_load(args.config.read_text())
    config["samples"] = {key: value * args.samples_scale for key, value in config["samples"].items()}
    torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("A GPU is required for this timing experiment")
    def synchronize():
        if args.device == "cuda":
            torch.cuda.synchronize()
    cls = next(getattr(m, name) for name in ("WaveEquation2D", "NavierStokes2D", "ClimatePDE") if hasattr(m, name))
    params = {**getattr(m, "DEFAULT_PHYSICS", {}), **config.get("physics", {})}
    pde = cls(reference=True, **({"params": params} if params else {}))
    informer = m.create_informer(pde, args.device)
    model = m.create_model(2 if m.TIME_END is None else 3, len(m.FIELD_NAMES), config, args.device)
    optimizer = torch.optim.LBFGS(model.parameters(), lr=1., max_iter=1, max_eval=20,
                                 history_size=100, line_search_fn="strong_wolfe",
                                 tolerance_grad=1e-9, tolerance_change=1e-12)
    evaluations = 0

    def closure():
        nonlocal evaluations
        optimizer.zero_grad(set_to_none=True)
        # Freeze training collocation points; leave held-out samples independent.
        with torch.random.fork_rng(devices=[torch.cuda.current_device()] if args.device == "cuda" else []):
            torch.manual_seed(args.seed + 101)
            losses = m.loss_terms(model, informer, config, args.device)
        loss = sum(losses.values())
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite fixed-batch loss")
        loss.backward()
        evaluations += 1
        return loss

    rows = []
    synchronize()
    start = time.perf_counter()
    paused = 0.
    for step in range(1, max(args.checkpoints) + 1):
        optimizer.step(closure)
        if step in args.checkpoints:
            synchronize()
            checkpoint_start = time.perf_counter()
            losses = m.heldout_losses(m.loss_terms, model, informer, config, args.device, args.seed)
            row = {"calls": step, "closure_evaluations": evaluations,
                   "training_seconds": checkpoint_start - start - paused,
                   "heldout_rmse": {key: value.sqrt().item() for key, value in losses.items()}}
            if hasattr(m, "exact_reference"):
                xy, _ = m.evaluation_grid(args.device, m.TIME_END)
                reference = m.exact_reference if not params else lambda p, t: m.exact_reference(p, t, params)
                row["reference_over_time"] = m.reference_errors_over_time(model, xy, m.TIME_END, m.FIELD_NAMES, reference)
            synchronize()
            paused += time.perf_counter() - checkpoint_start
            rows.append(row)
            print("BENCHMARK " + json.dumps(row), flush=True)
    args.output_dir.mkdir(parents=True)
    report = {"lesson": str(lesson), "optimizer": "fixed_batch_lbfgs", "seed": args.seed,
              "dtype": "float32", "device": torch.cuda.get_device_name() if args.device == "cuda" else "cpu", "config": config,
              "checkpoints": rows, "scope": "optimizer experiment, not certification of convergence"}
    (args.output_dir / "benchmark.json").write_text(json.dumps(report, indent=2) + "\n")
    torch.save(model.state_dict(), args.output_dir / "model.pt")


if __name__ == "__main__":
    main()
