# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""Small, explicit PyTorch training utilities shared by the four Labs."""
import argparse
import csv
import json
import math
from importlib.metadata import version
from pathlib import Path
import random

import numpy as np
import torch
from physicsnemo.models.mlp import FullyConnected
from physicsnemo.sym.eq.phy_informer import PhysicsInformer
from ETC.runtime.artifacts import staged_output


def parser(description, default_config=None):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--steps", type=int)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=Path, default=Path("outputs") / "labs")
    p.add_argument("--config", type=Path, default=default_config)
    return p


def setup(args, *, defaults=None):
    if args.output_dir.exists() or args.output_dir.is_symlink():
        raise FileExistsError(f"Output directory already exists: {args.output_dir}. Choose a new --output-dir; existing runs are never overwritten.")
    cfg = {"steps": 2000, "batch_size": 128, "learning_rate": 0.001,
           "layer_size": 64, "num_layers": 3}
    if defaults is not None:
        unknown = set(defaults) - set(cfg)
        if unknown:
            raise ValueError(f"Unknown Lab defaults: {sorted(unknown)}")
        cfg.update(defaults)
    if args.config is not None:
        import yaml
        supplied = yaml.safe_load(args.config.read_text(encoding="utf-8"))
        supplied = {} if supplied is None else supplied
        if not isinstance(supplied, dict):
            raise ValueError("Lab config must be a mapping of setting names to values")
        unknown = set(supplied) - set(cfg)
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        cfg.update(supplied)
    if args.steps is not None:
        cfg["steps"] = args.steps
    for key in ("steps", "batch_size", "layer_size", "num_layers"):
        if type(cfg[key]) is not int or cfg[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    rate = cfg["learning_rate"]
    if isinstance(rate, bool) or not isinstance(rate, (int, float)) or not math.isfinite(rate) or rate <= 0:
        raise ValueError("learning_rate must be a positive finite number")
    if type(args.seed) is not int or not 0 <= args.seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    if args.device not in ("auto", "cpu", "cuda"):
        raise ValueError("device must be auto, cpu or cuda")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; use --device cpu")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cpu":
        torch.set_num_threads(min(4, torch.get_num_threads()))
    return cfg, device


def mlp(nin, nout, cfg):
    return FullyConnected(in_features=nin, out_features=nout,
                          layer_size=cfg["layer_size"], num_layers=cfg["num_layers"],
                          activation_fn="tanh")


def derivative(value, coordinate):
    """Differentiable first derivative; supports constant analytical test fields."""
    if not value.requires_grad:
        return coordinate * 0.0
    result = torch.autograd.grad(value, coordinate, torch.ones_like(value),
                                 create_graph=True, retain_graph=True,
                                 allow_unused=True)[0]
    return coordinate * 0.0 if result is None else result


def informer(pde, device):
    return PhysicsInformer(required_outputs=list(pde.equations), equations=pde,
                           grad_method="autodiff", device=device)


def optimize(model, loss_fn, cfg):
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    history = []
    for step in range(1, cfg["steps"] + 1):
        optimizer.zero_grad(set_to_none=True)
        terms = loss_fn()
        if not terms or any(not torch.is_tensor(v) or v.ndim != 0 for v in terms.values()):
            raise ValueError("Each named loss term must be a scalar tensor")
        loss = sum(terms.values())
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite training loss at step {step}")
        if not loss.requires_grad:
            raise ValueError("The loss is disconnected from the model parameters")
        loss.backward()
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        if not gradients:
            raise ValueError("The loss is disconnected from the model parameters")
        if any(not torch.isfinite(grad).all() for grad in gradients):
            raise FloatingPointError(f"Nonfinite gradient at step {step}")
        optimizer.step()
        if any(not torch.isfinite(p).all() for p in model.parameters()):
            raise FloatingPointError(f"Nonfinite model parameter after step {step}")
        row = {"step": step, "loss": float(loss.detach()),
               **{name: float(v.detach()) for name, v in terms.items()}}
        history.append(row)
        if step == 1 or step == cfg["steps"] or step % 500 == 0:
            print(json.dumps(row), flush=True)
    return history


def save_run(args, cfg, model, history, arrays, metrics, plot=None):
    """Publish a complete run only after every artifact has been written."""
    if not history:
        raise ValueError("Cannot save a run without training history")
    arrays = {k: (v.detach().cpu().numpy() if torch.is_tensor(v) else np.asarray(v))
              for k, v in arrays.items()}
    if not arrays or not all(a.size and np.isfinite(a).all() for a in arrays.values()):
        raise FloatingPointError("Predictions must be nonempty and finite")
    state = model.state_dict()
    if any(torch.is_tensor(value) and not torch.isfinite(value).all() for value in state.values()):
        raise FloatingPointError("Cannot save nonfinite model parameters")
    summary = dict(metrics)
    summary.update({"steps": cfg["steps"], "seed": args.seed,
                    "device": str(next(model.parameters()).device),
                    "initial_loss": metrics["heldout_before"]["objective"],
                    "final_loss": metrics["heldout_after"]["objective"],
                    "loss_definition": "fixed_evaluation_objective_before_training_and_after_final_update",
                    "initial_minibatch_loss": history[0]["loss"],
                    "final_minibatch_loss_before_update": history[-1]["loss"],
                    "physicsnemo_version": version("nvidia-physicsnemo"), "torch_version": str(torch.__version__),
                    "status": "execution_complete_not_convergence_certified"})
    serialized_metrics = json.dumps(summary, indent=2, allow_nan=False) + "\n"
    with staged_output(args.output_dir) as destination:
        np.savez_compressed(destination / "predictions.npz", **arrays)
        torch.save({"model_state_dict": state, "config": cfg,
                    "seed": args.seed, "physicsnemo_version": version("nvidia-physicsnemo"),
                    "torch_version": str(torch.__version__), "steps": cfg["steps"],
                    "device": str(next(model.parameters()).device)}, destination / "model.pt")
        with (destination / "loss.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)
        if plot is not None:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig = plot(plt, arrays)
            try:
                fig.savefig(destination / "preview.png", dpi=130, bbox_inches="tight")
            finally:
                plt.close(fig)
        (destination / "metrics.json").write_text(serialized_metrics, encoding="utf-8")
    print(f"Saved artifacts to {args.output_dir.resolve()}", flush=True)
