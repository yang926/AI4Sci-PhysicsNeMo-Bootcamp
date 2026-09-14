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

"""Small, explicit PyTorch training utilities shared by the four tutorial labs."""
import argparse
import csv
import json
from importlib.metadata import version
from pathlib import Path
import random

import numpy as np
import torch
from physicsnemo.models.mlp import FullyConnected
from physicsnemo.sym.eq.phy_informer import PhysicsInformer


def parser(description, default_config=None):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--steps", type=int)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=Path, default=Path("outputs") / "tutorial")
    p.add_argument("--config", type=Path, default=default_config)
    return p


def setup(args):
    if args.output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {args.output_dir}. Choose a new --output-dir; existing runs are never overwritten.")
    cfg = {"steps": 2000, "batch_size": 128, "learning_rate": 0.001,
           "layer_size": 64, "num_layers": 3}
    if args.config is not None:
        import yaml
        supplied = yaml.safe_load(args.config.read_text()) or {}
        unknown = set(supplied) - set(cfg)
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        cfg.update(supplied)
    if args.steps is not None:
        cfg["steps"] = args.steps
    if cfg["steps"] < 1 or cfg["batch_size"] < 1:
        raise ValueError("steps and batch_size must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; use --device cpu")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cpu":
        torch.set_num_threads(min(4, torch.get_num_threads()))
    args.output_dir.mkdir(parents=True, exist_ok=False)
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
        loss = sum(terms.values())
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite training loss at step {step}")
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all()
               for p in model.parameters()):
            raise FloatingPointError(f"Nonfinite gradient at step {step}")
        optimizer.step()
        row = {"step": step, "loss": float(loss.detach()),
               **{name: float(v.detach()) for name, v in terms.items()}}
        history.append(row)
        if step == 1 or step == cfg["steps"] or step % 500 == 0:
            print(json.dumps(row), flush=True)
    return history


def save_run(args, cfg, model, history, arrays, metrics, plot=None):
    arrays = {k: (v.detach().cpu().numpy() if torch.is_tensor(v) else np.asarray(v))
              for k, v in arrays.items()}
    if not all(np.isfinite(a).all() for a in arrays.values()):
        raise FloatingPointError("Nonfinite predictions")
    np.savez_compressed(args.output_dir / "predictions.npz", **arrays)
    torch.save({"model_state_dict": model.state_dict(), "config": cfg,
                "seed": args.seed, "physicsnemo_version": version("nvidia-physicsnemo"),
                "torch_version": str(torch.__version__), "steps": cfg["steps"],
                "device": str(next(model.parameters()).device)},
               args.output_dir / "model.pt")
    with (args.output_dir / "loss.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    metrics.update({"steps": cfg["steps"], "seed": args.seed,
                    "device": str(next(model.parameters()).device),
                    "initial_loss": metrics["heldout_before"]["objective"],
                    "final_loss": metrics["heldout_after"]["objective"],
                    "loss_definition": "fixed_evaluation_objective_before_training_and_after_final_update",
                    "initial_minibatch_loss": history[0]["loss"],
                    "final_minibatch_loss_before_update": history[-1]["loss"],
                    "physicsnemo_version": version("nvidia-physicsnemo"), "torch_version": str(torch.__version__),
                    "status": "execution_complete_not_convergence_certified"})
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False))
    if plot is not None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig = plot(plt, arrays)
        fig.savefig(args.output_dir / "preview.png", dpi=130, bbox_inches="tight")
        plt.close(fig)
    print(f"Saved artifacts to {args.output_dir.resolve()}", flush=True)
