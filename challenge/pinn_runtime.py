"""Small CLI, sampling, derivative and output helpers for the PINN challenges.

PDEs, conditions and explicit optimization loops live in each Level script.
PhysicsNeMo 2.2.2 PhysicsInformer computes x/y spatial derivatives. Time
derivatives are supplied explicitly as field__t and field__t__t tensors.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import importlib.metadata
import json
import math
from pathlib import Path
import random

import numpy as np
import torch
from physicsnemo.models.mlp import FullyConnected
from physicsnemo.sym.eq.phy_informer import PhysicsInformer


def parse_args(script, description, config_name):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--reference", action="store_true", help="Use the complete reference PDE; default uses your student_equations function.")
    parser.add_argument("--config", type=Path, default=Path(script).parent / "conf" / config_name)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    import yaml
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        parser.error("config must contain a mapping")
    args.steps = args.steps if args.steps is not None else config["training"]["steps"]
    if args.steps <= 0:
        parser.error("--steps must be positive")
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable. Select the event GPU session or use --device cpu for a small smoke run.")
    for section in ("model", "samples"):
        for key, value in config[section].items():
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                parser.error(f"{section}.{key} must be a positive integer")
    if not math.isfinite(float(config["training"]["learning_rate"])) or config["training"]["learning_rate"] <= 0:
        parser.error("training.learning_rate must be positive and finite")
    args.output_dir = args.output_dir or Path(script).parent / "outputs" / (Path(script).stem + "-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f"))
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if args.device == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    else:
        torch.set_num_threads(min(4, torch.get_num_threads()))
    return args, config


def create_model(inputs, outputs, config, device):
    return FullyConnected(in_features=inputs, out_features=outputs,
                          layer_size=config["model"]["width"],
                          num_layers=config["model"]["layers"],
                          activation_fn="tanh").to(device)


def create_informer(pde, device):
    return PhysicsInformer(required_outputs=list(pde.equations), equations=pde,
                           grad_method="autodiff", device=device)


def gradient(value, variable):
    if not value.requires_grad:
        return variable * 0
    result = torch.autograd.grad(value, variable, torch.ones_like(value),
                                 create_graph=True, retain_graph=True,
                                 allow_unused=True)[0]
    return variable * 0 if result is None else result


def evaluate_fields(model, coordinates, time, names):
    inputs = coordinates if time is None else torch.cat((coordinates, time), dim=1)
    output = model(inputs)
    return {name: output[:, i:i + 1] for i, name in enumerate(names)}


def residuals(model, informer, coordinates, time, names):
    coordinates = coordinates.detach().requires_grad_(True)
    time = None if time is None else time.detach().requires_grad_(True)
    fields = evaluate_fields(model, coordinates, time, names)
    inputs = {**fields, "coordinates": coordinates,
              "x": coordinates[:, :1], "y": coordinates[:, 1:2]}
    if time is not None:
        inputs["t"] = time
        for name, value in fields.items():
            first = gradient(value, time)
            inputs[name + "__t"] = first
            inputs[name + "__t__t"] = gradient(first, time)
    return informer.forward(inputs)


def sample_square(count, device, boundary=False, length=math.pi):
    coordinates = torch.rand(count, 2, device=device) * length
    if boundary:
        side = torch.arange(count, device=device) % 4
        coordinates[side == 0, 0] = 0
        coordinates[side == 1, 0] = length
        coordinates[side == 2, 1] = 0
        coordinates[side == 3, 1] = length
    return coordinates


def sample_circle(count, device, boundary=False):
    theta = 2 * math.pi * torch.rand(count, 1, device=device)
    radius = torch.ones_like(theta) if boundary else torch.sqrt(torch.rand_like(theta))
    return torch.cat((radius * torch.cos(theta), radius * torch.sin(theta)), dim=1)


def sample_time(count, end, device):
    return torch.rand(count, 1, device=device) * end


def evaluation_grid(device, time_end, circle=False, fluid_blocks=None, side=24):
    if fluid_blocks is not None:
        axes = (torch.linspace(-2.5, 2.5, side, device=device), torch.linspace(-.5, .5, side, device=device))
    else:
        bound = (-1., 1.) if circle else (0., math.pi)
        axes = tuple(torch.linspace(*bound, side, device=device) for _ in range(2))
    coordinates = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).reshape(-1, 2)
    if circle:
        coordinates = coordinates[coordinates.square().sum(1) <= 1]
    if fluid_blocks is not None:
        keep = torch.ones(len(coordinates), dtype=torch.bool, device=device)
        for xmin, xmax, ymax in fluid_blocks:
            keep &= ~((coordinates[:, 0] >= xmin) & (coordinates[:, 0] <= xmax) & (coordinates[:, 1] <= ymax))
        coordinates = coordinates[keep]
    time = None if time_end is None else torch.full((len(coordinates), 1), time_end / 2, device=device)
    return coordinates, time


def save_results(args, config, model, history, coordinates, time, names, metrics, reference=None, notes=()):
    """Save only actual results, refusing to overwrite any existing directory."""
    destination = args.output_dir.resolve()
    if destination.exists():
        raise FileExistsError(f"Output directory already exists: {destination}. Choose a new --output-dir.")
    destination.mkdir(parents=True)
    model.eval()
    with torch.no_grad():
        fields = evaluate_fields(model, coordinates, time, names)
    prediction = np.concatenate([fields[name].cpu().numpy() for name in names], axis=1)
    if not np.isfinite(prediction).all():
        raise FloatingPointError("Non-finite final predictions; no successful result is claimed.")
    arrays = {"coordinates": coordinates.detach().cpu().numpy(), "prediction": prediction,
              "field_names": np.asarray(names)}
    if time is not None:
        arrays["t"] = time.detach().cpu().numpy()
    if reference is not None:
        arrays["reference"] = reference.detach().cpu().numpy()
        metrics["reference_rmse"] = float(np.sqrt(np.mean((prediction - arrays["reference"]) ** 2)))
    np.savez_compressed(destination / "predictions.npz", **arrays)
    torch.save({"state_dict": model.state_dict(), "config": config, "seed": args.seed,
                "field_names": names, "steps": args.steps, "device": args.device,
                "physicsnemo_version": importlib.metadata.version("nvidia-physicsnemo")}, destination / "model.pt")
    with (destination / "loss.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    metrics.update({"steps": args.steps, "seed": args.seed, "device": args.device,
                    "initial_loss": history[0]["total"], "final_loss": history[-1]["total"],
                    "loss_comparison": "identical held-out samples before/after training; training minibatches are separate CSV rows",
                    "torch_version": torch.__version__,
                    "physicsnemo_version": importlib.metadata.version("nvidia-physicsnemo"),
                    "reference_implementation": args.reference,
                    "exact_reference_available": reference is not None,
                    "evaluation_points": len(coordinates), "notes": list(notes),
                    "convergence_certified": False})
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        panels = 3 if reference is not None else len(names)
        fig, axes = plt.subplots(1, panels, figsize=(4 * panels, 3.5), squeeze=False, constrained_layout=True)
        if reference is not None:
            values = [arrays["reference"][:, 0], prediction[:, 0], np.abs(prediction[:, 0] - arrays["reference"][:, 0])]
            titles = ["Reference " + names[0], "Prediction " + names[0], "Absolute error"]
            # A shared scale exposes amplitude differences between solution and prediction.
            comparison_limits = {"vmin": min(float(values[0].min()), float(values[1].min())),
                                 "vmax": max(float(values[0].max()), float(values[1].max()))}
        else:
            values = [prediction[:, i] for i in range(len(names))]
            titles = ["Prediction " + name for name in names]
        for index, (ax, value, title) in enumerate(zip(axes.flat, values, titles)):
            limits = comparison_limits if reference is not None and index < 2 else {}
            image = ax.scatter(arrays["coordinates"][:, 0], arrays["coordinates"][:, 1], c=value,
                               s=10, cmap="viridis", **limits)
            ax.set(title=title, xlabel="x", ylabel="y", aspect="equal")
            fig.colorbar(image, ax=ax)
        fig.savefig(destination / "preview.png", dpi=130)
        plt.close(fig)
        metrics["preview_created"] = True
    except ImportError:
        metrics["preview_created"] = False
    (destination / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("Results:", destination)


def record_step(step, losses, history):
    total = sum(losses.values())
    if not torch.isfinite(total):
        raise FloatingPointError("Non-finite loss; no successful result is claimed.")
    row = {"step": step, "phase": "training_minibatch_before_update", "total": float(total.detach()), **{k: float(v.detach()) for k, v in losses.items()}}
    history.append(row)
    if step == 1 or step % 100 == 0:
        print(f"step {step}: loss={row['total']:.6g}")
    return total


def heldout_losses(loss_function, model, informer, config, device, seed):
    """Recreate identical held-out points without perturbing training RNG."""
    devices = [torch.cuda.current_device()] if device == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed + 100000)
        result = loss_function(model, informer, config, device)
    return {name: value.detach() for name, value in result.items()}


def heldout_row(step, phase, losses):
    return {"step": step, "phase": phase, "total": float(sum(losses.values())),
            **{name: float(value) for name, value in losses.items()}}


def reference_errors(model, coordinates, time, names, reference):
    if reference is None:
        return {}
    with torch.no_grad():
        fields = evaluate_fields(model, coordinates, time, names)
        prediction = torch.cat([fields[name] for name in names], dim=1)
        error = prediction - reference
        return {"rmse": float(error.square().mean().sqrt()),
                "relative_l2": float(torch.linalg.vector_norm(error) / torch.linalg.vector_norm(reference).clamp_min(1e-12))}
