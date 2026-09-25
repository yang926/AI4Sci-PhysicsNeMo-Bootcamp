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
from ETC.runtime.artifacts import staged_output


def parse_args(script, description, config_name):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--reference", action="store_true", help="Use all instructor exercise functions; default uses your equations, conditions, and level-specific geometry/parameters.")
    parser.add_argument("--config", type=Path, default=Path(script).parent / "conf" / config_name)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    import yaml
    try:
        config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
        args.steps = validate_config(config, args.steps)
    except (OSError, yaml.YAMLError, ValueError) as exc:
        parser.error(str(exc))
    if not 0 <= args.seed < 2 ** 32:
        parser.error("--seed must be an integer between 0 and 4294967295")
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable. Select the event GPU session or use --device cpu for a small smoke run.")
    args.output_dir = args.output_dir or Path(script).parent / "outputs" / (Path(script).stem + "-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f"))
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if args.device == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    else:
        torch.set_num_threads(min(4, torch.get_num_threads()))
    return args, config


def validate_config(config, steps=None):
    """Validate the small shared config before constructing a model or optimizer."""
    if not isinstance(config, dict):
        raise ValueError("config must contain a mapping")
    required = {"training": ("learning_rate",), "model": ("width", "layers"),
                "samples": ("interior", "boundary", "initial")}
    for section, keys in required.items():
        if not isinstance(config.get(section), dict):
            raise ValueError(f"{section} must contain a mapping")
        for key in keys:
            if key not in config[section]:
                raise ValueError(f"Missing config value: {section}.{key}")
    steps = config["training"].get("steps") if steps is None else steps
    if not isinstance(steps, int) or isinstance(steps, bool) or steps <= 0:
        raise ValueError("training.steps / --steps must be a positive integer")
    for section in ("model", "samples"):
        for key, value in config[section].items():
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{section}.{key} must be a positive integer")
    rate = config["training"]["learning_rate"]
    if (not isinstance(rate, (int, float)) or isinstance(rate, bool)
            or not math.isfinite(rate) or rate <= 0):
        raise ValueError("training.learning_rate must be a positive finite number")
    if "physics" in config and not isinstance(config["physics"], dict):
        raise ValueError("physics must contain a mapping")
    return steps


def create_model(inputs, outputs, config, device):
    return FullyConnected(in_features=inputs, out_features=outputs,
                          layer_size=config["model"]["width"],
                          num_layers=config["model"]["layers"],
                          activation_fn="tanh").to(device)


def create_informer(pde, device):
    if (not isinstance(pde.equations, dict) or not pde.equations
            or any(not isinstance(name, str) or not name for name in pde.equations)):
        raise ValueError("Equations must be a nonempty mapping of residual names to SymPy expressions")
    # residuals() supplies first/second time derivatives explicitly. Register
    # only those actually used, retaining errors for absent tensors instead of
    # hiding PhysicsInformer warnings globally.
    from sympy import Derivative
    from ETC.runtime.labs import informer
    supplied = set()
    for expression in pde.equations.values():
        for derivative in getattr(expression, "atoms", lambda *_: set())(Derivative):
            if all(str(variable) == "t" for variable in derivative.variables) and len(derivative.variables) <= 2:
                supplied.add(derivative.expr.func.__name__ + "__t" * len(derivative.variables))
    return informer(pde, device, supplied_derivatives=supplied)


def create_evaluation_informer(pde_type, device, **parameters):
    """Evaluate the stated PDE independently of the learner's equation builder.

    This is local feedback, not a trusted competition service: reference code
    and configuration remain editable in the learner's checkout.
    """
    if "reference" in parameters:
        raise ValueError("Evaluation always uses the provided reference equations")
    return create_informer(pde_type(reference=True, **parameters), device)


def gradient(value, variable):
    if not value.requires_grad:
        return variable * 0
    result = torch.autograd.grad(value, variable, torch.ones_like(value),
                                 create_graph=True, retain_graph=True,
                                 allow_unused=True)[0]
    return variable * 0 if result is None else result


def evaluate_fields(model, coordinates, time, names):
    """Keep the field contract explicit; never broadcast or truncate outputs."""
    if coordinates.ndim != 2 or coordinates.shape[1] != 2 or not len(coordinates):
        raise ValueError("coordinates must have nonempty shape [points, 2]")
    if not names or any(not isinstance(name, str) or not name for name in names) or len(set(names)) != len(names):
        raise ValueError("Field names must be nonempty, unique strings")
    if time is not None and time.shape != (len(coordinates), 1):
        raise ValueError("time must have shape [points, 1]")
    inputs = coordinates if time is None else torch.cat((coordinates, time), dim=1)
    output = model(inputs)
    expected = (len(coordinates), len(names))
    if not isinstance(output, torch.Tensor) or output.shape != expected:
        raise ValueError(f"Model output must have shape {expected}, one column per field {list(names)}")
    if not output.is_floating_point() or output.device != coordinates.device:
        raise ValueError("Model output must be floating point on the coordinates' device")
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


def save_results(args, config, model, history, coordinates, time, names, metrics, reference=None, notes=(), extra_artifacts=None):
    """Publish complete results only, preserving existing results on any failure.

    All serialization and plotting finish in a temporary sibling directory.
    The final directory is reserved exclusively before moving the artifacts;
    a failed publication cleans up only artifacts created by this call.
    """
    destination = args.output_dir.absolute()
    if not history or any(set(row) != set(history[0]) for row in history):
        raise ValueError("Loss history must contain rows with identical columns")
    if not {"step", "phase", "total"} <= set(history[0]):
        raise ValueError("Loss history must include step, phase, and total")
    for row in history:
        if any(not math.isfinite(float(value)) for key, value in row.items() if key != "phase"):
            raise FloatingPointError("Loss history contains non-finite values")
    was_training = model.training
    try:
        model.eval()
        with staged_output(destination) as staging:
            _write_results(staging, args, config, model, history, coordinates, time,
                           names, dict(metrics), reference, notes)
            if extra_artifacts is not None:
                extra_artifacts(staging)
    finally:
        model.train(was_training)
    print("Results:", destination)


def _write_results(destination, args, config, model, history, coordinates, time,
                   names, metrics, reference, notes):
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
        _check_reference(torch.from_numpy(prediction), reference.detach().cpu())
        arrays["reference"] = reference.detach().cpu().numpy()
        metrics["reference_rmse"] = float(np.sqrt(np.mean((prediction.astype(np.float64) - arrays["reference"]) ** 2)))
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
                    "assessment": {"kind": "local_practice_feedback",
                                   "official_score": None, "ranking_ready": False},
                    "convergence_certified": False})
    if (history[0]["phase"] == "heldout_before_training"
            and history[-1]["phase"] == "heldout_after_training"):
        for name, row in (("heldout_before", history[0]), ("heldout_after", history[-1])):
            metrics[name] = {
                "objective": row["total"],
                "loss_components": {key: value for key, value in row.items()
                                    if key not in ("step", "phase", "total")},
            }
        metrics["metric_definition"] = (
            "heldout_before/heldout_after use identical fixed samples and canonical equations; "
            "loss_components are weighted training-objective terms, not unweighted RMSE. "
            "objective is the sum of those terms."
        )
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        rows, columns = (len(names), 3) if reference is not None else (1, len(names))
        fig, axes = plt.subplots(rows, columns, figsize=(4 * columns, 3.5 * rows),
                                 squeeze=False, constrained_layout=True)
        values, titles, limits = [], [], []
        if reference is not None:
            for i, name in enumerate(names):
                truth, estimate = arrays["reference"][:, i], prediction[:, i]
                values.extend((truth, estimate, np.abs(estimate - truth)))
                titles.extend(("Reference " + name, "Prediction " + name, "Absolute error " + name))
                # A shared scale exposes amplitude differences for each field.
                shared = {"vmin": min(float(truth.min()), float(estimate.min())),
                          "vmax": max(float(truth.max()), float(estimate.max()))}
                limits.extend((shared, shared, {}))
        else:
            values = [prediction[:, i] for i in range(len(names))]
            titles = ["Prediction " + name for name in names]
            limits = [{} for _ in names]
        try:
            for ax, value, title, scale in zip(axes.flat, values, titles, limits):
                image = ax.scatter(arrays["coordinates"][:, 0], arrays["coordinates"][:, 1], c=value,
                                   s=10, cmap="viridis", **scale)
                ax.set(title=title, xlabel="x", ylabel="y", aspect="equal")
                fig.colorbar(image, ax=ax)
            fig.savefig(destination / "preview.png", dpi=130)
        finally:
            plt.close(fig)
        metrics["preview_created"] = True
        metrics["preview_fields"] = list(names)
    except ImportError:
        metrics["preview_created"] = False
    (destination / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def record_step(step, losses, history):
    total = checked_total_loss(losses)
    row = {"step": step, "phase": "training_minibatch_before_update", "total": float(total.detach()), **{k: float(v.detach()) for k, v in losses.items()}}
    history.append(row)
    if step == 1 or step % 100 == 0:
        print(f"step {step}: loss={row['total']:.6g}")
    return total


def checked_total_loss(losses):
    """Losses are named scalar tensors; malformed student results fail clearly."""
    if not isinstance(losses, dict) or not losses:
        raise ValueError("Losses must be a nonempty mapping of scalar tensors")
    for name, value in losses.items():
        if not isinstance(value, torch.Tensor) or value.ndim != 0:
            raise ValueError(f"Loss {name!r} must be a scalar tensor; reduce residuals with square().mean()")
    values = torch.stack(list(losses.values()))
    total = values.sum()
    if not torch.isfinite(values).all() or not torch.isfinite(total):
        raise FloatingPointError("Non-finite loss; no successful result is claimed.")
    return total


def heldout_losses(loss_function, model, informer, config, device, seed):
    """Recreate identical held-out points without perturbing training RNG."""
    device_spec = torch.device(device)
    devices = [device_spec.index if device_spec.index is not None else torch.cuda.current_device()] if device_spec.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        # torch.manual_seed also seeds every CUDA device, outside a CPU-only
        # fork. Seed only the generators whose states this context restores.
        torch.random.default_generator.manual_seed(seed + 100000)
        if devices:
            with torch.cuda.device(devices[0]):
                torch.cuda.manual_seed(seed + 100000)
        result = loss_function(model, informer, config, device)
    checked_total_loss(result)
    return {name: value.detach() for name, value in result.items()}


def heldout_row(step, phase, losses):
    return {"step": step, "phase": phase, "total": float(checked_total_loss(losses)),
            **{name: float(value) for name, value in losses.items()}}


def _check_reference(prediction, reference):
    if not isinstance(reference, torch.Tensor) or reference.shape != prediction.shape:
        raise ValueError("Reference and prediction shapes must match")
    if not torch.isfinite(prediction).all() or not torch.isfinite(reference).all():
        raise ValueError("Reference evaluation requires finite fields")


def reference_errors(model, coordinates, time, names, reference):
    if reference is None:
        return {}
    with torch.no_grad():
        fields = evaluate_fields(model, coordinates, time, names)
        prediction = torch.cat([fields[name] for name in names], dim=1)
        _check_reference(prediction, reference)
        reference = reference.double()
        error = prediction.double() - reference
        return {"rmse": float(error.square().mean().sqrt()),
                "relative_l2": float(torch.linalg.vector_norm(error) / torch.linalg.vector_norm(reference).clamp_min(1e-12))}


def reference_errors_over_time(model, coordinates, time_end, names, reference_function,
                               time_fractions=(0., .25, .5, .75, 1.)):
    """Compare a solution at several times, not just the preview's middle slice.

    Aggregate relative L2 uses all reference values together, so a nearly
    decayed final temperature does not become the entire benchmark. An empty
    mapping means the problem has no analytical reference, not zero error.
    """
    if not math.isfinite(time_end) or time_end <= 0:
        raise ValueError("time_end must be positive and finite")
    if not time_fractions or any(not math.isfinite(t) or not 0 <= t <= 1
                                 for t in time_fractions):
        raise ValueError("time_fractions must contain finite values in [0, 1]")
    errors, references, per_time, availability = [], [], [], []
    with torch.no_grad():
        for fraction in time_fractions:
            time = coordinates.new_full((len(coordinates), 1), time_end * fraction)
            reference = reference_function(coordinates, time)
            availability.append(reference is not None)
            if reference is None:
                continue
            fields = evaluate_fields(model, coordinates, time, names)
            prediction = torch.cat([fields[name] for name in names], dim=1)
            _check_reference(prediction, reference)
            reference = reference.double()
            error = prediction.double() - reference
            errors.append(error)
            references.append(reference)
            per_time.append({"time": time_end * fraction,
                             "rmse": float(error.square().mean().sqrt()),
                             "relative_l2": float(error.norm() / reference.norm().clamp_min(1e-12))})
    if not any(availability):
        return {}
    if not all(availability):
        raise ValueError("Reference availability must not change across times")
    error = torch.cat(errors)
    reference = torch.cat(references)
    return {"rmse": float(error.square().mean().sqrt()),
            "relative_l2": float(error.norm() / reference.norm().clamp_min(1e-12)),
            "per_field": {name: {
                "rmse": float(error[:, i].square().mean().sqrt()),
                "relative_l2": float(error[:, i].norm() / reference[:, i].norm().clamp_min(1e-12))
            } for i, name in enumerate(names)},
            "spatial_points": len(coordinates), "per_time": per_time,
            "scope": "fixed spatial grid at multiple times; not a full-domain error bound"}
