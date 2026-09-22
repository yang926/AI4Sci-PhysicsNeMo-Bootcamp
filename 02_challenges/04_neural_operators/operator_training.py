"""Shared explicit PyTorch training for the three neural-operator exercises.

Supported teaching baseline: nvidia-physicsnemo==2.2.2. PDE and model imports
use the unified package. No legacy Sym training graph or Hydra runtime.
"""
from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import math
import random
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

try:
    from .generate_data import SCHEMA, spectral_residual
except ImportError:
    from generate_data import SCHEMA, spectral_residual

MODULE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(MODULE_DIR.parents[1]))
from ETC.runtime.artifacts import staged_output


class BatchedPhysicsInformer:
    """Apply the scalar-field PhysicsInformer adapter independently per sample.

    PhysicsNeMo 2.2.2's spectral gradient adapter extracts u[0, 0]. Calling it
    on an entire batch can broadcast the first sample's derivatives. Separate
    calls preserve every field and its autograd graph; our FFT check catches
    any sample-mixing regression. This lesson has exactly one field channel.
    """
    def __init__(self, informer):
        self.informer = informer

    def forward(self, fields):
        u, f = fields["u"], fields["f"]
        if u.shape != f.shape or u.ndim != 4 or u.shape[1] != 1 or u.shape[0] == 0:
            raise ValueError("Physics residual expects matching [batch, 1, nx, ny] fields")
        values = [self.informer.forward({"u": u[index:index + 1], "f": f[index:index + 1]})[
                  "reaction_diffusion"] for index in range(u.shape[0])]
        return {"reaction_diffusion": torch.cat(values, dim=0)}


class UnfinishedExerciseError(RuntimeError):
    """A participant has not yet implemented a documented learning task."""


def reference_datasets(train_pairs, val_pairs, test_pairs):
    return tuple(TensorDataset(*pairs) for pairs in (train_pairs, val_pairs, test_pairs))


def checked_datasets(normalized_pairs, dataset_builder):
    """Check the dataset exercise without letting it define evaluation targets.

    The exercise is to wrap the provided splits, not to change their contents.
    Private copies keep accidental in-place edits out of the canonical data.
    Validation/test loaders are constructed here from canonical tensors, never
    from participant dataset objects. This is a local correctness check, not an
    isolation boundary against arbitrary Python code or edited source files.
    """
    exercise_pairs = [tuple(tensor.clone() for tensor in pair) for pair in normalized_pairs]
    datasets = dataset_builder(*exercise_pairs)
    if not isinstance(datasets, (tuple, list)) or len(datasets) != 3:
        raise ValueError("build_datasets must return train, validation and test TensorDataset objects")
    for name, dataset, expected in zip(("train", "val", "test"), datasets, normalized_pairs):
        if type(dataset) is not TensorDataset or len(dataset.tensors) != 2:
            raise ValueError(f"{name}: return TensorDataset(f, u), without a custom dataset wrapper")
        for field, actual, canonical in zip(("f", "u"), dataset.tensors, expected):
            if (actual.shape != canonical.shape or actual.dtype != canonical.dtype
                    or actual.device != canonical.device or not torch.equal(actual, canonical)):
                raise ValueError(
                    f"{name}.{field}: preserve the provided split, sample order and normalized values; "
                    "do not swap splits, replace targets or normalize again"
                )
    return (datasets[0], TensorDataset(*normalized_pairs[1]), TensorDataset(*normalized_pairs[2]))


def reference_fno(model_config, grid_size):
    from physicsnemo.models.fno import FNO
    return FNO(in_channels=1, out_channels=1, dimension=2, **model_config)


def reference_afno(model_config, grid_size):
    patch = model_config.get("patch_size", [4, 4])
    if (not isinstance(patch, (list, tuple)) or len(patch) != 2
            or any(isinstance(size, bool) or not isinstance(size, int) or size < 1 for size in patch)):
        raise ValueError("AFNO patch_size must contain two positive integers")
    if any(grid_size % size for size in patch):
        raise ValueError("Grid size must be divisible by both AFNO patch dimensions; do not crop a periodic domain.")
    from physicsnemo.models.afno import AFNO
    return AFNO(inp_shape=[grid_size, grid_size], in_channels=1, out_channels=1, **model_config)


def reference_physics(device):
    from sympy import Function, Symbol
    from physicsnemo.sym.eq.pde import PDE
    from physicsnemo.sym.eq.phy_informer import PhysicsInformer

    class ReactionDiffusion(PDE):
        def __init__(self):
            self.dim = 2
            x, y = Symbol("x"), Symbol("y")
            u, f = Function("u")(x, y), Function("f")(x, y)
            self.equations = {"reaction_diffusion": u - u.diff(x, 2) - u.diff(y, 2) - f}

    return BatchedPhysicsInformer(PhysicsInformer(
        required_outputs=["reaction_diffusion"], equations=ReactionDiffusion(),
        grad_method="spectral", bounds=[1.0, 1.0], device=str(device)))


def load_pair(path, expected_split):
    with h5py.File(path, "r") as hf:
        if hf.attrs.get("schema") != SCHEMA or hf.attrs.get("split") != expected_split:
            raise ValueError(f"{path}: wrong dataset schema or split. Run generate_data.py into a fresh directory.")
        f, u = torch.from_numpy(hf["f"][:]), torch.from_numpy(hf["u"][:])
        seed = int(hf.attrs["seed"])
    if f.shape != u.shape or f.ndim != 4 or f.shape[1] != 1 or f.shape[-2] != f.shape[-1]:
        raise ValueError(f"{path}: expected matching [samples, 1, n, n] tensors")
    if f.dtype != torch.float32 or u.dtype != torch.float32 or f.shape[-1] < 4:
        raise ValueError(f"{path}: expected float32 fields on a grid of at least 4 points per axis")
    if len(f) < 1 or not torch.isfinite(f).all() or not torch.isfinite(u).all():
        raise ValueError(f"{path}: empty or non-finite data")
    return (f, u), seed


def load_data(data_dir):
    pairs, seeds = [], []
    for split in ("train", "val", "test"):
        path = data_dir / f"{split}.hdf5"
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}. Run python generate_data.py --output-dir {data_dir} first.")
        pair, seed = load_pair(path, split)
        pairs.append(pair)
        seeds.append(seed)
    if len(set(seeds)) != 3 or len({pair[0].shape[-1] for pair in pairs}) != 1:
        raise ValueError("Train, validation and test must have independent seeds and matching grids")
    return pairs


def predict_field(model, normalized_forcing):
    prediction = model(normalized_forcing)
    if not isinstance(prediction, torch.Tensor) or prediction.shape != normalized_forcing.shape:
        raise ValueError("The operator must return a tensor with the same [batch, 1, n, n] shape as its input")
    if not torch.isfinite(prediction).all():
        raise FloatingPointError("Non-finite operator prediction")
    return prediction


def physics_residual(physics, prediction, forcing):
    residual = physics.forward({"u": prediction, "f": forcing})["reaction_diffusion"]
    if not isinstance(residual, torch.Tensor) or residual.shape != prediction.shape:
        raise ValueError("The physics residual must preserve every [batch, 1, n, n] field")
    if not torch.isfinite(residual).all():
        raise FloatingPointError("Non-finite physics residual")
    return residual


def evaluate(model, loader, device, stats, physics=None, *, collect_predictions=True):
    """Accumulate metrics by grid point, retaining fields only when exporting.

    Float64 sums avoid float32 overflow in squared errors. Weighting by point
    count (not batch count) also handles a smaller final minibatch correctly.
    """
    predictions, truths, forcings = [], [], []
    error_sum, truth_sum, sym_mse_sum, fft_mse_sum, count = 0.0, 0.0, 0.0, 0.0, 0
    model.eval()
    with torch.no_grad():
        for normalized_f, normalized_u in loader:
            normalized_f, normalized_u = normalized_f.to(device), normalized_u.to(device)
            pred = predict_field(model, normalized_f) * stats["u_std"] + stats["u_mean"]
            truth = normalized_u * stats["u_std"] + stats["u_mean"]
            forcing = normalized_f * stats["f_std"] + stats["f_mean"]
            fft_residual = spectral_residual(pred, forcing)
            error_sum += (pred.double() - truth.double()).square().sum().item()
            truth_sum += truth.double().square().sum().item()
            fft_mse_sum += fft_residual.double().square().sum().item()
            if physics is not None:
                sym_residual = physics_residual(physics, pred, forcing)
                if not torch.allclose(sym_residual, fft_residual, rtol=3e-4, atol=3e-4):
                    raise RuntimeError("PhysicsInformer residual disagrees with independent FFT residual")
                sym_mse_sum += sym_residual.double().square().sum().item()
            count += truth.numel()
            if collect_predictions:
                predictions.append(pred.cpu())
                truths.append(truth.cpu())
                forcings.append(forcing.cpu())
    if count == 0:
        raise ValueError("Cannot evaluate an empty dataset")
    metrics = {"rmse": (error_sum / count) ** 0.5,
               "relative_l2": error_sum ** 0.5 / max(truth_sum ** 0.5, 1e-12),
               "pde_rmse_fft": (fft_mse_sum / count) ** 0.5}
    if physics is not None:
        metrics["pde_rmse_physicsinformer"] = (sym_mse_sum / count) ** 0.5
    if not all(math.isfinite(value) for value in metrics.values()):
        raise FloatingPointError("Non-finite evaluation metric; check data and normalization")
    arrays = ({name: torch.cat(items).numpy() for name, items in
               (("f", forcings), ("u_true", truths), ("u_pred", predictions))}
              if collect_predictions else None)
    return metrics, arrays


def validate_config(config, steps):
    """Reject common YAML editing mistakes before constructing a model."""
    if not isinstance(config, dict) or any(not isinstance(config.get(key), dict)
                                           for key in ("data", "model", "training")):
        raise ValueError("Config must contain data, model and training mappings")
    training = config["training"]
    integers = {"steps": steps, "grid_size": config["data"].get("grid_size"),
                "batch_size": training.get("batch_size"), "cpu_threads": training.get("cpu_threads", 2)}
    for name, value in integers.items():
        minimum = 4 if name == "grid_size" else 1
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    for name, value in (("learning_rate", training.get("learning_rate")),
                        ("physics_weight", training.get("physics_weight", 1.0))):
        if (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
                or value < 0 or (name == "learning_rate" and value == 0)):
            raise ValueError(f"{name} must be finite and {'positive' if name == 'learning_rate' else 'nonnegative'}")


def run(level, model_builder, *, dataset_builder=reference_datasets, physics_builder=None,
        argv=None):
    parser = argparse.ArgumentParser(description=f"Level {level}: reaction-diffusion neural operator")
    config_names = {1: "FNO", 2: "AFNO", 3: "PINO"}
    parser.add_argument("--config", type=Path, default=MODULE_DIR / f"conf/config_{config_names[level]}.yaml")
    parser.add_argument("--data-dir", type=Path, default=MODULE_DIR / "datasets/Reaction_Diffusion")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=MODULE_DIR / f"outputs/level{level}")
    parser.add_argument("--reference", action="store_true", help="Use the instructor implementations of exercise functions")
    args = parser.parse_args(argv)
    output = args.output_dir.absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Choose a fresh --output-dir; previous results are preserved: {output}")
    with args.config.open() as handle:
        config = yaml.safe_load(handle)
    steps = args.steps if args.steps is not None else (
        config.get("training", {}).get("steps") if isinstance(config, dict)
        and isinstance(config.get("training"), dict) else None)
    validate_config(config, steps)
    if not 0 <= args.seed < 2 ** 32:
        parser.error("--seed must be between 0 and 2**32 - 1")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    # Small classroom grids benefit from a bounded thread count on shared CPUs.
    torch.set_num_threads(int(config["training"].get("cpu_threads", 2)))
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("--device cuda requested, but CUDA is unavailable. Use --device cpu.")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    if args.reference:
        model_builder = reference_afno if level == 2 else reference_fno
        dataset_builder = reference_datasets
        physics_builder = reference_physics if level == 3 else None
    # Resolve exercises before loading data so unfinished work yields an actionable error.
    grid_size = int(config["data"]["grid_size"])
    model = model_builder(config["model"], grid_size).to(device)
    physics = physics_builder(device) if physics_builder is not None else None
    pairs = load_data(args.data_dir.resolve())
    actual_grid_size = pairs[0][0].shape[-1]
    if actual_grid_size != grid_size:
        raise ValueError(f"Dataset grid {actual_grid_size} != config grid {grid_size}; use a matching --config.")
    train_f, train_u = pairs[0]
    stats = {"f_mean": train_f.mean().item(), "f_std": train_f.std(unbiased=False).clamp_min(1e-12).item(),
             "u_mean": train_u.mean().item(), "u_std": train_u.std(unbiased=False).clamp_min(1e-12).item()}
    if not all(math.isfinite(value) for value in stats.values()):
        raise FloatingPointError("Non-finite training normalization statistics")
    normalized = [((f - stats["f_mean"]) / stats["f_std"], (u - stats["u_mean"]) / stats["u_std"])
                  for f, u in pairs]
    datasets = checked_datasets(normalized, dataset_builder)
    # Dataset wrappers now own the required normalized tensors. Release raw
    # fields and the temporary canonical training copy before optimization.
    del pairs, normalized, train_f, train_u
    batch_size = int(config["training"]["batch_size"])
    train_loader = DataLoader(datasets[0], batch_size=batch_size, shuffle=True,
                              generator=torch.Generator().manual_seed(args.seed))
    val_loader = DataLoader(datasets[1], batch_size=batch_size)
    test_loader = DataLoader(datasets[2], batch_size=batch_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["training"]["learning_rate"]))
    physics_weight = float(config["training"].get("physics_weight", 1.0))
    initial_test, _ = evaluate(model, test_loader, device, stats, physics, collect_predictions=False)
    history, iterator = [], iter(train_loader)
    for step in range(1, steps + 1):
        model.train()
        try:
            normalized_f, normalized_u = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            normalized_f, normalized_u = next(iterator)
        normalized_f, normalized_u = normalized_f.to(device), normalized_u.to(device)
        optimizer.zero_grad(set_to_none=True)
        normalized_prediction = predict_field(model, normalized_f)
        data_loss = (normalized_prediction - normalized_u).square().mean()
        physics_loss = torch.zeros((), device=device)
        if physics is not None:
            prediction = normalized_prediction * stats["u_std"] + stats["u_mean"]
            forcing = normalized_f * stats["f_std"] + stats["f_mean"]
            residual = physics_residual(physics, prediction, forcing)
            physics_loss = (residual / stats["f_std"]).square().mean()
        loss = data_loss + physics_weight * physics_loss
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at step {step}")
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise FloatingPointError(f"Non-finite gradient at step {step}")
        optimizer.step()
        history.append({"step": step, "loss": loss.item(), "data_loss": data_loss.item(),
                        "physics_loss": physics_loss.item()})
        if step == 1 or step == steps or step % 100 == 0:
            print(json.dumps(history[-1]), flush=True)
    validation, _ = evaluate(model, val_loader, device, stats, physics, collect_predictions=False)
    test, arrays = evaluate(model, test_loader, device, stats, physics)
    if any(not np.isfinite(values).all() for values in arrays.values()):
        raise FloatingPointError("Non-finite predictions")
    metrics = {"level": level, "method": config_names[level], "pde": "u-laplacian(u)=f",
               "physicsnemo_version": importlib.metadata.version("nvidia-physicsnemo"),
               "torch_version": str(torch.__version__), "device": str(device), "steps": steps, "seed": args.seed,
               "reference": args.reference, "status": "completed", "convergence_claim": False,
               "assessment": {"kind": "local_practice_feedback", "official_score": None,
                              "ranking_ready": False},
               "evaluation_equations": "provided_reference_equations",
               "evaluation_data": "canonical_local_validation_and_test_splits",
               "dataset_exercise_checked": True,
               "evaluation_protocol": "reaction_diffusion_v2_canonical_splits",
               "initial_loss": history[0]["loss"], "final_loss": history[-1]["loss"],
               "loss_measurement": "training minibatch loss before each optimizer step",
               "initial_test": initial_test, "test_relative_l2_before": initial_test["relative_l2"],
               "test_relative_l2_after": test["relative_l2"],
               "validation": validation, "test": test, "normalization_train_only": stats,
               "split_samples": dict(zip(("train", "val", "test"), map(len, datasets)))}
    # json allow_nan=False rejects non-finite scalar metrics as well.
    serialized = json.dumps(metrics, indent=2, allow_nan=False) + "\n"
    with staged_output(output) as destination:
        with (destination / "loss.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=history[0].keys())
            writer.writeheader()
            writer.writerows(history)
        np.savez_compressed(destination / "predictions.npz", **arrays)
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            metrics["preview"] = "unavailable: install matplotlib"
            serialized = json.dumps(metrics, indent=2, allow_nan=False) + "\n"
        else:
            fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), constrained_layout=True)
            try:
                fields = [arrays["u_true"][0, 0], arrays["u_pred"][0, 0],
                          arrays["u_pred"][0, 0] - arrays["u_true"][0, 0]]
                scale = max(np.abs(fields[0]).max(), np.abs(fields[1]).max(), 1e-12)
                for ax, values, title in zip(axes, fields, ("Exact u", "Predicted u", "Prediction error")):
                    im = ax.imshow(values.T, origin="lower", extent=(0, 1, 0, 1), cmap="coolwarm",
                                   vmin=-scale if title != "Prediction error" else None,
                                   vmax=scale if title != "Prediction error" else None)
                    ax.set(title=title, xlabel="x", ylabel="y")
                    fig.colorbar(im, ax=ax)
                fig.suptitle(f"{config_names[level]}: {steps} training steps (convergence not certified)")
                fig.savefig(destination / "preview.png", dpi=140)
            finally:
                plt.close(fig)
        torch.save({"model_state_dict": model.state_dict(), "config": config,
                    "normalization": stats, "metrics": metrics, "steps": steps, "seed": args.seed,
                    "physicsnemo_version": metrics["physicsnemo_version"], "device": str(device)},
                   destination / "model.pt")
        (destination / "metrics.json").write_text(serialized)
    print(serialized, flush=True)
    return metrics
