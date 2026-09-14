"""Shared explicit PyTorch training for the three neural-operator exercises.

Supported teaching baseline: nvidia-physicsnemo==2.2.2. PDE and model imports
use the unified package. No legacy Sym training graph or Hydra runtime.
"""
from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import random
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
        if u.shape != f.shape or u.ndim != 4 or u.shape[1] != 1:
            raise ValueError("Physics residual expects matching [batch, 1, nx, ny] fields")
        values = [self.informer.forward({"u": u[index:index + 1], "f": f[index:index + 1]})[
                  "reaction_diffusion"] for index in range(u.shape[0])]
        return {"reaction_diffusion": torch.cat(values, dim=0)}


class UnfinishedExerciseError(RuntimeError):
    """A participant has not yet implemented a documented learning task."""


def reference_datasets(train_pairs, val_pairs, test_pairs):
    return tuple(TensorDataset(*pairs) for pairs in (train_pairs, val_pairs, test_pairs))


def reference_fno(model_config, grid_size):
    from physicsnemo.models.fno import FNO
    return FNO(in_channels=1, out_channels=1, dimension=2, **model_config)


def reference_afno(model_config, grid_size):
    from physicsnemo.models.afno import AFNO
    patch = model_config.get("patch_size", [4, 4])
    if any(grid_size % size for size in patch):
        raise ValueError("Grid size must be divisible by both AFNO patch dimensions; do not crop a periodic domain.")
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


def evaluate(model, loader, device, stats, physics=None):
    predictions, truths, forcings = [], [], []
    sym_mse_sum, fft_mse_sum, count = 0.0, 0.0, 0
    model.eval()
    with torch.no_grad():
        for normalized_f, normalized_u in loader:
            normalized_f, normalized_u = normalized_f.to(device), normalized_u.to(device)
            pred = model(normalized_f) * stats["u_std"] + stats["u_mean"]
            truth = normalized_u * stats["u_std"] + stats["u_mean"]
            forcing = normalized_f * stats["f_std"] + stats["f_mean"]
            fft_residual = spectral_residual(pred, forcing)
            fft_mse_sum += fft_residual.square().sum().item()
            if physics is not None:
                sym_residual = physics.forward({"u": pred, "f": forcing})["reaction_diffusion"]
                if not torch.allclose(sym_residual, fft_residual, rtol=3e-4, atol=3e-4):
                    raise RuntimeError("PhysicsInformer residual disagrees with independent FFT residual")
                sym_mse_sum += sym_residual.square().sum().item()
            count += truth.numel()
            predictions.append(pred.cpu())
            truths.append(truth.cpu())
            forcings.append(forcing.cpu())
    predicted, true, forcing = (torch.cat(items) for items in (predictions, truths, forcings))
    mse = (predicted - true).square().mean()
    metrics = {"rmse": mse.sqrt().item(),
               "relative_l2": ((predicted - true).norm() / true.norm().clamp_min(1e-12)).item(),
               "pde_rmse_fft": (fft_mse_sum / count) ** 0.5}
    if physics is not None:
        metrics["pde_rmse_physicsinformer"] = (sym_mse_sum / count) ** 0.5
    return metrics, {"f": forcing.numpy(), "u_true": true.numpy(), "u_pred": predicted.numpy()}


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
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"Choose a fresh --output-dir; previous results are preserved: {output}")
    with args.config.open() as handle:
        config = yaml.safe_load(handle)
    steps = args.steps if args.steps is not None else int(config["training"]["steps"])
    if steps < 1:
        parser.error("--steps must be positive")
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
    normalized = [((f - stats["f_mean"]) / stats["f_std"], (u - stats["u_mean"]) / stats["u_std"])
                  for f, u in pairs]
    datasets = dataset_builder(*normalized)
    batch_size = int(config["training"]["batch_size"])
    train_loader = DataLoader(datasets[0], batch_size=batch_size, shuffle=True,
                              generator=torch.Generator().manual_seed(args.seed))
    val_loader = DataLoader(datasets[1], batch_size=batch_size)
    test_loader = DataLoader(datasets[2], batch_size=batch_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["training"]["learning_rate"]))
    physics_weight = float(config["training"].get("physics_weight", 1.0))
    initial_test, _ = evaluate(model, test_loader, device, stats, physics)
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
        normalized_prediction = model(normalized_f)
        data_loss = (normalized_prediction - normalized_u).square().mean()
        physics_loss = torch.zeros((), device=device)
        if physics is not None:
            prediction = normalized_prediction * stats["u_std"] + stats["u_mean"]
            forcing = normalized_f * stats["f_std"] + stats["f_mean"]
            residual = physics.forward({"u": prediction, "f": forcing})["reaction_diffusion"]
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
    validation, _ = evaluate(model, val_loader, device, stats, physics)
    test, arrays = evaluate(model, test_loader, device, stats, physics)
    if any(not np.isfinite(values).all() for values in arrays.values()):
        raise FloatingPointError("Non-finite predictions")
    metrics = {"level": level, "method": config_names[level], "pde": "u-laplacian(u)=f",
               "physicsnemo_version": importlib.metadata.version("nvidia-physicsnemo"),
               "torch_version": str(torch.__version__), "device": str(device), "steps": steps, "seed": args.seed,
               "reference": args.reference, "status": "completed", "convergence_claim": False,
               "initial_loss": history[0]["loss"], "final_loss": history[-1]["loss"],
               "loss_measurement": "training minibatch loss before each optimizer step",
               "initial_test": initial_test, "test_relative_l2_before": initial_test["relative_l2"],
               "test_relative_l2_after": test["relative_l2"],
               "validation": validation, "test": test, "normalization_train_only": stats,
               "split_samples": dict(zip(("train", "val", "test"), map(len, datasets)))}
    # json allow_nan=False rejects non-finite scalar metrics as well.
    serialized = json.dumps(metrics, indent=2, allow_nan=False) + "\n"
    output.mkdir(parents=True)
    with (output / "loss.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    np.savez_compressed(output / "predictions.npz", **arrays)
    torch.save({"model_state_dict": model.state_dict(), "config": config,
                "normalization": stats, "metrics": metrics, "steps": steps, "seed": args.seed,
                "physicsnemo_version": metrics["physicsnemo_version"], "device": str(device)}, output / "model.pt")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        metrics["preview"] = "unavailable: install matplotlib"
        serialized = json.dumps(metrics, indent=2, allow_nan=False) + "\n"
    else:
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), constrained_layout=True)
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
        fig.savefig(output / "preview.png", dpi=140)
        plt.close(fig)
    (output / "metrics.json").write_text(serialized)
    print(serialized, flush=True)
    return metrics
