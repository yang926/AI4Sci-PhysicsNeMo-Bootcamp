# Adapted from OpenHackathons AI-Powered-Physics-Bootcamp, commit 9cae27f.
# Original attribution and license statements are retained at repository root.
"""Completed, bounded wave PINN exercise for the PhysicsNeMo 25.11 environment.

Training time and convergence must be measured on the event GPU before release.
"""

import importlib.metadata
import json
from pathlib import Path
import re
import time

import numpy as np
import torch
from sympy import Function, Symbol, sin

import physicsnemo.sym
from physicsnemo.sym.hydra import PhysicsNeMoConfig, instantiate_arch
from physicsnemo.sym.domain import Domain
from physicsnemo.sym.domain.constraint import PointwiseBoundaryConstraint, PointwiseInteriorConstraint
from physicsnemo.sym.domain.validator import PointwiseValidator
from physicsnemo.sym.eq.pde import PDE
from physicsnemo.sym.geometry.primitives_2d import Rectangle
from physicsnemo.sym.key import Key
from physicsnemo.sym.solver import Solver

from wave_reference import exact_solution


class WaveEquation2D(PDE):
    name = "WaveEquation2D"

    def __init__(self, c):
        x, y, t = Symbol("x"), Symbol("y"), Symbol("t")
        u = Function("u")(x, y, t)
        self.equations = {
            "wave_equation": u.diff(t, 2) - c**2 * (u.diff(x, 2) + u.diff(y, 2))
        }


def pde_rmse(network, xyz, c, device):
    """Evaluate the equation independently with PyTorch automatic differentiation."""
    values = {
        key: torch.tensor(xyz[:, i:i+1], dtype=torch.float32, device=device, requires_grad=True)
        for i, key in enumerate(("x", "y", "t"))
    }
    u = network(values)["u"]
    second = {}
    for key, coordinate in values.items():
        first = torch.autograd.grad(u, coordinate, torch.ones_like(u), create_graph=True)[0]
        second[key] = torch.autograd.grad(first, coordinate, torch.ones_like(first), create_graph=True)[0]
    residual = second["t"] - c**2 * (second["x"] + second["y"])
    return torch.sqrt(torch.mean(residual**2)).item()


@physicsnemo.sym.main(config_path="conf", config_name="config_wave")
def run(cfg: PhysicsNeMoConfig):
    c, t_end = float(cfg.custom.c), float(cfg.custom.t_end)
    exact_solution(0.0, 0.0, 0.0, c)  # Validate c before creating any output.
    if not np.isfinite(t_end) or t_end <= 0:
        raise ValueError("t_end must be finite and positive.")
    run_name = str(cfg.custom.run_name)
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_name):
        raise ValueError("run_name accepts letters, digits, underscores, and hyphens.")
    output = Path(__file__).resolve().parent / "runs" / run_name
    if output.exists():
        raise FileExistsError(f"Choose a new custom.run_name to keep the previous run: {output}")
    output.mkdir(parents=True)
    cfg.network_dir = str(output / "checkpoints")
    np.random.seed(int(cfg.custom.seed))
    torch.manual_seed(int(cfg.custom.seed))

    x, y, t = Symbol("x"), Symbol("y"), Symbol("t")
    network = instantiate_arch(
        input_keys=[Key("x"), Key("y"), Key("t")], output_keys=[Key("u")],
        cfg=cfg.arch.fully_connected,
    )
    nodes = WaveEquation2D(c).make_nodes() + [network.make_node(name="wave_network")]
    domain = Domain()
    geometry = Rectangle((0, 0), (float(np.pi), float(np.pi)))
    domain.add_constraint(PointwiseInteriorConstraint(
        nodes=nodes, geometry=geometry,
        outvar={"u": sin(x)*sin(y), "u__t": sin(x)*sin(y)},
        batch_size=cfg.batch_size.IC, parameterization={t: 0.0},
    ), "initial")
    domain.add_constraint(PointwiseBoundaryConstraint(
        nodes=nodes, geometry=geometry, outvar={"u": 0.0},
        batch_size=cfg.batch_size.BC, parameterization={t: (0.0, t_end)},
    ), "boundary")
    domain.add_constraint(PointwiseInteriorConstraint(
        nodes=nodes, geometry=geometry, outvar={"wave_equation": 0.0},
        batch_size=cfg.batch_size.interior, parameterization={t: (0.0, t_end)},
    ), "interior")

    grid = np.meshgrid(np.linspace(0, np.pi, 20), np.linspace(0, np.pi, 20),
                       np.linspace(0, t_end, 5), indexing="ij")
    invar = {key: data.reshape(-1, 1) for key, data in zip(("x", "y", "t"), grid)}
    true_u = exact_solution(invar["x"], invar["y"], invar["t"], c)
    domain.add_validator(PointwiseValidator(
        nodes=nodes, invar=invar, true_outvar={"u": true_u}, batch_size=128,
    ), "reference")

    started = time.perf_counter()
    Solver(cfg, domain).solve()
    elapsed = time.perf_counter() - started
    network.eval()
    device = next(network.parameters()).device
    with torch.no_grad():
        pred = network({k: torch.as_tensor(v, dtype=torch.float32, device=device)
                        for k, v in invar.items()})["u"].cpu().numpy()
    rng = np.random.default_rng(43)
    test_xyz = rng.random((128, 3)) * np.array([np.pi, np.pi, t_end])
    metrics = {
        "c": c, "t_end": t_end, "seed": int(cfg.custom.seed),
        "training_steps_requested": int(cfg.training.max_steps),
        "training_seconds": elapsed, "device": str(device),
        "validation_rmse": float(np.sqrt(np.mean((pred - true_u)**2))),
        "pde_rmse_128_points": pde_rmse(network, test_xyz, c, device),
        "torch_version": torch.__version__,
    }
    for package in ("nvidia-physicsnemo", "nvidia-physicsnemo.sym"):
        try:
            metrics[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            metrics[package] = "not recorded"
    np.savez(output / "prediction.npz", **invar, truth=true_u, prediction=pred)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))
    print(f"Results: {output}")


if __name__ == "__main__":
    run()
