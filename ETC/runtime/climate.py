"""Optimization and collocation points for the Climate exercises."""
from __future__ import annotations

import math
import time

import torch

from ETC.runtime.pinn import checked_total_loss, reference_errors_over_time


def optimization_budget(total_steps):
    if isinstance(total_steps, bool) or not isinstance(total_steps, int) or total_steps <= 0:
        raise ValueError("total_steps must be a positive integer")
    # A tiny execution check should not allocate a full L-BFGS dataset.
    adam_steps = total_steps if total_steps < 10 else min(1500, total_steps // 2)
    return adam_steps, total_steps - adam_steps


class ClimateModel(torch.nn.Module):
    """Scale physical coordinates inside autograd before the unchanged MLP."""
    def __init__(self, network, time_end):
        super().__init__()
        self.network = network
        self.register_buffer("coordinate_extent", torch.tensor(
            [math.pi, math.pi, time_end], device=next(network.parameters()).device))

    def forward(self, inputs):
        return self.network(2 * inputs / self.coordinate_extent - 1)


def temperature_errors(model, coordinates, time_end, names, comparison):
    """Show the changing signal amplitude beside absolute and relative error."""
    fractions = (0., .25 / time_end, .5 / time_end, 1. / time_end, .25, .5, .75, 1.)
    errors = reference_errors_over_time(
        model, coordinates, time_end, names, comparison, time_fractions=fractions)
    if not errors:
        return errors
    with torch.no_grad():
        initial = comparison(coordinates, coordinates.new_zeros((len(coordinates), 1)))
        initial_rms = float(initial.double().square().mean().sqrt())
        for row in errors["per_time"]:
            reference = comparison(coordinates, coordinates.new_full((len(coordinates), 1), row["time"]))
            row["reference_max"] = float(reference.abs().max())
            row["initial_scale_rmse"] = row["rmse"] / max(initial_rms, 1e-12)
    return errors


def collocation_points(config, device, time_end, *, multiplier=1):
    if isinstance(multiplier, bool) or not isinstance(multiplier, int) or multiplier <= 0:
        raise ValueError("multiplier must be a positive integer")
    counts = config["samples"]
    unit_points = torch.quasirandom.SobolEngine(3, scramble=True, seed=87291).draw(
        counts["interior"] * multiplier).to(device)
    initial_side = max(2, math.ceil(math.sqrt(counts["initial"] * multiplier)))
    initial_axis = torch.linspace(0, math.pi, initial_side, device=device)
    initial_xy = torch.stack(torch.meshgrid(initial_axis, initial_axis, indexing="ij"), dim=-1).reshape(-1, 2)
    boundary_side = max(2, math.ceil(math.sqrt(counts["boundary"] * multiplier / 4)))
    edge_axis = torch.linspace(0, math.pi, boundary_side, device=device)
    time_axis = torch.linspace(0, time_end, boundary_side, device=device)
    edge, times = torch.meshgrid(edge_axis, time_axis, indexing="ij")
    edge, times = edge.reshape(-1), times.reshape(-1, 1)
    zero, top = torch.zeros_like(edge), torch.full_like(edge, math.pi)
    boundary = torch.cat([torch.stack(pair, dim=-1) for pair in (
        (zero, edge), (top, edge), (edge, zero), (edge, top))])
    return {
        "interior_xy": unit_points[:, :2] * math.pi,
        "interior_t": unit_points[:, 2:] * time_end,
        "initial_xy": initial_xy,
        "boundary_xy": boundary,
        "boundary_t": times.repeat(4, 1),
    }


def optimize_climate(model, loss_function, config, device, time_end, *,
                     adam_steps, lbfgs_steps, multiplier=32, lbfgs_max_iter=2, callback=None):
    """Train from equations and conditions; no analytic solution is used.

    Adam uses fresh samples. L-BFGS reuses one larger set for every closure,
    so its line search and curvature history refer to the same objective.
    """
    for name, value in (("adam_steps", adam_steps), ("lbfgs_steps", lbfgs_steps)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    if not adam_steps + lbfgs_steps:
        raise ValueError("At least one optimization step is required")
    if isinstance(lbfgs_max_iter, bool) or not isinstance(lbfgs_max_iter, int) or lbfgs_max_iter <= 0:
        raise ValueError("lbfgs_max_iter must be a positive integer")
    started = time.perf_counter()
    adam = torch.optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        adam, T_max=max(1, adam_steps), eta_min=1e-5)
    for step in range(1, adam_steps + 1):
        adam.zero_grad(set_to_none=True)
        losses = loss_function()
        total = checked_total_loss(losses)
        total.backward()
        adam.step()
        scheduler.step()
        if callback is not None:
            callback(step, "adam_before_update", {k: v.detach() for k, v in losses.items()})

    points = collocation_points(config, device, time_end, multiplier=multiplier) if lbfgs_steps else None
    def new_lbfgs():
        return torch.optim.LBFGS(
            model.parameters(), lr=1.0, max_iter=lbfgs_max_iter, max_eval=20,
            history_size=100, tolerance_grad=1e-9, tolerance_change=1e-12,
            line_search_fn="strong_wolfe")

    optimizer = new_lbfgs()
    evaluations = 0
    for step in range(1, lbfgs_steps + 1):
        def closure():
            nonlocal evaluations
            optimizer.zero_grad(set_to_none=True)
            losses = loss_function(points=points)
            total = checked_total_loss(losses)
            # Uniform scaling preserves the minimizer and relative weights.
            # It keeps L-BFGS's absolute stopping tests useful at small losses.
            total = total * 1000
            total.backward()
            evaluations += 1
            return total

        optimizer.step(closure)
        losses = loss_function(points=points)
        if callback is not None:
            callback(adam_steps + step, "lbfgs_after_update", {k: v.detach() for k, v in losses.items()})
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()
    return {"adam_steps": adam_steps, "lbfgs_steps": lbfgs_steps,
            "lbfgs_closure_evaluations": evaluations,
            "lbfgs_max_iter": lbfgs_max_iter, "lbfgs_history_size": 100,
            "lbfgs_objective_scale": 1000,
            "lbfgs_samples": {} if points is None else {"interior": len(points["interior_xy"]),
                               "initial": len(points["initial_xy"]),
                               "boundary": len(points["boundary_xy"])},
            "collocation_scheme": "scrambled Sobol interior; fixed initial/boundary grids including endpoints",
            "training_seconds": time.perf_counter() - started}
