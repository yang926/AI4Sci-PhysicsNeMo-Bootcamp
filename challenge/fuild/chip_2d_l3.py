"""Level 3: Time-dependent channel flow around one chip, PhysicsNeMo 2.2.2.

The blocks are fixed. No structural-deformation model is implied by this flow
exercise. Complete student_equations or select --reference.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch
from sympy import symbols, Function
from physicsnemo.sym.eq.pde import PDE
from pinn_runtime import (parse_args, create_model, create_informer, residuals,
    evaluate_fields, sample_time, evaluation_grid, save_results, record_step,
    heldout_losses, heldout_row)
from fluid_geometry import (SINGLE_BLOCK, THREE_BLOCKS, sample_interior,
    sample_inlet, sample_walls, sample_flux, inlet_velocity, sdf_weight, openfoam_metrics)

LEVEL = 3
BLOCKS = SINGLE_BLOCK
TIME_END = 10.0
FIELD_NAMES = ["u", "v", "p"]


def reference_equations(x, y, t, u, v, p, nu=.01, rho=1.0):
    return {
        "continuity": u.diff(x) + v.diff(y),
        "momentum_x": u.diff(t) + u*u.diff(x) + v*u.diff(y) + p.diff(x)/rho - nu*(u.diff(x, 2)+u.diff(y, 2)),
        "momentum_y": v.diff(t) + u*v.diff(x) + v*v.diff(y) + p.diff(y)/rho - nu*(v.diff(x, 2)+v.diff(y, 2)),
    }


def student_equations(x, y, t, u, v, p, nu=.01, rho=1.0):
    # FIXME: return continuity, momentum_x, momentum_y residuals.
    # Level 3 includes u.diff(t), v.diff(t); levels 1/2 are steady.
    raise NotImplementedError("Complete student_equations in {} and save, or add --reference for the completed PDE.".format(Path(__file__).name))


class NavierStokes2D(PDE):
    def __init__(self, reference=False):
        self.dim = 2
        x, y, t = symbols("x y t")
        coordinates = (x, y, t) if TIME_END is not None else (x, y)
        u, v, p = [Function(name)(*coordinates) for name in FIELD_NAMES]
        self.equations = (reference_equations if reference else student_equations)(x, y, t, u, v, p)


def time_sample(count, device):
    return None if TIME_END is None else sample_time(count, TIME_END, device)


def loss_terms(model, informer, config, device):
    count = config["samples"]
    xy = sample_interior(count["interior"], BLOCKS, device)
    pde = residuals(model, informer, xy, time_sample(len(xy), device), FIELD_NAMES)
    weighting = sdf_weight(xy, BLOCKS)
    losses = {"pde_" + name + "_weighted": (weighting * value.square()).mean() for name, value in pde.items()}
    inlet = sample_inlet(count["boundary"], device)
    incoming = evaluate_fields(model, inlet, time_sample(len(inlet), device), FIELD_NAMES)
    losses["inlet_u"] = (incoming["u"] - inlet_velocity(inlet)).square().mean()
    losses["inlet_v"] = incoming["v"].square().mean()
    outlet = sample_inlet(count["boundary"], device, outlet=True)
    losses["outlet_pressure"] = evaluate_fields(model, outlet, time_sample(len(outlet), device), FIELD_NAMES)["p"].square().mean()
    walls = sample_walls(count["boundary"], BLOCKS, device)
    velocity = evaluate_fields(model, walls, time_sample(len(walls), device), FIELD_NAMES)
    losses["no_slip"] = velocity["u"].square().mean() + velocity["v"].square().mean()
    flux_xy, flux_time, height = sample_flux(count["flux_lines"], count["flux_points"], BLOCKS, device, TIME_END)
    flux_u = evaluate_fields(model, flux_xy, flux_time, FIELD_NAMES)["u"]
    flux = flux_u.reshape(count["flux_lines"], count["flux_points"]).mean(1, keepdim=True) * height
    losses["integral_continuity"] = (flux - 1.0).square().mean()
    if TIME_END is not None:
        initial_xy = sample_interior(count["initial"], BLOCKS, device)
        fields = evaluate_fields(model, initial_xy, torch.zeros(len(initial_xy), 1, device=device), FIELD_NAMES)
        losses["initial_rest"] = sum(value.square().mean() for value in fields.values())
    return losses


def main():
    args, config = parse_args(__file__, __doc__, "config_chip_2d.yaml")
    pde = NavierStokes2D(reference=args.reference)
    informer = create_informer(pde, args.device)
    model = create_model(3, 3, config, args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])
    if args.output_dir.exists():
        raise FileExistsError("Choose a new --output-dir; previous results are preserved.")
    initial = heldout_losses(loss_terms, model, informer, config, args.device, args.seed)
    history = [heldout_row(0, "heldout_before_training", initial)]
    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        losses = loss_terms(model, informer, config, args.device)
        total = record_step(step, losses, history)
        total.backward()
        optimizer.step()
    final = heldout_losses(loss_terms, model, informer, config, args.device, args.seed)
    history.append(heldout_row(args.steps, "heldout_after_training", final))
    metrics = {name + "_rmse": float(value.sqrt()) for name, value in final.items()}
    # Separate unweighted PDE residuals from the SDF-weighted training objective.
    torch.manual_seed(args.seed + 200000)
    xy_check = sample_interior(256, BLOCKS, args.device)
    check = residuals(model, informer, xy_check, time_sample(len(xy_check), args.device), FIELD_NAMES)
    metrics.update({name + "_unweighted_rmse": float(value.detach().square().mean().sqrt()) for name, value in check.items()})
    metrics["openfoam_reference_available"] = False
    xy, time = evaluation_grid(args.device, TIME_END, fluid_blocks=BLOCKS)
    save_results(args, config, model, history, xy, time, FIELD_NAMES, metrics, notes=[
        "No analytic solution is claimed for channel flow around these blocks.",
        "Inlet peak=1.5, zero outlet pressure, no-slip walls, nu=.01, rho=1, integral flux=1.",
        'Original impulsive start retained: interior is at rest at t=0 while the inlet is prescribed for positive time; compatibility at the initial inlet corner is not smooth.'])


if __name__ == "__main__":
    try:
        main()
    except NotImplementedError as exc:
        raise SystemExit(str(exc))
