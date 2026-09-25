"""Level 2: Steady channel flow around three chips, PhysicsNeMo 2.2.2.

The blocks are fixed. No structural-deformation model is implied by this flow
exercise. Complete equations, conditions and geometry, or select --reference.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch
from sympy import symbols, Function
from ETC.runtime.exercises import conditions, condition_tensor, block_geometry
from physicsnemo.sym.eq.pde import PDE
from ETC.runtime.pinn import (parse_args, create_model, create_informer, residuals,
    evaluate_fields, sample_time, evaluation_grid, save_results, record_step,
    heldout_losses, heldout_row, create_evaluation_informer)
from fluid_geometry import (SINGLE_BLOCK, THREE_BLOCKS, sample_interior,
    sample_inlet, sample_walls, sample_flux, inlet_velocity, sdf_weight, openfoam_metrics)

LEVEL = 2
BLOCKS = THREE_BLOCKS
TIME_END = None
FIELD_NAMES = ["u", "v", "p"]


def reference_equations(x, y, t, u, v, p, nu=.02, rho=1.0):
    return {
        "continuity": u.diff(x) + v.diff(y),
        "momentum_x": u*u.diff(x) + v*u.diff(y) + p.diff(x)/rho - nu*(u.diff(x, 2)+u.diff(y, 2)),
        "momentum_y": u*v.diff(x) + v*v.diff(y) + p.diff(y)/rho - nu*(v.diff(x, 2)+v.diff(y, 2)),
    }


def student_equations(x, y, t, u, v, p, nu=.02, rho=1.0):
    # FIXME: return continuity, momentum_x, momentum_y residuals.
    # Level 3 includes u.diff(t), v.diff(t); levels 1/2 are steady.
    raise NotImplementedError("Complete student_equations in {} and save, or add --reference for the completed PDE.".format(Path(__file__).name))


def reference_conditions(x, y, t):
    return {"inlet_u": 1.5 * (1 - 4*y**2), "inlet_v": 0,
            "outlet_pressure": 0, "wall_u": 0, "wall_v": 0, "flux": 1}


def student_conditions(x, y, t):
    # FIXME: inlet_u, inlet_v, outlet_pressure, wall_u, wall_v and integral flux.
    # The inlet peak is 1.5 and channel height is 1; integrate the profile for flux.
    raise NotImplementedError("Complete student_conditions: inlet, outlet, walls, flux and any initial values.")


def reference_geometry():
    return {"blocks": ((-1.0, -.4, -.1), (.2, .7, 0.0), (1.2, 1.6, -.15))}


def student_geometry():
    # FIXME: {"blocks": ((xmin, xmax, top), ...)}. Every chip starts at y=-0.5.
    # Three cutouts: start/width/height = (-1,.6,.4), (.2,.5,.5), (1.2,.4,.35).
    raise NotImplementedError("Complete student_geometry: subtract the specified chip rectangles from the channel.")


class NavierStokes2D(PDE):
    def __init__(self, reference=False):
        self.dim = 2
        x, y, t = symbols("x y t")
        coordinates = (x, y, t) if TIME_END is not None else (x, y)
        u, v, p = [Function(name)(*coordinates) for name in FIELD_NAMES]
        self.equations = (reference_equations if reference else student_equations)(x, y, t, u, v, p)


def time_sample(count, device):
    return None if TIME_END is None else sample_time(count, TIME_END, device)


def loss_terms(model, informer, config, device, exercise=None, geometry=None):
    exercise = conditions(reference_conditions) if exercise is None else exercise
    geometry = block_geometry(reference_geometry) if geometry is None else geometry
    count = config["samples"]
    xy = sample_interior(count["interior"], geometry, device)
    pde = residuals(model, informer, xy, time_sample(len(xy), device), FIELD_NAMES)
    weighting = sdf_weight(xy, geometry)
    losses = {"pde_" + name + "_weighted": (weighting * value.square()).mean() for name, value in pde.items()}
    inlet = sample_inlet(count["boundary"], device)
    inlet_time = time_sample(len(inlet), device)
    incoming = evaluate_fields(model, inlet, inlet_time, FIELD_NAMES)
    losses["inlet_u"] = (incoming["u"] - condition_tensor(exercise, "inlet_u", inlet, inlet_time)).square().mean()
    losses["inlet_v"] = (incoming["v"] - condition_tensor(exercise, "inlet_v", inlet, inlet_time)).square().mean()
    outlet = sample_inlet(count["boundary"], device, outlet=True)
    outlet_time = time_sample(len(outlet), device)
    pressure = evaluate_fields(model, outlet, outlet_time, FIELD_NAMES)["p"]
    losses["outlet_pressure"] = (pressure - condition_tensor(exercise, "outlet_pressure", outlet, outlet_time)).square().mean()
    walls = sample_walls(count["boundary"], geometry, device)
    wall_time = time_sample(len(walls), device)
    velocity = evaluate_fields(model, walls, wall_time, FIELD_NAMES)
    losses["no_slip"] = sum((velocity[name] - condition_tensor(exercise, "wall_" + name, walls, wall_time)).square().mean() for name in ("u", "v"))
    flux_xy, flux_time, height = sample_flux(count["flux_lines"], count["flux_points"], geometry, device, TIME_END)
    flux_u = evaluate_fields(model, flux_xy, flux_time, FIELD_NAMES)["u"]
    flux = flux_u.reshape(count["flux_lines"], count["flux_points"]).mean(1, keepdim=True) * height
    section_xy = flux_xy.reshape(count["flux_lines"], count["flux_points"], 2)[:, 0, :]
    section_time = None if flux_time is None else flux_time.reshape(count["flux_lines"], count["flux_points"])[:, :1]
    losses["integral_continuity"] = (flux - condition_tensor(exercise, "flux", section_xy, section_time)).square().mean()
    if TIME_END is not None:
        initial_xy = sample_interior(count["initial"], geometry, device)
        initial_t = torch.zeros(len(initial_xy), 1, device=device)
        fields = evaluate_fields(model, initial_xy, initial_t, FIELD_NAMES)
        losses["initial_rest"] = sum((value - condition_tensor(exercise, "initial_" + name, initial_xy, initial_t)).square().mean() for name, value in fields.items())
    return losses


def main():
    args, config = parse_args(__file__, __doc__, "config_chip_2d.yaml")
    exercise = conditions(reference_conditions if args.reference else student_conditions)
    geometry = block_geometry(reference_geometry if args.reference else student_geometry)
    pde = NavierStokes2D(reference=args.reference)
    informer = create_informer(pde, args.device)
    eval_informer = create_evaluation_informer(NavierStokes2D, args.device)
    model = create_model(2, 3, config, args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])
    if args.output_dir.exists():
        raise FileExistsError("Choose a new --output-dir; previous results are preserved.")
    initial = heldout_losses(loss_terms, model, eval_informer, config, args.device, args.seed)
    history = [heldout_row(0, "heldout_before_training", initial)]
    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        losses = loss_terms(model, informer, config, args.device, exercise, geometry)
        total = record_step(step, losses, history)
        total.backward()
        optimizer.step()
    final = heldout_losses(loss_terms, model, eval_informer, config, args.device, args.seed)
    history.append(heldout_row(args.steps, "heldout_after_training", final))
    metrics = {name + "_rmse": float(value.sqrt()) for name, value in final.items()}
    metrics["evaluation_equations"] = "provided_reference_equations"
    metrics["evaluation_conditions"] = "provided_reference_conditions_and_geometry"
    metrics["training_blocks"] = geometry
    # Separate unweighted PDE residuals from the SDF-weighted training objective.
    torch.manual_seed(args.seed + 200000)
    xy_check = sample_interior(256, BLOCKS, args.device)
    check = residuals(model, eval_informer, xy_check, time_sample(len(xy_check), args.device), FIELD_NAMES)
    metrics.update({name + "_unweighted_rmse": float(value.detach().square().mean().sqrt()) for name, value in check.items()})
    metrics["openfoam_reference_available"] = False
    xy, time = evaluation_grid(args.device, TIME_END, fluid_blocks=BLOCKS)
    save_results(args, config, model, history, xy, time, FIELD_NAMES, metrics, notes=[
        "No analytic solution is claimed for channel flow around these blocks.",
        "Inlet peak=1.5, zero outlet pressure, no-slip walls, nu=.02, rho=1, integral flux=1.",
        'Blocks are stationary; this exercise does not solve structural deformation.'])


if __name__ == "__main__":
    try:
        main()
    except NotImplementedError as exc:
        raise SystemExit(str(exc))
