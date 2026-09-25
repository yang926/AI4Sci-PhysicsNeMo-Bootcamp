"""Level 2: Spatially varying wave speed. PhysicsNeMo 2.2.2 explicit-loop PINN.

Complete all student_* functions. --reference supplies the full problem.
A short run checks execution only; low loss and convergence are not promised.
"""
from pathlib import Path
import math
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from sympy import symbols, Function, sin, cos, exp
from ETC.runtime.exercises import conditions, condition_tensor
from physicsnemo.sym.eq.pde import PDE
from ETC.runtime.pinn import (parse_args, create_model, create_informer, residuals,
    evaluate_fields, gradient, sample_square, sample_circle, sample_time,
    evaluation_grid, save_results, record_step, heldout_losses, heldout_row, reference_errors,
    create_evaluation_informer, reference_errors_over_time)

LEVEL = 2
TIME_END = 2 * math.pi
FIELD_NAMES = ["u"]


def reference_equations(x, y, t, u, c=1.0):
    """Residual is zero: u_tt - c(x,y)^2 (u_xx + u_yy)."""
    return {"wave": u.diff(t, 2) - c ** 2 * (u.diff(x, 2) + u.diff(y, 2))}


def student_equations(x, y, t, u, c=1.0):
    # FIXME: return {"wave": ...} using u.diff(t, 2), u.diff(x, 2), u.diff(y, 2).
    # Use the supplied c from student_speed; do not differentiate c in this non-divergence PDE.
    raise NotImplementedError("Complete student_equations in {} then save, or add --reference to run the completed implementation.".format(Path(__file__).name))


def reference_speed(x, y):
    return {"c": 1 + .5 * sin(x) * cos(y)}


def student_speed(x, y):
    # FIXME: return {"c": ...} for c(x,y)=1+0.5*sin(x)*cos(y).
    raise NotImplementedError("Complete student_speed with the stated wave-speed field, or use --reference.")


def reference_conditions(x, y, t, u):
    return {"initial_u": sin(x) * sin(y),
            "initial_ut": 0, "boundary": u}


def student_conditions(x, y, t, u):
    # FIXME: return initial_u, initial_ut, and the boundary residual (target zero).
    # The square has zero displacement on all four edges.
    raise NotImplementedError("Complete student_conditions: initial displacement, velocity and boundary condition.")


class WaveEquation2D(PDE):
    def __init__(self, reference=False, c=1.0):
        if isinstance(c, bool) or not isinstance(c, (int, float)) or not math.isfinite(c) or c <= 0:
            raise ValueError("wave speed c must be positive and finite")
        self.dim = 2
        x, y, t = symbols("x y t")
        u = Function("u")(x, y, t)
        build = reference_equations if reference else student_equations
        speed = (reference_speed if reference else student_speed)(x, y)
        if not isinstance(speed, dict) or set(speed) != {"c"}:
            raise ValueError("student_speed must return {'c': speed_expression}")
        self.equations = build(x, y, t, u, c * speed["c"])


def initial_displacement(coordinates):
    x, y = coordinates[:, :1], coordinates[:, 1:2]
    return torch.sin(x) * torch.sin(y)


def exact_reference(coordinates, time, c=1.0):
    # No verified closed-form reference for this variable-speed Dirichlet problem.
    return None


def spatial_sample(count, device, boundary=False):
    return sample_square(count, device, boundary=boundary)


def boundary_residual(model, coordinates, time, exercise=None):
    coordinates = coordinates.detach().requires_grad_(True)
    fields = evaluate_fields(model, coordinates, time, FIELD_NAMES)
    exercise = conditions(reference_conditions, wave=True) if exercise is None else exercise
    return condition_tensor(exercise, "boundary", coordinates, time, fields)


def loss_terms(model, informer, config, device, exercise=None):
    exercise = conditions(reference_conditions, wave=True) if exercise is None else exercise
    count = config["samples"]
    xy = spatial_sample(count["interior"], device)
    pde = residuals(model, informer, xy, sample_time(len(xy), TIME_END, device), FIELD_NAMES)
    initial_xy = spatial_sample(count["initial"], device)
    initial_t = torch.zeros(len(initial_xy), 1, device=device, requires_grad=True)
    initial_u = evaluate_fields(model, initial_xy, initial_t, FIELD_NAMES)["u"]
    target = condition_tensor(exercise, "initial_u", initial_xy, initial_t)
    velocity_target = condition_tensor(exercise, "initial_ut", initial_xy, initial_t)
    initial_velocity = gradient(initial_u, initial_t)
    boundary_xy = spatial_sample(count["boundary"], device, boundary=True)
    bc = boundary_residual(model, boundary_xy, sample_time(len(boundary_xy), TIME_END, device), exercise)
    return {"pde": pde["wave"].square().mean(),
            "initial_displacement": (initial_u - target).square().mean(),
            "initial_velocity": (initial_velocity - velocity_target).square().mean(),
            "boundary": bc.square().mean()}


def main():
    args, config = parse_args(__file__, __doc__, "config_wave.yaml")
    exercise = conditions(reference_conditions if args.reference else student_conditions, wave=True)
    pde = WaveEquation2D(reference=args.reference)
    informer = create_informer(pde, args.device)
    eval_informer = create_evaluation_informer(WaveEquation2D, args.device)
    model = create_model(3, 1, config, args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])
    if args.output_dir.exists():
        raise FileExistsError("Choose a new --output-dir; previous results are preserved.")
    initial = heldout_losses(loss_terms, model, eval_informer, config, args.device, args.seed)
    xy, time = evaluation_grid(args.device, TIME_END, circle=False)
    reference = exact_reference(xy, time)
    before_error = reference_errors(model, xy, time, FIELD_NAMES, reference)
    history = [heldout_row(0, "heldout_before_training", initial)]
    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        losses = loss_terms(model, informer, config, args.device, exercise)
        total = record_step(step, losses, history)
        total.backward()
        optimizer.step()
    evaluation = heldout_losses(loss_terms, model, eval_informer, config, args.device, args.seed)
    history.append(heldout_row(args.steps, "heldout_after_training", evaluation))
    metrics = {name + "_rmse": float(value.detach().sqrt()) for name, value in evaluation.items()}
    xy, time = evaluation_grid(args.device, TIME_END, circle=False)
    metrics.update({"initial_reference_error": before_error,
                    "final_reference_error": reference_errors(model, xy, time, FIELD_NAMES, reference),
                    "evaluation_equations": "provided_reference_equations",
                    "evaluation_conditions": "provided_reference_conditions",
                    "reference_over_time": reference_errors_over_time(model, xy, TIME_END, FIELD_NAMES, exact_reference)})
    save_results(args, config, model, history, xy, time, FIELD_NAMES, metrics,
                 reference=exact_reference(xy, time), notes=["Independent sampled PDE/IC/BC checks; prediction grid is a mid-time slice.",
                 'No closed-form reference is asserted; PDE and condition residuals do not replace independent solution validation.'])


if __name__ == "__main__":
    try:
        main()
    except NotImplementedError as exc:
        raise SystemExit(str(exc))
