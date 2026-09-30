"""Climate Level 2: coupled atmosphere-ocean temperature PINN.

This simplified PDE system models temperature transport and heat exchange.
The original uncoupled baseline is the default. Then compare a nonzero heat
exchange coefficient with the same initial data and diffusivities.
"""
from pathlib import Path
import math
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
from sympy import symbols, Function, sin, exp
from ETC.runtime.exercises import conditions, condition_tensor, physical_parameters
from physicsnemo.sym.eq.pde import PDE
from ETC.runtime.pinn import (parse_args, create_model, create_informer, residuals,
    evaluate_fields, sample_square, sample_time, evaluation_grid, save_results,
    record_step, heldout_losses, heldout_row, reference_errors,
    create_evaluation_informer, reference_errors_over_time)

LEVEL = 2
FIELD_NAMES = ['Ta', 'To']
TIME_END = 2 * math.pi
DEFAULT_PHYSICS = {'u0': 0.0, 'v0': 0.0, 'kappa_a': 1.0, 'kappa_o': 0.5, 'lam_a': 0.0, 'Q_a0': 0.0, 'Q_o0': 0.0, 'Teq_a0': 0.0, 'gamma0': 0.0}


############################################################################
# EDIT HERE 1/4: student_equations (PDE residuals)
# Keep the function signature. Replace the raise ... placeholder below
# with your implementation; do not leave it after your return/assignment.
############################################################################
def student_equations(x, y, t, fields, params):
    # FIXME: write the ADR residuals with equal and opposite atmosphere/ocean exchange terms.
    raise NotImplementedError("Complete student_equations in {} and save.".format(Path(__file__).name))
# END EDIT HERE 1/4: student_equations
############################################################################


############################################################################
# EDIT HERE 2/4: student_conditions (initial and boundary conditions)
# Keep the function signature. Replace the raise ... placeholder below
# with your implementation; do not leave it after your return/assignment.
############################################################################
def student_conditions(x, y, t):
    # FIXME: return the initial_<field> and boundary_<field> targets for every field.
    # Initial values are sin(x)*sin(y); all four edges have temperature zero.
    raise NotImplementedError("Complete student_conditions for all initial and boundary temperatures.")
# END EDIT HERE 2/4: student_conditions
############################################################################


############################################################################
# EDIT HERE 3/4: student_parameters (physical coefficients)
# Keep the function signature. Replace the raise ... placeholder below
# with your implementation; do not leave it after your return/assignment.
############################################################################
def student_parameters():
    # FIXME: specify the original diffusion-only validation coefficients.
    # u0=v0=lam_a=Q_a0=Q_o0=Teq_a0=gamma0=0; kappa_a=1; kappa_o=0.5.
    # Restore these after any separate coefficient experiment.
    raise NotImplementedError("Complete student_parameters with every coefficient named in DEFAULT_PHYSICS.")
# END EDIT HERE 3/4: student_parameters
############################################################################


############################################################################
# EDIT HERE 4/4: student_solution (baseline analytic solution)
# Keep the function signature. Replace the raise ... placeholder below
# with your implementation; do not leave it after your return/assignment.
############################################################################
def student_solution(x, y, t, params):
    # FIXME: derive the baseline sine-mode exact solution for Ta and To.
    # This expression supplies the comparison plot; it is not independently verified.
    raise NotImplementedError("Complete student_solution for the original diffusion-only baseline.")
# END EDIT HERE 4/4: student_solution
############################################################################


class ClimatePDE(PDE):
    def __init__(self, params=None):
        self.dim = 2
        if params is not None and not isinstance(params, dict):
            raise ValueError("Physics parameters must be a mapping")
        unknown = set(params or {}) - DEFAULT_PHYSICS.keys()
        if unknown:
            raise ValueError(f"Unknown physics parameters: {sorted(unknown)}")
        params = DEFAULT_PHYSICS.copy() if params is None else {**DEFAULT_PHYSICS, **params}
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
               for v in params.values()):
            raise ValueError("Physics parameters must be finite numbers")
        for name, value in params.items():
            if name.startswith("kappa") and value <= 0:
                raise ValueError("Diffusivity must be positive")
        if params["gamma0"] < 0:
            raise ValueError("Heat-exchange coefficient gamma0 must be nonnegative")
        x, y, t = symbols("x y t")
        fields = {name: Function(name)(x, y, t) for name in FIELD_NAMES}
        self.equations = student_equations(x, y, t, fields, params)


def loss_terms(model, informer, config, device, exercise=None):
    exercise = conditions(student_conditions) if exercise is None else exercise
    count = config["samples"]
    xy = sample_square(count["interior"], device)
    pde = residuals(model, informer, xy, sample_time(len(xy), TIME_END, device), FIELD_NAMES)
    losses = {"pde_" + name: value.square().mean() for name, value in pde.items()}
    initial_xy = sample_square(count["initial"], device)
    fields = evaluate_fields(model, initial_xy, torch.zeros(len(initial_xy), 1, device=device), FIELD_NAMES)
    for name, value in fields.items():
        target = condition_tensor(exercise, "initial_" + name, initial_xy)
        losses["initial_" + name] = (value - target).square().mean()
    boundary_xy = sample_square(count["boundary"], device, boundary=True)
    boundary_t = sample_time(len(boundary_xy), TIME_END, device)
    boundary = evaluate_fields(model, boundary_xy, boundary_t, FIELD_NAMES)
    for name, value in boundary.items():
        target = condition_tensor(exercise, "boundary_" + name, boundary_xy, boundary_t)
        losses["boundary_" + name] = (value - target).square().mean()
    return losses


def main():
    args, config = parse_args(__file__, __doc__, "config_coupled.yaml")
    params = physical_parameters(student_parameters, DEFAULT_PHYSICS)
    exercise = conditions(student_conditions)
    solution = conditions(lambda x, y, t: student_solution(x, y, t, params))
    if set(solution) != set(FIELD_NAMES):
        raise ValueError("student_solution must return exactly: " + ", ".join(FIELD_NAMES))

    def comparison(points, times):
        # The assigned solution is for the uncoupled diffusion-only baseline.
        if any(params[name] != 0 for name in ('u0', 'v0', 'lam_a', 'Q_a0', 'Q_o0', 'gamma0')):
            return None
        return torch.cat([condition_tensor(solution, name, points, times)
                          for name in FIELD_NAMES], dim=1)

    pde = ClimatePDE(params=params)
    informer = create_informer(pde, args.device)
    evaluation_informer = create_evaluation_informer(ClimatePDE, args.device, params=params)
    model = create_model(3, len(FIELD_NAMES), config, args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])
    if args.output_dir.exists():
        raise FileExistsError("Choose a new --output-dir; previous results are preserved.")
    initial = heldout_losses(loss_terms, model, evaluation_informer, config, args.device, args.seed)
    xy, time = evaluation_grid(args.device, TIME_END)
    reference = comparison(xy, time)
    before_error = reference_errors(model, xy, time, FIELD_NAMES, reference)
    history = [heldout_row(0, "heldout_before_training", initial)]
    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        losses = loss_terms(model, informer, config, args.device, exercise)
        total = record_step(step, losses, history)
        total.backward()
        optimizer.step()
    final = heldout_losses(loss_terms, model, evaluation_informer, config, args.device, args.seed)
    history.append(heldout_row(args.steps, "heldout_after_training", final))
    metrics = {name + "_rmse": float(value.sqrt()) for name, value in final.items()}
    metrics.update({"physics": params, "training_physics": params,
                    "analytic_exercise": {"source": "student_solution", "independently_checked": False},
                    "comparison_source": "student_solution",
                    "comparison_label": "Learner-supplied solution",
                    "evaluation_conditions": "student_conditions", "initial_reference_error": before_error,
                    "final_reference_error": reference_errors(model, xy, time, FIELD_NAMES, reference),
                    "evaluation_equations": "student_equations",
                    "reference_over_time": reference_errors_over_time(
                        model, xy, TIME_END, FIELD_NAMES,
                        comparison)})
    save_results(args, config, model, history, xy, time, FIELD_NAMES, metrics, reference=reference, notes=[
        "The comparison uses student_solution and does not independently certify its correctness.",
        "The baseline comparison is disabled when coupling, advection, sources, or relaxation are enabled.",
        "Held-out residuals and condition errors use the learner's implementation on fresh points; a small error does not prove the implementation is correct.",
        "Original baseline gamma0=0. Compare gamma0=.5 as a separate local experiment, then restore the baseline."])


if __name__ == "__main__":
    try:
        main()
    except NotImplementedError as exc:
        raise SystemExit(str(exc))
