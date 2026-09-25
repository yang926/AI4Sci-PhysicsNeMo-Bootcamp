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
from ETC.runtime.exercises import conditions, condition_tensor, physical_parameters, analytic_expression_checks
from physicsnemo.sym.eq.pde import PDE
from ETC.runtime.pinn import (parse_args, create_model, create_informer, residuals,
    evaluate_fields, sample_square, sample_time, evaluation_grid, save_results,
    record_step, heldout_losses, heldout_row, reference_errors,
    create_evaluation_informer, reference_errors_over_time)

LEVEL = 2
FIELD_NAMES = ['Ta', 'To']
TIME_END = 2 * math.pi
DEFAULT_PHYSICS = {'u0': 0.0, 'v0': 0.0, 'kappa_a': 1.0, 'kappa_o': 0.5, 'lam_a': 0.0, 'Q_a0': 0.0, 'Q_o0': 0.0, 'Teq_a0': 0.0, 'gamma0': 0.0}


def reference_equations(x, y, t, fields, params):
    Ta, To = fields["Ta"], fields["To"]
    exchange = params["gamma0"] * (Ta - To)
    return {
        "atmosphere": Ta.diff(t) + params["u0"]*Ta.diff(x) + params["v0"]*Ta.diff(y)
            - params["kappa_a"]*(Ta.diff(x, 2)+Ta.diff(y, 2)) - params["Q_a0"]
            + params["lam_a"]*(Ta-params["Teq_a0"]) + exchange,
        "ocean": To.diff(t) - params["kappa_o"]*(To.diff(x, 2)+To.diff(y, 2))
            - params["Q_o0"] - exchange,
    }


def student_equations(x, y, t, fields, params):
    # FIXME: write the ADR residuals with equal and opposite atmosphere/ocean exchange terms.
    raise NotImplementedError("Complete student_equations in {} and save, or use --reference to run the completed PDE.".format(Path(__file__).name))


def reference_conditions(x, y, t):
    return {"initial_Ta": sin(x)*sin(y), "initial_To": sin(x)*sin(y),
            "boundary_Ta": 0, "boundary_To": 0}


def student_conditions(x, y, t):
    # FIXME: return the initial_<field> and boundary_<field> targets for every field.
    # Initial values are sin(x)*sin(y); all four edges have temperature zero.
    raise NotImplementedError("Complete student_conditions for all initial and boundary temperatures.")


def reference_parameters():
    return {'u0': 0.0, 'v0': 0.0, 'kappa_a': 1.0, 'kappa_o': 0.5, 'lam_a': 0.0, 'Q_a0': 0.0, 'Q_o0': 0.0, 'Teq_a0': 0.0, 'gamma0': 0.0}


def student_parameters():
    # FIXME: specify the original diffusion-only validation coefficients.
    # Set advection, sources, relaxation and heat exchange to zero; kappa_a=1, kappa_o=.5.
    raise NotImplementedError("Complete student_parameters with every coefficient named in DEFAULT_PHYSICS.")


def reference_solution(x, y, t, params):
    # Original no-advection/source/relaxation (and no-coupling) validation case.
    return {"Ta": sin(x)*sin(y)*exp(-2*params["kappa_a"]*t),
            "To": sin(x)*sin(y)*exp(-2*params["kappa_o"]*t)}


def student_solution(x, y, t, params):
    # FIXME: derive the baseline sine-mode exact solution for Ta and To.
    # This exercise is checked independently; it never replaces the held-out truth.
    raise NotImplementedError("Complete student_solution for the original diffusion-only baseline.")


class ClimatePDE(PDE):
    def __init__(self, reference=False, params=None):
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
        self.equations = (reference_equations if reference else student_equations)(x, y, t, fields, params)


def exact_reference(coordinates, time, params):
    """Independent sine-mode solution of the coupled diffusion equations.

    With Ta=a(t)*sin(x)*sin(y), To=o(t)*sin(x)*sin(y), the amplitudes
    obey z'=-K z, z(0)=(1, 1). Diagonalizing the symmetric decay matrix
    avoids division by gamma or an eigenvalue gap, including gamma=0 and
    equal diffusivities. This is a reference solution, not training data.
    """
    if any(params[name] != 0 for name in ['u0', 'v0', 'lam_a', 'Q_a0', 'Q_o0']):
        return None
    initial = torch.sin(coordinates[:, :1]) * torch.sin(coordinates[:, 1:2])
    gamma = params["gamma0"]
    decay_matrix = coordinates.new_tensor([
        [2 * params["kappa_a"] + gamma, -gamma],
        [-gamma, 2 * params["kappa_o"] + gamma],
    ])
    rates, modes = torch.linalg.eigh(decay_matrix)
    initial_modes = coordinates.new_ones(2) @ modes
    amplitudes = (torch.exp(-time * rates) * initial_modes) @ modes.T
    return initial * amplitudes


def loss_terms(model, informer, config, device, exercise=None):
    exercise = conditions(reference_conditions) if exercise is None else exercise
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
    params = {**DEFAULT_PHYSICS, **config.get("physics", {})}
    exercise = conditions(reference_conditions if args.reference else student_conditions)
    training_params = params if args.reference else physical_parameters(student_parameters, DEFAULT_PHYSICS)
    solution_checks = analytic_expression_checks(reference_solution if args.reference else student_solution, reference_solution, DEFAULT_PHYSICS)
    pde = ClimatePDE(reference=args.reference, params=training_params)
    informer = create_informer(pde, args.device)
    evaluation_informer = create_evaluation_informer(ClimatePDE, args.device, params=params)
    model = create_model(3, len(FIELD_NAMES), config, args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])
    if args.output_dir.exists():
        raise FileExistsError("Choose a new --output-dir; previous results are preserved.")
    initial = heldout_losses(loss_terms, model, evaluation_informer, config, args.device, args.seed)
    xy, time = evaluation_grid(args.device, TIME_END)
    reference = exact_reference(xy, time, params)
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
    metrics.update({"physics": params, "training_physics": training_params,
                    "analytic_exercise": solution_checks,
                    "evaluation_conditions": "provided_reference_conditions", "initial_reference_error": before_error,
                    "final_reference_error": reference_errors(model, xy, time, FIELD_NAMES, reference),
                    "evaluation_equations": "provided_reference_equations",
                    "reference_over_time": reference_errors_over_time(
                        model, xy, TIME_END, FIELD_NAMES,
                        lambda points, times: exact_reference(points, times, params))})
    save_results(args, config, model, history, xy, time, FIELD_NAMES, metrics, reference=reference, notes=[
        "Reference uses the exact coupled sine mode with no advection, source, or relaxation; gamma0=0 also remains valid.",
        "Held-out residuals use the provided equations, independently of student_equations. These are local practice metrics, not an official competition score.",
        "Original baseline gamma0=0. Compare gamma0=.5 as a separate local experiment; submission checks the stated baseline."])


if __name__ == "__main__":
    try:
        main()
    except NotImplementedError as exc:
        raise SystemExit(str(exc))
