"""Climate Level 1: simple atmosphere ADR PINN.

This is a simplified temperature PDE, not FourCastNet weather forecasting.
The original diffusion-only validation case is the default. Other settings need separate validation.
"""
from pathlib import Path
import math
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from sympy import symbols, Function
from physicsnemo.sym.eq.pde import PDE
from pinn_runtime import (parse_args, create_model, create_informer, residuals,
    evaluate_fields, sample_square, sample_time, evaluation_grid, save_results,
    record_step, heldout_losses, heldout_row, reference_errors)

LEVEL = 1
FIELD_NAMES = ['T']
TIME_END = 2 * math.pi
DEFAULT_PHYSICS = {'u0': 0.0, 'v0': 0.0, 'kappa': 1.0, 'lam': 0.0, 'Q0': 0.0, 'Teq': 0.0}


def reference_equations(x, y, t, fields, params):
    T = fields["T"]
    return {"adr": T.diff(t) + params["u0"]*T.diff(x) + params["v0"]*T.diff(y)
            - params["kappa"]*(T.diff(x, 2)+T.diff(y, 2)) - params["Q0"]
            + params["lam"]*(T-params["Teq"])}


def student_equations(x, y, t, fields, params):
    # FIXME: write the ADR residual.
    raise NotImplementedError("Complete student_equations in {} and save, or use --reference to run the completed PDE.".format(Path(__file__).name))


class ClimatePDE(PDE):
    def __init__(self, reference=False, params=None):
        self.dim = 2
        params = DEFAULT_PHYSICS.copy() if params is None else {**DEFAULT_PHYSICS, **params}
        if not all(math.isfinite(float(v)) for v in params.values()):
            raise ValueError("Physics parameters must be finite")
        for name, value in params.items():
            if name.startswith("kappa") and value <= 0:
                raise ValueError("Diffusivity must be positive")
        x, y, t = symbols("x y t")
        fields = {name: Function(name)(x, y, t) for name in FIELD_NAMES}
        self.equations = (reference_equations if reference else student_equations)(x, y, t, fields, params)


def exact_reference(coordinates, time, params):
    # This formula is verified only for the original no-advection/no-source case.
    if any(params[name] != 0 for name in ['u0', 'v0', 'lam', 'Q0']):
        return None
    initial = torch.sin(coordinates[:, :1]) * torch.sin(coordinates[:, 1:2])
    return initial * torch.exp(-2 * params["kappa"] * time)


def loss_terms(model, informer, config, device):
    count = config["samples"]
    xy = sample_square(count["interior"], device)
    pde = residuals(model, informer, xy, sample_time(len(xy), TIME_END, device), FIELD_NAMES)
    losses = {"pde_" + name: value.square().mean() for name, value in pde.items()}
    initial_xy = sample_square(count["initial"], device)
    fields = evaluate_fields(model, initial_xy, torch.zeros(len(initial_xy), 1, device=device), FIELD_NAMES)
    target = torch.sin(initial_xy[:, :1]) * torch.sin(initial_xy[:, 1:2])
    for name, value in fields.items():
        losses["initial_" + name] = (value - target).square().mean()
    boundary_xy = sample_square(count["boundary"], device, boundary=True)
    boundary = evaluate_fields(model, boundary_xy, sample_time(len(boundary_xy), TIME_END, device), FIELD_NAMES)
    for name, value in boundary.items():
        losses["boundary_" + name] = value.square().mean()
    return losses


def main():
    args, config = parse_args(__file__, __doc__, "config_atmos.yaml")
    params = {**DEFAULT_PHYSICS, **config.get("physics", {})}
    pde = ClimatePDE(reference=args.reference, params=params)
    informer = create_informer(pde, args.device)
    model = create_model(3, len(FIELD_NAMES), config, args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])
    if args.output_dir.exists():
        raise FileExistsError("Choose a new --output-dir; previous results are preserved.")
    initial = heldout_losses(loss_terms, model, informer, config, args.device, args.seed)
    xy, time = evaluation_grid(args.device, TIME_END)
    reference = exact_reference(xy, time, params)
    before_error = reference_errors(model, xy, time, FIELD_NAMES, reference)
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
    metrics.update({"physics": params, "initial_reference_error": before_error,
                    "final_reference_error": reference_errors(model, xy, time, FIELD_NAMES, reference)})
    save_results(args, config, model, history, xy, time, FIELD_NAMES, metrics, reference=reference, notes=[
        "Reference is the sine-decay solution only when its no-advection/no-source/no-relaxation assumptions hold.",
        'The full configurable ADR equation is retained; the default validation case reduces to diffusion.'])


if __name__ == "__main__":
    try:
        main()
    except NotImplementedError as exc:
        raise SystemExit(str(exc))
