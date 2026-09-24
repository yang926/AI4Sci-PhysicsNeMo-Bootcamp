# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""Lab 3: steady conduction with exact endpoint temperatures and learned interfaces."""
import json
import math
import sys
from pathlib import Path
import torch
from sympy import Function, Symbol
from physicsnemo.sym.eq.pde import PDE
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from ETC.runtime.labs import parser, setup, mlp, derivative, informer, save_run

D2, TA, TC = 0.1, 0.0, 100.0
DEFAULT_STEPS = 300
ACCURACY_LIMITS = {"left_relative_rmse": .05, "right_relative_rmse": .005,
                   "profile_max_abs": .5, "boundary_max_abs": .1,
                   "temperature_jump_abs": .1, "flux_jump_relative": .02,
                   "left_normalized_pde_rmse": .01, "right_normalized_pde_rmse": .01}


class Diffusion(PDE):
    def __init__(self, field="u", conductivity="D1"):
        self.dim = 1
        x = Symbol("x")
        u = Function(field)(x)
        d = Symbol(conductivity) if isinstance(conductivity, str) else conductivity
        self.equations = {"diffusion": -d * u.diff(x, 2)}


class DiffusionInterface(PDE):
    def __init__(self):
        self.dim = 1
        x, d1 = Symbol("x"), Symbol("D1")
        a, b = Function("u_1")(x), Function("u_2")(x)
        self.equations = {"temperature_jump": a - b,
                          "flux_jump": d1 * a.diff(x) - D2 * b.diff(x)}


class CompositeBar(torch.nn.Module):
    def __init__(self, cfg, parameterized=False):
        super().__init__()
        self.parameterized = parameterized
        self.left = mlp(2 if parameterized else 1, 1, cfg)
        self.right = mlp(2 if parameterized else 1, 1, cfg)

    def forward(self, x, d1):
        # Normalize the resistance ratio over D1 in [5, 25]. This is an input
        # feature, not a temperature solution or an interface constraint.
        resistance = (D2 / d1 - .012) / .008
        inputs = torch.cat((x - 1, resistance), dim=1) if self.parameterized else x - 1
        # Only the prescribed endpoints are exact. Both temperature fields,
        # their spatial curvature and both interface conditions remain learned.
        return (TA + x * (100 * D2 / d1) * self.left(inputs),
                TC + (2 - x) * 100 * self.right(inputs))


def analytical(x, d1=10.0):
    tb = (TC + (d1 / D2) * TA) / (1 + d1 / D2)
    return torch.where(x <= 1, x * tb + (1 - x) * TA,
                       (x - 1) * TC + (2 - x) * tb)


def loss_terms(model, physics, batch_size, device, *, points=None):
    dtype = next(model.parameters()).dtype
    if points is None:
        xl = torch.rand(batch_size, 1, device=device, dtype=dtype)
        d1 = (5 + 20 * torch.rand_like(xl) if model.parameterized
              else torch.full_like(xl, 10.0))
    else:
        xl, d1 = points
    xl = xl.detach().requires_grad_()
    xr = (1 + xl.detach()).requires_grad_()
    ul, _ = model(xl, d1)
    _, ur = model(xr, d1)
    rl = physics[0].forward({"coordinates": xl, "u_1": ul, "D1": d1})["diffusion"]
    rr = physics[1].forward({"coordinates": xr, "u_2": ur})["diffusion"]
    xi = torch.ones_like(xl, requires_grad=True)
    ui, vi = model(xi, d1)
    interface = physics[2].forward({"coordinates": xi, "u_1": ui, "u_2": vi, "D1": d1})
    at_left, _ = model(torch.zeros_like(xl), d1)
    _, at_right = model(torch.full_like(xr, 2), d1)
    return {"physics": (rl / (100 * D2)).square().mean() + (rr / (100 * D2)).square().mean(),
            "boundary": ((at_left - TA) / (100 * D2 / d1)).square().mean() + ((at_right - TC) / 100).square().mean(),
            "interface_temperature": (interface["temperature_jump"] / 10).square().mean(),
            "interface_flux": (interface["flux_jump"] / (100 * D2)).square().mean()}


def optimize_lab(model, physics, cfg, device):
    """Adam warmup, then deterministic L-BFGS; references are evaluation-only."""
    dtype = next(model.parameters()).dtype
    if model.parameterized:
        x, d = torch.meshgrid(torch.linspace(0, 1, 17, device=device, dtype=dtype),
                              torch.linspace(5, 25, 17, device=device, dtype=dtype), indexing="ij")
        points = (x.reshape(-1, 1), d.reshape(-1, 1))
    else:
        x = torch.linspace(0, 1, cfg["batch_size"], device=device, dtype=dtype)[:, None]
        points = (x, torch.full_like(x, 10.0))
    adam_steps = min(200, max(1, 2 * cfg["steps"] // 3))
    adam = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    lbfgs = torch.optim.LBFGS(model.parameters(), lr=1, max_iter=20, history_size=50,
                             line_search_fn="strong_wolfe", tolerance_grad=1e-9,
                             tolerance_change=1e-14)
    history = []
    for step in range(1, cfg["steps"] + 1):
        warming_up = step <= adam_steps
        optimizer = adam if warming_up else lbfgs
        evaluations, latest, first_terms = 0, {}, None

        def closure():
            nonlocal evaluations, latest, first_terms
            optimizer.zero_grad(set_to_none=True)
            latest = loss_terms(model, physics, cfg["batch_size"], device,
                                points=None if warming_up else points)
            objective = sum(latest.values())
            if first_terms is None:
                first_terms = {name: value.detach() for name, value in latest.items()}
            # A constant multiplier improves numerical resolution of L-BFGS's
            # stopping tests without changing relative physical loss weights.
            loss = objective if warming_up else 100 * objective
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite training loss at step {step}")
            loss.backward()
            gradients = [p.grad.reshape(-1) for p in model.parameters() if p.grad is not None]
            if not gradients or not torch.isfinite(torch.cat(gradients)).all():
                raise FloatingPointError(f"Missing or nonfinite gradient at step {step}")
            evaluations += 1
            return loss

        if warming_up:
            closure()
            optimizer.step()
        else:
            optimizer.step(closure)
        if not torch.isfinite(torch.cat([p.detach().reshape(-1) for p in model.parameters()])).all():
            raise FloatingPointError(f"Nonfinite model parameter at step {step}")
        values = {name: float(value) for name, value in first_terms.items()}
        row = {"step": step, "phase": "adam" if warming_up else "lbfgs",
               "closure_evaluations": evaluations, "loss": sum(values.values()), **values}
        history.append(row)
        if step == 1 or step == cfg["steps"] or step % 100 == 0:
            print(json.dumps(row), flush=True)
    return history


def evaluation_parameters(parameterized=False):
    return [5 + .5 * i for i in range(41)] if parameterized else [10.0]


def evaluate(model, physics, device):
    errors, residual_values, objectives, records = [], [], [], []
    dtype = next(model.parameters()).dtype
    for d in evaluation_parameters(model.parameterized):
        xl = torch.linspace(0, 1, 201, device=device, dtype=dtype)[:, None].requires_grad_()
        xr = (xl.detach() + 1).requires_grad_()
        dl, dr = torch.full_like(xl, d), torch.full_like(xr, d)
        ul, _ = model(xl, dl)
        _, ur = model(xr, dr)
        el, er = (ul - analytical(xl, d)).detach(), (ur - analytical(xr, d)).detach()
        errors.extend((el, er))
        rl = physics[0].forward({"coordinates": xl, "u_1": ul, "D1": dl})["diffusion"]
        rr = physics[1].forward({"coordinates": xr, "u_2": ur})["diffusion"]
        residual_values.extend((rl.detach(), rr.detach()))
        xi = torch.ones(1, 1, device=device, dtype=dtype, requires_grad=True)
        di = torch.full_like(xi, d)
        ui, vi = model(xi, di)
        jumps = physics[2].forward({"coordinates": xi, "u_1": ui, "u_2": vi, "D1": di})
        at_left, _ = model(torch.zeros_like(xi), di)
        _, at_right = model(torch.full_like(xi, 2), di)
        objective = (rl / (100 * D2)).square().mean() + (rr / (100 * D2)).square().mean()
        objective = objective + ((at_left - TA) / (100 * D2 / d)).square().mean() + ((at_right - TC) / 100).square().mean()
        objective = objective + (jumps["temperature_jump"] / 10).square().mean() + (jumps["flux_jump"] / (100 * D2)).square().mean()
        objectives.append(objective.detach())
        tb = float(analytical(torch.ones(1), d))
        left_rmse, right_rmse = float(el.square().mean().sqrt()), float(er.square().mean().sqrt())
        records.append({"D1": d, "left_rmse": left_rmse, "right_rmse": right_rmse,
                        "left_relative_rmse": left_rmse / (tb - TA),
                        "right_relative_rmse": right_rmse / (TC - tb),
                        "left_normalized_pde_rmse": float((rl / (100 * D2)).square().mean().sqrt().detach()),
                        "right_normalized_pde_rmse": float((rr / (100 * D2)).square().mean().sqrt().detach()),
                        "profile_max_abs": float(torch.cat((el, er)).abs().max()),
                        "boundary_max_abs": float(torch.cat((at_left - TA, at_right - TC)).abs().max().detach()),
                        "temperature_jump_abs": float(jumps["temperature_jump"].abs().max().detach()),
                        "flux_jump_abs": float(jumps["flux_jump"].abs().max().detach()),
                        "flux_jump_relative": float(jumps["flux_jump"].abs().max().detach()) / (D2 * (TC - tb))})
    return {"objective": float(torch.stack(objectives).mean().detach()),
            "solution_rmse": float(torch.cat(errors).square().mean().sqrt().detach()),
            "pde_rmse": float(torch.cat(residual_values).square().mean().sqrt().detach()),
            "per_conductivity": records}


def accuracy_checks(metrics, parameterized=False):
    """Require every material/parameter check, including both range endpoints."""
    checks = []
    records = metrics.get("per_conductivity", [])
    if not isinstance(records, list) or any(not isinstance(row, dict) for row in records):
        records = []
    expected = evaluation_parameters(parameterized)
    coverage = (len(records) == len(expected) and
                all(type(row.get("D1")) in (int, float) and math.isfinite(row["D1"]) for row in records) and
                sorted(row.get("D1", float("nan")) for row in records) == expected)
    checks.append({"metric": "parameter_coverage", "value": len(records),
                   "limit": len(expected), "passed": coverage})
    for d in expected:
        found = [row for row in records if row.get("D1") == d]
        row = found[0] if len(found) == 1 else {}
        for metric, limit in ACCURACY_LIMITS.items():
            value = row.get(metric)
            passed = (type(value) in (int, float) and math.isfinite(value) and 0 <= value <= limit)
            checks.append({"D1": d, "metric": metric, "value": value,
                           "limit": limit, "passed": passed})
    return {"passed": all(check["passed"] for check in checks), "checks": checks,
            "scope": "201 points per material; D1=5:0.5:25" if parameterized
                     else "201 points per material; D1=10"}


def main(parameterized=False):
    filename = "config_param.yaml" if parameterized else "config.yaml"
    p = parser(__doc__, Path(__file__).parent / "conf" / filename)
    p.set_defaults(output_dir=Path("outputs/diffusion_bar_parameterized" if parameterized else "outputs/diffusion_bar"))
    args = p.parse_args()
    cfg, device = setup(args, defaults={"steps": DEFAULT_STEPS})
    cfg["lab3_parameterized"] = parameterized
    cfg["lab3_dtype"] = "float32"
    model = CompositeBar(cfg, parameterized).to(device=device, dtype=torch.float32)
    physics = (informer(Diffusion("u_1", "D1"), device),
               informer(Diffusion("u_2", D2), device), informer(DiffusionInterface(), device))
    heldout_before = evaluate(model, physics, device)
    history = optimize_lab(model, physics, cfg, device)
    x = torch.linspace(0, 2, 201, device=device)[:, None]
    dvals = [5.0, 10.0, 25.0] if parameterized else [10.0]
    predictions, references, jumps, flux_jumps = [], [], [], []
    for d in dvals:
        d1 = torch.full_like(x, d)
        with torch.no_grad():
            left, right = model(x, d1)
            predictions.append(torch.where(x <= 1, left, right))
            references.append(analytical(x, d))
        xi = torch.ones(1, 1, device=device, requires_grad=True)
        left, right = model(xi, torch.full_like(xi, d))
        jumps.append(float((left - right).detach()))
        flux_jumps.append(float((d * derivative(left, xi) - D2 * derivative(right, xi)).detach()))
    pred, exact = torch.stack(predictions), torch.stack(references)
    def plot(plt, a):
        fig, ax = plt.subplots(figsize=(8, 4))
        for i, d in enumerate(dvals):
            ax.plot(a["x"], a["reference"][i, :, 0], "--", label=f"exact D1={d:g}")
            ax.plot(a["x"], a["prediction"][i, :, 0], label=f"PINN D1={d:g}")
        ax.axvline(1, color="grey", alpha=.4)
        ax.set(xlabel="x", ylabel="temperature")
        ax.legend()
        return fig
    heldout_after = evaluate(model, physics, device)
    save_run(args, cfg, model, history, {"x": x, "D1": torch.tensor(dvals, device=device, dtype=torch.float32), "prediction": pred, "reference": exact},
             {"problem": "parameterized_composite_bar" if parameterized else "composite_bar",
              "heldout_before": heldout_before, "heldout_after": heldout_after,
              "accuracy": accuracy_checks(heldout_after, parameterized), "training_dtype": "float32",
              "training_recipe": {"optimizer": "adam_then_lbfgs", "step_unit": "optimizer_calls", "dtype": "float32",
                                  "adam_steps": min(200, max(1, 2 * cfg["steps"] // 3)),
                                  "lbfgs_max_iter_per_call": 20,
                                  "closure_evaluations": sum(row["closure_evaluations"] for row in history)},
              "validation_rmse": float((pred - exact).square().mean().sqrt()),
              "temperature_jumps": jumps, "physical_flux_jumps": flux_jumps,
              "parameter_training_range": [5, 25] if parameterized else None}, plot)


if __name__ == "__main__":
    main()
