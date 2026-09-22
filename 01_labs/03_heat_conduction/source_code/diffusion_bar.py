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

"""Lab 3: steady heat conduction across two materials, with soft BC/interface losses."""
import sys
from pathlib import Path
import torch
from sympy import Function, Symbol
from physicsnemo.sym.eq.pde import PDE
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from ETC.runtime.labs import parser, setup, mlp, derivative, informer, optimize, save_run

D2, TA, TC = 0.1, 0.0, 100.0


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
        inputs = torch.cat((x - 1, (d1 - 15) / 10), dim=1) if self.parameterized else x - 1
        return 100 * self.left(inputs), 100 * self.right(inputs)


def analytical(x, d1=10.0):
    tb = (TC + (d1 / D2) * TA) / (1 + d1 / D2)
    return torch.where(x <= 1, x * tb + (1 - x) * TA,
                       (x - 1) * TC + (2 - x) * tb)


def loss_terms(model, physics, batch_size, device):
    d1 = 5 + 20 * torch.rand(batch_size, 1, device=device) if model.parameterized else torch.full((batch_size, 1), 10.0, device=device)
    xl = torch.rand(batch_size, 1, device=device, requires_grad=True)
    xr = (1 + torch.rand(batch_size, 1, device=device)).requires_grad_()
    ul, _ = model(xl, d1)
    _, ur = model(xr, d1)
    rl = physics[0].forward({"coordinates": xl, "u_1": ul, "D1": d1})["diffusion"]
    rr = physics[1].forward({"coordinates": xr, "u_2": ur})["diffusion"]
    xi = torch.ones_like(xl, requires_grad=True)
    ui, vi = model(xi, d1)
    interface = physics[2].forward({"coordinates": xi, "u_1": ui, "u_2": vi, "D1": d1})
    at_left, _ = model(torch.zeros_like(xl), d1)
    _, at_right = model(torch.full_like(xr, 2), d1)
    return {"physics": (rl / (100 * d1)).square().mean() + (rr / (100 * D2)).square().mean(),
            "boundary": ((at_left - TA) / 100).square().mean() + ((at_right - TC) / 100).square().mean(),
            "interface_temperature": (interface["temperature_jump"] / 100).square().mean(),
            "interface_flux": (interface["flux_jump"] / (100 * d1)).square().mean()}


def evaluate(model, physics, device):
    errors, residual_values, objectives = [], [], []
    for d in ([7.5, 17.5, 22.5] if model.parameterized else [10.0]):
        xl = torch.linspace(.013, .987, 71, device=device)[:, None].requires_grad_()
        xr = (xl.detach() + 1).requires_grad_()
        dl, dr = torch.full_like(xl, d), torch.full_like(xr, d)
        ul, _ = model(xl, dl)
        _, ur = model(xr, dr)
        errors.extend((ul - analytical(xl, d), ur - analytical(xr, d)))
        rl = physics[0].forward({"coordinates": xl, "u_1": ul, "D1": dl})["diffusion"]
        rr = physics[1].forward({"coordinates": xr, "u_2": ur})["diffusion"]
        residual_values.extend((rl, rr))
        xi = torch.ones(1, 1, device=device, requires_grad=True)
        di = torch.full_like(xi, d)
        ui, vi = model(xi, di)
        jumps = physics[2].forward({"coordinates": xi, "u_1": ui, "u_2": vi, "D1": di})
        at_left, _ = model(torch.zeros_like(xi), di)
        _, at_right = model(torch.full_like(xi, 2), di)
        objective = (rl / (100 * d)).square().mean() + (rr / (100 * D2)).square().mean()
        objective = objective + ((at_left - TA) / 100).square().mean() + ((at_right - TC) / 100).square().mean()
        objective = objective + (jumps["temperature_jump"] / 100).square().mean() + (jumps["flux_jump"] / (100 * d)).square().mean()
        objectives.append(objective)
    return {"objective": float(torch.stack(objectives).mean().detach()),
            "solution_rmse": float(torch.cat(errors).square().mean().sqrt().detach()),
            "pde_rmse": float(torch.cat(residual_values).square().mean().sqrt().detach())}


def main(parameterized=False):
    filename = "config_param.yaml" if parameterized else "config.yaml"
    p = parser(__doc__, Path(__file__).parent / "conf" / filename)
    p.set_defaults(output_dir=Path("outputs/diffusion_bar_parameterized" if parameterized else "outputs/diffusion_bar"))
    args = p.parse_args()
    cfg, device = setup(args)
    model = CompositeBar(cfg, parameterized).to(device)
    physics = (informer(Diffusion("u_1", "D1"), device),
               informer(Diffusion("u_2", D2), device), informer(DiffusionInterface(), device))
    heldout_before = evaluate(model, physics, device)
    history = optimize(model, lambda: loss_terms(model, physics, cfg["batch_size"], device), cfg)
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
    save_run(args, cfg, model, history, {"x": x, "D1": dvals, "prediction": pred, "reference": exact},
             {"problem": "parameterized_composite_bar" if parameterized else "composite_bar",
              "heldout_before": heldout_before, "heldout_after": evaluate(model, physics, device),
              "validation_rmse": float((pred - exact).square().mean().sqrt()),
              "temperature_jumps": jumps, "physical_flux_jumps": flux_jumps,
              "parameter_training_range": [5, 25] if parameterized else None}, plot)


if __name__ == "__main__":
    main()
