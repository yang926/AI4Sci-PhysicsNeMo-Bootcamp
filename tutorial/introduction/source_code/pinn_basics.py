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

"""Lab 1: forward, parameterized and inverse PINNs for u_xx = f(x)."""
import math
import sys
from pathlib import Path
import torch
from sympy import Function, Symbol
from physicsnemo.sym.eq.pde import PDE
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from tutorial.runtime import parser, setup, mlp, informer, optimize, save_run


class Poisson1D(PDE):
    def __init__(self, inverse=False):
        self.dim = 1
        x = Symbol("x")
        u = Function("u")(x)
        f = Function("f")(x) if inverse else 1
        self.equations = {"poisson": u.diff(x, 2) - f}


class BasicPINN(torch.nn.Module):
    def __init__(self, cfg, mode="forward"):
        super().__init__()
        self.mode = mode
        self.solution = mlp(2 if mode == "parameterized" else 1, 1, cfg)
        if mode == "inverse":
            self.source = mlp(1, 1, cfg)

    def forward(self, x, length=None):
        inputs = torch.cat((x, length), dim=1) if self.mode == "parameterized" else x
        return self.solution(inputs)


def analytical(x, length=1.0):
    return 0.5 * (x - length) * x


def inverse_reference(x):
    u = (8 * x * (x**2 - 1) - 3 * torch.sin(4 * math.pi * x) / math.pi**2) / 48
    return u, x + torch.sin(4 * math.pi * x)


def loss_terms(model, physics, batch_size, device, observations=None):
    length = 1 + torch.rand(batch_size, 1, device=device) if model.mode == "parameterized" else torch.ones(batch_size, 1, device=device)
    # Sampling density p(l,x)=1/l requires weight l for the domain integral.
    x = (length * torch.rand(batch_size, 1, device=device)).requires_grad_()
    u = model(x, length)
    variables = {"coordinates": x, "u": u}
    if model.mode == "inverse":
        variables["f"] = model.source(x)
    residual = physics.forward(variables)["poisson"]
    terms = {"physics": (length * residual.square()).mean(),
             "boundary": model(torch.zeros_like(x), length).square().mean()
                         + model(length, length).square().mean()}
    if observations is not None:
        ox, ou = observations
        terms["data"] = 100 * (model(ox) - ou).square().mean()
    return terms


def evaluate(model, physics, device):
    length = 1.5 if model.mode == "parameterized" else 1.0
    x = torch.linspace(.013, length - .013, 91, device=device)[:, None].requires_grad_()
    u = model(x, torch.full_like(x, length))
    target = inverse_reference(x)[0] if model.mode == "inverse" else analytical(x, length)
    variables = {"coordinates": x, "u": u}
    if model.mode == "inverse":
        variables["f"] = model.source(x)
    residual = physics.forward(variables)["poisson"]
    edge_l = torch.full((1, 1), length, device=device)
    objective = length * residual.square().mean()
    objective = objective + model(torch.zeros_like(edge_l), edge_l).square().mean() + model(edge_l, edge_l).square().mean()
    if model.mode == "inverse":
        objective = objective + 100 * (u - target).square().mean()
    result = {"objective": float(objective.detach()),
              "solution_rmse": float((u - target).square().mean().sqrt().detach()),
              "pde_rmse": float(residual.square().mean().sqrt().detach())}
    if model.mode == "inverse":
        result["source_rmse"] = float((variables["f"] - inverse_reference(x)[1]).square().mean().sqrt().detach())
    return result


def main():
    p = parser(__doc__)
    p.add_argument("--mode", choices=("forward", "parameterized", "inverse"), default="forward")
    p.set_defaults(output_dir=Path("outputs/pinn_basics"))
    args = p.parse_args()
    cfg, device = setup(args)
    model = BasicPINN(cfg, args.mode).to(device)
    physics = informer(Poisson1D(args.mode == "inverse"), device)
    ox = torch.rand(100, 1, device=device)
    observations = (ox, inverse_reference(ox)[0]) if args.mode == "inverse" else None
    heldout_before = evaluate(model, physics, device)
    history = optimize(model, lambda: loss_terms(model, physics, cfg["batch_size"], device, observations), cfg)
    length = 1.5 if args.mode == "parameterized" else 1.0
    x = torch.linspace(0, length, 201, device=device)[:, None]
    with torch.no_grad():
        pred = model(x, torch.full_like(x, length))
        exact = inverse_reference(x)[0] if args.mode == "inverse" else analytical(x, length)
        arrays = {"x": x, "prediction": pred, "reference": exact}
        metrics = {"problem": "pinn_basics", "heldout_before": heldout_before, "mode": args.mode, "validation_length": length,
                   "validation_rmse": float((pred - exact).square().mean().sqrt())}
        if args.mode == "inverse":
            arrays.update(source_prediction=model.source(x), source_reference=inverse_reference(x)[1])
            metrics["source_rmse"] = float((arrays["source_prediction"] - arrays["source_reference"]).square().mean().sqrt())
    metrics["heldout_after"] = evaluate(model, physics, device)
    def plot(plt, a):
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(a["x"], a["reference"], label="analytical u")
        ax.plot(a["x"], a["prediction"], label="PINN u")
        ax.set(xlabel="x", ylabel="u", title=args.mode)
        ax.legend()
        return fig
    save_run(args, cfg, model, history, arrays, metrics, plot)


if __name__ == "__main__":
    main()
