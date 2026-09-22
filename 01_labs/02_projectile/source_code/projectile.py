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

"""Lab 2: projectile PINN with the PhysicsNeMo 2.2.2 model and symbolic PDE API."""
import math
import sys
from pathlib import Path

import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from ETC.runtime.labs import parser, setup, mlp, derivative, informer, optimize, save_run
from projectile_eqn import ProjectileEquation


class ProjectileModel(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.network = mlp(1, 2, cfg)

    def forward(self, t):
        return 100.0 * self.network(2.0 * t / 5.0 - 1.0)


def analytical(t):
    return torch.cat((20.0 * t, 40.0 * math.sin(math.pi / 3) * t - 4.905 * t**2), dim=1)


def residuals(field, t, physics):
    x, y = field[:, :1], field[:, 1:]
    # PhysicsInformer differentiates spatial x/y/z; supply the time derivatives.
    return physics.forward({"coordinates": t, "x": x, "y": y,
                            "x__t__t": derivative(derivative(x, t), t),
                            "y__t__t": derivative(derivative(y, t), t)})


def loss_terms(model, physics, batch_size, device):
    t = (5.0 * torch.rand(batch_size, 1, device=device)).requires_grad_()
    res = residuals(model(t), t, physics)
    t0 = torch.zeros(batch_size, 1, device=device, requires_grad=True)
    xy0 = model(t0)
    vx, vy = derivative(xy0[:, :1], t0), derivative(xy0[:, 1:], t0)
    return {"physics": sum((v / 9.81).square().mean() for v in res.values()),
            "initial_position": (xy0 / 100).square().mean(),
            "initial_velocity": ((vx - 20) / 40).square().mean()
                                + ((vy - 40 * math.sin(math.pi / 3)) / 40).square().mean()}


def evaluate(model, physics, device):
    t = torch.linspace(0.017, 4.983, 101, device=device)[:, None].requires_grad_()
    pred = model(t)
    res = residuals(pred, t, physics)
    t0 = torch.zeros(1, 1, device=device, requires_grad=True)
    initial = model(t0)
    vx, vy = derivative(initial[:, :1], t0), derivative(initial[:, 1:], t0)
    objective = sum((v / 9.81).square().mean() for v in res.values())
    objective = objective + (initial / 100).square().mean()
    objective = objective + ((vx - 20) / 40).square().mean() + ((vy - 40 * math.sin(math.pi / 3)) / 40).square().mean()
    return {"objective": float(objective.detach()),
            "solution_rmse": float((pred - analytical(t)).square().mean().sqrt().detach()),
            "pde_rmse": float(torch.cat(list(res.values()), dim=1).square().mean().sqrt().detach())}


def main():
    p = parser(__doc__, Path(__file__).parent / "conf/config.yaml")
    p.set_defaults(output_dir=Path("outputs/projectile"))
    args = p.parse_args()
    cfg, device = setup(args)
    model = ProjectileModel(cfg).to(device)
    physics = informer(ProjectileEquation(), device)
    heldout_before = evaluate(model, physics, device)
    history = optimize(model, lambda: loss_terms(model, physics, cfg["batch_size"], device), cfg)
    t = torch.linspace(0, 8, 401, device=device)[:, None]
    with torch.no_grad():
        pred, exact = model(t), analytical(t)
    in_domain = t[:, 0] <= 5
    metrics = {"problem": "projectile", "heldout_before": heldout_before,
               "heldout_after": evaluate(model, physics, device), "training_interval": [0, 5],
               "extrapolation_interval": [5, 8],
               "in_domain_rmse": float((pred[in_domain] - exact[in_domain]).square().mean().sqrt()),
               "extrapolation_rmse": float((pred[~in_domain] - exact[~in_domain]).square().mean().sqrt())}
    def plot(plt, a):
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        for j, name in enumerate(("x", "y")):
            ax[j].plot(a["t"], a["reference"][:, j], label="analytical")
            ax[j].plot(a["t"], a["prediction"][:, j], label="PINN")
            ax[j].axvline(5, color="grey", linestyle="--", label="training interval ends")
            ax[j].set(xlabel="time (s)", ylabel=f"{name} (m)")
            ax[j].legend()
        return fig
    save_run(args, cfg, model, history, {"t": t, "prediction": pred, "reference": exact}, metrics, plot)


if __name__ == "__main__":
    main()
