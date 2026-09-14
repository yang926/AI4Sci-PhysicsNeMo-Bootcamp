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

"""Lab 4: periodic 2-D incompressible flow, not a validated weather forecast."""
import math
import sys
from pathlib import Path
import numpy as np
import torch
from sympy import Function, Symbol
from physicsnemo.sym.eq.pde import PDE
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from tutorial.runtime import parser, setup, mlp, derivative, informer, optimize, save_run

LENGTH = 1.440
LOWER = -0.720
LENGTH_SCALE = 12742000 / LENGTH
TIME_SCALE = 60 * 60 * 60
VELOCITY_SCALE = LENGTH_SCALE / TIME_SCALE
PRESSURE_SCALE = 1.1614 * VELOCITY_SCALE**2
LEGACY_PRESSURE_FACTOR = 0.10197
REAL_NU = 1.655e-5 / (LENGTH_SCALE**2 / TIME_SCALE)


class NavierStokes(PDE):
    """The constant-density, 2-D incompressible equations used by this lab.

    PhysicsNeMo 2.2.2 exposes the PDE base class rather than the legacy
    physicsnemo.sym.eq.pdes.navier_stokes module, so write the equations directly.
    """
    def __init__(self, nu, rho=1.0, dim=2, time=True):
        if dim != 2 or not time:
            raise ValueError("This tutorial defines the unsteady 2-D equations only")
        self.dim = 2
        x, y, t = Symbol("x"), Symbol("y"), Symbol("t")
        u, v, p = (Function(name)(x, y, t) for name in ("u", "v", "p"))
        self.equations = {
            "continuity": u.diff(x) + v.diff(y),
            "momentum_x": u.diff(t) + u * u.diff(x) + v * u.diff(y)
                          + p.diff(x) / rho - nu * (u.diff(x, 2) + u.diff(y, 2)),
            "momentum_y": v.diff(t) + u * v.diff(x) + v * v.diff(y)
                          + p.diff(y) / rho - nu * (v.diff(x, 2) + v.diff(y, 2)),
        }


def read_wf_data(velocity_scale=VELOCITY_SCALE, pressure_scale=PRESSURE_SCALE, data_path=None):
    """Preserve upstream tiled input coordinates and normalization, including its
    unvalidated 0.10197 pressure factor. See DATA_PROVENANCE.md before interpreting units.
    """
    path = Path(data_path) if data_path else Path(__file__).resolve().parents[1] / "data_lat.npy"
    if not path.is_file():
        raise FileNotFoundError(f"Missing original data: {path}. Use --smoke-data only for a synthetic execution check.")
    ic = np.load(path, allow_pickle=False).astype(np.float32)
    if ic.ndim != 3 or ic.shape[0] != 3 or not np.isfinite(ic).all():
        raise ValueError("Expected finite upstream data shaped (3, H, W) for u, v, p")
    mesh_y, mesh_x = np.meshgrid(np.linspace(-.720, .719, ic.shape[1]),
                                np.linspace(-.720, .719, ic.shape[2]), indexing="ij")
    xy = np.column_stack((mesh_x.ravel(), mesh_y.ravel())).astype(np.float32)
    fields = np.column_stack((ic[0].ravel() / velocity_scale,
                              ic[1].ravel() / velocity_scale,
                              ic[2].ravel() * LEGACY_PRESSURE_FACTOR / pressure_scale))
    return xy, fields.astype(np.float32)


class PeriodicFlow(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.network = mlp(5, 3, cfg)

    def forward(self, xy, t):
        phase = 2 * math.pi * (xy - LOWER) / LENGTH
        # Periodic feature map replaces the retired architecture periodicity keyword.
        return self.network(torch.cat((phase.sin(), phase.cos(), t), dim=1))


def taylor_green(xy, t, nu=0.01):
    """Exact periodic incompressible solution used only for synthetic smoke checks."""
    k = 2 * math.pi / LENGTH
    x, y = k * (xy[:, :1] - LOWER), k * (xy[:, 1:] - LOWER)
    decay = torch.exp(-2 * nu * k**2 * t)
    u = -torch.cos(x) * torch.sin(y) * decay
    v = torch.sin(x) * torch.cos(y) * decay
    p = -.25 * (torch.cos(2 * x) + torch.cos(2 * y)) * decay**2
    return torch.cat((u, v, p), dim=1)


def residuals(field, xy, t, physics):
    u, v, p = field.split(1, dim=1)
    return physics.forward({"coordinates": xy, "u": u, "v": v, "p": p,
                            "u__t": derivative(u, t), "v__t": derivative(v, t)})


def loss_terms(model, physics, batch_size, device, initial_data=None):
    xy = (LOWER + LENGTH * torch.rand(batch_size, 2, device=device)).requires_grad_()
    t = torch.rand(batch_size, 1, device=device, requires_grad=True)
    res = residuals(model(xy, t), xy, t, physics)
    if initial_data is None:
        ix = LOWER + LENGTH * torch.rand(batch_size, 2, device=device)
        target = taylor_green(ix, torch.zeros(batch_size, 1, device=device))
    else:
        coords, values = initial_data
        index = torch.randint(len(coords), (batch_size,))
        ix, target = coords[index].to(device), values[index].to(device)
    prediction = model(ix, torch.zeros(batch_size, 1, device=device))
    return {"physics": sum(v.square().mean() for v in res.values()),
            "initial_data": (prediction - target).square().mean()}


def evaluate(model, physics, device, synthetic, initial_data=None):
    grid = torch.linspace(LOWER + .017, LOWER + LENGTH - .017, 9, device=device)
    xx, yy = torch.meshgrid(grid, grid, indexing="xy")
    xy = torch.stack((xx.ravel(), yy.ravel()), dim=1).requires_grad_()
    t = torch.full_like(xy[:, :1], .37, requires_grad=True)
    pred = model(xy, t)
    res = residuals(pred, xy, t, physics)
    physics_loss = sum(v.square().mean() for v in res.values())
    if synthetic:
        initial_xy = xy.detach()
        initial_target = taylor_green(initial_xy, torch.zeros_like(t))
    else:
        coords, values = initial_data
        index = torch.linspace(0, len(coords) - 1, min(257, len(coords))).long()
        initial_xy, initial_target = coords[index].to(device), values[index].to(device)
    initial_prediction = model(initial_xy, torch.zeros_like(initial_xy[:, :1]))
    objective = physics_loss + (initial_prediction - initial_target).square().mean()
    result = {"objective": float(objective.detach()),
              "pde_rmse": float(torch.cat(list(res.values()), dim=1).square().mean().sqrt().detach())}
    if synthetic:
        result["synthetic_solution_rmse"] = float((pred - taylor_green(xy, t)).square().mean().sqrt().detach())
    return result


def main():
    p = parser(__doc__, Path(__file__).parent / "conf/config.yaml")
    p.add_argument("--smoke-data", action="store_true", help="Use labeled analytic Taylor-Green fixture; no weather data")
    p.add_argument("--data-path", type=Path)
    p.set_defaults(output_dir=Path("outputs/navier_stokes"))
    args = p.parse_args()
    if args.smoke_data and args.data_path is not None:
        p.error("--smoke-data and --data-path are mutually exclusive")
    cfg, device = setup(args)
    nu = 0.01 if args.smoke_data else REAL_NU
    initial_data = None if args.smoke_data else tuple(torch.from_numpy(a) for a in read_wf_data(data_path=args.data_path))
    model = PeriodicFlow(cfg).to(device)
    physics = informer(NavierStokes(nu=nu, rho=1.0, dim=2, time=True), device)
    heldout_before = evaluate(model, physics, device, args.smoke_data, initial_data)
    history = optimize(model, lambda: loss_terms(model, physics, cfg["batch_size"], device, initial_data), cfg)
    axis = torch.linspace(LOWER, LOWER + LENGTH, 33, device=device)[:-1]
    xx, yy = torch.meshgrid(axis, axis, indexing="xy")
    xy = torch.stack((xx.ravel(), yy.ravel()), dim=1)
    # Original tutorial aimed at six-hour snapshots across a sixty-hour scale.
    # 11 points now give exactly 6 hours between frames, including both endpoints.
    times = torch.linspace(0, 1, 11, device=device)
    with torch.no_grad():
        pred = torch.stack([model(xy, torch.full_like(xy[:, :1], t)) for t in times])
        arrays = {"xy": xy, "times": times, "prediction": pred}
        metrics = {"problem": "periodic_2d_navier_stokes", "heldout_before": heldout_before, "nu": nu,
                   "data_kind": "synthetic_taylor_green" if args.smoke_data else "upstream_data_lat_legacy_normalization",
                   "weather_forecast_validated": False,
                   "legacy_pressure_factor": None if args.smoke_data else LEGACY_PRESSURE_FACTOR}
        if args.smoke_data:
            exact = torch.stack([taylor_green(xy, torch.full_like(xy[:, :1], t), nu) for t in times])
            arrays["reference"] = exact
            metrics["synthetic_reference_rmse"] = float((pred - exact).square().mean().sqrt())
    metrics["heldout_after"] = evaluate(model, physics, device, args.smoke_data, initial_data)
    def plot(plt, a):
        fig, ax = plt.subplots(figsize=(6, 5))
        h = ax.scatter(a["xy"][:, 0], a["xy"][:, 1], c=a["prediction"][-1, :, 0], s=10)
        fig.colorbar(h, ax=ax, label="predicted nondimensional u")
        ax.set(xlabel="x", ylabel="y", title=f"{metrics['data_kind']} / t=1 / not weather validation")
        return fig
    save_run(args, cfg, model, history, arrays, metrics, plot)


if __name__ == "__main__":
    main()
