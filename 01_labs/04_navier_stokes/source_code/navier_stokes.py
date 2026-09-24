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
import json
import sys
from pathlib import Path
import numpy as np
import torch
from sympy import Function, Symbol
from physicsnemo.sym.eq.pde import PDE
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from ETC.runtime.labs import parser, setup, mlp, derivative, informer, save_run

LENGTH = 1.440
LOWER = -0.720
LENGTH_SCALE = 12742000 / LENGTH
TIME_SCALE = 60 * 60 * 60
VELOCITY_SCALE = LENGTH_SCALE / TIME_SCALE
PRESSURE_SCALE = 1.1614 * VELOCITY_SCALE**2
LEGACY_PRESSURE_FACTOR = 0.10197
REAL_NU = 1.655e-5 / (LENGTH_SCALE**2 / TIME_SCALE)
INITIAL_DATA_WEIGHT = 10.0
EVALUATION_TIMES = (0.0, 0.13, 0.37, 0.61, 0.83, 1.0)


class NavierStokes(PDE):
    """The constant-density, 2-D incompressible equations used by this lab.

    PhysicsNeMo 2.2.2 exposes the PDE base class rather than the legacy
    physicsnemo.sym.eq.pdes.navier_stokes module, so write the equations directly.
    """
    def __init__(self, nu, rho=1.0, dim=2, time=True):
        if dim != 2 or not time:
            raise ValueError("This Lab defines the unsteady 2-D equations only")
        if isinstance(nu, bool) or not isinstance(nu, (int, float)) or not math.isfinite(nu) or nu < 0:
            raise ValueError("nu must be finite and nonnegative")
        if isinstance(rho, bool) or not isinstance(rho, (int, float)) or not math.isfinite(rho) or rho <= 0:
            raise ValueError("rho must be finite and positive")
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
    for name, value in (("velocity_scale", velocity_scale), ("pressure_scale", pressure_scale)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    path = Path(data_path) if data_path else Path(__file__).resolve().parents[1] / "data_lat.npy"
    if not path.is_file():
        raise FileNotFoundError(f"Missing original data: {path}. Use --smoke-data only for a synthetic execution check.")
    ic = np.load(path, allow_pickle=False).astype(np.float32)
    if ic.ndim != 3 or ic.shape[0] != 3 or not ic.size or not np.isfinite(ic).all():
        raise ValueError("Expected finite upstream data shaped (3, H, W) for u, v, p")
    mesh_y, mesh_x = np.meshgrid(np.linspace(-.720, .719, ic.shape[1]),
                                np.linspace(-.720, .719, ic.shape[2]), indexing="ij")
    xy = np.column_stack((mesh_x.ravel(), mesh_y.ravel())).astype(np.float32)
    fields = np.column_stack((ic[0].ravel() / velocity_scale,
                              ic[1].ravel() / velocity_scale,
                              ic[2].ravel() * LEGACY_PRESSURE_FACTOR / pressure_scale))
    if not np.isfinite(fields).all():
        raise ValueError("Normalized initial fields must remain finite; check the data scales")
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
    """Exact synthetic reference: training may use its t=0 initial data only."""
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


def training_points(batch_size, device, initial_data=None, *, initial_size=None):
    """Sample the PDE domain and initial observations; no future solution labels."""
    initial_size = batch_size if initial_size is None else initial_size
    xy = LOWER + LENGTH * torch.rand(batch_size, 2, device=device, dtype=torch.float32)
    t = torch.rand(batch_size, 1, device=device, dtype=torch.float32)
    if initial_data is None:
        ix = LOWER + LENGTH * torch.rand(initial_size, 2, device=device, dtype=torch.float32)
        target = taylor_green(ix, torch.zeros_like(ix[:, :1]))
    else:
        coords, values = initial_data
        index = torch.randint(len(coords), (initial_size,))
        ix, target = coords[index].to(device), values[index].to(device)
    return xy, t, ix, target


def loss_terms(model, physics, batch_size, device, initial_data=None, *, points=None):
    points = training_points(batch_size, device, initial_data) if points is None else points
    raw_xy, raw_t, ix, target = points
    xy, t = raw_xy.detach().requires_grad_(), raw_t.detach().requires_grad_()
    res = residuals(model(xy, t), xy, t, physics)
    prediction = model(ix, torch.zeros_like(ix[:, :1]))
    return {"physics": sum(v.square().mean() for v in res.values()),
            "initial_data": INITIAL_DATA_WEIGHT * (prediction - target).square().mean()}


def optimize_lab(model, physics, cfg, device, initial_data=None):
    """Stochastic Adam followed by deterministic full-batch L-BFGS in FP32.

    One recorded step is one optimizer.step call. Each line-search closure is
    counted separately. L-BFGS uses fixed samples so all trial losses compare
    the same objective; the held-out grids never select or train the model.
    """
    adam_steps = min(1000, cfg["steps"] // 3)
    adam = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    lbfgs = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=1, max_eval=20,
                            history_size=50, line_search_fn="strong_wolfe",
                            tolerance_grad=1e-9, tolerance_change=1e-12)
    fixed = None
    history = []
    for step in range(1, cfg["steps"] + 1):
        phase = "adam" if step <= adam_steps else "lbfgs"
        optimizer = adam if phase == "adam" else lbfgs
        if phase == "lbfgs" and fixed is None:
            fixed = training_points(max(2048, cfg["batch_size"]), device, initial_data,
                                    initial_size=max(1024, cfg["batch_size"]))
        first_terms, evaluations = None, 0

        def closure():
            nonlocal first_terms, evaluations
            optimizer.zero_grad(set_to_none=True)
            terms = loss_terms(model, physics, cfg["batch_size"], device, initial_data,
                               points=fixed if phase == "lbfgs" else None)
            loss = sum(terms.values())
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite {phase} loss at step {step}")
            loss.backward()
            gradients = [p.grad for p in model.parameters() if p.grad is not None]
            if not gradients or not torch.isfinite(torch.cat([g.detach().reshape(-1) for g in gradients])).all():
                raise FloatingPointError(f"Missing or nonfinite {phase} gradients at step {step}")
            if first_terms is None:
                values = torch.stack([loss, *terms.values()]).detach().cpu().tolist()
                first_terms = dict(zip(("loss", *terms), values))
            evaluations += 1
            return loss

        if phase == "adam":
            closure()
            optimizer.step()
        else:
            optimizer.step(closure)
        if not torch.isfinite(torch.cat([p.detach().reshape(-1) for p in model.parameters()])).all():
            raise FloatingPointError(f"Nonfinite parameters at step {step}")
        row = {"step": step, "phase": phase, "closure_evaluations": evaluations, **first_terms}
        history.append(row)
        if step == 1 or step == cfg["steps"] or step % 500 == 0:
            print(json.dumps(row), flush=True)
    return history


def solution_errors(prediction, reference):
    """Separate velocity accuracy from pressure's spatially constant gauge.

    Inputs have shape [..., spatial_points, 3]. Align pressure independently at
    each time by subtracting the spatial mean error, not the velocity or the
    spatial pressure variation. Keep raw pressure error visible as a diagnostic.
    """
    if prediction.shape != reference.shape or prediction.ndim < 2 or prediction.shape[-1] != 3:
        raise ValueError("Expected matching [..., spatial_points, 3] velocity-pressure fields")
    if not prediction.numel() or not torch.isfinite(prediction).all() or not torch.isfinite(reference).all():
        raise ValueError("Velocity-pressure fields must be nonempty and finite")
    error = prediction - reference
    pressure_error = error[..., 2:3]
    pressure_offset = pressure_error.mean(dim=-2, keepdim=True)
    values = {
        "velocity_rmse": error[..., :2].square().mean().sqrt(),
        "pressure_rmse": pressure_error.square().mean().sqrt(),
        "pressure_gauge_aligned_rmse": (pressure_error - pressure_offset).square().mean().sqrt(),
        "pressure_offset_rmse": pressure_offset.square().mean().sqrt(),
    }
    return {name: float(value.detach()) for name, value in values.items()}


def periodic_errors(model, device, time):
    """Opposite-edge values and spatial first derivatives at 33 edge points."""
    line = torch.linspace(LOWER, LOWER + LENGTH, 33, device=device, dtype=torch.float32)
    value_error, gradient_error = [], []
    for axis in (0, 1):
        low = torch.stack((torch.full_like(line, LOWER), line), dim=1)
        high = torch.stack((torch.full_like(line, LOWER + LENGTH), line), dim=1)
        if axis == 1:
            low, high = low.flip(1), high.flip(1)
        low, high = low.requires_grad_(), high.requires_grad_()
        a, b = model(low, torch.full_like(low[:, :1], time)), model(high, torch.full_like(high[:, :1], time))
        value_error.append((a - b).abs().max())
        for column in range(3):
            gradient_error.append((derivative(a[:, column:column + 1], low)
                                   - derivative(b[:, column:column + 1], high)).abs().max())
    return {"periodic_value_max_abs": float(torch.stack(value_error).max().detach()),
            "periodic_gradient_max_abs": float(torch.stack(gradient_error).max().detach())}


def evaluate(model, physics, device, synthetic, initial_data=None):
    """Held-out 24x24 cell-center grid at six times, including both endpoints."""
    grid = LOWER + LENGTH * (torch.arange(24, device=device, dtype=torch.float32) + .5) / 24
    xx, yy = torch.meshgrid(grid, grid, indexing="xy")
    base = torch.stack((xx.ravel(), yy.ravel()), dim=1)
    records = []
    for time in EVALUATION_TIMES:
        xy = base.detach().requires_grad_()
        t = torch.full_like(xy[:, :1], time, requires_grad=True)
        pred = model(xy, t)
        res = residuals(pred, xy, t, physics)
        row = {"time": time,
               "pde_rmse": float(torch.cat(list(res.values()), dim=1).square().mean().sqrt().detach()),
               **{f"{name}_rmse": float(value.square().mean().sqrt().detach())
                  for name, value in res.items()},
               **periodic_errors(model, device, time)}
        if synthetic:
            exact = taylor_green(xy, t)
            row.update({f"synthetic_{name}": value for name, value in solution_errors(pred, exact).items()})
            row["synthetic_solution_rmse"] = float((pred - exact).square().mean().sqrt().detach())
        records.append(row)
    if synthetic:
        initial_xy = base
        initial_target = taylor_green(initial_xy, torch.zeros_like(initial_xy[:, :1]))
    else:
        coords, values = initial_data
        index = torch.linspace(0, len(coords) - 1, min(257, len(coords))).long()
        initial_xy, initial_target = coords[index].to(device), values[index].to(device)
    initial_prediction = model(initial_xy, torch.zeros_like(initial_xy[:, :1]))
    initial_rmse = float((initial_prediction - initial_target).square().mean().sqrt().detach())
    # Equal grid sizes: pool squared errors, never average RMSEs directly.
    result = {name: math.sqrt(sum(row[name] ** 2 for row in records) / len(records))
              for name in records[0] if name.endswith("rmse")}
    result.update({name: max(row[name] for row in records)
                   for name in ("periodic_value_max_abs", "periodic_gradient_max_abs")})
    result.update(objective=3 * result["pde_rmse"] ** 2 + INITIAL_DATA_WEIGHT * initial_rmse ** 2,
                  initial_data_rmse=initial_rmse, per_time=records,
                  scope="24x24 held-out spatial cell centers at six times; not a continuous-domain bound")
    return result


def accuracy_checks(metrics):
    """Fixed Taylor-Green lesson criteria, declared before optimizer tuning."""
    records = metrics.get("per_time", [])
    if (not isinstance(records, list) or len(records) != len(EVALUATION_TIMES)
            or any(not isinstance(row, dict) or type(row.get("time")) not in (int, float)
                   or row["time"] != time for row, time in zip(records, EVALUATION_TIMES))):
        raise ValueError("Accuracy checks require all six evaluation times")
    limits = {"synthetic_velocity_rmse": .02,
              "synthetic_pressure_gauge_aligned_rmse": .02,
              "pde_rmse": .05, "periodic_value_max_abs": 2e-5,
              "periodic_gradient_max_abs": 1e-4}

    def check(row, name, limit):
        value = row.get(name)
        valid = type(value) in (int, float) and math.isfinite(value) and 0 <= value <= limit
        return {"time": row.get("time", 0.0), "metric": name, "value": value,
                "limit": limit, "passed": valid}

    checks = [check(row, name, limit) for row in records for name, limit in limits.items()]
    checks.append(check(metrics, "initial_data_rmse", .02))
    return {"passed": all(item["passed"] for item in checks), "checks": checks,
            "scope": "synthetic Taylor-Green fixture only; six times and a 24x24 held-out grid"}


def main():
    p = parser(__doc__, Path(__file__).parent / "conf/config.yaml")
    p.add_argument("--smoke-data", action="store_true", help="Use labeled analytic Taylor-Green fixture; no weather data")
    p.add_argument("--data-path", type=Path)
    p.set_defaults(output_dir=Path("outputs/navier_stokes"))
    args = p.parse_args()
    if args.smoke_data and args.data_path is not None:
        p.error("--smoke-data and --data-path are mutually exclusive")
    cfg, device = setup(args, defaults={"steps": 3000})
    cfg.update(lab4_dtype="float32", lab4_recipe="adam_lbfgs_fp32_v1")
    nu = 0.01 if args.smoke_data else REAL_NU
    initial_data = None if args.smoke_data else tuple(torch.from_numpy(a) for a in read_wf_data(data_path=args.data_path))
    model = PeriodicFlow(cfg).to(device=device, dtype=torch.float32)
    physics = informer(NavierStokes(nu=nu, rho=1.0, dim=2, time=True), device)
    heldout_before = evaluate(model, physics, device, args.smoke_data, initial_data)
    history = optimize_lab(model, physics, cfg, device, initial_data)
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
                   "pressure_evaluation": "spatial_mean_error_removed_separately_at_each_evaluated_time",
                   "legacy_mixed_rmse": "synthetic_solution_rmse and synthetic_reference_rmse mix velocity and raw pressure; use separate diagnostics",
                   "legacy_pressure_factor": None if args.smoke_data else LEGACY_PRESSURE_FACTOR}
        if args.smoke_data:
            exact = torch.stack([taylor_green(xy, torch.full_like(xy[:, :1], t), nu) for t in times])
            arrays["reference"] = exact
            metrics["synthetic_reference_rmse"] = float((pred - exact).square().mean().sqrt())
            metrics.update({f"synthetic_{name}": value for name, value in solution_errors(pred, exact).items()})
    metrics["heldout_after"] = evaluate(model, physics, device, args.smoke_data, initial_data)
    metrics["training_recipe"] = {
        "optimizer": "Adam then full-batch L-BFGS", "dtype": "float32",
        "adam_step_calls": min(1000, cfg["steps"] // 3),
        "lbfgs_step_calls": cfg["steps"] - min(1000, cfg["steps"] // 3),
        "closure_evaluations": sum(row["closure_evaluations"] for row in history),
        "lbfgs_pde_points": max(2048, cfg["batch_size"]),
        "lbfgs_initial_points": max(1024, cfg["batch_size"]),
        "initial_data_weight": INITIAL_DATA_WEIGHT,
        "loss_history": "before each optimizer step; initial_data is weighted MSE",
        "positive_time_reference_targets_used_for_training": False,
    }
    if args.smoke_data:
        metrics["accuracy"] = accuracy_checks(metrics["heldout_after"])
        print("Lesson accuracy checks: " + ("PASS" if metrics["accuracy"]["passed"] else "NOT MET"), flush=True)
    def plot(plt, a):
        fig, ax = plt.subplots(figsize=(6, 5))
        h = ax.scatter(a["xy"][:, 0], a["xy"][:, 1], c=a["prediction"][-1, :, 0], s=10)
        fig.colorbar(h, ax=ax, label="predicted nondimensional u")
        ax.set(xlabel="x", ylabel="y", title=f"{metrics['data_kind']} / t=1 / not weather validation")
        return fig
    save_run(args, cfg, model, history, arrays, metrics, plot)


if __name__ == "__main__":
    main()
