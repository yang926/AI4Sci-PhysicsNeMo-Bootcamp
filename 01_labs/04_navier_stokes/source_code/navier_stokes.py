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
import time
from pathlib import Path
import numpy as np
import torch
from sympy import Function, Symbol
from physicsnemo.sym.eq.pde import PDE
from physicsnemo.models.mlp import FullyConnected
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
        raise FileNotFoundError(f"Missing original data: {path}. Restore data_lat.npy from the course repository before running Lab 4.")
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


def sample_initial_field(initial_data, grid_size=128):
    """Sample rows AND columns of the original x-fast grid without interpolation.

    A linspace over flattened indices visits mostly a diagonal. Keep a Cartesian
    subset instead, for both the input preview and spatial initial-fit checks.
    """
    if type(grid_size) is not int or grid_size < 2:
        raise ValueError("grid_size must be an integer of at least two")
    coords, values = (torch.as_tensor(item) for item in initial_data)
    if coords.ndim != 2 or coords.shape[1] != 2 or values.shape != (len(coords), 3):
        raise ValueError("Expected initial coordinates (N, 2) and fields (N, 3)")
    x, y = torch.unique(coords[:, 0], sorted=True), torch.unique(coords[:, 1], sorted=True)
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2 or nx * ny != len(coords):
        raise ValueError("Expected a rectangular initial-data grid")
    grid = coords.reshape(ny, nx, 2)
    if not (torch.equal(grid[:, :, 0], x.expand(ny, nx))
            and torch.equal(grid[:, :, 1], y[:, None].expand(ny, nx))):
        raise ValueError("Expected x-fast initial-data ordering")
    ix = torch.linspace(0, nx - 1, min(grid_size, nx), device=coords.device).long()
    iy = torch.linspace(0, ny - 1, min(grid_size, ny), device=coords.device).long()
    indices = (iy[:, None] * nx + ix[None, :]).reshape(-1)
    return coords[indices], values[indices]


def split_initial_observations(initial_data, grid_size=32):
    """Reserve a Cartesian initial grid; never use it for training or scaling.

    The full input remains unchanged for plotting. This is spatial validation
    at t=0, not a held-out future-weather dataset.
    """
    if type(grid_size) is not int or grid_size < 2:
        raise ValueError("Initial validation grid_size must be an integer of at least two")
    coords, fields = initial_data
    # Also validate the rectangular x-fast ordering through the common sampler.
    sample_initial_field(initial_data, 2)
    nx = torch.unique(coords[:, 0]).numel()
    ny = len(coords) // nx
    count = min(grid_size, max(2, min(nx, ny) // 4))
    if count ** 2 >= len(coords):
        raise ValueError("Initial validation needs at least two points per axis and a nonempty training set")
    ix = torch.linspace(0, nx - 1, count, device=coords.device).long()
    iy = torch.linspace(0, ny - 1, count, device=coords.device).long()
    indices = (iy[:, None] * nx + ix[None, :]).reshape(-1)
    keep = torch.ones(len(coords), dtype=torch.bool, device=coords.device)
    keep[indices] = False
    return ((coords[keep], fields[keep]), (coords[indices], fields[indices]))


def configure_efficient_representation(cfg, training_initial):
    """Reparameterize the same physical outputs using training statistics only."""
    _, fields = training_initial
    scale = fields.std(dim=0)
    if not torch.isfinite(fields).all() or not (scale > 0).all():
        raise ValueError("Efficient training needs finite, nonconstant initial u/v/p fields")
    cfg.update(lab4_feature_bands=[1, 2, 4, 8],
               lab4_field_center=fields.mean(dim=0).tolist(),
               lab4_field_scale=scale.tolist())


class PeriodicFlow(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        bands = cfg.get("lab4_feature_bands")
        in_features = 5
        if bands is not None:
            if (not isinstance(bands, (tuple, list)) or not bands
                    or any(type(k) is not int or k <= 0 for k in bands)
                    or len(set(bands)) != len(bands)):
                raise ValueError("Periodic feature bands must be distinct positive integers")
            center = torch.as_tensor(cfg["lab4_field_center"], dtype=torch.float32)
            scale = torch.as_tensor(cfg["lab4_field_scale"], dtype=torch.float32)
            if (center.shape != (3,) or scale.shape != (3,)
                    or not torch.isfinite(center).all() or not torch.isfinite(scale).all()
                    or not (scale > 0).all()):
                raise ValueError("Field center and positive scale must contain finite u/v/p values")
            self.register_buffer("bands", torch.tensor(bands, dtype=torch.float32))
            self.register_buffer("center", center)
            self.register_buffer("scale", scale)
            in_features = 4 * len(bands) + 1
        if cfg.get("lab4_architecture") == "upstream_silu_weight_norm":
            self.network = FullyConnected(in_features=in_features, out_features=3,
                                          layer_size=cfg["layer_size"], num_layers=cfg["num_layers"],
                                          activation_fn="silu", weight_norm=True)
        else:
            # The small analytic fixture is separate from the original-data lesson.
            self.network = mlp(in_features, 3, cfg)

    def forward(self, xy, t):
        phase = 2 * math.pi * (xy - LOWER) / LENGTH
        # Periodic feature map replaces the retired architecture periodicity keyword.
        if hasattr(self, "bands"):
            phase = (phase[:, :, None] * self.bands).flatten(1)
            values = self.network(torch.cat((phase.sin(), phase.cos(), t), dim=1))
            # Return the original normalized physical fields before PDE derivatives.
            return self.center + self.scale * values
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


def original_loss_terms(model, physics, points):
    """Upstream PointwiseConstraint scaling, not the synthetic fixture weights.

    from_numpy gives each observed sample area=1, so its loss sums the batch.
    The interior samples integrate over the rectangle, area LENGTH**2. Both
    constraints sum their three component losses. See LAB4_RESTORATION.md.
    """
    raw_xy, raw_t, ix, target = points
    xy, t = raw_xy.detach().requires_grad_(), raw_t.detach().requires_grad_()
    residual = residuals(model(xy, t), xy, t, physics)
    prediction = model(ix, torch.zeros_like(ix[:, :1]))
    return {"physics": LENGTH**2 * sum(value.square().mean() for value in residual.values()),
            "initial_data": (prediction - target).square().sum()}


def optimize_original(model, physics, cfg, device, initial_data):
    """Original lesson: Adam, 2048 samples per constraint, exponential decay.

    The current API uses a direct training loop instead of the retired Solver.
    Points are resampled each step; this is not a bitwise legacy replay.
    """
    if initial_data is None:
        raise ValueError("The student lesson requires the supplied initial field")
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    history = []
    for step in range(1, cfg["steps"] + 1):
        rate = cfg["learning_rate"] * .95 ** ((step - 1) / 3000)
        for group in optimizer.param_groups:
            group["lr"] = rate
        optimizer.zero_grad(set_to_none=True)
        points = training_points(cfg["batch_size"], device, initial_data)
        terms = original_loss_terms(model, physics, points)
        loss = sum(terms.values())
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite Adam loss at step {step}")
        loss.backward()
        gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
        if not gradients or not torch.stack([torch.isfinite(gradient).all() for gradient in gradients]).all():
            raise FloatingPointError(f"Missing or nonfinite Adam gradients at step {step}")
        optimizer.step()
        values = torch.stack([loss, *terms.values()]).detach().cpu().tolist()
        history.append({"step": step, "learning_rate": rate,
                        **dict(zip(("loss", *terms), values))})
        if step == 1 or step == cfg["steps"] or step % 500 == 0:
            print(json.dumps(history[-1]), flush=True)
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


def evaluate(model, physics, device, synthetic, initial_data=None, *, initial_batch_size=2048,
             initial_validation=None, grid_size=24):
    """Fixed cell-center PDE grid at six times, including both endpoints."""
    if type(grid_size) is not int or grid_size < 2:
        raise ValueError("PDE evaluation grid_size must be an integer of at least two")
    grid = LOWER + LENGTH * (torch.arange(grid_size, device=device, dtype=torch.float32) + .5) / grid_size
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
    elif initial_validation is not None:
        initial_xy, initial_target = (item.to(device) for item in initial_validation)
    else:
        initial_xy, initial_target = (item.to(device) for item in sample_initial_field(initial_data, 32))
    initial_prediction = model(initial_xy, torch.zeros_like(initial_xy[:, :1]))
    initial_rmse = float((initial_prediction - initial_target).square().mean().sqrt().detach())
    # Equal grid sizes: pool squared errors, never average RMSEs directly.
    result = {name: math.sqrt(sum(row[name] ** 2 for row in records) / len(records))
              for name in records[0] if name.endswith("rmse")}
    result.update({name: max(row[name] for row in records)
                   for name in ("periodic_value_max_abs", "periodic_gradient_max_abs")})
    pde_weight = 1.0 if synthetic else LENGTH**2
    initial_weight = INITIAL_DATA_WEIGHT if synthetic else 3 * initial_batch_size
    result.update(objective=3 * pde_weight * result["pde_rmse"] ** 2 + initial_weight * initial_rmse ** 2,
                  initial_data_rmse=initial_rmse, per_time=records,
                  scope=f"{grid_size}x{grid_size} held-out spatial cell centers at six times; not a continuous-domain bound")
    result.update({f"initial_{name}": value for name, value in
                   solution_errors(initial_prediction, initial_target).items()})
    if not synthetic:
        result["initial_evaluation_scope"] = (
            "reserved spatial t=0 observations excluded from training and scaling; not future forecast accuracy"
            if initial_validation is not None else
            "32x32 spatial subset of the supplied t=0 observations; may overlap training, not future forecast accuracy")
        for column, name in enumerate(("u", "v", "p")):
            error = initial_prediction[:, column] - initial_target[:, column]
            result[f"initial_{name}_rmse"] = float(error.square().mean().sqrt().detach())
            target_std = float(initial_target[:, column].std().detach())
            if target_std > 0:
                result[f"initial_{name}_nrmse"] = result[f"initial_{name}_rmse"] / target_std
    return result


def accuracy_checks(metrics):
    """Fixed criteria for the separate Taylor-Green regression fixture only."""
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
    p = parser(__doc__)
    p.add_argument("--smoke-data", action="store_true", help="Use labeled analytic Taylor-Green fixture; no weather data")
    p.add_argument("--data-path", type=Path)
    p.add_argument("--recipe", choices=("efficient", "upstream"), default="efficient",
                   help="Measured 3000-update multiscale class preset, or original 50000-update settings")
    p.set_defaults(output_dir=Path("outputs/navier_stokes"))
    args = p.parse_args()
    if args.smoke_data and args.data_path is not None:
        p.error("--smoke-data and --data-path are mutually exclusive")
    if args.config is None:
        filename = ("synthetic_fixture.yaml" if args.smoke_data else
                    "upstream.yaml" if args.recipe == "upstream" else "config.yaml")
        args.config = Path(__file__).parent / "conf" / filename
    default_steps = 50000 if not args.smoke_data and args.recipe == "upstream" else 3000
    cfg, device = setup(args, defaults={"steps": default_steps})
    cfg.update(lab4_dtype="float32",
               lab4_recipe=("adam_lbfgs_fp32_v1" if args.smoke_data else
                            "upstream_adam_fp32_v1" if args.recipe == "upstream" else "multiscale_adam_fp32_v1"),
               lab4_architecture="tanh_fixture" if args.smoke_data else "upstream_silu_weight_norm")
    nu = 0.01 if args.smoke_data else REAL_NU
    initial_data = None if args.smoke_data else tuple(torch.from_numpy(a) for a in read_wf_data(data_path=args.data_path))
    training_initial = initial_data
    evaluation_kwargs = {"initial_batch_size": cfg["batch_size"]}
    if not args.smoke_data and args.recipe == "efficient":
        initial_data = tuple(item.to(device) for item in initial_data)
        training_initial, validation_initial = split_initial_observations(initial_data)
        configure_efficient_representation(cfg, training_initial)
        evaluation_kwargs.update(initial_validation=validation_initial, grid_size=64)
    model = PeriodicFlow(cfg).to(device=device, dtype=torch.float32)
    physics = informer(NavierStokes(nu=nu, rho=1.0, dim=2, time=True), device,
                       supplied_derivatives=("u__t", "v__t"))
    heldout_before = evaluate(model, physics, device, args.smoke_data, initial_data,
                              **evaluation_kwargs)
    training_started = time.perf_counter()
    history = (optimize_lab(model, physics, cfg, device) if args.smoke_data
               else optimize_original(model, physics, cfg, device, training_initial))
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)
    training_seconds = time.perf_counter() - training_started
    grid_size = 32 if args.smoke_data else 128
    axis = torch.linspace(LOWER, LOWER + LENGTH, grid_size + 1, device=device)[:-1]
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
                   "time_unit": "dimensionless" if args.smoke_data else "hours",
                   "time_scale_hours": None if args.smoke_data else TIME_SCALE / 3600,
                   "pressure_evaluation": ("spatial_mean_error_removed_separately_at_each_evaluated_time" if args.smoke_data
                                           else "initial pressure fit only; no future pressure reference"),
                   "legacy_mixed_rmse": "synthetic_solution_rmse and synthetic_reference_rmse mix velocity and raw pressure; use separate diagnostics",
                   "legacy_pressure_factor": None if args.smoke_data else LEGACY_PRESSURE_FACTOR}
        if args.smoke_data:
            exact = torch.stack([taylor_green(xy, torch.full_like(xy[:, :1], t), nu) for t in times])
            arrays["reference"] = exact
            metrics["synthetic_reference_rmse"] = float((pred - exact).square().mean().sqrt())
            metrics.update({f"synthetic_{name}": value for name, value in solution_errors(pred, exact).items()})
        else:
            initial_xy, initial_fields = sample_initial_field(initial_data)
            arrays.update(initial_xy=initial_xy, initial_fields=initial_fields,
                          times_hours=torch.linspace(0, TIME_SCALE / 3600, 11, dtype=torch.float32))
            metrics["initial_data_source"] = "data_lat.npy (upstream normalization retained)" if args.data_path is None else str(args.data_path)
            metrics["initial_data_shape"] = [len(initial_data[0]), 3]
    metrics["heldout_after"] = evaluate(model, physics, device, args.smoke_data, initial_data,
                                      **evaluation_kwargs)
    metrics["training_seconds"] = training_seconds
    if args.smoke_data:
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
    else:
        metrics["training_recipe"] = {
            "name": cfg["lab4_recipe"], "optimizer": "Adam", "dtype": "float32",
            "optimizer_step_calls": len(history), "learning_rate_initial": cfg["learning_rate"],
            "decay_rate": .95, "decay_steps": 3000,
            "architecture": "periodic MLP, SiLU, weight normalization",
            "hidden_layers": cfg["num_layers"], "layer_width": cfg["layer_size"],
            "pde_batch_size": cfg["batch_size"], "initial_batch_size": cfg["batch_size"],
            "initial_data_loss": "sum of squared errors over samples and u/v/p components",
            "physics_loss": "rectangle area times sum of three residual mean squares",
            "sampling": "fresh uniform samples each step; not the legacy fixed shuffled dataset",
            "positive_time_reference_targets_used_for_training": False,
        }
        if args.recipe == "efficient":
            metrics["training_recipe"].update(
                periodic_feature_bands=cfg["lab4_feature_bands"],
                output_scaling="training-field mean/std; converted back before PDE and original loss",
                initial_training_points=len(training_initial[0]),
                initial_validation_points=len(validation_initial[0]),
                selection="bounded class-time/initial-fit tradeoff, not convergence or forecast validation")
    if args.smoke_data:
        metrics["accuracy"] = accuracy_checks(metrics["heldout_after"])
        print("Synthetic fixture accuracy checks: " + ("PASS" if metrics["accuracy"]["passed"] else "NOT MET"), flush=True)
    def plot(plt, a):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
        first_xy = a["xy"] if args.smoke_data else a["initial_xy"]
        first_field = a["reference"][0] if args.smoke_data else a["initial_fields"]
        maximum = max(float(np.hypot(first_field[:, 0], first_field[:, 1]).max()),
                      float(np.hypot(a["prediction"][:, :, 0], a["prediction"][:, :, 1]).max()), 1e-8)
        from ETC.runtime.flow_visualization import _draw_flow
        _draw_flow(axes[0], first_xy, first_field, maximum, "Initial input")
        image, _, _ = _draw_flow(axes[1], a["xy"], a["prediction"][-1], maximum,
                                "PINN: t=1 (synthetic)" if args.smoke_data else "PINN: 60 hours (simplified flow)")
        fig.colorbar(image, ax=axes, label="Speed (normalized; fixed scale)")
        return fig
    save_run(args, cfg, model, history, arrays, metrics, plot)


if __name__ == "__main__":
    main()
