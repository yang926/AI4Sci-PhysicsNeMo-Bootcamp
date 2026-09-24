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
import json
import math
import sys
from pathlib import Path
import torch
from sympy import Function, Symbol
from physicsnemo.sym.eq.pde import PDE
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from ETC.runtime.labs import parser, setup, mlp, informer, save_run


INVERSE_DATA_WEIGHT = 1000.0
BOUNDARY_WEIGHT = 10.0


class FourierMLP(torch.nn.Module):
    """Smooth coordinate features; neither coefficients nor source values are given."""

    def __init__(self, cfg, dtype=None):
        super().__init__()
        self.register_buffer("frequencies", math.pi * torch.arange(1, 5, dtype=dtype).reshape(1, -1))
        layers = []
        inputs = 9  # x plus sine/cosine features at k*pi, k=1,...,4
        for _ in range(cfg["num_layers"]):
            layers.extend((torch.nn.Linear(inputs, cfg["layer_size"], dtype=dtype), torch.nn.Tanh()))
            inputs = cfg["layer_size"]
        layers.append(torch.nn.Linear(inputs, 1, dtype=dtype))
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x):
        angles = x * self.frequencies
        return self.layers(torch.cat((2 * x - 1, angles.sin(), angles.cos()), dim=1))


class Poisson1D(PDE):
    def __init__(self, inverse=False):
        self.dim = 1
        x = Symbol("x")
        u = Function("u")(x)
        f = Function("f")(x) if inverse else 1
        self.equations = {"poisson": u.diff(x, 2) - f}


class BasicPINN(torch.nn.Module):
    def __init__(self, cfg, mode="forward", *, dtype=None):
        super().__init__()
        if mode not in ("forward", "parameterized", "inverse"):
            raise ValueError("mode must be forward, parameterized or inverse")
        self.mode = mode
        if mode == "inverse":
            self.solution = FourierMLP(cfg, dtype)
            self.source = FourierMLP(cfg, dtype)
        else:
            self.solution = mlp(2 if mode == "parameterized" else 1, 1, cfg)
            if dtype is not None:
                self.solution.to(dtype=dtype)

    def forward(self, x, length=None):
        if self.mode == "inverse":
            # This uses only the supplied u(0)=u(1)=0, not the analytical solution.
            return x * (1 - x) * 0.1 * self.solution(x)
        inputs = (torch.cat((2 * x / length - 1, 2 * length - 3), dim=1)
                  if self.mode == "parameterized" else 2 * x - 1)
        return self.solution(inputs)


def model_dtype(model):
    parameter = next(model.parameters(), None)
    return parameter.dtype if parameter is not None else torch.get_default_dtype()


def analytical(x, length=1.0):
    return 0.5 * (x - length) * x


def inverse_reference(x):
    u = (8 * x * (x**2 - 1) - 3 * torch.sin(4 * math.pi * x) / math.pi**2) / 48
    return u, x + torch.sin(4 * math.pi * x)


def observation_loss(model, ox, ou):
    """Fit the supplied u observations after factoring out the known zero BCs.

    u=x(1-x)*v, so observations constrain v=u/[x(1-x)]. This gives near-edge
    observations adequate weight when recovering curvature. It is intended for
    this noise-free example, not as a general prescription for noisy data.
    Endpoint observations add nothing to the exactly enforced zero boundaries.
    """
    interior = ((ox > 0) & (ox < 1)).squeeze(1)
    if not interior.any():
        raise ValueError("The inverse problem needs interior u observations")
    x, u = ox[interior], ou[interior]
    target = u / (x * (1 - x))
    return INVERSE_DATA_WEIGHT * (0.1 * model.solution(x) - target).square().mean()


def loss_terms(model, physics, batch_size, device, observations=None, *, points=None):
    dtype = model_dtype(model)
    if points is None:
        length = (1 + torch.rand(batch_size, 1, device=device, dtype=dtype)
                  if model.mode == "parameterized"
                  else torch.ones(batch_size, 1, device=device, dtype=dtype))
        x = length * torch.rand(batch_size, 1, device=device, dtype=dtype)
    else:
        x, length = points
    # A fresh leaf avoids accumulating coordinate gradients in L-BFGS closures.
    x = x.detach().requires_grad_()
    u = model(x, length)
    variables = {"coordinates": x, "u": u}
    if model.mode == "inverse":
        variables["f"] = model.source(x)
    residual = physics.forward(variables)["poisson"]
    terms = {"physics": (length * residual.square()).mean(),
             "boundary": BOUNDARY_WEIGHT * (model(torch.zeros_like(x), length).square().mean()
                         + model(length, length).square().mean())}
    if observations is not None:
        ox, ou = observations
        terms["data"] = observation_loss(model, ox, ou)
    return terms


def optimize_lab(model, physics, cfg, device, observations=None):
    """Joint PINN learning: Adam, then L-BFGS on a fixed training set.

    Each history row describes the loss before one optimizer step. L-BFGS may
    evaluate several trial points within a step; those are counted separately.
    Neither analytical evaluation errors nor source targets select the model.
    """
    dtype = model_dtype(model)
    warmup = min(1000, cfg["steps"] // 3)
    adam = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    lbfgs = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=1, max_eval=20,
                            history_size=100, line_search_fn="strong_wolfe",
                            tolerance_grad=1e-10, tolerance_change=1e-14)
    fixed = None
    history = []
    for step in range(1, cfg["steps"] + 1):
        phase = "adam" if step <= warmup else "lbfgs"
        optimizer = adam if phase == "adam" else lbfgs
        if phase == "lbfgs" and fixed is None:
            if model.mode == "parameterized":
                # Include endpoint lengths so the family is learned at l=1 and 2 too.
                lengths = torch.linspace(1, 2, 17, device=device, dtype=dtype)[:, None]
                positions = (torch.arange(32, device=device, dtype=dtype) + .5) / 32
                length = lengths.repeat_interleave(32, dim=0)
                x = (lengths * positions[None, :]).reshape(-1, 1)
            else:
                x = torch.linspace(0, 1, 256, device=device, dtype=dtype)[:, None]
                length = torch.ones_like(x)
            fixed = (x, length)
        evaluations = 0
        first_terms = None

        def closure():
            nonlocal evaluations, first_terms
            optimizer.zero_grad(set_to_none=True)
            terms = loss_terms(model, physics, cfg["batch_size"], device, observations,
                               points=fixed if phase == "lbfgs" else None)
            loss = sum(terms.values())
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite {phase} loss at step {step}")
            loss.backward()
            gradients = [p.grad for p in model.parameters() if p.grad is not None]
            # One host/device check rather than synchronizing for every tensor.
            if not gradients or not torch.isfinite(torch.cat([g.detach().reshape(-1) for g in gradients])).all():
                raise FloatingPointError(f"Missing or nonfinite {phase} gradients at step {step}")
            if first_terms is None:
                first_terms = {"loss": loss.detach().item(),
                               **{key: value.detach().item() for key, value in terms.items()}}
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


def evaluate_at_length(model, physics, device, length):
    """Fixed interior and endpoint checks for one bar length."""
    dtype = model_dtype(model)
    x = torch.linspace(0, length, 401, device=device, dtype=dtype)[:, None].requires_grad_()
    u = model(x, torch.full_like(x, length))
    target = inverse_reference(x)[0] if model.mode == "inverse" else analytical(x, length)
    variables = {"coordinates": x, "u": u}
    if model.mode == "inverse":
        variables["f"] = model.source(x)
    residual = physics.forward(variables)["poisson"]
    edge_l = torch.full((1, 1), length, device=device, dtype=dtype)
    edge_values = torch.cat((model(torch.zeros_like(edge_l), edge_l), model(edge_l, edge_l)))
    objective = length * residual.square().mean()
    objective = objective + BOUNDARY_WEIGHT * edge_values.square().sum()
    if model.mode == "inverse":
        objective = objective + observation_loss(model, x, target)
    result = {"objective": float(objective.detach()),
              "solution_rmse": float((u - target).square().mean().sqrt().detach()),
              "pde_rmse": float(residual.square().mean().sqrt().detach()),
              "boundary_max_abs": float(edge_values.abs().max().detach())}
    if model.mode == "inverse":
        result["source_rmse"] = float((variables["f"] - inverse_reference(x)[1]).square().mean().sqrt().detach())
        result["source_max_abs"] = float((variables["f"] - inverse_reference(x)[1]).abs().max().detach())
    return result


def evaluate(model, physics, device):
    if model.mode != "parameterized":
        return evaluate_at_length(model, physics, device, 1.0)
    lengths = (1.0, 1.25, 1.5, 1.75, 2.0)
    records = [{"length": length, **evaluate_at_length(model, physics, device, length)}
               for length in lengths]
    # Every length has the same number of points: pool squared errors, not RMSEs.
    return {"objective": sum(row["objective"] for row in records) / len(records),
            **{name: math.sqrt(sum(row[name] ** 2 for row in records) / len(records))
               for name in ("solution_rmse", "pde_rmse")},
            "boundary_max_abs": max(row["boundary_max_abs"] for row in records),
            "per_length": records,
            "scope": "five fixed lengths in [1, 2]; not a continuous-family error bound"}


def accuracy_checks(metrics, mode):
    """Lesson acceptance checks, not competition scores or a general error bound."""
    if mode not in ("forward", "parameterized", "inverse"):
        raise ValueError("Unknown Lab 1 mode")
    limits = {"solution_rmse": 1e-3, "pde_rmse": 1e-2, "boundary_max_abs": 1e-3}
    if mode == "inverse":
        limits.update(solution_rmse=1e-4, source_rmse=5e-2, source_max_abs=1e-1)
    records = metrics["per_length"] if mode == "parameterized" else [metrics]
    if mode == "parameterized" and [row.get("length") for row in records] != [1., 1.25, 1.5, 1.75, 2.]:
        raise ValueError("Accuracy checks require all five evaluation lengths")
    checks = [{"length": row.get("length", 1.0), "metric": key,
               "value": row[key], "limit": limit,
               "passed": math.isfinite(row[key]) and 0 <= row[key] <= limit}
              for row in records for key, limit in limits.items()]
    return {"passed": all(check["passed"] for check in checks), "checks": checks,
            "scope": "401 evaluation points per length including endpoints; this noise-free teaching example only"}


def main():
    p = parser(__doc__)
    p.add_argument("--mode", choices=("forward", "parameterized", "inverse"), default="forward")
    p.set_defaults(output_dir=Path("outputs/pinn_basics"))
    args = p.parse_args()
    defaults = {"steps": 3000}
    if args.mode == "inverse":
        defaults.update(layer_size=32, num_layers=2)
    cfg, device = setup(args, defaults=defaults)
    cfg.update(lab1_mode=args.mode, lab1_dtype="float32", lab1_recipe="adam_lbfgs_fp32_v2")
    if device.type == "cuda":
        probe = torch.ones(1, device=device, requires_grad=True)
        probe.square().sum().backward()
        torch.cuda.synchronize(device)
    model = BasicPINN(cfg, args.mode, dtype=torch.float32).to(device)
    physics = informer(Poisson1D(args.mode == "inverse"), device)
    ox = torch.rand(100, 1, device=device, dtype=torch.float32) if args.mode == "inverse" else None
    observations = (ox, inverse_reference(ox)[0]) if args.mode == "inverse" else None
    heldout_before = evaluate(model, physics, device)
    history = optimize_lab(model, physics, cfg, device, observations)
    length = 1.5 if args.mode == "parameterized" else 1.0
    x = torch.linspace(0, length, 401, device=device, dtype=torch.float32)[:, None]
    with torch.no_grad():
        pred = model(x, torch.full_like(x, length))
        exact = inverse_reference(x)[0] if args.mode == "inverse" else analytical(x, length)
        arrays = {"x": x, "prediction": pred, "reference": exact}
        metrics = {"problem": "pinn_basics", "heldout_before": heldout_before, "mode": args.mode, "validation_length": length,
                   "validation_rmse": float((pred - exact).square().mean().sqrt()),
                   "preview_scope": "single length; heldout metrics cover five lengths in parameterized mode"}
        if args.mode == "inverse":
            arrays.update(source_prediction=model.source(x), source_reference=inverse_reference(x)[1])
            metrics["source_rmse"] = float((arrays["source_prediction"] - arrays["source_reference"]).square().mean().sqrt())
    metrics["heldout_after"] = evaluate(model, physics, device)
    metrics["accuracy"] = accuracy_checks(metrics["heldout_after"], args.mode)
    metrics["training_recipe"] = {"optimizer": "Adam then full-batch L-BFGS", "dtype": "float32",
                                  "adam_step_calls": min(1000, cfg["steps"] // 3),
                                  "lbfgs_step_calls": cfg["steps"] - min(1000, cfg["steps"] // 3),
                                  "closure_evaluations": sum(row["closure_evaluations"] for row in history),
                                  "loss_history": "before each optimizer step; L-BFGS trial evaluations counted separately",
                                  "inverse_data_weight": INVERSE_DATA_WEIGHT,
                                  "inverse_data_target": "observed u / (x * (1 - x)); interior observations only",
                                  "boundary_weight": BOUNDARY_WEIGHT,
                                  "source_targets_used_for_training": False}
    accuracy_message = "PASS" if metrics["accuracy"]["passed"] else "NOT MET: inspect the curves and errors"
    print("Lesson accuracy checks: " + accuracy_message, flush=True)
    def plot(plt, a):
        columns = 2 if args.mode == "inverse" else 1
        fig, axes = plt.subplots(1, columns, figsize=(11, 4) if columns == 2 else (7, 4), squeeze=False)
        ax = axes[0, 0]
        ax.plot(a["x"], a["reference"], label="analytical u")
        ax.plot(a["x"], a["prediction"], "--", label="PINN u")
        ax.set(xlabel="x", ylabel="u", title=args.mode)
        ax.legend()
        if args.mode == "inverse":
            ax = axes[0, 1]
            ax.plot(a["x"], a["source_reference"], label="analytical f")
            ax.plot(a["x"], a["source_prediction"], "--", label="learned f")
            ax.set(xlabel="x", ylabel="f", title="Recovered source (not a training target)")
            ax.legend()
        fig.tight_layout()
        return fig
    save_run(args, cfg, model, history, arrays, metrics, plot)


if __name__ == "__main__":
    main()
