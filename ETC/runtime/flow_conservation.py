"""Optional integrated periodic-flow constraints in normalized lesson units.

For unforced incompressible flow on a periodic domain, the continuous PDE gives
``d<E>/dt + nu <|grad u|**2 + |grad v|**2> = 0`` and constant mean velocity.
Here ``E = .5 * (u**2 + v**2)`` and brackets denote a spatial mean. The uniform
cell-center grid only approximates these integrals; its penalties neither
guarantee conservation nor certify convergence or future forecast accuracy.

Initial training velocities set a fixed normalization scale, not an energy
target at any time. These helpers do not modify observations, use future labels,
or replace the local PDE/initial-data losses. Models must act independently on
each (x, y, t) row, as the lesson's pointwise MLP does.
"""

from dataclasses import dataclass
import math
from numbers import Integral, Real

import torch


@dataclass(frozen=True)
class ConservationGrid:
    """Prepared quadrature coordinates and detached initial-data energy scale."""

    xy: torch.Tensor
    energy_scale: torch.Tensor


def _real_scalar(value, name, *, positive=False, nonnegative=False):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite real scalar.")
    value = float(value)
    if positive and value <= 0:
        raise ValueError(f"{name} must be positive.")
    if nonnegative and value < 0:
        raise ValueError(f"{name} must be nonnegative.")
    return value


def prepare_conservation(initial_training_fields, device, dtype=torch.float32,
                         *, lower, length, grid_size=32):
    """Prepare a cell-center grid using only the unmodified training fields.

    ``initial_training_fields`` has shape (N, 3), ordered as u, v, p. Its velocity
    energy ``.5 * mean(u**2 + v**2)`` must be finite and positive; zero-energy data
    cannot normalize these penalties. ``lower`` and ``length`` describe each
    axis of the square periodic domain, in the same normalized units as the PDE.
    Floating-point reduction uses ``dtype`` (FP32 by default), as in training.
    """
    if dtype not in (torch.float32, torch.float64):
        raise ValueError("dtype must be torch.float32 or torch.float64.")
    lower = _real_scalar(lower, "lower")
    length = _real_scalar(length, "length", positive=True)
    if not math.isfinite(lower + length):
        raise ValueError("The domain endpoints must be finite.")
    if isinstance(grid_size, bool) or not isinstance(grid_size, Integral) or grid_size < 2:
        raise ValueError("grid_size must be an integer of at least two.")

    fields = torch.as_tensor(initial_training_fields)
    if fields.dtype == torch.bool or fields.is_complex():
        raise ValueError("initial_training_fields must contain real numeric values.")
    if fields.ndim != 2 or fields.shape[1] != 3 or len(fields) == 0:
        raise ValueError("initial_training_fields must have nonempty shape (N, 3).")
    fields = torch.as_tensor(initial_training_fields, device=device, dtype=dtype).detach()
    if not torch.isfinite(fields).all():
        raise ValueError("initial_training_fields must contain only finite values.")
    energy_scale = .5 * fields[:, :2].square().sum(-1).mean().detach()
    if not torch.isfinite(energy_scale) or energy_scale <= 0:
        raise ValueError("The training velocity energy scale must be finite and positive.")

    axis = lower + length * (torch.arange(grid_size, device=device, dtype=dtype) + .5) / grid_size
    if not torch.isfinite(axis).all() or not (axis[1:] > axis[:-1]).all():
        raise ValueError("The domain grid must remain finite and distinct in the requested dtype.")
    xx, yy = torch.meshgrid(axis, axis, indexing="xy")
    return ConservationGrid(torch.stack((xx.flatten(), yy.flatten()), dim=1), energy_scale)


def _derivative(value, coordinate):
    """Match labs.derivative, including differentiable zero for constant fields."""
    if not value.requires_grad:
        return coordinate * 0.0
    result = torch.autograd.grad(value, coordinate, torch.ones_like(value),
                                 create_graph=True, retain_graph=True,
                                 allow_unused=True)[0]
    return coordinate * 0.0 if result is None else result


def conservation_terms(model, prepared, nu, time=None):
    """Return the experiment's energy-balance and mean-momentum penalties.

    With no ``time`` argument, sample one shared normalized time in [0, 1).
    A supplied scalar time must lie in [0, 1]. Each quadrature row gets its own
    cloned time coordinate for pointwise autograd; differentiating a single
    shared scalar would incorrectly multiply the spatial mean by grid size.
    No prediction is detached, so both scalar penalties retain parameter
    gradients. ``nu`` is the unchanged nonnegative normalized PDE viscosity.
    """
    if not isinstance(prepared, ConservationGrid):
        raise ValueError("prepared must be returned by prepare_conservation.")
    nu = _real_scalar(nu, "nu", nonnegative=True)
    xy = prepared.xy.detach().requires_grad_()
    if time is None:
        shared_time = torch.rand(1, 1, device=xy.device, dtype=xy.dtype)
    else:
        shared_time = torch.as_tensor(time)
        if (shared_time.numel() != 1 or shared_time.dtype == torch.bool
                or shared_time.is_complex()):
            raise ValueError("time must be one finite real scalar in [0, 1].")
        shared_time = torch.as_tensor(time, device=xy.device, dtype=xy.dtype).detach().reshape(1, 1)
        if not torch.isfinite(shared_time).all() or (shared_time < 0).any() or (shared_time > 1).any():
            raise ValueError("time must be one finite real scalar in [0, 1].")
    time_rows = shared_time.expand(len(xy), 1).clone().requires_grad_()
    prediction = model(xy, time_rows)
    if prediction.shape != (len(xy), 3):
        raise ValueError("model must return shape (N, 3) for u, v, p.")
    uv = prediction[:, :2]
    kinetic = .5 * uv.square().sum(-1, keepdim=True)
    energy_rate = _derivative(kinetic, time_rows).mean()
    dissipation = nu * sum(_derivative(uv[:, i:i + 1], xy).square().sum(-1).mean()
                           for i in (0, 1))
    mean_rates = torch.stack([_derivative(uv[:, i:i + 1], time_rows).mean()
                              for i in (0, 1)])
    return {
        "energy_balance": ((energy_rate + dissipation) / prepared.energy_scale).square(),
        "mean_momentum": mean_rates.square().sum() / (2 * prepared.energy_scale),
    }
