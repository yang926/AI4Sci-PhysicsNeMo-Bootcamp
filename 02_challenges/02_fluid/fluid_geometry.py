"""Sampling for the original channel and its bottom-attached rectangular chips.

The inlet and outlet are excluded from no-slip sampling. The flux target is
one: integral_{-.5}^{.5} 1.5*(1-4*y**2) dy = 1.
"""
import math
from pathlib import Path
import numpy as np
import torch


SINGLE_BLOCK = [(-1.0, 0.0, 0.1)]  # xmin, xmax, top; bottom=-0.5
THREE_BLOCKS = [(-1.0, -.4, -.1), (.2, .7, 0.0), (1.2, 1.6, -.15)]


def outside_blocks(coordinates, blocks):
    keep = torch.ones(len(coordinates), dtype=torch.bool, device=coordinates.device)
    for xmin, xmax, top in blocks:
        keep &= ~((coordinates[:, 0] >= xmin) & (coordinates[:, 0] <= xmax) & (coordinates[:, 1] <= top))
    return keep


def sample_interior(count, blocks, device):
    selected = []
    remaining = count
    while remaining:
        points = torch.rand(max(2 * remaining, 16), 2, device=device)
        points[:, 0] = 5 * points[:, 0] - 2.5
        points[:, 1] -= .5
        points = points[outside_blocks(points, blocks)][:remaining]
        selected.append(points)
        remaining -= len(points)
    return torch.cat(selected)


def sample_inlet(count, device, outlet=False):
    return torch.cat((torch.full((count, 1), 2.5 if outlet else -2.5, device=device),
                      torch.rand(count, 1, device=device) - .5), dim=1)


def inlet_velocity(coordinates):
    return 1.5 * (1 - 4 * coordinates[:, 1:2].square())


def wall_segments(blocks):
    segments = [((-2.5, .5), (2.5, .5))]
    previous = -2.5
    for xmin, xmax, top in sorted(blocks):
        segments += [((previous, -.5), (xmin, -.5)), ((xmin, -.5), (xmin, top)),
                     ((xmin, top), (xmax, top)), ((xmax, top), (xmax, -.5))]
        previous = xmax
    segments.append(((previous, -.5), (2.5, -.5)))
    return segments


def sample_walls(count, blocks, device):
    segments = torch.tensor(wall_segments(blocks), dtype=torch.float32, device=device)
    lengths = torch.linalg.vector_norm(segments[:, 1] - segments[:, 0], dim=1)
    choice = torch.multinomial(lengths, count, replacement=True)
    selected = segments[choice]
    return selected[:, 0] + torch.rand(count, 1, device=device) * (selected[:, 1] - selected[:, 0])


def sdf_weight(coordinates, blocks):
    distance = .5 - coordinates[:, 1:2].abs()
    for xmin, xmax, top in blocks:
        dx = torch.maximum((xmin - coordinates[:, :1]).clamp_min(0), (coordinates[:, :1] - xmax).clamp_min(0))
        dy = (coordinates[:, 1:2] - top).clamp_min(0)
        distance = torch.minimum(distance, (dx.square() + dy.square()).sqrt())
    return 2 * distance.clamp_min(0)


def sample_flux(lines, points, blocks, device, time_end=None):
    """Midpoint quadrature across fluid portions, with one time per whole line."""
    x = torch.rand(lines, 1, device=device) * 5 - 2.5
    bottom = torch.full_like(x, -.5)
    for xmin, xmax, top in blocks:
        bottom = torch.where((x >= xmin) & (x <= xmax), torch.maximum(bottom, torch.full_like(bottom, top)), bottom)
    height = .5 - bottom
    quadrature = (torch.arange(points, device=device).float() + .5) / points
    y = bottom + height * quadrature[None, :]
    coordinates = torch.stack((x.expand(-1, points), y), dim=-1).reshape(-1, 2)
    time = None if time_end is None else (torch.rand(lines, 1, device=device) * time_end).repeat_interleave(points, dim=0)
    return coordinates, time, height


def openfoam_metrics(model, device):
    """Compare L1 with supplied CFD data, not an analytic solution."""
    from ETC.runtime.pinn import evaluate_fields
    source = Path(__file__).parent / "examples_sym/chip_2d/openfoam/2D_chip_fluid0.csv"
    if not source.is_file():
        return {"openfoam_reference_available": False}
    data = np.loadtxt(source, delimiter=",", skiprows=1)
    data = data[np.linspace(0, len(data) - 1, min(1024, len(data)), dtype=int)]
    xy = torch.tensor(data[:, [5, 6]] - [2.5, .5], dtype=torch.float32, device=device)
    with torch.no_grad():
        prediction = evaluate_fields(model, xy, None, ["u", "v", "p"])
    result = {"openfoam_reference_available": True, "openfoam_evaluation_points": len(data)}
    for name, column in (("u", 1), ("v", 2), ("p", 4)):
        result["openfoam_" + name + "_rmse"] = float(np.sqrt(np.mean((prediction[name].cpu().numpy().ravel() - data[:, column]) ** 2)))
    return result
