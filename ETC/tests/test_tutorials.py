"""Analytical residual/condition checks and real PhysicsNeMo finite-gradient tests."""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from ETC.runtime.labs import informer


def load(name, path):
    location = ROOT / path
    sys.path.insert(0, str(location.parent))
    spec = importlib.util.spec_from_file_location(name, location)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


basic = load("tutorial_basic_test", "01_labs/01_pinn/source_code/pinn_basics.py")
projectile = load("tutorial_projectile_test", "01_labs/02_projectile/source_code/projectile.py")
diffusion = load("tutorial_diffusion_test", "01_labs/03_heat_conduction/source_code/diffusion_bar.py")
navier = load("tutorial_navier_test", "01_labs/04_navier_stokes/source_code/navier_stokes.py")
CFG = {"layer_size": 12, "num_layers": 2}
DEVICE = torch.device("cpu")


def assert_backward(model, terms):
    total = sum(terms.values())
    assert torch.isfinite(total)
    total.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    assert any(g.abs().sum() > 0 for g in gradients)


@pytest.mark.parametrize("length", [1.0, 1.37, 2.0])
def test_forward_and_parameterized_poisson_exact(length):
    x = torch.linspace(.01, length - .01, 23)[:, None].requires_grad_()
    physics = informer(basic.Poisson1D(), DEVICE)
    u = basic.analytical(x, length)
    r = physics.forward({"coordinates": x, "u": u})["poisson"]
    assert r.abs().max() < 1e-6
    assert basic.analytical(torch.tensor([[0.0], [length]]), length).abs().max() == 0


def test_inverse_source_matches_analytical_solution():
    x = torch.linspace(.007, .993, 57)[:, None].requires_grad_()
    u, f = basic.inverse_reference(x)
    r = informer(basic.Poisson1D(True), DEVICE).forward({"coordinates": x, "u": u, "f": f})["poisson"]
    assert r.abs().max() < 2e-6
    assert basic.inverse_reference(torch.tensor([[0.0], [1.0]]))[0].abs().max() < 1e-7


def test_projectile_equations_and_initial_conditions():
    t = torch.linspace(.013, 4.987, 27)[:, None].requires_grad_()
    field = projectile.analytical(t)
    residual = projectile.residuals(field, t, informer(projectile.ProjectileEquation(), DEVICE))
    assert all(r.abs().max() < 2e-6 for r in residual.values())
    t0 = torch.zeros(1, 1, requires_grad=True)
    xy = projectile.analytical(t0)
    assert torch.equal(xy, torch.zeros_like(xy))
    assert torch.allclose(projectile.derivative(xy[:, :1], t0), torch.tensor([[20.0]]))
    assert torch.allclose(projectile.derivative(xy[:, 1:], t0), torch.tensor([[34.641016]]), atol=1e-5)


@pytest.mark.parametrize("d1", [5.0, 10.0, 17.5, 25.0])
def test_composite_bar_equations_boundary_and_flux(d1):
    x = torch.linspace(.01, .99, 31)[:, None].requires_grad_()
    tb = 100 / (1 + d1 / .1)
    ul = x * tb + 0 * x**3
    ur = x * 100 + (1 - x) * tb + 0 * x**3
    left = informer(diffusion.Diffusion("u_1", "D1"), DEVICE)
    right = informer(diffusion.Diffusion("u_2", .1), DEVICE)
    assert left.forward({"coordinates": x, "u_1": ul, "D1": torch.full_like(x, d1)})["diffusion"].abs().max() < 1e-6
    assert right.forward({"coordinates": x, "u_2": ur})["diffusion"].abs().max() < 1e-6
    xi = torch.ones(3, 1, requires_grad=True)
    ui, vi = xi * tb, (xi - 1) * 100 + (2 - xi) * tb
    jump = informer(diffusion.DiffusionInterface(), DEVICE).forward({"coordinates": xi, "u_1": ui, "u_2": vi, "D1": torch.full_like(xi, d1)})
    assert jump["temperature_jump"].abs().max() < 1e-6
    assert jump["flux_jump"].abs().max() < 2e-6
    assert torch.allclose(diffusion.analytical(torch.tensor([[0.0], [2.0]]), d1), torch.tensor([[0.0], [100.0]]))


def test_taylor_green_satisfies_navier_stokes_and_periodicity():
    torch.manual_seed(21)
    xy = (navier.LOWER + navier.LENGTH * torch.rand(49, 2)).requires_grad_()
    t = torch.full((49, 1), .27, requires_grad=True)
    field = navier.taylor_green(xy, t)
    physics = informer(navier.NavierStokes(nu=.01, rho=1, dim=2, time=True), DEVICE)
    res = navier.residuals(field, xy, t, physics)
    assert all(v.abs().max() < 3e-5 for v in res.values())
    shifted = xy.detach() + torch.tensor([[navier.LENGTH, 0]])
    assert torch.allclose(field.detach(), navier.taylor_green(shifted, t.detach()), atol=2e-6)
    model = navier.PeriodicFlow(CFG)
    assert torch.allclose(model(xy.detach(), t.detach()), model(shifted, t.detach()), atol=1e-6)


@pytest.mark.parametrize("mode", ["forward", "parameterized", "inverse"])
def test_basic_forward_backward(mode):
    model = basic.BasicPINN(CFG, mode)
    physics = informer(basic.Poisson1D(mode == "inverse"), DEVICE)
    x = torch.linspace(0, 1, 100)[:, None]
    observations = (x, basic.inverse_reference(x)[0]) if mode == "inverse" else None
    assert_backward(model, basic.loss_terms(model, physics, 8, DEVICE, observations))


def test_projectile_forward_backward():
    model = projectile.ProjectileModel(CFG)
    assert_backward(model, projectile.loss_terms(model, informer(projectile.ProjectileEquation(), DEVICE), 8, DEVICE))


@pytest.mark.parametrize("parameterized", [False, True])
def test_diffusion_forward_backward(parameterized):
    model = diffusion.CompositeBar(CFG, parameterized)
    physics = (informer(diffusion.Diffusion("u_1", "D1"), DEVICE), informer(diffusion.Diffusion("u_2", .1), DEVICE), informer(diffusion.DiffusionInterface(), DEVICE))
    assert_backward(model, diffusion.loss_terms(model, physics, 8, DEVICE))


def test_navier_forward_backward():
    model = navier.PeriodicFlow(CFG)
    physics = informer(navier.NavierStokes(nu=.01, rho=1, dim=2, time=True), DEVICE)
    assert_backward(model, navier.loss_terms(model, physics, 8, DEVICE))


def test_original_data_loader_path_and_normalization(tmp_path):
    # Independent tiny fixture checks legacy channel order and conversion exactly.
    data = np.stack([np.full((2, 3), v) for v in (2, 3, 4)])
    path = tmp_path / "fixture.npy"
    np.save(path, data)
    coords, fields = navier.read_wf_data(2, 4, path)
    assert coords.shape == (6, 2)
    assert np.allclose(coords[0], [-.720, -.720])
    assert np.allclose(coords[-1], [.719, .719])
    assert np.allclose(fields, np.tile([1, 1.5, .10197], (6, 1)))


def test_missing_weather_data_fails_explicitly(tmp_path):
    with pytest.raises(FileNotFoundError, match="Restore data_lat.npy"):
        navier.read_wf_data(data_path=tmp_path / "absent.npy")


def test_existing_output_directory_is_preserved(tmp_path):
    from argparse import Namespace
    from ETC.runtime.labs import setup
    output = tmp_path / "previous-run"
    output.mkdir()
    artifact = output / "metrics.json"
    artifact.write_text('{"previous_result": true}')
    args = Namespace(output_dir=output, config=None, steps=1, device="cpu", seed=42)
    with pytest.raises(FileExistsError, match="never overwritten"):
        setup(args)
    assert artifact.read_text() == '{"previous_result": true}'
