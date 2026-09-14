"""Physics and finite-backward checks for all eight migrated PINN challenges.

These CPU tests exercise the installed PhysicsNeMo PDE/PhysicsInformer and MLP,
not a mock. They do not certify training convergence or event GPU readiness.
"""
import importlib
import math
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "challenge"))

import torch
from pinn_runtime import create_model, create_informer, residuals, gradient, heldout_losses
from challenge.wave import wave_l1, wave_l2, wave_l3
from challenge.fuild import chip_2d_l1, chip_2d_l2, chip_2d_l3
from challenge.fuild.fluid_geometry import (SINGLE_BLOCK, THREE_BLOCKS, outside_blocks,
    sample_interior, sample_walls, sample_flux, inlet_velocity)
from challenge.climate import climate_l1, climate_l2


class AnalyticField(torch.nn.Module):
    def __init__(self, function):
        super().__init__()
        self.function = function

    def forward(self, inputs):
        return self.function(inputs)


SMALL_CONFIG = {"model": {"width": 12, "layers": 2},
                "samples": {"interior": 8, "initial": 8, "boundary": 8,
                            "flux_lines": 2, "flux_points": 8}}


class WavePhysicsTests(unittest.TestCase):
    def test_corrected_constant_speed_reference_pde_initial_and_boundary(self):
        torch.manual_seed(31)
        xy = .1 + torch.rand(12, 2) * 2.8
        time = torch.rand(12, 1) * 6
        for speed in (.5, 1., 2.):
            model = AnalyticField(lambda a: wave_l1.exact_reference(a[:, :2], a[:, 2:3], speed))
            informer = create_informer(wave_l1.WaveEquation2D(reference=True, c=speed), "cpu")
            residual = residuals(model, informer, xy, time, ["u"])["wave"]
            self.assertLess(float(residual.abs().max()), 3e-5)
            initial_t = torch.zeros(len(xy), 1, requires_grad=True)
            initial = wave_l1.exact_reference(xy, initial_t, speed)
            target = torch.sin(xy[:, :1]) * torch.sin(xy[:, 1:2])
            torch.testing.assert_close(initial, target)
            torch.testing.assert_close(gradient(initial, initial_t), target)
            for dimension in (0, 1):
                for edge in (0., math.pi):
                    boundary = xy.clone()
                    boundary[:, dimension] = edge
                    self.assertLess(float(wave_l1.exact_reference(boundary, time, speed).abs().max()), 1e-6)

    def test_variable_speed_matches_nondivergence_equation(self):
        xy = torch.tensor([[.3, .7], [1.2, 2.1]])
        time = torch.tensor([[.2], [.8]])
        model = AnalyticField(lambda a: a.square().sum(1, keepdim=True))
        informer = create_informer(wave_l2.WaveEquation2D(reference=True), "cpu")
        actual = residuals(model, informer, xy, time, ["u"])["wave"]
        c = 1 + .5 * torch.sin(xy[:, :1]) * torch.cos(xy[:, 1:2])
        torch.testing.assert_close(actual, 2 - 4 * c.square())
        self.assertIsNone(wave_l2.exact_reference(xy, time))

    def test_circular_robin_normal_and_gaussian_initial_condition(self):
        boundary = torch.tensor([[1., 0.], [0., 1.], [-1., 0.], [0., -1.]])
        time = torch.zeros(4, 1)
        model = AnalyticField(lambda a: a[:, :2].square().sum(1, keepdim=True))
        # u=r^2=1 and outward derivative=2 on the unit circle: 1+.5*2=2.
        torch.testing.assert_close(wave_l3.boundary_residual(model, boundary, time), torch.full((4, 1), 2.))
        origin = torch.zeros(1, 2)
        torch.testing.assert_close(wave_l3.initial_displacement(origin), torch.tensor([[2 * math.exp(-1.8)]]))
        self.assertIsNone(wave_l3.exact_reference(boundary, time))


class FluidPhysicsTests(unittest.TestCase):
    def test_channel_geometry_and_flux_quadrature(self):
        for blocks in (SINGLE_BLOCK, THREE_BLOCKS):
            xy = sample_interior(256, blocks, "cpu")
            self.assertTrue(bool(outside_blocks(xy, blocks).all()))
            self.assertTrue(bool(((xy[:, 0] > -2.5) & (xy[:, 0] < 2.5)).all()))
            walls = sample_walls(256, blocks, "cpu")
            self.assertTrue(bool(((walls[:, 0] > -2.5) & (walls[:, 0] < 2.5)).all()))
            # Every no-slip sample lies on a channel wall or an exposed chip face.
            on_wall = torch.isclose(walls[:, 1].abs(), torch.tensor(.5))
            for xmin, xmax, top in blocks:
                on_wall |= (torch.isclose(walls[:, 0], torch.tensor(xmin)) | torch.isclose(walls[:, 0], torch.tensor(xmax))) & (walls[:, 1] <= top + 1e-6)
                on_wall |= torch.isclose(walls[:, 1], torch.tensor(top)) & (walls[:, 0] >= xmin - 1e-6) & (walls[:, 0] <= xmax + 1e-6)
            self.assertTrue(bool(on_wall.all()))
            flux_xy, time, height = sample_flux(4, 32, blocks, "cpu", time_end=10)
            self.assertTrue(bool(outside_blocks(flux_xy, blocks).all()))
            torch.testing.assert_close(time.reshape(4, 32), time.reshape(4, 32)[:, :1].expand(-1, 32))
            self.assertTrue(bool((height > 0).all()))
        xy, _, height = sample_flux(2, 1024, [], "cpu")
        flux = inlet_velocity(xy).reshape(2, 1024).mean(1, keepdim=True) * height
        torch.testing.assert_close(flux, torch.ones_like(flux), atol=2e-6, rtol=0)

    def test_navier_stokes_residual_against_manufactured_channel_field(self):
        # This is an exact unobstructed channel field, NOT a solution with chips.
        xy = torch.tensor([[-1.3, .1], [.3, -.2], [1.7, .3]])
        for module in (chip_2d_l1, chip_2d_l2, chip_2d_l3):
            # Upstream L3 reduces viscosity to .01; L1/2 retain .02.
            nu = .01 if module is chip_2d_l3 else .02
            def poiseuille(a):
                x, y = a[:, :1], a[:, 1:2]
                return torch.cat((1.5*(1-4*y.square()) + 0*x.pow(3), 0*(x.pow(3)+y.pow(3)), -12*nu*x + 0*y.pow(3)), dim=1)
            model = AnalyticField(poiseuille)
            informer = create_informer(module.NavierStokes2D(reference=True), "cpu")
            time = torch.ones(3, 1) if module.TIME_END is not None else None
            result = residuals(model, informer, xy, time, module.FIELD_NAMES)
            for name, value in result.items():
                self.assertLess(float(value.abs().max()), 1e-5, name)


class ClimatePhysicsTests(unittest.TestCase):
    def test_default_reference_pde_initial_and_boundary(self):
        xy = torch.tensor([[.3, .8], [1.2, 2.1], [2.4, .7]])
        time = torch.tensor([[.1], [.7], [1.4]])
        for module in (climate_l1, climate_l2):
            params = module.DEFAULT_PHYSICS.copy()
            model = AnalyticField(lambda a: module.exact_reference(a[:, :2], a[:, 2:3], params))
            informer = create_informer(module.ClimatePDE(reference=True, params=params), "cpu")
            for value in residuals(model, informer, xy, time, module.FIELD_NAMES).values():
                self.assertLess(float(value.abs().max()), 1e-5)
            initial = module.exact_reference(xy, torch.zeros_like(time), params)
            target = (torch.sin(xy[:, :1]) * torch.sin(xy[:, 1:2])).expand_as(initial)
            torch.testing.assert_close(initial, target)
            for axis in (0, 1):
                for edge in (0., math.pi):
                    boundary = xy.clone()
                    boundary[:, axis] = edge
                    self.assertLess(float(module.exact_reference(boundary, time, params).abs().max()), 1e-6)

    def test_coupling_equal_and_opposite_and_reference_disabled(self):
        params = {**climate_l2.DEFAULT_PHYSICS, "gamma0": .3}
        xy, time = torch.ones(3, 2), torch.ones(3, 1)
        # Constant field retains a zero-valued differentiable polynomial path.
        def fields(a):
            zero = 0 * a.pow(3).sum(1, keepdim=True)
            return torch.cat((2 + zero, 1 + zero), dim=1)
        informer = create_informer(climate_l2.ClimatePDE(reference=True, params=params), "cpu")
        result = residuals(AnalyticField(fields), informer, xy, time, ["Ta", "To"])
        torch.testing.assert_close(result["atmosphere"], torch.full((3, 1), .3))
        torch.testing.assert_close(result["ocean"], torch.full((3, 1), -.3))
        self.assertIsNone(climate_l2.exact_reference(xy, time, params))
        changed = {**climate_l1.DEFAULT_PHYSICS, "u0": 1.}
        self.assertIsNone(climate_l1.exact_reference(xy, time, changed))


class AllLevelTrainingTests(unittest.TestCase):
    def test_every_level_has_finite_loss_and_backward_and_repeatable_validation(self):
        modules = [wave_l1, wave_l2, wave_l3, chip_2d_l1, chip_2d_l2,
                   chip_2d_l3, climate_l1, climate_l2]
        for module in modules:
            with self.subTest(level=module.__name__):
                torch.manual_seed(7)
                pde_type = getattr(module, "WaveEquation2D", None) or getattr(module, "NavierStokes2D", None) or module.ClimatePDE
                with self.assertRaisesRegex(NotImplementedError, "--reference"):
                    pde_type()
                pde = pde_type(reference=True)
                informer = create_informer(pde, "cpu")
                model = create_model(2 if module.TIME_END is None else 3, len(module.FIELD_NAMES), SMALL_CONFIG, "cpu")
                losses = module.loss_terms(model, informer, SMALL_CONFIG, "cpu")
                total = sum(losses.values())
                self.assertTrue(bool(torch.isfinite(total)))
                total.backward()
                gradients = [p.grad for p in model.parameters() if p.grad is not None]
                self.assertTrue(gradients)
                self.assertTrue(all(bool(torch.isfinite(g).all()) for g in gradients))
                self.assertTrue(any(bool((g != 0).any()) for g in gradients))
                before = torch.random.get_rng_state().clone()
                first = heldout_losses(module.loss_terms, model, informer, SMALL_CONFIG, "cpu", 15)
                self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
                second = heldout_losses(module.loss_terms, model, informer, SMALL_CONFIG, "cpu", 15)
                for name in first:
                    torch.testing.assert_close(first[name], second[name])


if __name__ == "__main__":
    unittest.main()
