"""CPU checks for restored original Wave initial field and Fluid startup conditions.

These analytic and small backward tests do not claim training convergence.
"""
import importlib
import math
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
import yaml
from ETC.runtime.pinn import create_informer, create_model, gradient, residuals
from ETC.runtime.exercises import conditions, condition_tensor

wave_l3 = importlib.import_module("02_challenges.01_wave.wave_l3")
fluid_l1, fluid_l2, fluid_l3 = (
    importlib.import_module(f"02_challenges.02_fluid.chip_2d_l{level}")
    for level in (1, 2, 3)
)


class AnalyticField(torch.nn.Module):
    def __init__(self, function):
        super().__init__()
        self.function = function

    def forward(self, inputs):
        return self.function(inputs)


class WaveInitialCompatibilityTests(unittest.TestCase):
    def test_original_initial_boundary_mismatch_is_visible_not_silently_tapered(self):
        angle = torch.linspace(0, 2 * math.pi, 257, dtype=torch.float64)
        boundary = torch.stack((angle.cos(), angle.sin()), dim=1).requires_grad_()
        value = wave_l3.initial_displacement(boundary)
        normal = (gradient(value, boundary) * boundary).sum(1, keepdim=True)
        self.assertGreater(float(value.detach().abs().max()), 1e-6)
        self.assertGreater(float(normal.detach().abs().max()), 1e-5)
        model = AnalyticField(lambda inputs: wave_l3.initial_displacement(inputs[:, :2]))
        robin = wave_l3.boundary_residual(model, boundary, torch.zeros(257, 1, dtype=torch.float64))
        torch.testing.assert_close(robin, value + .5 * normal)
        self.assertGreater(float(robin.detach().abs().max()), 1e-5)

    def test_original_gaussians_retain_symmetric_interior_pulses(self):
        points = torch.tensor([[-.25, 0.], [0., 0.], [.25, 0.], [.2, .3]], dtype=torch.float64)
        x, y = points[:, :1], points[:, 1:2]
        gaussians = (-20 * ((x - .3).square() + y.square())).exp()
        gaussians += (-20 * ((x + .3).square() + y.square())).exp()
        expected = gaussians
        actual = wave_l3.initial_displacement(points)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(actual[0], actual[2])
        self.assertGreater(float(actual[0]), float(actual[1]))
        self.assertGreater(float(actual.min()), 0)


class FluidStartupTests(unittest.TestCase):
    def test_original_startup_has_rest_initial_data_and_nonzero_inlet(self):
        expression = conditions(fluid_l3.reference_conditions)
        xy = torch.tensor([[-2.5, 0.], [-2.5, .25]])
        for time in (0., .1, 10.):
            t = torch.full((2, 1), time)
            inlet = condition_tensor(expression, "inlet_u", xy, t)
            torch.testing.assert_close(inlet, fluid_l3.inlet_velocity(xy))
            for name in ("u", "v", "p"):
                torch.testing.assert_close(condition_tensor(expression, "initial_" + name, xy, t), torch.zeros(2, 1))
        self.assertGreater(float(inlet[0]), 0.)

    def test_no_unrequested_ramp_parameter_remains(self):
        config = yaml.safe_load((ROOT / "02_challenges/02_fluid/conf/config_chip_2d.yaml").read_text())
        self.assertNotIn("inlet_ramp_time", config.get("physics", {}))
        self.assertFalse(hasattr(fluid_l3, "startup_ramp"))

    def test_original_inlet_integral_equals_unit_flux(self):
        count = 4096
        y = (torch.arange(count, dtype=torch.float64) + .5) / count - .5
        coordinates = torch.stack((torch.full_like(y, -2.5), y), dim=1)
        expression = conditions(fluid_l3.reference_conditions)
        for time in (0., .05, .5, 3.):
            t = torch.full((count, 1), time, dtype=torch.float64)
            inlet = condition_tensor(expression, "inlet_u", coordinates, t)
            flux = condition_tensor(expression, "flux", coordinates[:1], t[:1])
            torch.testing.assert_close(inlet.mean(), flux.squeeze(), atol=4e-8, rtol=0)

    def test_loss_uses_original_unit_flux_and_exposes_initial_mismatch(self):
        # This unobstructed profile tests targets, not the chip-flow solution.
        config = {"samples": {"interior": 4, "initial": 4, "boundary": 8,
                              "flux_lines": 3, "flux_points": 1024}}
        def profile(inputs):
            zero = 0 * inputs.sum(1, keepdim=True)
            return torch.cat((fluid_l3.inlet_velocity(inputs[:, :2]), zero, zero), dim=1)
        def zero_residuals(model, informer, xy, time, names):
            return {name: torch.zeros(len(xy), 1) for name in ("continuity", "momentum_x", "momentum_y")}
        torch.manual_seed(24)
        with patch.object(fluid_l3, "residuals", zero_residuals):
            losses = fluid_l3.loss_terms(AnalyticField(profile), None, config, "cpu", geometry=[])
        # SymPy expands 1.5*(1-4*y**2) to 1.5-6*y**2; FP32 roundoff differs.
        self.assertLess(float(losses["inlet_u"]), 1e-12)
        self.assertGreater(float(losses["initial_rest"]), 0.)
        self.assertLess(float(losses["integral_continuity"]), 1e-11)

    def test_rest_field_satisfies_pde_but_not_nonzero_inlet(self):
        model = AnalyticField(lambda inputs: (0 * inputs.pow(3).sum(1, keepdim=True)).expand(-1, 3))
        config = {"samples": {"interior": 4, "initial": 4, "boundary": 8,
                              "flux_lines": 2, "flux_points": 8}}
        informer = create_informer(fluid_l3.NavierStokes2D(reference=True), "cpu")
        losses = fluid_l3.loss_terms(model, informer, config, "cpu")
        self.assertEqual(float(losses["initial_rest"]), 0.)
        self.assertGreater(float(losses["inlet_u"]), 0.)
        self.assertEqual(float(losses["integral_continuity"]), 1.)

    def test_steady_levels_retain_unit_flux_target(self):
        config = {"samples": {"interior": 4, "initial": 4, "boundary": 8,
                              "flux_lines": 2, "flux_points": 8}}
        for module in (fluid_l1, fluid_l2):
            with self.subTest(level=module.LEVEL):
                model = AnalyticField(
                    lambda inputs: (0 * inputs.pow(3).sum(1, keepdim=True)).expand(-1, 3)
                )
                informer = create_informer(module.NavierStokes2D(reference=True), "cpu")
                losses = module.loss_terms(model, informer, config, "cpu")
                self.assertEqual(float(losses["integral_continuity"]), 1.)
                self.assertNotIn("initial_rest", losses)

    def test_restored_conditions_have_finite_backward(self):
        config = {"model": {"width": 8, "layers": 2},
                  "samples": {"interior": 4, "initial": 4, "boundary": 4,
                              "flux_lines": 2, "flux_points": 8}}
        for module, pde_type in ((wave_l3, wave_l3.WaveEquation2D),
                                 (fluid_l3, fluid_l3.NavierStokes2D)):
            with self.subTest(module=module.__name__):
                model = create_model(3, len(module.FIELD_NAMES), config, "cpu")
                informer = create_informer(pde_type(reference=True), "cpu")
                total = sum(module.loss_terms(model, informer, config, "cpu").values())
                self.assertTrue(bool(torch.isfinite(total)))
                total.backward()
                grads = [p.grad for p in model.parameters() if p.grad is not None]
                self.assertTrue(grads)
                self.assertTrue(all(bool(torch.isfinite(g).all()) for g in grads))


if __name__ == "__main__":
    unittest.main()
