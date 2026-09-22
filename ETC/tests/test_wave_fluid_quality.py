"""CPU checks for the revised Wave initial field and Fluid startup conditions.

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
    def test_initial_value_and_normal_derivative_satisfy_robin(self):
        angle = torch.linspace(0, 2 * math.pi, 257, dtype=torch.float64)
        boundary = torch.stack((angle.cos(), angle.sin()), dim=1).requires_grad_()
        value = wave_l3.initial_displacement(boundary)
        normal = (gradient(value, boundary) * boundary).sum(1, keepdim=True)
        self.assertLess(float(value.detach().abs().max()), 1e-25)
        self.assertLess(float(normal.detach().abs().max()), 1e-14)
        model = AnalyticField(lambda inputs: wave_l3.initial_displacement(inputs[:, :2]))
        robin = wave_l3.boundary_residual(model, boundary, torch.zeros(257, 1, dtype=torch.float64))
        self.assertLess(float(robin.detach().abs().max()), 1e-14)

    def test_envelope_retains_symmetric_interior_pulses(self):
        points = torch.tensor([[-.25, 0.], [0., 0.], [.25, 0.], [.2, .3]], dtype=torch.float64)
        x, y = points[:, :1], points[:, 1:2]
        gaussians = (-20 * ((x - .3).square() + y.square())).exp()
        gaussians += (-20 * ((x + .3).square() + y.square())).exp()
        expected = (1 - x.square() - y.square()).square() * gaussians
        actual = wave_l3.initial_displacement(points)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(actual[0], actual[2])
        self.assertGreater(float(actual[0]), float(actual[1]))
        self.assertGreater(float(actual.min()), 0)


class FluidStartupTests(unittest.TestCase):
    def test_ramp_begins_with_zero_value_and_acceleration(self):
        for tau in (.25, 1., 2.):
            with self.subTest(tau=tau):
                time = torch.tensor([[0.], [.1], [1.], [10.]], dtype=torch.float64, requires_grad=True)
                ramp = fluid_l3.startup_ramp(time, tau)
                rate = gradient(ramp, time)
                self.assertEqual(float(ramp[0].detach()), 0.)
                self.assertEqual(float(rate[0].detach()), 0.)
                self.assertTrue(bool((ramp.detach().diff(dim=0) >= 0).all()))
                self.assertTrue(bool(((ramp >= 0) & (ramp <= 1)).all()))
                self.assertAlmostEqual(float(ramp[-1].detach()), 1., places=9)

    def test_timescale_has_expected_scaling_and_is_validated(self):
        config_path = ROOT / "02_challenges/02_fluid/conf/config_chip_2d.yaml"
        config = yaml.safe_load(config_path.read_text())
        self.assertEqual(fluid_l3.inlet_ramp_time(config), 1.)
        self.assertEqual(fluid_l3.inlet_ramp_time({}), 1.)
        for tau in (.25, 1., 2.):
            target = fluid_l3.startup_ramp(torch.tensor([[tau]], dtype=torch.float64), tau)
            self.assertAlmostEqual(float(target), 1 - math.exp(-1), places=12)
        for invalid in (0, -1, float("nan"), float("inf"), True, "1", None):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "positive and finite"):
                    fluid_l3.inlet_ramp_time({"physics": {"inlet_ramp_time": invalid}})
                with self.assertRaisesRegex(ValueError, "positive and finite"):
                    fluid_l3.startup_ramp(torch.zeros(1, 1), invalid)
        with self.assertRaisesRegex(ValueError, "mapping"):
            fluid_l3.inlet_ramp_time({"physics": None})

    def test_inlet_integral_equals_time_dependent_flux(self):
        count = 4096
        y = (torch.arange(count, dtype=torch.float64) + .5) / count - .5
        coordinates = torch.stack((torch.full_like(y, -2.5), y), dim=1)
        for tau in (.5, 2.):
            for time in (0., .05, .5, 3.):
                ramp = fluid_l3.startup_ramp(torch.tensor([[time]], dtype=torch.float64), tau)
                inlet = fluid_l3.inlet_velocity(coordinates) * ramp
                torch.testing.assert_close(inlet.mean(), ramp.squeeze(), atol=4e-8, rtol=0)

    def test_loss_uses_same_ramp_for_inlet_and_flux(self):
        # Unobstructed manufactured field tests the two condition targets only;
        # it is NOT asserted to solve the chip-flow PDE at positive times.
        config = {"physics": {"inlet_ramp_time": 2.},
                  "samples": {"interior": 4, "initial": 4, "boundary": 8,
                              "flux_lines": 3, "flux_points": 1024}}
        def ramped_profile(inputs):
            zero = 0 * inputs.sum(1, keepdim=True)
            u = fluid_l3.inlet_velocity(inputs[:, :2]) * fluid_l3.startup_ramp(inputs[:, 2:3], 2.)
            return torch.cat((u, zero, zero), dim=1)
        def zero_residuals(model, informer, xy, time, names):
            return {name: torch.zeros(len(xy), 1) for name in ("continuity", "momentum_x", "momentum_y")}
        torch.manual_seed(24)
        with patch.object(fluid_l3, "BLOCKS", []), patch.object(fluid_l3, "residuals", zero_residuals):
            losses = fluid_l3.loss_terms(AnalyticField(ramped_profile), None, config, "cpu")
        self.assertEqual(float(losses["inlet_u"]), 0.)
        self.assertEqual(float(losses["initial_rest"]), 0.)
        self.assertLess(float(losses["integral_continuity"]), 1e-11)

    def test_zero_initial_pressure_agrees_with_zero_initial_acceleration(self):
        def fields(inputs):
            zero = 0 * inputs.pow(3).sum(1, keepdim=True)
            u = fluid_l3.inlet_velocity(inputs[:, :2]) * fluid_l3.startup_ramp(inputs[:, 2:3])
            return torch.cat((u + zero, zero, zero), dim=1)
        xy = torch.tensor([[-2., .2], [.3, .3], [1., -.2]])
        time = torch.zeros(3, 1)
        informer = create_informer(fluid_l3.NavierStokes2D(reference=True), "cpu")
        actual = residuals(AnalyticField(fields), informer, xy, time, fluid_l3.FIELD_NAMES)
        for name, value in actual.items():
            self.assertLess(float(value.detach().abs().max()), 1e-7, name)

    def test_steady_levels_retain_unit_flux_target(self):
        config = {"physics": {"inlet_ramp_time": 2.},
                  "samples": {"interior": 4, "initial": 4, "boundary": 8,
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

    def test_revised_conditions_have_finite_backward(self):
        config = {"model": {"width": 8, "layers": 2},
                  "physics": {"inlet_ramp_time": .7},
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
