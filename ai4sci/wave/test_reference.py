"""CPU numerical checks for the physical reference; no training dependencies."""

import unittest
import numpy as np
from wave_reference import exact_solution


class WaveReferenceTests(unittest.TestCase):
    def test_initial_displacement_and_velocity(self):
        x, y = np.array([0.3, 1.2]), np.array([0.7, 2.3])
        expected = np.sin(x) * np.sin(y)
        for c in (0.5, 1.0, 2.0):
            np.testing.assert_allclose(exact_solution(x, y, 0, c), expected)
            h = 1e-5
            velocity = (exact_solution(x, y, h, c)-exact_solution(x, y, -h, c))/(2*h)
            np.testing.assert_allclose(velocity, expected, rtol=1e-8)

    def test_boundary_and_pde_for_multiple_speeds(self):
        x, y, t = 0.8, 1.3, 0.6
        h = 1e-4
        for c in (0.5, 1.0, 2.0):
            for edge in (0.0, np.pi):
                self.assertAlmostEqual(float(exact_solution(edge, y, t, c)), 0.0)
                self.assertAlmostEqual(float(exact_solution(x, edge, t, c)), 0.0)
            u = exact_solution(x, y, t, c)
            dtt = (exact_solution(x,y,t+h,c)-2*u+exact_solution(x,y,t-h,c))/h**2
            dxx = (exact_solution(x+h,y,t,c)-2*u+exact_solution(x-h,y,t,c))/h**2
            dyy = (exact_solution(x,y+h,t,c)-2*u+exact_solution(x,y-h,t,c))/h**2
            self.assertLess(abs(dtt-c**2*(dxx+dyy)), 2e-6)

    def test_invalid_speed(self):
        for c in (0.0, -1.0, np.inf, np.nan):
            with self.assertRaises(ValueError):
                exact_solution(0, 0, 0, c)


if __name__ == "__main__":
    unittest.main()
