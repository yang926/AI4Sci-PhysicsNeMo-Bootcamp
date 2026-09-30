"""Reference-free summaries of saved Lab 4 velocity predictions.

The inputs use the lesson's dimensionless coordinates, times, and velocities.
Speed has normalized velocity units; kinetic energy is per unit mass and has
squared normalized velocity units. No density or physical unit conversion is
applied. Pressure is checked for finite values but is not part of these velocity
statistics. If a physical velocity scale U is known, multiply speed/mean velocity
by U and energy by U**2; normalized time requires its own supplied time scale.

Retention compares each prediction with the *predicted* t=0 frame. It does not
measure fit to the supplied initial observations, accuracy at a future time, or
convergence. In particular, preserved energy alone does not establish accuracy.
"""

import numpy as np


def _finite_array(value, name):
    """Accept real numeric arrays and calculate reductions in float64."""
    array = np.asarray(value)
    if array.dtype.kind not in "fiu":
        raise ValueError(f"{name} must contain real numeric values.")
    array = array.astype(np.float64, copy=False)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values.")
    return array


def _validate_grid(xy):
    """Validate equal-weight samples on a complete uniform Cartesian grid."""
    x, y = np.unique(xy[:, 0]), np.unique(xy[:, 1])
    if (len(x) < 2 or len(y) < 2 or len(x) * len(y) != len(xy)
            or len(np.unique(xy, axis=0)) != len(xy)):
        raise ValueError("xy must form a complete rectangular grid without duplicates.")
    for axis in (x, y):
        # The saved grid is float32; allow its coordinate rounding, not
        # nonuniform point distributions that need quadrature weights.
        spacing = (axis[-1] - axis[0]) / (len(axis) - 1)
        if not np.allclose(np.diff(axis), spacing, rtol=2e-5,
                           atol=1e-7 * abs(spacing)):
            raise ValueError("xy must have uniform spacing along each axis.")


def summarize_flow(prediction, times, xy):
    """Return JSON-safe, per-frame statistics from arrays in predictions.npz.

    Args:
        prediction: Finite (T, N, 3) values ordered as u, v, p.
        times: Finite (T,) normalized times, strictly increasing from zero.
        xy: Finite (N, 2) coordinates on a complete uniform rectangular grid.
            Any point ordering is accepted if predictions use the same ordering.

    Spatial means give every saved grid point equal weight. For a periodic
    spatial integral, the caller must supply an endpoint-excluded grid so that
    periodic boundary points are not counted twice. This routine does not infer
    periodic domain lengths or estimate derivatives from ambiguous endpoints.

    speed_rms = sqrt(mean(u**2 + v**2)); mean_kinetic_energy is half that
    mean-square speed. Fluctuating kinetic energy first subtracts each frame's
    spatial mean velocity. Retention ratios divide by the corresponding value
    in the predicted t=0 frame; a zero baseline yields None at every frame.
    No tolerance or pass/fail convergence decision is imposed.
    """
    prediction = _finite_array(prediction, "prediction")
    times = _finite_array(times, "times")
    xy = _finite_array(xy, "xy")
    if prediction.ndim != 3 or prediction.shape[2] != 3:
        raise ValueError("prediction must have shape (T, N, 3) for u, v, p.")
    frames, points, _ = prediction.shape
    if frames == 0 or points == 0:
        raise ValueError("prediction must contain at least one frame and one point.")
    if times.shape != (frames,):
        raise ValueError("times must have shape (T,) matching prediction.")
    if times[0] != 0 or np.any(np.diff(times) <= 0):
        raise ValueError("times must start at zero and be strictly increasing.")
    if xy.shape != (points, 2):
        raise ValueError("xy must have shape (N, 2) matching prediction.")
    _validate_grid(xy)

    velocity = prediction[:, :, :2]
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        mean_velocity = np.mean(velocity, axis=1)
        speed_squared = np.mean(np.sum(velocity**2, axis=2), axis=1)
        energy = .5 * speed_squared
        speed_rms = np.sqrt(speed_squared)
        fluctuations = velocity - mean_velocity[:, None, :]
        fluctuating_energy = .5 * np.mean(np.sum(fluctuations**2, axis=2), axis=1)
    if not all(np.isfinite(values).all() for values in
               (mean_velocity, speed_rms, energy, fluctuating_energy)):
        raise ValueError("Velocity statistics overflowed; check the input scale.")

    def retention(values, index):
        if values[0] == 0:
            return None
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            ratio = float(values[index] / values[0])
        if not np.isfinite(ratio):
            raise ValueError("A retention ratio overflowed; check the input scale.")
        return ratio

    return {
        "baseline": "predicted_t0",
        "spatial_average": "equal_weight_saved_grid_points",
        "units": {"time": "dimensionless", "velocity": "dimensionless",
                  "kinetic_energy": "dimensionless_velocity_squared_per_unit_mass"},
        "per_time": [
            {"time": float(time),
             "speed_rms": float(speed_rms[index]),
             "mean_kinetic_energy": float(energy[index]),
             "mean_u": float(mean_velocity[index, 0]),
             "mean_v": float(mean_velocity[index, 1]),
             "fluctuating_kinetic_energy": float(fluctuating_energy[index]),
             "speed_retention_ratio": retention(speed_rms, index),
             "energy_retention_ratio": retention(energy, index),
             "fluctuating_energy_retention_ratio": retention(fluctuating_energy, index)}
            for index, time in enumerate(times)
        ],
    }
