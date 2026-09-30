"""Explicitly derived periodic incompressible initial conditions, using NumPy.

The caller supplies the original normalized fields. Nothing reads, writes,
normalizes, caches, or replaces a source dataset implicitly. Returned pressure
is recalculated planar pressure, not the source weather pressure.
"""
from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np

DERIVED_DATA_KIND = "derived_periodic_incompressible_initial_conditions"


def _modes(shape, length):
    ny, nx = shape
    kx = 2 * np.pi * np.fft.fftfreq(nx, d=length / nx)[None, :]
    ky = 2 * np.pi * np.fft.fftfreq(ny, d=length / ny)[:, None]
    return kx, ky, kx**2 + ky**2


def _nyquist_mask(shape):
    ny, nx = shape
    mask = np.zeros(shape, dtype=bool)
    if ny % 2 == 0:
        mask[ny // 2, :] = True
    if nx % 2 == 0:
        mask[:, nx // 2] = True
    return mask


def _real_inverse(spectrum):
    value = np.fft.ifft2(spectrum, axes=(-2, -1))
    imaginary = float(np.max(np.abs(value.imag)))
    if imaginary > 1e-10 * max(1., float(np.max(np.abs(value.real)))):
        raise ArithmeticError(f'Fourier array lost real Hermitian symmetry: {imaginary}')
    return value.real


def _energy(velocity):
    return float(.5 * np.mean(np.sum(velocity**2, axis=0)))


def _divergence_metrics(velocity, length):
    spectra = np.fft.fft2(velocity, axes=(-2, -1))
    spectra[:, _nyquist_mask(velocity.shape[-2:])] = 0
    kx, ky, _ = _modes(velocity.shape[-2:], length)
    divergence = _real_inverse(1j * (kx * spectra[0] + ky * spectra[1]))
    return {'rms': float(np.sqrt(np.mean(divergence**2))),
            'max_abs': float(np.max(np.abs(divergence)))}


def _project_velocity(velocity, length):
    velocity = np.asarray(velocity, dtype=np.float64)
    spectra = np.fft.fft2(velocity, axes=(-2, -1))
    count = np.prod(velocity.shape[-2:])
    mask = _nyquist_mask(velocity.shape[-2:])
    nyquist_energy = float(.5 * np.sum(np.abs(spectra[:, mask])**2) / count**2)
    spectra[:, mask] = 0
    retained = _real_inverse(spectra)
    kx, ky, k_squared = _modes(velocity.shape[-2:], length)
    potential = (kx * spectra[0] + ky * spectra[1]) / np.where(k_squared == 0, 1., k_squared)
    projected_h = np.stack((spectra[0] - kx * potential, spectra[1] - ky * potential))
    projected = _real_inverse(projected_h)
    removed = retained - projected
    original_energy = _energy(velocity)
    diagnostics = {
        'original_energy': original_energy,
        'nyquist_planes_removed_energy': nyquist_energy,
        'nyquist_planes_removed_fraction_of_original_energy': nyquist_energy / original_energy if original_energy else None,
        'retained_before_projection_energy': _energy(retained),
        'projected_energy': _energy(projected),
        'divergent_component_removed_energy': _energy(removed),
        'projected_energy_fraction_of_original': _energy(projected) / original_energy if original_energy else None,
        'orthogonal_energy_identity_abs_error': abs(original_energy - nyquist_energy - _energy(projected) - _energy(removed)),
        'original_velocity_mean': velocity.mean(axis=(1, 2)).tolist(),
        'projected_velocity_mean': projected.mean(axis=(1, 2)).tolist(),
        'maximum_mean_change': float(np.max(np.abs(velocity.mean(axis=(1, 2)) - projected.mean(axis=(1, 2))))),
        'original_divergence': _divergence_metrics(velocity, length),
        'projected_float64_divergence': _divergence_metrics(projected, length),
    }
    return projected, diagnostics


def _resize_spectrum(spectrum, shape):
    """Preserve signed modes and physical amplitudes when resizing FFT grids."""
    old_y, old_x = spectrum.shape
    new_y, new_x = shape
    if new_y >= old_y and new_x >= old_x:
        iy = np.rint(np.fft.fftfreq(old_y) * old_y).astype(int) % new_y
        ix = np.rint(np.fft.fftfreq(old_x) * old_x).astype(int) % new_x
        result = np.zeros(shape, dtype=np.complex128)
        result[np.ix_(iy, ix)] = spectrum
    elif new_y <= old_y and new_x <= old_x:
        iy = np.rint(np.fft.fftfreq(new_y) * new_y).astype(int) % old_y
        ix = np.rint(np.fft.fftfreq(new_x) * new_x).astype(int) % old_x
        result = spectrum[np.ix_(iy, ix)].copy()
    else:
        raise ValueError('Both axes must be upsampled or downsampled together')
    return result * ((new_y * new_x) / (old_y * old_x))


def _pressure_from_velocity(velocity, pressure_mean, length):
    """Solve Delta p = -partial_i partial_j(u_i u_j), with 3/2 dealiasing."""
    velocity = np.asarray(velocity, dtype=np.float64)
    shape = velocity.shape[-2:]
    padded_shape = tuple(int(np.ceil(1.5 * value)) for value in shape)
    spectra = np.fft.fft2(velocity, axes=(-2, -1))
    spectra[:, _nyquist_mask(shape)] = 0
    up, vp = (_real_inverse(_resize_spectrum(value, padded_shape)) for value in spectra)
    kx, ky, k_squared = _modes(shape, length)
    quadratic = np.zeros(shape, dtype=np.complex128)
    for product, multiplier in ((up * up, kx**2), (up * vp, 2 * kx * ky), (vp * vp, ky**2)):
        product_h = _resize_spectrum(np.fft.fft2(product), shape)
        quadratic += multiplier * product_h
    mask = _nyquist_mask(shape)
    pressure_h = -quadratic / np.where(k_squared == 0, 1., k_squared)
    discarded = float(np.sum(np.abs(pressure_h[mask])**2))
    total = float(np.sum(np.abs(pressure_h)**2))
    pressure_h[mask] = 0
    quadratic[mask] = 0
    pressure_h[0, 0] = 0
    centered = _real_inverse(pressure_h)
    pressure = centered + float(pressure_mean)
    residual = _real_inverse(-k_squared * pressure_h - quadratic)
    rhs = _real_inverse(quadratic)
    return pressure, {
        'padded_product_grid_shape': list(padded_shape),
        'dealiasing': '3/2 zero padding; physical-space quadratic products; signed-mode truncation with FFT normalization preserved',
        'poisson_definition': 'Delta p = -partial_i partial_j(u_i u_j), using projected divergence-free velocity',
        'pressure_mean_gauge_requested': float(pressure_mean),
        'pressure_mean_gauge_actual_float64': float(pressure.mean()),
        'centered_pressure_rms': float(np.sqrt(np.mean(centered**2))),
        'pressure_nyquist_planes_discarded_spectral_norm_fraction': discarded / total if total else 0.,
        'poisson_rhs_rms': float(np.sqrt(np.mean(rhs**2))),
        'poisson_residual_float64_rms': float(np.sqrt(np.mean(residual**2))),
        'pressure_semantics': 'Derived planar incompressible pressure, NOT original ERA5-derived pressure; original pressure mean retained only as gauge.',
    }


def derived_initial_conditions(xy, fields, *, length=1.440, source_context=None):
    """Return new FP32 normalized fields and explicit derivation diagnostics.

    ``xy`` is an x-fast, uniformly spaced, endpoint-excluded periodic square
    grid, shaped (N,2); ``fields`` is its already-normalized (N,3) u/v/p array.
    No low-pass cutoff is applied. Only ambiguous even-grid Nyquist planes are
    removed; their energy is reported. Inputs remain untouched. The optional
    source context is copied into the report without claiming it was verified.
    """
    if isinstance(length, bool) or not isinstance(length, (int, float)) or not math.isfinite(length) or length <= 0:
        raise ValueError('length must be finite and positive')
    if source_context is not None and not isinstance(source_context, Mapping):
        raise ValueError('source_context must be a mapping or None')
    xy, fields = np.asarray(xy), np.asarray(fields)
    if (xy.ndim != 2 or xy.shape[1] != 2 or fields.shape != (len(xy), 3)
            or xy.dtype.kind not in 'fiu' or fields.dtype.kind not in 'fiu'
            or not np.isfinite(xy).all() or not np.isfinite(fields).all()):
        raise ValueError('Expected finite real coordinates (N,2) and normalized fields (N,3)')
    x, y = np.unique(xy[:, 0]), np.unique(xy[:, 1])
    nx, ny = len(x), len(y)
    if nx < 4 or ny < 4 or nx * ny != len(xy):
        raise ValueError('Expected a complete rectangular grid with at least four points per axis')
    grid = xy.reshape(ny, nx, 2)
    if (not np.array_equal(grid[:, :, 0], np.broadcast_to(x, (ny, nx)))
            or not np.array_equal(grid[:, :, 1], np.broadcast_to(y[:, None], (ny, nx)))):
        raise ValueError('Expected x-fast Cartesian coordinate ordering')
    tolerance = 4 * np.finfo(np.float32).eps * max(1., length)
    for axis in (x, y):
        expected = float(axis[0]) + length * np.arange(len(axis)) / len(axis)
        if not np.allclose(axis, expected, atol=tolerance, rtol=0):
            raise ValueError('Grid must be uniform and exclude the repeated periodic endpoint')
    original = fields.astype(np.float64).T.reshape(3, ny, nx)
    projected, projection_info = _project_velocity(original[:2], length)
    pressure, pressure_info = _pressure_from_velocity(projected, original[2].mean(), length)
    compatible = np.column_stack((projected[0].ravel(), projected[1].ravel(), pressure.ravel())).astype(np.float32)
    if not np.isfinite(compatible).all():
        raise FloatingPointError('Derived initial fields are nonfinite in FP32')
    projection_info['saved_float32_divergence'] = _divergence_metrics(compatible[:, :2].astype(np.float64).T.reshape(2, ny, nx), length)
    pressure_info['pressure_mean_gauge_actual_saved_float32'] = float(compatible[:, 2].astype(np.float64).mean())
    return compatible, {
        'data_kind': DERIVED_DATA_KIND,
        'source_context': dict(source_context or {}),
        'source_context_verified_by_helper': False,
        'grid_shape_yx': [ny, nx], 'periodic_length': float(length),
        'input_units': 'already normalized; no velocity or pressure conversion performed',
        'output_dtype': 'float32',
        'projection': projection_info, 'pressure': pressure_info,
        'nyquist_policy': 'Remove full even-grid Nyquist row/column in velocity and derived pressure; no arbitrary low-pass band. Removed energy and pressure spectral fraction are reported.',
        'physical_change': 'Original wind is replaced in this explicit derived experiment by its periodic divergence-free Helmholtz component; pressure is recalculated. This changes the initial-value problem.',
        'weather_forecast_validated': False,
    }
