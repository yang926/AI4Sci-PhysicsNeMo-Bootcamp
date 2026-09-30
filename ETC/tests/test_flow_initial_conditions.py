"""CPU checks of derived-input math and parity with the measured experiment."""
import hashlib
import os
from pathlib import Path

import numpy as np
import pytest

from ETC.runtime import flow_initial_conditions as initial


def grid(n=32, length=1.44, lower=-.72):
    axis = lower + length * np.arange(n) / n
    x, y = np.meshgrid(axis, axis, indexing='xy')
    return np.column_stack((x.ravel(), y.ravel()))


def test_projection_divergence_idempotence_mean_and_energy():
    velocity = np.random.default_rng(319).normal(size=(2, 32, 32))
    projected, info = initial._project_velocity(velocity, 1.44)
    twice, _ = initial._project_velocity(projected, 1.44)
    np.testing.assert_allclose(twice, projected, atol=2e-15, rtol=2e-15)
    np.testing.assert_allclose(projected.mean(axis=(1, 2)), velocity.mean(axis=(1, 2)), atol=2e-16)
    assert info['projected_float64_divergence']['rms'] < 2e-13
    assert info['projected_energy'] <= info['retained_before_projection_energy'] <= info['original_energy']
    assert info['orthogonal_energy_identity_abs_error'] < 5e-15


def test_even_nyquist_removal_is_explicit_and_measured():
    alternating = (-1.) ** np.arange(32)
    velocity = np.stack((np.broadcast_to(alternating, (32, 32)), np.zeros((32, 32))))
    projected, info = initial._project_velocity(velocity, 1.44)
    np.testing.assert_allclose(projected, 0, atol=1e-15)
    assert info['nyquist_planes_removed_energy'] == pytest.approx(.5)
    assert info['nyquist_planes_removed_fraction_of_original_energy'] == pytest.approx(1.)


def test_analytic_pressure_poisson_sign_gauge_and_unchanged_compatible_velocity():
    axis = 2 * np.pi * np.arange(32) / 32
    x, y = np.meshgrid(axis, axis, indexing='xy')
    velocity = np.stack((-np.cos(x) * np.sin(y), np.sin(x) * np.cos(y)))
    projected, _ = initial._project_velocity(velocity, 2 * np.pi)
    np.testing.assert_allclose(projected, velocity, atol=2e-15)
    pressure, info = initial._pressure_from_velocity(projected, 5.3, 2 * np.pi)
    exact = -.25 * (np.cos(2 * x) + np.cos(2 * y)) + 5.3
    np.testing.assert_allclose(pressure, exact, atol=3e-15)
    assert info['poisson_residual_float64_rms'] < 1e-13


def test_three_halves_products_exclude_aliases():
    axis = 2 * np.pi * np.arange(32) / 32
    x, _ = np.meshgrid(axis, axis, indexing='xy')
    spectrum = np.fft.fft2(np.cos(11 * x))
    spectrum[initial._nyquist_mask(spectrum.shape)] = 0
    padded = initial._real_inverse(initial._resize_spectrum(spectrum, (48, 48)))
    product_h = initial._resize_spectrum(np.fft.fft2(padded**2), (32, 32))
    product_h[initial._nyquist_mask(product_h.shape)] = 0
    np.testing.assert_allclose(initial._real_inverse(product_h), .5, atol=8e-15)


def test_public_function_preserves_inputs_and_reports_derived_semantics():
    xy = grid().astype(np.float32)
    fields = np.random.default_rng(20).normal(size=(len(xy), 3)).astype(np.float32)
    fields[:, 2] += 5.3
    old_xy, old_fields = xy.copy(), fields.copy()
    result, report = initial.derived_initial_conditions(xy, fields, source_context={'source': 'unit-test'})
    np.testing.assert_array_equal(xy, old_xy)
    np.testing.assert_array_equal(fields, old_fields)
    assert result.dtype == np.float32 and result.shape == fields.shape
    assert report['data_kind'] == initial.DERIVED_DATA_KIND
    assert report['source_context'] == {'source': 'unit-test'}
    assert report['source_context_verified_by_helper'] is False
    assert 'NOT original ERA5' in report['pressure']['pressure_semantics']
    assert report['pressure']['pressure_mean_gauge_actual_saved_float32'] == pytest.approx(float(fields[:, 2].astype(np.float64).mean()), abs=1e-7)


def test_zero_velocity_has_defined_pressure_and_undefined_energy_ratios():
    xy = grid()
    fields = np.zeros((len(xy), 3), dtype=np.float32)
    fields[:, 2] = 5.3
    result, report = initial.derived_initial_conditions(xy, fields)
    np.testing.assert_array_equal(result, fields)
    assert report['projection']['projected_energy_fraction_of_original'] is None
    assert report['pressure']['centered_pressure_rms'] == 0.


@pytest.mark.parametrize('change', ['nan', 'complex', 'missing_point', 'reordered', 'repeated_endpoint', 'nonuniform'])
def test_invalid_grids_or_fields_are_rejected(change):
    xy = grid()
    fields = np.zeros((len(xy), 3))
    if change == 'nan':
        fields[0, 0] = np.nan
    elif change == 'complex':
        fields = fields.astype(complex)
    elif change == 'missing_point':
        xy, fields = xy[:-1], fields[:-1]
    elif change == 'reordered':
        xy = xy[::-1]
    elif change == 'repeated_endpoint':
        xy = xy * (32 / 31)
    else:
        xy[0, 0] += .001
    with pytest.raises(ValueError):
        initial.derived_initial_conditions(xy, fields)


def test_original_full_array_matches_measured_experiment_bit_for_bit():
    source = Path(__file__).resolve().parents[2] / 'ETC/reference_labs/04_navier_stokes/data_lat.npy'
    reference_path = os.environ.get('AI4SCI_LAB4_COMPATIBLE_REFERENCE')
    if not reference_path:
        pytest.skip('Set AI4SCI_LAB4_COMPATIBLE_REFERENCE for full measured-artifact parity')
    measured = Path(reference_path)
    if not source.exists() or not measured.exists():
        pytest.skip('Full original source and separately measured preparation are not available')
    before_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    raw = np.load(source, allow_pickle=False).astype(np.float32)
    velocity_scale = (12742000 / 1.44) / (60 * 60 * 60)
    pressure_scale = 1.1614 * velocity_scale**2
    fields = np.column_stack((raw[0].ravel() / velocity_scale, raw[1].ravel() / velocity_scale,
                              raw[2].ravel() * .10197 / pressure_scale)).astype(np.float32)
    with np.load(measured, allow_pickle=False) as expected:
        xy = expected['xy']
        np.testing.assert_array_equal(fields, expected['original_normalized'])
        result, report = initial.derived_initial_conditions(xy, fields, source_context={'sha256': before_hash})
        np.testing.assert_array_equal(result, expected['compatible_normalized'])
    assert report['projection']['projected_energy_fraction_of_original'] == pytest.approx(.6690646981466705, rel=1e-13)
    assert report['projection']['saved_float32_divergence']['rms'] < 1e-5
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before_hash
