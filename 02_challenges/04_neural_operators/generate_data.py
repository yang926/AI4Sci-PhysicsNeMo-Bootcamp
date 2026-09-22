"""Reproducible periodic data for u - Laplacian(u) = f on [0, 1)^2.

Generate the forcing f as a random tensor-product Fourier series, and divide
its coefficients by 1 + (2*pi)^2*(k^2+l^2) to obtain the exact solution u.
The endpoint 1 is excluded: periodic FFTs require n distinct points, dx=1/n.
The historical datasets/Poisson_Fourier files are not used by this generator.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import h5py
import numpy as np
import torch

SCHEMA = "reaction_diffusion_periodic_v2"
MODULE_DIR = Path(__file__).resolve().parent


def _positive_integer(value, name, *, minimum=1):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _validate_fourier_settings(grid_size, max_mode, batch_size):
    _positive_integer(grid_size, "grid_size", minimum=4)
    _positive_integer(max_mode, "max_mode")
    _positive_integer(batch_size, "batch_size")
    if 2 * max_mode >= grid_size:
        raise ValueError("max_mode must be strictly below the grid's Nyquist frequency")


def make_unit_square_grid(n: int, *, dtype=torch.float64):
    _positive_integer(n, "grid_size", minimum=4)
    axis = torch.arange(n, dtype=dtype) / n
    return torch.meshgrid(axis, axis, indexing="ij")


@torch.no_grad()
def generate_batch_fourier_series(batch_size: int, grid_size: int, max_mode: int,
                                  *, generator=None, dtype=torch.float64):
    """Return (f, u), both [batch, grid_size, grid_size], from exact coefficients.

    k and l index Fourier modes; x and y index physical grid points. The
    contraction bkl,kx,ly->bxy keeps both physical axes instead of summing one.
    Frequencies stay strictly below Nyquist so derivatives are unambiguous.
    """
    _validate_fourier_settings(grid_size, max_mode, batch_size)
    x = torch.arange(grid_size, dtype=dtype) / grid_size
    modes = torch.arange(max_mode + 1, dtype=dtype)
    phase = 2 * math.pi * modes[:, None] * x[None, :]
    sin_basis, cos_basis = phase.sin(), phase.cos()
    eigenvalues = 1 + (2 * math.pi) ** 2 * (modes[:, None] ** 2 + modes[None, :] ** 2)
    f = torch.zeros(batch_size, grid_size, grid_size, dtype=dtype)
    u = torch.zeros_like(f)
    for basis_x, basis_y in [(sin_basis, sin_basis), (sin_basis, cos_basis),
                             (cos_basis, sin_basis), (cos_basis, cos_basis)]:
        coeff = torch.randn(batch_size, max_mode + 1, max_mode + 1,
                            generator=generator, dtype=dtype) * 0.5
        # Choose a zero-mean forcing distribution. The reaction term also
        # permits nonzero constant modes; this is a dataset choice, not a gauge.
        coeff[:, 0, 0] = 0.0
        f += torch.einsum("bkl,kx,ly->bxy", coeff, basis_x, basis_y)
        u += torch.einsum("bkl,kx,ly->bxy", coeff / eigenvalues, basis_x, basis_y)
    return f, u


def spectral_residual(u: torch.Tensor, f: torch.Tensor):
    """Independent torch FFT evaluation of u - Delta(u) - f on a unit torus."""
    if u.shape != f.shape or u.ndim < 2 or min(u.shape[-2:]) < 1:
        raise ValueError("Spectral residual requires matching fields with two nonempty spatial axes")
    if u.dtype not in (torch.float32, torch.float64) or u.dtype != f.dtype or u.device != f.device:
        raise ValueError("Spectral fields must share a float32/float64 dtype and device")
    nx, ny = u.shape[-2:]
    kx = 2 * math.pi * torch.fft.fftfreq(nx, d=1 / nx, device=u.device, dtype=u.dtype)
    ky = 2 * math.pi * torch.fft.fftfreq(ny, d=1 / ny, device=u.device, dtype=u.dtype)
    multiplier = -(kx[:, None] ** 2 + ky[None, :] ** 2)
    laplacian = torch.fft.ifft2(torch.fft.fft2(u) * multiplier).real
    return u - laplacian - f


def generate_and_save_dataset(filename, num_samples, grid_size=64, max_mode=6,
                              batch_size=200, *, seed=42, split="train"):
    """Write batches without retaining the whole dataset in memory."""
    # Validate before creating a file: an invalid exercise setting must not
    # leave a partial dataset that blocks the student's corrected command.
    _positive_integer(num_samples, "num_samples")
    _validate_fourier_settings(grid_size, max_mode, batch_size)
    if split not in ("train", "val", "test"):
        raise ValueError("split must be train, val or test")
    filename = Path(filename)
    if filename.exists():
        raise FileExistsError(f"Dataset exists: {filename}. Choose a new --output-dir; existing data is preserved.")
    filename.parent.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator().manual_seed(seed)
    max_relative_residual = 0.0
    with h5py.File(filename, "x") as hf:
        hf.attrs.update(schema=SCHEMA, pde="u-laplacian(u)=f", domain="[0,1)^2",
                        grid_size=grid_size, max_mode=max_mode, seed=seed, split=split,
                        endpoint_included=False)
        shape = (num_samples, 1, grid_size, grid_size)
        f_store = hf.create_dataset("f", shape=shape, dtype="f4", compression="gzip")
        u_store = hf.create_dataset("u", shape=shape, dtype="f4", compression="gzip")
        for start in range(0, num_samples, batch_size):
            count = min(batch_size, num_samples - start)
            f, u = generate_batch_fourier_series(count, grid_size, max_mode, generator=generator)
            relative = spectral_residual(u, f).square().mean().sqrt() / f.square().mean().sqrt()
            max_relative_residual = max(max_relative_residual, relative.item())
            f_store[start:start + count] = f[:, None].float().numpy()
            u_store[start:start + count] = u[:, None].float().numpy()
        hf.attrs["analytical_relative_pde_residual_float64"] = max_relative_residual
    return {"file": str(filename), "samples": num_samples, "seed": seed,
            "analytical_relative_pde_residual_float64": max_relative_residual}


def generate_splits(output_dir, *, train_samples=8000, val_samples=1000, test_samples=1000,
                    grid_size=64, max_mode=6, seed=42, batch_size=200):
    _validate_fourier_settings(grid_size, max_mode, batch_size)
    for name, count in (("train_samples", train_samples), ("val_samples", val_samples),
                        ("test_samples", test_samples)):
        _positive_integer(count, name)
    output_dir = Path(output_dir)
    files = [output_dir / f"{split}.hdf5" for split in ("train", "val", "test")]
    if any(path.exists() for path in files):
        raise FileExistsError(f"Dataset already exists in {output_dir}; choose a fresh --output-dir.")
    children = np.random.SeedSequence(seed).spawn(3)
    records = []
    for path, count, split, child in zip(files, (train_samples, val_samples, test_samples),
                                        ("train", "val", "test"), children):
        split_seed = int(child.generate_state(1, dtype=np.uint32)[0])
        records.append(generate_and_save_dataset(path, count, grid_size, max_mode, batch_size,
                                                 seed=split_seed, split=split))
    manifest = {"schema": SCHEMA, "pde": "u-laplacian(u)=f", "grid_size": grid_size,
                "max_mode": max_mode, "seed": seed, "batch_size": batch_size, "splits": records}
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=MODULE_DIR / "datasets/Reaction_Diffusion")
    parser.add_argument("--train-samples", type=int, default=8000)
    parser.add_argument("--val-samples", type=int, default=1000)
    parser.add_argument("--test-samples", type=int, default=1000)
    parser.add_argument("--grid-size", type=int, default=64)
    parser.add_argument("--max-mode", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(generate_splits(**vars(args)), indent=2))


if __name__ == "__main__":
    main()
