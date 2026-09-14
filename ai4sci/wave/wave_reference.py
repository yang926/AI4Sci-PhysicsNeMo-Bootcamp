"""Analytical reference for the AI4Sci two-dimensional wave exercise."""

import numpy as np


def exact_solution(x, y, t, c=1.0):
    """Solve u_tt=c²(u_xx+u_yy), u(0)=sin(x)sin(y), u_t(0)=sin(x)sin(y)."""
    if not np.isfinite(c) or c <= 0:
        raise ValueError("Wave speed c must be finite and positive.")
    omega = np.sqrt(2.0) * c
    return np.sin(x) * np.sin(y) * (
        np.cos(omega * t) + np.sin(omega * t) / omega
    )
