"""Optional plotting example. Uses only actual arrays from the current run."""
import matplotlib.pyplot as plt
import numpy as np

def plot_pinn(x, u_pred, u_exact):
    x, u_pred, u_exact = (np.asarray(a).reshape(-1) for a in (x, u_pred, u_exact))
    if x.shape != u_pred.shape or x.shape != u_exact.shape:
        raise ValueError("x, u_pred and u_exact must have matching lengths")
    if not all(np.isfinite(a).all() for a in (x, u_pred, u_exact)):
        raise ValueError("Use finite results from a completed run")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), constrained_layout=True)
    axes[0].plot(x, u_exact, label="Analytical solution")
    axes[0].plot(x, u_pred, "--", label="PINN prediction")
    axes[0].set(xlabel="x", ylabel="u", title="Solution comparison")
    axes[0].legend()
    axes[1].plot(x, u_pred-u_exact)
    axes[1].axhline(0, color="gray", linewidth=0.8)
    axes[1].set(xlabel="x", ylabel="Prediction - reference", title="Signed error")
    return fig
