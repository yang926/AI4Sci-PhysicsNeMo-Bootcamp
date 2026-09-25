"""Lab 4 views of the original initial field and the learned flow over time.

All spatial coordinates and field values remain in the lesson's normalized
units. Hours label the original 60-hour time scale, not verified forecast lead
times. No analytical fixture or invented future reference is shown here.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np


ORIGINAL_DATA_KIND = "upstream_data_lat_legacy_normalization"
DOMAIN_LENGTH = 1.440


def _grid(xy, fields, max_side=128):
    """Read the saved x-fast rectangular grid; subsample only for display."""
    xy, fields = np.asarray(xy), np.asarray(fields)
    if xy.ndim != 2 or xy.shape[1] != 2 or fields.shape != (len(xy), 3):
        raise ValueError("Expected coordinates (N, 2) and fields (N, 3).")
    if not np.isfinite(xy).all() or not np.isfinite(fields).all():
        raise ValueError("Flow coordinates and fields must be finite.")
    x, y = np.unique(xy[:, 0]), np.unique(xy[:, 1])
    if len(x) < 2 or len(y) < 2 or len(x) * len(y) != len(xy):
        raise ValueError("The flow view needs a rectangular grid.")
    coordinates = xy.reshape(len(y), len(x), 2)
    if not (np.array_equal(coordinates[:, :, 0], np.broadcast_to(x, (len(y), len(x))))
            and np.array_equal(coordinates[:, :, 1], np.broadcast_to(y[:, None], (len(y), len(x))))):
        raise ValueError("Expected the saved x-fast grid ordering.")
    ix = np.unique(np.linspace(0, len(x) - 1, min(max_side, len(x))).astype(int))
    iy = np.unique(np.linspace(0, len(y) - 1, min(max_side, len(y))).astype(int))
    values = fields.reshape(len(y), len(x), 3)[np.ix_(iy, ix)]
    return x[ix], y[iy], values


def load_original_flow(output_dir):
    """Reject stale fixture results before displaying this student lesson."""
    output = Path(output_dir)
    metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    if metrics.get("data_kind") != ORIGINAL_DATA_KIND:
        raise ValueError("This is not the original-data Lab 4 run. Rerun the Lab 4 training cell.")
    required = {"xy", "times", "times_hours", "prediction", "initial_xy", "initial_fields"}
    with np.load(output / "predictions.npz", allow_pickle=False) as saved:
        if not required <= set(saved.files):
            raise ValueError("This older result lacks the original input/time snapshots. Rerun Lab 4.")
        if "reference" in saved.files:
            raise ValueError("A future analytical reference does not belong to this original-data run.")
        data = {key: saved[key].copy() for key in required}
    if data["times"].shape != (11,) or data["times_hours"].shape != (11,):
        raise ValueError("Expected 11 saved frames from 0 to 60 hours.")
    if not (np.allclose(data["times"], np.linspace(0, 1, 11), atol=2e-6, rtol=0)
            and np.allclose(data["times_hours"], np.linspace(0, 60, 11), atol=2e-4, rtol=0)):
        raise ValueError("Frame times must cover 0 to 60 hours at six-hour intervals.")
    if data["prediction"].shape != (11, len(data["xy"]), 3):
        raise ValueError("Expected eleven predicted u, v, p fields.")
    for field in data["prediction"]:
        _grid(data["xy"], field)
    _grid(data["initial_xy"], data["initial_fields"])
    return data


def _speed(fields):
    return np.hypot(fields[..., 0], fields[..., 1])


def _draw_flow(ax, xy, fields, maximum, title):
    x, y, values = _grid(xy, fields)
    surface = ax.pcolormesh(x, y, _speed(values), shading="auto", cmap="viridis",
                            vmin=0, vmax=maximum, rasterized=True)
    stride = max(1, int(np.ceil(max(len(x), len(y)) / 17)))
    xx, yy = np.meshgrid(x[::stride], y[::stride], indexing="xy")
    arrows = ax.quiver(xx, yy, values[::stride, ::stride, 0], values[::stride, ::stride, 1],
                       color="white", angles="xy", scale_units="xy",
                       scale=maximum * 15 / DOMAIN_LENGTH, width=.003)
    ax.set(title=title, xlabel="x (normalized)", ylabel="y (normalized)", aspect="equal")
    return surface, arrows, stride


def plot_initial_flow(xy, fields):
    """Show the supplied input before any network has been trained."""
    maximum = max(float(_speed(np.asarray(fields)).max()), 1e-8)
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    surface, _, _ = _draw_flow(ax, xy, fields, maximum, "Original input wind at 0 hours")
    fig.colorbar(surface, ax=ax, label="Speed (normalized)")
    return fig


def build_flow_animation(data):
    """Keep the input fixed on the left and play eleven predicted frames."""
    maximum = max(float(_speed(data["initial_fields"]).max()),
                  float(_speed(data["prediction"]).max()), 1e-8)
    # Keep geometry fixed while animation frames are serialized. Re-running
    # constrained layout for each frame can push the shared title off-canvas.
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.subplots_adjust(left=.065, right=.875, bottom=.12, top=.83, wspace=.20)
    _draw_flow(axes[0], data["initial_xy"], data["initial_fields"], maximum,
               "Original input at 0 hours (fixed)")
    surface, arrows, stride = _draw_flow(axes[1], data["xy"], data["prediction"][0],
                                         maximum, "PINN prediction: 0 hours")
    colorbar_axis = fig.add_axes((.905, .18, .016, .59))
    fig.colorbar(surface, cax=colorbar_axis, label="Speed (normalized; fixed scale)")
    fig.suptitle("Original-data flow evolution · arrows show velocity direction and magnitude", y=.97)

    def update(index):
        _, _, values = _grid(data["xy"], data["prediction"][index])
        surface.set_array(_speed(values).ravel())
        arrows.set_UVC(values[::stride, ::stride, 0], values[::stride, ::stride, 1])
        axes[1].set_title(f"PINN prediction: {data['times_hours'][index]:g} hours")
        return surface, arrows

    animation = FuncAnimation(fig, update, frames=len(data["times"]), interval=600,
                               blit=False, repeat=True)
    return fig, animation


def animate_flow(data):
    """Return self-contained Jupyter playback; no ffmpeg/server is required."""
    from IPython.display import HTML

    fig, animation = build_flow_animation(data)
    try:
        return HTML(animation.to_jshtml(default_mode="loop"))
    finally:
        plt.close(fig)
