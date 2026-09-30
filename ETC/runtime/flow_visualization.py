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

from ETC.runtime.flow_diagnostics import summarize_flow


ORIGINAL_DATA_KIND = "upstream_data_lat_legacy_normalization"
DERIVED_DATA_KIND = "derived_periodic_incompressible_initial_conditions"
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


def load_derived_flow(output_dir):
    """Load the separately labeled six-hour, derived-initial-condition demo."""
    output = Path(output_dir)
    metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    if metrics.get("data_kind") != DERIVED_DATA_KIND:
        raise ValueError("This is not the derived-initial-condition six-hour Lab 4 run.")
    if (metrics.get("time_unit", "hours") != "hours"
            or metrics.get("time_scale_hours", 60) != 60
            or metrics.get("original_time_scale_hours", 60) != 60):
        raise ValueError("Derived flow time metadata must use hours with the original sixty-hour scale.")
    required = {"xy", "times", "times_hours", "prediction", "initial_xy", "initial_fields"}
    with np.load(output / "predictions.npz", allow_pickle=False) as saved:
        if not required <= set(saved.files):
            raise ValueError("Derived flow requires saved initial fields and explicit six-hour frame times.")
        if "reference" in saved.files:
            raise ValueError("A future reference must be displayed separately from derived flow predictions.")
        data = {key: saved[key].copy() for key in required}
    if any(value.dtype.kind not in "fiu" or not np.isfinite(value).all() for value in data.values()):
        raise ValueError("Derived flow arrays must contain finite real numbers.")
    if data["times"].shape != (11,) or data["times_hours"].shape != (11,):
        raise ValueError("Expected 11 derived-flow frames from 0 to 6 hours.")
    if not (np.allclose(data["times"], np.linspace(0, .1, 11), atol=2e-7, rtol=0)
            and np.allclose(data["times_hours"], np.linspace(0, 6, 11), atol=2e-5, rtol=0)):
        raise ValueError("Derived frame times must cover 0 to 6 hours in the original sixty-hour units.")
    if data["prediction"].shape != (11, len(data["xy"]), 3):
        raise ValueError("Expected eleven derived-input predicted u, v, p fields.")
    for field in data["prediction"]:
        _grid(data["xy"], field)
    _grid(data["initial_xy"], data["initial_fields"])
    data.update(initial_label="Derived incompressible input at 0 hours (fixed)",
                evolution_label="Derived-input six-hour flow · arrows show velocity direction and magnitude")
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


def plot_initial_flow(xy, fields, *, title="Original input wind at 0 hours"):
    """Show the supplied input before any network has been trained."""
    maximum = max(float(_speed(np.asarray(fields)).max()), 1e-8)
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    surface, _, _ = _draw_flow(ax, xy, fields, maximum, title)
    fig.colorbar(surface, ax=ax, label="Speed (normalized)")
    return fig


def _flow_diagnostic_data(data):
    """Use predictions only, with the explicitly saved display-time scale."""
    rows = summarize_flow(data["prediction"], data["times"], data["xy"])["per_time"]
    hours = np.asarray(data["times_hours"], dtype=np.float64)
    if (hours.shape != (len(rows),) or not np.isfinite(hours).all()
            or hours[0] != 0 or np.any(np.diff(hours) <= 0)):
        raise ValueError("Diagnostic hours must be finite, increasing, and start at zero.")
    final_fraction = rows[-1]["energy_retention_ratio"]
    review_flag = final_fraction is not None and final_fraction < .5
    return rows, hours, review_flag


def plot_flow_diagnostics(data):
    """Plot weakening without rescaling frames or claiming future accuracy.

    The caller owns the returned figure and should close it after display.
    The energy ratios use the predicted initial frame, not initial observations.
    """
    rows, hours, review_flag = _flow_diagnostic_data(data)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    try:
        fig.subplots_adjust(left=.08, right=.98, bottom=.25, top=.80, wspace=.30)
        axes[0].plot(hours, [row["speed_rms"] for row in rows], marker="o")
        axes[0].set(ylabel="RMS speed (normalized)", ylim=(0, None), title="Predicted speed")
        for key, label in (("energy_retention_ratio", "Total kinetic energy"),
                           ("fluctuating_energy_retention_ratio", "Fluctuating kinetic energy")):
            if rows[0][key] is not None:
                axes[1].plot(hours, [row[key] for row in rows], marker="o", label=label)
        axes[1].set(ylabel="Energy / corresponding predicted initial energy",
                    ylim=(0, None), title="Energy retention")
        if axes[1].lines:
            axes[1].legend(fontsize=9)
        else:
            axes[1].text(.5, .5, "Energy retention is undefined:\npredicted initial energy is zero.",
                         ha="center", va="center", transform=axes[1].transAxes)
        for ax in axes:
            ax.set_xlabel("Time (hours; original scale)")
            ax.grid(alpha=.25)
        fig.suptitle("Flow diagnostics: not forecast accuracy", fontsize=13)
        note = "Baseline: predicted 0-hour frame. Preserved energy alone does not establish accuracy."
        if review_flag:
            note += (f"\nReview flag: only {100 * rows[-1]['energy_retention_ratio']:.1f}% of initial energy remains. "
                     "The 50% flag is heuristic, not a physics threshold.")
        fig.text(.08, .06, note, fontsize=9, color="#8a4400" if review_flag else "#444444")
        return fig
    except Exception:
        plt.close(fig)
        raise


def flow_diagnostic_summary(data):
    """Return a prominent HTML summary without accessing future references."""
    from IPython.display import HTML

    rows, hours, review_flag = _flow_diagnostic_data(data)
    first, last = rows[0], rows[-1]

    def retention(value):
        return "undefined (zero predicted initial energy)" if value is None else f"{100 * value:.1f}%"

    heading = "Review flag: strong weakening" if review_flag else "Flow diagnostics"
    flag = ("<p>The final total energy is below 50% of the predicted initial energy. "
            "This is a heuristic review flag, not a physics threshold or a convergence test.</p>"
            if review_flag else "")
    color = "#b35a00" if review_flag else "#687888"
    return HTML(
        f'<div style="border-left:4px solid {color};padding:0.6em 1em;margin:0.8em 0">'
        f"<strong>{heading}: not forecast accuracy</strong>"
        f"<p>At {hours[-1]:g} hours: <strong>{retention(last['energy_retention_ratio'])} total "
        "kinetic energy retained</strong>; "
        f"{retention(last['fluctuating_energy_retention_ratio'])} fluctuating kinetic energy retained.</p>"
        f"<p>RMS speed: {first['speed_rms']:.4g} → {last['speed_rms']:.4g}. "
        f"Mean velocity (u, v): ({first['mean_u']:.4g}, {first['mean_v']:.4g}) → "
        f"({last['mean_u']:.4g}, {last['mean_v']:.4g}), all in normalized units.</p>"
        "<p>Ratios compare with the predicted 0-hour frame. Fluctuating energy excludes each frame's "
        "mean flow. These diagnostics use predictions only; they do not measure initial-data fit "
        "or future accuracy. Preserved energy alone does not establish accuracy.</p>"
        f"{flag}</div>"
    )


def build_flow_animation(data):
    """Keep the input fixed on the left and play eleven predicted frames."""
    maximum = max(float(_speed(data["initial_fields"]).max()),
                  float(_speed(data["prediction"]).max()), 1e-8)
    # Keep geometry fixed while animation frames are serialized. Re-running
    # constrained layout for each frame can push the shared title off-canvas.
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.subplots_adjust(left=.065, right=.875, bottom=.12, top=.83, wspace=.20)
    _draw_flow(axes[0], data["initial_xy"], data["initial_fields"], maximum,
               data.get("initial_label", "Original input at 0 hours (fixed)"))
    surface, arrows, stride = _draw_flow(axes[1], data["xy"], data["prediction"][0],
                                         maximum, "PINN prediction: 0 hours")
    colorbar_axis = fig.add_axes((.905, .18, .016, .59))
    fig.colorbar(surface, cax=colorbar_axis, label="Speed (normalized; fixed scale)")
    fig.suptitle(data.get("evolution_label",
                         "Original-data flow evolution · arrows show velocity direction and magnitude"), y=.97)

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
