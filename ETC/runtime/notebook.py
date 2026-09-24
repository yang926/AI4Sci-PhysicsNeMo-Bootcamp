"""Small, model-independent helpers for the course's Jupyter result cells.

Training stays in the visible notebook cells and lesson programs. This module
only validates run identity and presents the saved evidence; it never grades it.
"""

from html import escape
import json
import math
from pathlib import Path


def validate_settings(device, steps, reference=None):
    """Reject accidental string/bool controls before launching a training job."""
    if device not in {"auto", "cpu", "cuda"}:
        raise ValueError('DEVICE must be "auto", "cpu", or "cuda".')
    if type(steps) is not int or steps < 1:
        raise ValueError("STEPS must be a positive integer.")
    if reference is not None and type(reference) is not bool:
        raise ValueError("USE_REFERENCE must be True or False, not a string.")


def require_current_mode(metrics, reference, level=None):
    """Do not display an old instructor answer as the current student result."""
    prefix = f"Level {level}: " if level is not None else ""
    flags = [metrics[key] for key in ("reference", "reference_implementation") if key in metrics]
    if not flags or any(type(flag) is not bool for flag in flags):
        raise RuntimeError(prefix + "saved mode is unknown; rerun training in the current mode.")
    if len(set(flags)) != 1:
        raise RuntimeError(prefix + "saved mode flags conflict; rerun training.")
    if flags[0] != reference:
        raise RuntimeError(prefix + "saved mode differs from the current mode; rerun training.")


def load_results(output, *, steps=None, seed=None, reference=None):
    """Read one run without replacing missing or mismatched results with older files."""
    path = Path(output) / "metrics.json"
    try:
        metrics = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise RuntimeError(f"Cannot read this run's metrics: {path}. Check the training error above.") from error
    if not isinstance(metrics, dict):
        raise RuntimeError(f"Expected an object in {path}; rerun training.")
    for name, expected in (("steps", steps), ("seed", seed)):
        if expected is not None and (type(metrics.get(name)) is not int or metrics[name] != expected):
            raise RuntimeError(f"Saved {name} differs from the current setting; rerun training.")
    if reference is not None:
        require_current_mode(metrics, reference)
    return metrics


def completed_output(run_dirs, completed, key):
    """Keep each Lab subproblem's plots attached to its own successful attempt."""
    if not completed.get(key, False) or key not in run_dirs:
        raise RuntimeError(f"The current {key} training run has not completed.")
    return run_dirs[key]


def require_artifacts(output):
    required = ("loss.csv", "model.pt", "predictions.npz")
    missing = [name for name in required
               if not (Path(output) / name).is_file() or (Path(output) / name).stat().st_size == 0]
    if missing:
        raise RuntimeError(f"Incomplete result directory {output}: missing or empty {', '.join(missing)}. Check the training output.")


def _numbers(values, prefix=""):
    """Flatten numeric diagnostics, excluding flags and string metadata."""
    result = {}
    if isinstance(values, dict):
        for key, value in values.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(value, dict):
                result.update(_numbers(value, name))
            elif type(value) in (int, float):
                result[name] = value
    return result


def _format(value):
    if value is None:
        return "—"
    if not math.isfinite(value):
        return "non-finite — inspect this run"
    return f"{value:.5g}"


def result_html(metrics, output):
    """Render escaped, accessible HTML; no widget server or JS extension needed."""
    before = _numbers(metrics.get("initial_test", metrics.get("heldout_before", {})))
    after = _numbers(metrics.get("test", metrics.get("heldout_after", {})))
    if not before and not after:
        # Earlier PINN outputs stored a weighted objective and slice errors at
        # the top level. Keep them useful without calling them physical RMSE.
        before = _numbers({"weighted_objective": metrics.get("initial_loss")})
        after = _numbers({"weighted_objective": metrics.get("final_loss")})
        before.update(_numbers(metrics.get("initial_reference_error", {}), "analytical_slice"))
        after.update(_numbers(metrics.get("final_reference_error", {}), "analytical_slice"))
        after.update({name: value for name, value in _numbers(metrics).items()
                      if "." not in name and name.endswith("_rmse")})
    rows = []
    for name in dict.fromkeys((*before, *after)):
        rows.append(f'<tr><th scope="row">{escape(name)}</th>'
                    f"<td>{_format(before.get(name))}</td><td>{_format(after.get(name))}</td></tr>")
    table = ("<table><caption>Fixed evaluation data: before and after training</caption>"
             '<thead><tr><th scope="col">Metric</th><th scope="col">Before</th>'
             '<th scope="col">After</th></tr></thead><tbody>' + "".join(rows) + "</tbody></table>")
    if not rows:
        table = "<p>No before/after summary is available. Read the full metrics below.</p>"
    analytical = metrics.get("reference_over_time")
    if isinstance(analytical, dict) and analytical:
        errors = {key: analytical[key] for key in ("rmse", "relative_l2", "per_field") if key in analytical}
        table += "<p>Analytical comparison across evaluation times:</p><pre>" + escape(json.dumps(errors, indent=2)) + "</pre>"
    elif analytical == {}:
        table += "<p>No analytical comparison is available for this case. This does not mean zero error.</p>"
    mode = metrics.get("reference", metrics.get("reference_implementation"))
    label = "Instructor reference" if mode is True else "Student" if mode is False else "Worked example"
    mode_notice = ("Instructor demonstration. The provided answer ran; student edits were not evaluated."
                   if mode is True else "Local practice result. This run has not been submitted to the scoring server.")
    return (f"<section><h4>{label} · {escape(str(metrics.get('steps', '?')))} training steps</h4>"
            f"<p><strong>{mode_notice}</strong></p>"
            "<p>Practice feedback, not official points. Compare errors only for the same problem and evaluation settings.</p>"
            + table + f"<p>Saved run: <code>{escape(str(output))}</code></p>"
            "<details><summary>Full metrics and run settings</summary><pre>"
            + escape(json.dumps(metrics, indent=2)) + "</pre></details></section>")


def show_results(output, *, steps=None, seed=None, reference=None, preview=True):
    """Return the full metrics while showing a compact table and optional preview."""
    from IPython.display import HTML, Image, display

    metrics = load_results(output, steps=steps, seed=seed, reference=reference)
    require_artifacts(output)
    display(HTML(result_html(metrics, output)))
    if preview:
        image = Path(output) / "preview.png"
        if image.is_file():
            display(Image(filename=str(image)))
        else:
            print("No preview image was saved. Inspect predictions.npz and the training output.")
    return metrics
