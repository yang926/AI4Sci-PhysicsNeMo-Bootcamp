"""Evaluate an unmodified weather forecast against ERA5 reanalysis, not observations.

NPZ contract: prediction/reference [9,4,latitude,longitude], variables
[u10m,v10m,msl,t2m], lead_hours [0,6,...,48], times datetime64, and
latitude/longitude on the same global, endpoint-excluded regular grid.
No spatial alignment, statistical calibration, rescaling or clipping is performed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


VARIABLES = ("u10m", "v10m", "msl", "t2m")
LEAD_HOURS = np.arange(0, 49, 6, dtype=np.int64)
SI_UNITS = {"u10m": "m s-1", "v10m": "m s-1", "msl": "Pa", "t2m": "K"}
PLOT_UNITS = {"u10m": "m/s", "v10m": "m/s", "msl": "hPa", "t2m": "deg C", "wind_speed": "m/s"}
EAST_ASIA = {"latitude_min": 10., "latitude_max": 55., "longitude_min": 100., "longitude_max": 160.}


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def read_fields(path, field_key):
    with np.load(path, allow_pickle=False) as source:
        required = {field_key, "variables", "times", "lead_hours", "latitude", "longitude"}
        if not required.issubset(source.files):
            raise ValueError(f"{path}: missing keys {sorted(required - set(source.files))}")
        result = {key: source[key] for key in required}
    field = result[field_key]
    if field.ndim != 4 or field.shape[:2] != (len(LEAD_HOURS), len(VARIABLES)):
        raise ValueError(f"{field_key} must have shape [9,4,latitude,longitude]")
    if field.dtype != np.float32:
        raise ValueError(f"{field_key} must contain raw physical-unit float32 values")
    if result["variables"].dtype.kind != "U" or result["variables"].tolist() != list(VARIABLES):
        raise ValueError("Variable order must be [u10m,v10m,msl,t2m] as Unicode strings")
    leads = result["lead_hours"]
    if leads.dtype.kind not in "iu" or not np.array_equal(leads, LEAD_HOURS):
        raise ValueError("All nine leads [0,6,...,48] are required; do not select favorable times")
    times = result["times"]
    if times.shape != (9,) or not np.issubdtype(times.dtype, np.datetime64) or np.isnat(times).any():
        raise ValueError("times must be a finite datetime64 vector of length nine")
    if not np.array_equal(times, times.astype("datetime64[h]")):
        raise ValueError("Timestamp precision must describe whole hours")
    if not np.array_equal(times, times[0] + leads.astype("timedelta64[h]")):
        raise ValueError("Valid timestamps do not agree with initialization plus lead_hours")
    latitude, longitude = result["latitude"], result["longitude"]
    for name, coordinate, size in (("latitude", latitude, field.shape[2]), ("longitude", longitude, field.shape[3])):
        if coordinate.ndim != 1 or len(coordinate) != size or size < 2:
            raise ValueError(f"{name} must be one-dimensional and match the field grid")
        if coordinate.dtype.kind not in "fiu" or not np.isfinite(coordinate).all():
            raise ValueError(f"{name} must be finite and numeric")
    # Smaller grids are allowed for CPU unit tests; production uses 720 x 1440.
    if not np.allclose(latitude, 90. - np.arange(len(latitude)) * 180. / len(latitude), rtol=0., atol=1e-6):
        raise ValueError("latitude must decrease from 90 on an endpoint-excluded global grid")
    if not np.allclose(longitude, np.arange(len(longitude)) * 360. / len(longitude), rtol=0., atol=1e-6):
        raise ValueError("longitude must increase from 0 on an endpoint-excluded global grid")
    for frame in field:
        if not np.isfinite(frame).all():
            raise ValueError(f"{field_key} contains missing/nonfinite values; repair source data explicitly")
    return result


def load_pair(forecast_path, truth_path):
    forecast = read_fields(forecast_path, "prediction")
    truth = read_fields(truth_path, "reference")
    for key in ("variables", "times", "lead_hours", "latitude", "longitude"):
        if not np.array_equal(forecast[key], truth[key]):
            raise ValueError(f"Forecast/reanalysis {key} mismatch; implicit alignment is forbidden")
    if forecast["prediction"].shape != truth["reference"].shape:
        raise ValueError("Forecast/reanalysis grid shape mismatch")
    if not np.array_equal(forecast["prediction"][0], truth["reference"][0]):
        raise ValueError("Lead-zero forecast must exactly copy the supplied physical initial fields")
    return {**forecast, "reference": truth["reference"]}


def weighted_mean(values, latitude):
    """Area-weighted average for a regular longitude grid, with cos(latitude)."""
    values = np.asarray(values, dtype=np.float64)
    latitude = np.asarray(latitude, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != len(latitude) or values.shape[1] == 0:
        raise ValueError("Expected a nonempty [latitude,longitude] field")
    if not np.isfinite(values).all() or not np.isfinite(latitude).all():
        raise ValueError("Weighted metric requires finite values and latitude")
    weights = np.maximum(np.cos(np.deg2rad(latitude)), 0.)
    weights[np.abs(latitude) == 90.] = 0.
    denominator = weights.sum() * values.shape[1]
    if denominator <= 0:
        raise ValueError("Grid has no positive-area latitude rows")
    return float(np.sum(values * weights[:, None]) / denominator)


def scalar_metrics(prediction, reference, initial, latitude):
    error = np.asarray(prediction, dtype=np.float64) - reference
    persistence_error = np.asarray(initial, dtype=np.float64) - reference
    mse = weighted_mean(error * error, latitude)
    persistence_mse = weighted_mean(persistence_error * persistence_error, latitude)
    return {
        "rmse": float(np.sqrt(mse)),
        "bias_prediction_minus_reanalysis": weighted_mean(error, latitude),
        "persistence_rmse": float(np.sqrt(persistence_mse)),
        "mse_skill_vs_persistence": 1. - mse / persistence_mse if persistence_mse > 0 else None,
    }


def wind_metrics(prediction_uv, reference_uv, initial_uv, latitude):
    prediction_uv, reference_uv, initial_uv = (
        np.asarray(value, dtype=np.float64) for value in (prediction_uv, reference_uv, initial_uv)
    )
    error = prediction_uv - reference_uv
    persistence_error = initial_uv - reference_uv
    mse = weighted_mean(np.sum(error * error, axis=0), latitude)
    persistence_mse = weighted_mean(np.sum(persistence_error * persistence_error, axis=0), latitude)
    speed = lambda value: np.hypot(value[0], value[1])
    return {
        "vector_rmse": float(np.sqrt(mse)),
        "persistence_vector_rmse": float(np.sqrt(persistence_mse)),
        "vector_mse_skill_vs_persistence": 1. - mse / persistence_mse if persistence_mse > 0 else None,
        "speed": scalar_metrics(speed(prediction_uv), speed(reference_uv), speed(initial_uv), latitude),
        "unit": "m s-1",
    }


def region_indices(data):
    latitude, longitude = data["latitude"], data["longitude"]
    lat_ids = np.flatnonzero((latitude >= EAST_ASIA["latitude_min"]) & (latitude <= EAST_ASIA["latitude_max"]))
    lon_ids = np.flatnonzero((longitude >= EAST_ASIA["longitude_min"]) & (longitude <= EAST_ASIA["longitude_max"]))
    if not len(lat_ids) or not len(lon_ids):
        raise ValueError("Grid has no points in the declared East-Asia region")
    return lat_ids, lon_ids


def evaluate(data):
    p, r = data["prediction"], data["reference"]
    lat_ids, lon_ids = region_indices(data)
    regions = {
        "global": (np.arange(p.shape[-2]), np.arange(p.shape[-1])),
        "east_asia": (lat_ids, lon_ids),
    }
    output = {}
    for region, (rows, columns) in regions.items():
        latitude = data["latitude"][rows]
        crop = lambda field: field[..., rows[:, None], columns]
        initial = crop(r[0]).astype(np.float64)
        records = []
        for frame in range(1, len(LEAD_HOURS)):
            prediction, reference = crop(p[frame]), crop(r[frame])
            variables = {}
            for channel, variable in enumerate(VARIABLES):
                metrics = scalar_metrics(prediction[channel], reference[channel], initial[channel], latitude)
                factor = .01 if variable == "msl" else 1.
                variables[variable] = {
                    **metrics, "unit": SI_UNITS[variable],
                    "display_rmse": metrics["rmse"] * factor,
                    "display_persistence_rmse": metrics["persistence_rmse"] * factor,
                    "display_error_unit": "deg C difference" if variable == "t2m" else PLOT_UNITS[variable],
                }
            records.append({
                "lead_hours": int(data["lead_hours"][frame]),
                "valid_time_utc": np.datetime_as_string(data["times"][frame], unit="h") + "Z",
                "variables": variables,
                "wind_10m": wind_metrics(prediction[:2], reference[:2], initial[:2], latitude),
            })
        output[region] = records
    return {
        "problem": "pretrained weather inference against ERA5 reanalysis",
        "reference_kind": "ERA5 reanalysis; not direct observations",
        "initial_time_utc": np.datetime_as_string(data["times"][0], unit="h") + "Z",
        "variables": list(VARIABLES), "grid_shape": list(p.shape[-2:]),
        "forecast_lead_hours": LEAD_HOURS[1:].tolist(),
        "initial_frame_excluded_from_forecast_scores": True,
        "east_asia_bounds_degrees": EAST_ASIA,
        "metrics": output,
        "definitions": {
            "weighting": "cos(latitude) per cell, normalized over the stated global or regional grid; poles have zero weight",
            "rmse": "sqrt(weighted_mean((prediction - ERA5 reanalysis)^2)); physical SI units, never divided by absolute pressure or temperature",
            "wind_vector_rmse": "sqrt(weighted_mean((u_prediction-u_reanalysis)^2 + (v_prediction-v_reanalysis)^2)); no division by two",
            "wind_speed_rmse": "RMSE of hypot(u10m,v10m), distinct from vector RMSE and blind to direction by itself",
            "persistence": "Exact initial physical field held fixed at every positive lead",
            "skill": "1 - forecast_MSE / persistence_MSE; 1 perfect, 0 equals persistence, negative worse; null when persistence_MSE is zero",
            "display_units": "msl fields/errors Pa / 100 -> hPa; t2m fields K - 273.15 -> Celsius; temperature errors keep the same numerical magnitude",
            "scope": "Eight successive leads from one initialization are correlated, not eight independent weather cases; these scores alone do not establish broad forecast skill",
        },
    }


def geographic_lines(path):
    """Read optional local GeoJSON; never fetch packages or geographic data."""
    if path is None:
        return []
    content = json.loads(Path(path).read_text(encoding="utf-8"))
    lines = []

    def visit(item):
        kind = item.get("type")
        if kind == "FeatureCollection":
            for feature in item["features"]:
                visit(feature)
        elif kind == "Feature":
            if item.get("geometry") is not None:
                visit(item["geometry"])
        elif kind == "GeometryCollection":
            for geometry in item["geometries"]:
                visit(geometry)
        elif kind == "LineString":
            lines.append(np.asarray(item["coordinates"], dtype=float)[:, :2])
        elif kind in ("MultiLineString", "Polygon"):
            for line in item["coordinates"]:
                lines.append(np.asarray(line, dtype=float)[:, :2])
        elif kind == "MultiPolygon":
            for polygon in item["coordinates"]:
                for line in polygon:
                    lines.append(np.asarray(line, dtype=float)[:, :2])
        else:
            raise ValueError(f"Unsupported geographic geometry: {kind}")

    visit(content)
    normalized = []
    for line in lines:
        if line.ndim != 2 or line.shape[1] != 2 or not np.isfinite(line).all():
            raise ValueError("Invalid GeoJSON line coordinates")
        line = line.copy()
        line[:, 0] %= 360.
        breaks = np.flatnonzero(np.abs(np.diff(line[:, 0])) > 180.) + 1
        normalized.extend(piece for piece in np.split(line, breaks) if len(piece) >= 2)
    return normalized


def assess_acceptance(report, criteria):
    """Report prespecified checks without claiming to audit the forecast runner."""
    checks = []
    initialization_matches = np.datetime64(report["initial_time_utc"].removesuffix("Z")) == np.datetime64(criteria["initial_time_utc"].removesuffix("Z"))
    checks.append({"criterion": "prespecified_initialization", "passed": bool(initialization_matches),
                   "value": report["initial_time_utc"], "expected": criteria["initial_time_utc"]})
    checks.append({"criterion": "all_positive_leads_reported", "passed": report["forecast_lead_hours"] == criteria["forecast_leads_hours"],
                   "value": report["forecast_lead_hours"], "expected": criteria["forecast_leads_hours"]})
    limits = criteria["acceptance"]["global_and_east_asia_24h_and_48h"]
    for region in ("global", "east_asia"):
        by_lead = {row["lead_hours"]: row for row in report["metrics"][region]}
        for lead in (24, 48):
            row = by_lead[lead]
            values = (
                ("wind_vector_rmse_m_s_max", row["wind_10m"]["vector_rmse"]),
                ("mean_sea_level_pressure_rmse_hpa_max", row["variables"]["msl"]["display_rmse"]),
                ("temperature_2m_rmse_kelvin_max", row["variables"]["t2m"]["rmse"]),
            )
            for key, value in values:
                checks.append({"region": region, "lead_hours": lead, "criterion": key,
                               "value": value, "limit": limits[key], "passed": bool(np.isfinite(value) and value <= limits[key])})
            for key, forecast, persistence in (
                ("wind_vector_mse_less_than_persistence", row["wind_10m"]["vector_rmse"], row["wind_10m"]["persistence_vector_rmse"]),
                ("pressure_mse_less_than_persistence", row["variables"]["msl"]["rmse"], row["variables"]["msl"]["persistence_rmse"]),
                ("temperature_mse_less_than_persistence", row["variables"]["t2m"]["rmse"], row["variables"]["t2m"]["persistence_rmse"]),
            ):
                if limits[key] is not True:
                    raise ValueError(f"Expected strict prespecified persistence check: {key}")
                checks.append({"region": region, "lead_hours": lead, "criterion": key,
                               "forecast_rmse": forecast, "persistence_rmse": persistence,
                               "passed": bool(np.isfinite(forecast) and np.isfinite(persistence) and forecast < persistence)})
    return {
        "measured_forecast_criteria_passed": all(check["passed"] for check in checks),
        "checks": checks,
        "validated_on_load": ["all four selected variables finite at all nine leads", "initial selected fields exactly equal ERA5"],
        "not_certified_here": ["all 26 model channels finite", "forecast never read future truth", "checkpoint and preprocessing correctness", "GPU runtime or memory"],
        "scope": criteria["scope"],
    }


def plot_field(frames, field):
    if field == "wind_speed":
        return np.hypot(frames[:, 0].astype(np.float64), frames[:, 1].astype(np.float64))
    values = frames[:, VARIABLES.index(field)].astype(np.float64)
    return values / 100. if field == "msl" else values - 273.15 if field == "t2m" else values


def regional_plot_data(data, field):
    rows, columns = region_indices(data)
    p = data["prediction"][..., rows[:, None], columns]
    r = data["reference"][..., rows[:, None], columns]
    prediction, reference = plot_field(p, field), plot_field(r, field)
    error = prediction - reference
    if field in ("u10m", "v10m"):
        limit = max(float(np.max(np.abs(prediction[1:]))), float(np.max(np.abs(reference[1:]))), 1e-12)
        limits, cmap = (-limit, limit), "RdBu_r"
    else:
        lower = min(float(prediction[1:].min()), float(reference[1:].min()))
        upper = max(float(prediction[1:].max()), float(reference[1:].max()))
        if field == "wind_speed":
            lower = 0.
        if upper <= lower:
            upper = lower + 1.
        limits, cmap = (lower, upper), "viridis" if field in ("wind_speed", "msl") else "coolwarm"
    return dict(prediction=prediction, reference=reference, error=error,
                prediction_uv=p[:, :2], reference_uv=r[:, :2],
                latitude=data["latitude"][rows], longitude=data["longitude"][columns],
                limits=limits, cmap=cmap, error_limit=max(float(np.max(np.abs(error[1:]))), 1e-12))


def draw_map(ax, field, latitude, longitude, limits, cmap, lines, vector=None):
    dx = float(longitude[1] - longitude[0]) if len(longitude) > 1 else 1.
    dy = float(latitude[0] - latitude[1]) if len(latitude) > 1 else 1.
    image = ax.imshow(field, origin="upper", extent=(longitude[0] - dx / 2, longitude[-1] + dx / 2,
                      latitude[-1] - dy / 2, latitude[0] + dy / 2),
                      vmin=limits[0], vmax=limits[1], cmap=cmap, interpolation="nearest", aspect="auto")
    for line in lines:
        ax.plot(line[:, 0], line[:, 1], color="0.25", lw=.45)
    if vector is not None:
        stride = max(1, int(np.ceil(max(field.shape) / 12)))
        yy, xx = np.meshgrid(latitude[::stride], longitude[::stride], indexing="ij")
        arrows = ax.quiver(xx, yy, vector[0, ::stride, ::stride], vector[1, ::stride, ::stride],
                          angles="uv", scale=100, width=.003, pivot="middle", color="0.1")
        ax.quiverkey(arrows, .84, 1.04, 10., "10 m/s", labelpos="E", coordinates="axes", fontproperties={"size": 7})
    ax.set(xlim=(100, 160), ylim=(10, 55), xlabel="Longitude (degrees E)", ylabel="Latitude (degrees N)")
    ax.grid(alpha=.2)
    return image


def plot_time_curves(report, output_dir):
    import matplotlib.pyplot as plt
    paths = []
    for region, records in report["metrics"].items():
        leads = [row["lead_hours"] for row in records]
        fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
        for ax, variable in zip(axes.flat, VARIABLES):
            ax.plot(leads, [row["variables"][variable]["display_rmse"] for row in records], "o-", label="Forecast")
            ax.plot(leads, [row["variables"][variable]["display_persistence_rmse"] for row in records], "s--", label="Persistence")
            ax.set(title=variable, ylabel=f"RMSE ({PLOT_UNITS[variable]})")
        for ax, key, title in ((axes.flat[4], "vector", "10 m wind vector"), (axes.flat[5], "speed", "10 m wind speed")):
            forecast = [row["wind_10m"]["vector_rmse"] if key == "vector" else row["wind_10m"]["speed"]["rmse"] for row in records]
            baseline = [row["wind_10m"]["persistence_vector_rmse"] if key == "vector" else row["wind_10m"]["speed"]["persistence_rmse"] for row in records]
            ax.plot(leads, forecast, "o-", label="Forecast")
            ax.plot(leads, baseline, "s--", label="Persistence")
            ax.set(title=title, ylabel="RMSE (m/s)")
        for ax in axes.flat:
            ax.set(xlabel="Forecast lead (hours)", xticks=leads)
            ax.grid(alpha=.25)
            ax.legend(fontsize=8)
        fig.suptitle(f"{region}: cos(latitude)-weighted errors against ERA5 reanalysis\nInitialization {report['initial_time_utc']}; t=0 excluded")
        path = Path(output_dir) / f"rmse-by-lead-{region}.png"
        fig.savefig(path, dpi=130)
        plt.close(fig)
        paths.append(str(path.resolve()))
    return paths


def plot_maps(data, output_dir, fields=("msl", "t2m", "wind_speed"), lines=(), attribution=None):
    import matplotlib.pyplot as plt
    paths = []
    for field in fields:
        plot = regional_plot_data(data, field)
        for page, frames in enumerate((range(1, 5), range(5, 9)), start=1):
            fig, axes = plt.subplots(3, 4, figsize=(17, 10), constrained_layout=True)
            for column, frame in enumerate(frames):
                for row, (key, label) in enumerate((("reference", "ERA5 reanalysis"), ("prediction", "Forecast"), ("error", "Forecast minus ERA5"))):
                    limits = (-plot["error_limit"], plot["error_limit"]) if key == "error" else plot["limits"]
                    vector = None
                    if field == "wind_speed":
                        vector = plot["prediction_uv"][frame] - plot["reference_uv"][frame] if key == "error" else plot[key + "_uv"][frame]
                    image = draw_map(axes[row, column], plot[key][frame], plot["latitude"], plot["longitude"],
                                     limits, "RdBu_r" if key == "error" else plot["cmap"], lines, vector)
                    axes[row, column].set_title(f"{label} | +{int(data['lead_hours'][frame])} h", fontsize=10,
                                               pad=22 if field == "wind_speed" else 6)
                    if column == 3:
                        fig.colorbar(image, ax=axes[row, :].tolist(), shrink=.8, label=PLOT_UNITS[field] + (" difference" if key == "error" else ""))
            text = f"East Asia: {field} | initialization {np.datetime_as_string(data['times'][0], unit='h')} UTC\nFixed color scales across all eight leads; initial state is not a forecast"
            if attribution:
                text += "\n" + attribution
            fig.suptitle(text, fontsize=12)
            path = Path(output_dir) / f"east-asia-{field}-leads-page-{page}.png"
            fig.savefig(path, dpi=120)
            plt.close(fig)
            paths.append(str(path.resolve()))
    return paths


def save_gif(data, output_dir, lines=(), attribution=None):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    plot = regional_plot_data(data, "wind_speed")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), constrained_layout=True)
    # Colorbars are fixed and do not change as the forecast advances.
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    for ax, error in zip(axes, (False, False, True)):
        limits = (-plot["error_limit"], plot["error_limit"]) if error else plot["limits"]
        fig.colorbar(ScalarMappable(norm=Normalize(*limits), cmap="RdBu_r" if error else plot["cmap"]),
                     ax=ax, orientation="horizontal", shrink=.8, label="Speed error (m/s)" if error else "Wind speed (m/s)")

    def update(frame):
        for ax, key, label in zip(axes, ("reference", "prediction", "error"), ("ERA5 reanalysis", "Forecast", "Forecast minus ERA5")):
            ax.clear()
            error = key == "error"
            vector = plot["prediction_uv"][frame] - plot["reference_uv"][frame] if error else plot[key + "_uv"][frame]
            draw_map(ax, plot[key][frame], plot["latitude"], plot["longitude"],
                     (-plot["error_limit"], plot["error_limit"]) if error else plot["limits"],
                     "RdBu_r" if error else plot["cmap"], lines, vector)
            ax.set_title(label, pad=22)
        title = f"10 m wind | +{int(data['lead_hours'][frame])} h | {np.datetime_as_string(data['times'][frame], unit='h')} UTC"
        if attribution:
            title += "\n" + attribution
        fig.suptitle(title, fontsize=11)

    animation = FuncAnimation(fig, update, frames=range(1, 9), interval=700, repeat=True)
    path = Path(output_dir) / "east-asia-wind-6-to-48h.gif"
    animation.save(path, writer=PillowWriter(fps=1.5), dpi=110)
    plt.close(fig)
    return str(path.resolve())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forecast", type=Path, required=True)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--coastline", type=Path, help="Optional local GeoJSON; no automatic downloads")
    parser.add_argument("--coastline-attribution", help="Source/attribution to display for the supplied GeoJSON")
    parser.add_argument("--acceptance", type=Path, help="Optional prespecified weather-acceptance.json; thresholds are never adjusted")
    parser.add_argument("--map-fields", nargs="+", choices=(*VARIABLES, "wind_speed"), default=["msl", "t2m", "wind_speed"])
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--gif", action="store_true")
    args = parser.parse_args()
    if args.gif and args.no_plots:
        parser.error("--gif and --no-plots cannot be used together")
    if args.coastline and not args.coastline_attribution:
        parser.error("Provide --coastline-attribution with local geographic data")
    if args.output_dir.exists():
        raise FileExistsError(f"Preserve prior results: select a new output directory: {args.output_dir}")
    data = load_pair(args.forecast, args.truth)
    report = evaluate(data)
    report.update(forecast_sha256=digest(args.forecast), reanalysis_sha256=digest(args.truth),
                  evaluator_sha256=digest(__file__), coastline_attribution=args.coastline_attribution,
                  coastline_sha256=digest(args.coastline) if args.coastline else None)
    if args.acceptance:
        report["acceptance"] = assess_acceptance(report, json.loads(args.acceptance.read_text(encoding="utf-8")))
        report["acceptance_sha256"] = digest(args.acceptance)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report["figures"] = []
    if not args.no_plots:
        import matplotlib
        matplotlib.use("Agg")
        lines = geographic_lines(args.coastline)
        report["figures"] = plot_time_curves(report, args.output_dir)
        report["figures"] += plot_maps(data, args.output_dir, args.map_fields, lines, args.coastline_attribution)
        if args.gif:
            report["animation"] = save_gif(data, args.output_dir, lines, args.coastline_attribution)
    path = args.output_dir / "evaluation.json"
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(path.resolve()), "forecast_lead_hours": report["forecast_lead_hours"],
                      "reference_kind": report["reference_kind"], "figures": report["figures"]}), flush=True)


if __name__ == "__main__":
    main()
