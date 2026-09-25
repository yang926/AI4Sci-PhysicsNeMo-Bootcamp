"""Versioned pilot rules shared by the UI, queue and trusted evaluator."""
import hashlib
import importlib.metadata
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUBRIC = "bootcamp-task-completion-pilot-v3"
CHALLENGES = {
    "1": {"title": "Wave Dynamics", "directory": "01_wave", "files": ["wave_l1.py", "wave_l2.py", "wave_l3.py"]},
    "2": {"title": "Fluid Flow", "directory": "02_fluid", "files": ["chip_2d_l1.py", "chip_2d_l2.py", "chip_2d_l3.py"]},
    "3": {"title": "Climate Modeling", "directory": "03_climate", "files": ["climate_l1.py", "climate_l2.py"]},
    "4": {"title": "Neural Operators", "directory": "04_neural_operators", "files": ["fno_physicsnemo_l1.py", "fno_physicsnemo_l2.py", "fno_physicsnemo_l3.py"]},
}


def lesson_path(challenge, filename):
    spec = CHALLENGES.get(str(challenge))
    if spec is None or filename not in spec["files"]:
        raise ValueError("Unknown challenge or file. Use the original lesson filenames.")
    return ROOT / "02_challenges" / spec["directory"] / filename


def rules(steps=200, device="cpu"):
    if type(steps) is not int or not 1 <= steps <= 10000:
        raise ValueError("steps must be an integer from 1 to 10000")
    if device not in {"cpu", "cuda"}:
        raise ValueError("Choose cpu or cuda for this scoring session")
    # These are provisional scales, not calibrated claims about convergence.
    return {"rubric": RUBRIC, "status": "pilot_not_official", "steps": steps,
            "seed": 42, "timeout_seconds": 600, "device": device,
            "implementation_points": 100, "quality_points": 0,
            "quality_error_scale": 1.0, "challenge_max": 100, "overall_max": 100 * len(CHALLENGES),
            "operator_data": {"grid_size": 64, "max_mode": 6, "train_samples": 64,
                              "val_samples": 16, "test_samples": 16, "seed": 1729, "batch_size": 32},
            "description": "Each Level: 100 points for completion checks: PDE, initial/boundary conditions, "
            "geometry and coefficients where requested; or operator data/model/physics. "
            "Fixed-run numerical errors are feedback only, not additional points or a tuning competition. "
            "Numerical feedback is run only after every component matches. "
            "Levels have equal weight. Missing Levels earn zero. "
            "One best complete submission record per Challenge; joint ranks on rounded totals. "
            "All four Challenges are included. Operator evaluation uses a fixed 64/16/16 pilot dataset "
            "on the course 64x64 grid, not the full lesson dataset. Rules require event calibration and approval."}


def fingerprint(settings):
    """Prevent scores from different rules or trusted code revisions being mixed."""
    digest = hashlib.sha256(json.dumps(settings, sort_keys=True).encode())
    for package in ("torch", "sympy", "nvidia-physicsnemo"):
        digest.update((package + "=" + importlib.metadata.version(package)).encode())
    paths = list((ROOT / "ETC/runtime").glob("*.py")) + list(Path(__file__).parent.glob("*.py"))
    for spec in CHALLENGES.values():
        directory = ROOT / "02_challenges" / spec["directory"]
        paths.extend(directory.glob("*.py"))
        paths.extend(directory.glob("conf/*.yaml"))
        paths.extend(directory.glob("examples_sym/**/*.csv"))
    for path in sorted(set(paths)):
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def quality_metrics(challenge, filename):
    if challenge == "4":
        return ["test.relative_l2", "test.rmse", "test.pde_rmse_fft"]
    if challenge == "1":
        names = ["pde_rmse", "initial_displacement_rmse", "initial_velocity_rmse", "boundary_rmse"]
        if filename == "wave_l1.py":
            names.append("reference_over_time.relative_l2")
        return names
    if challenge == "2":
        names = ["continuity_unweighted_rmse", "momentum_x_unweighted_rmse", "momentum_y_unweighted_rmse",
                 "inlet_u_rmse", "inlet_v_rmse", "outlet_pressure_rmse", "no_slip_rmse", "integral_continuity_rmse"]
        if filename == "chip_2d_l3.py":
            names.append("initial_rest_rmse")
        return names
    if filename == "climate_l1.py":
        return ["pde_adr_rmse", "initial_T_rmse", "boundary_T_rmse", "reference_over_time.relative_l2"]
    return ["pde_atmosphere_rmse", "pde_ocean_rmse", "initial_Ta_rmse", "initial_To_rmse",
            "boundary_Ta_rmse", "boundary_To_rmse", "reference_over_time.relative_l2"]
