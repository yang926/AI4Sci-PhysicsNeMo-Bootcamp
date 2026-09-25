"""The original condition/geometry tasks must reach both training and grading."""
import ast
import inspect

import pytest
import sympy as sp
import torch

from ETC.judge.catalog import CHALLENGES, rules
from ETC.judge.contracts import check_setup, function_names, source_nodes
from ETC.judge.evaluate import check_equations, evaluate, level_points, load_lesson
from ETC.judge.expressions import SubmissionError
from ETC.runtime.exercises import block_geometry, condition_tensor, conditions
from ETC.runtime.submission import collect_submission
from ETC.tests.test_judge import answer


CASES = [(key, name) for key, spec in CHALLENGES.items() if key != "4" for name in spec["files"]]


@pytest.mark.parametrize("challenge,filename", CASES)
def test_all_completed_problem_components_are_accepted_and_collected(challenge, filename, tmp_path):
    source = answer(challenge, filename)
    module = load_lesson(challenge, filename)
    functions, checks = check_setup(module, source, challenge)
    assert checks and all(checks.values())
    assert set(functions) == set(function_names(challenge)) - {"student_equations"}
    (tmp_path / filename).write_text(source + '\nSECRET="never submit"\n')
    level = CHALLENGES[challenge]["files"].index(filename) + 1
    payload = collect_submission(challenge, tmp_path, levels=(level,))
    collected = payload["sources"][filename]
    assert "SECRET" not in collected
    assert {n.name for n in ast.parse(collected).body} == set(function_names(challenge))


def test_old_pde_only_submissions_require_an_explicit_upgrade():
    source = answer("1", "wave_l1.py").split("def student_conditions")[0]
    with pytest.raises(SubmissionError, match="format v3 requires student_conditions"):
        source_nodes(source, "1")


def test_wave_level_three_requires_robin_not_the_level_one_boundary(tmp_path):
    source = answer("1", "wave_l3.py").replace('u + .5 * (x * u.diff(x) + y * u.diff(y))', 'u')
    module = load_lesson("1", "wave_l3.py")
    assert all(check_equations(module, source, "1")[1].values())
    assert check_setup(module, source, "1")[1]["condition.boundary"] is False
    result = evaluate("1", {"wave_l3.py": source}, rules(2), tmp_path, "cpu")
    level = result["levels"]["wave_l3.py"]
    assert level["status"] == "incorrect_implementation"
    assert 0 < level["score"] < 100
    assert not list(tmp_path.rglob("metrics.json"))


def test_fluid_level_two_requires_its_three_obstacles(tmp_path):
    source = answer("2", "chip_2d_l2.py")
    source = source[:source.index("def student_geometry")] + 'def student_geometry():\n    return {"blocks": ((-1, 0, .1),)}\n'
    module = load_lesson("2", "chip_2d_l2.py")
    assert all(check_equations(module, source, "2")[1].values())
    checks = check_setup(module, source, "2")[1]
    assert not checks["geometry.block_count"] and not checks["geometry.block_2"]
    result = evaluate("2", {"chip_2d_l2.py": source}, rules(2), tmp_path, "cpu")
    level = result["levels"]["chip_2d_l2.py"]
    assert 0 < level["score"] < 100 and level["evaluation_errors"] == {}


def test_climate_coefficients_are_submitted_not_replaced_by_server_defaults():
    source = answer("3", "climate_l2.py").replace("'gamma0': 0.0", "'gamma0': 0.5")
    functions, checks = check_setup(load_lesson("3", "climate_l2.py"), source, "3")
    assert not checks["parameter.gamma0"]
    assert functions["student_parameters"]()["gamma0"] == sp.Float(.5)


def test_missing_time_derivative_not_masked_by_zero_initial_velocity():
    source = answer("2", "chip_2d_l3.py").replace("u.diff(t) + ", "")
    assert check_equations(load_lesson("2", "chip_2d_l3.py"), source, "2")[1]["momentum_x"] is False


def test_wave_level_two_speed_expression_is_not_supplied_by_the_grader():
    source = answer("1", "wave_l2.py").replace('1 + .5 * sin(x) * cos(y)', '1')
    module = load_lesson("1", "wave_l2.py")
    assert all(check_equations(module, source, "1")[1].values())
    assert not check_setup(module, source, "1")[1]["parameter.wave_speed"]


def test_learner_analytic_solution_is_checked_but_not_used_as_ground_truth():
    source = answer("3", "climate_l1.py").replace('exp(-2*params["kappa"]*t)', 'exp(-params["kappa"]*t)')
    module = load_lesson("3", "climate_l1.py")
    assert not check_setup(module, source, "3")[1]["solution.T"]
    xy, t = torch.ones(2, 2), torch.ones(2, 1)
    torch.testing.assert_close(module.exact_reference(xy, t, module.DEFAULT_PHYSICS),
        torch.sin(xy[:, :1])*torch.sin(xy[:, 1:2])*torch.exp(-2*t))


def test_conditions_and_geometry_have_no_import_or_default_execution(tmp_path):
    marker = tmp_path / "MUST_NOT_EXIST"
    source = answer("2", "chip_2d_l2.py")
    source = f'open({str(marker)!r}, "w").write("bad")\n' + source
    source = source.replace("def student_geometry():", f'def student_geometry(ignored=open({str(marker)!r}, "w")):')
    with pytest.raises(SubmissionError, match="arguments"):
        check_setup(load_lesson("2", "chip_2d_l2.py"), source, "2")
    assert not marker.exists()
    source = answer("2", "chip_2d_l2.py")
    source = source[:source.index("def student_geometry")] + 'def student_geometry():\n    return {"blocks": __import__("os").system("id")}\n'
    with pytest.raises(SubmissionError):
        check_setup(load_lesson("2", "chip_2d_l2.py"), source, "2")


@pytest.mark.parametrize("body", ['return {"blocks": ()}', 'return {"blocks": ((-3,0,.1),)}',
    'return {"blocks": ((-1,0,.5),)}', 'return {"blocks": ((-1,.5,.1),(0,1,.2))}',
    'return {"blocks": ((-1, "zero", .1),)}'])
def test_invalid_geometry_cannot_enter_sampling_loop(body):
    source = answer("2", "chip_2d_l1.py")
    source = source[:source.index("def student_geometry")] + 'def student_geometry():\n    ' + body
    with pytest.raises(SubmissionError):
        check_setup(load_lesson("2", "chip_2d_l1.py"), source, "2")


def test_learner_condition_changes_local_training_loss_but_not_reference_feedback():
    from ETC.runtime.pinn import create_model, create_informer
    module = load_lesson("1", "wave_l1.py")
    config = {"model": {"width": 8, "layers": 2}, "samples": {"interior": 8, "initial": 8, "boundary": 8}}
    model = create_model(3, 1, config, "cpu")
    informer = create_informer(module.WaveEquation2D(reference=True), "cpu")
    correct = conditions(module.reference_conditions, wave=True)
    changed = {**correct, "initial_u": sp.Integer(10)}
    torch.manual_seed(13)
    before = module.loss_terms(model, informer, config, "cpu", correct)
    torch.manual_seed(13)
    after = module.loss_terms(model, informer, config, "cpu", changed)
    assert after["initial_displacement"] > before["initial_displacement"] + 50
    torch.manual_seed(13)
    trusted = module.loss_terms(model, informer, config, "cpu")
    torch.testing.assert_close(trusted["initial_displacement"], before["initial_displacement"])


def test_robin_expression_tensor_evaluation_keeps_autograd():
    module = load_lesson("1", "wave_l3.py")
    xy = torch.tensor([[.3, .4], [.6, -.2]], requires_grad=True)
    time = torch.ones(2, 1)
    u = xy[:, :1]**2 + 3*xy[:, 1:2]**2
    actual = condition_tensor(conditions(module.reference_conditions, wave=True), "boundary", xy, time, {"u": u})
    torch.testing.assert_close(actual, 2*u)
    actual.sum().backward()
    assert xy.grad is not None and xy.grad.abs().sum() > 0


def test_learner_geometry_reaches_local_sampler(monkeypatch):
    from ETC.runtime.pinn import create_model, create_informer
    module = load_lesson("2", "chip_2d_l2.py")
    config = {"model": {"width": 8, "layers": 2}, "samples": {"interior": 8, "initial": 8, "boundary": 8, "flux_lines": 2, "flux_points": 8}}
    model = create_model(2, 3, config, "cpu")
    informer = create_informer(module.NavierStokes2D(reference=True), "cpu")
    selected = [(-1., 0., .1)]
    original = module.sample_interior
    calls = []
    def record(count, blocks, device):
        calls.append(blocks)
        return original(count, blocks, device)
    monkeypatch.setattr(module, "sample_interior", record)
    module.loss_terms(model, informer, config, "cpu", geometry=selected)
    assert calls == [selected]


def test_original_problem_conditions_are_not_silently_smoothed():
    wave = load_lesson("1", "wave_l3.py")
    xy = torch.tensor([[1., 0.]])
    assert wave.initial_displacement(xy).item() > 0  # no zero envelope
    fluid = load_lesson("2", "chip_2d_l3.py")
    x, y, t = sp.symbols("x y t")
    values = fluid.reference_conditions(x, y, t)
    assert values["flux"] == 1 and not values["inlet_u"].has(t)
    assert load_lesson("3", "climate_l2.py").DEFAULT_PHYSICS["gamma0"] == 0


def test_numerical_errors_do_not_create_a_hidden_tuning_competition():
    for error in (0, .01, 100):
        assert level_points({"pde": True, "boundary": True}, {"error": error}, ["error"], rules(2))["score"] == 100
    assert level_points({"pde": True, "boundary": False}, {}, ["error"], rules(2))["score"] == 50
