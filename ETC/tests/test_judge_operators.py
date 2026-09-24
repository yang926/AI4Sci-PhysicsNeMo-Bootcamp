"""Challenge 4 contract, restricted interpreter, export and real training."""
import ast
import json
import sys

import pytest
import yaml

from ETC.judge.catalog import CHALLENGES, lesson_path, rules
from ETC.judge.evaluate import load_lesson
from ETC.judge.expressions import SubmissionError
from ETC.judge.operators import check_operator, source_nodes
from ETC.judge.store import Store
from ETC.judge.worker import work_once
from ETC.runtime.submission import export_submission, submission_html


def operator_answer(level):
    datasets = '''def build_datasets(train_pairs, val_pairs, test_pairs):
    return tuple(TensorDataset(*pairs) for pairs in (train_pairs, val_pairs, test_pairs))
'''
    model = '''def build_model(model_config, grid_size):
    return FNO(in_channels=1, out_channels=1, dimension=2, **model_config)
'''
    if level == 2:
        model = '''def build_model(model_config, grid_size):
    patch = model_config.get("patch_size", [4, 4])
    if any(grid_size % size for size in patch):
        raise ValueError("Grid must be divisible by both patch dimensions")
    return AFNO(inp_shape=[grid_size, grid_size], in_channels=1, out_channels=1, **model_config)
'''
    physics = '''class ReactionDiffusionPDE(PDE):
    def __init__(self):
        self.dim = 2
        x, y = Symbol("x"), Symbol("y")
        u, f = Function("u")(x, y), Function("f")(x, y)
        self.equations = {"reaction_diffusion": u - u.diff(x, 2) - u.diff(y, 2) - f}
'''
    return datasets + "\n" + model + ("\n" + physics if level == 3 else "")


def check(source, level):
    load_lesson("4", f"fno_physicsnemo_l{level}.py")
    directory = lesson_path("4", f"fno_physicsnemo_l{level}.py").parent
    config = yaml.safe_load((directory / "conf" / f"config_{['FNO','AFNO','PINO'][level-1]}.yaml").read_text())
    return check_operator(source, level, config, sys.modules["operator_training"])


@pytest.mark.parametrize("level", (1, 2, 3))
def test_correct_factories_and_unfinished_templates(level):
    assert all(check(operator_answer(level), level)["checks"].values())
    unfinished = check(lesson_path("4", f"fno_physicsnemo_l{level}.py").read_text(), level)
    assert not any(unfinished["checks"].values())


@pytest.mark.parametrize("replacement", [
    '(TensorDataset(*train_pairs), TensorDataset(*val_pairs), TensorDataset(*test_pairs))',
    '[TensorDataset(train_pairs[0], train_pairs[1]), TensorDataset(val_pairs[0], val_pairs[1]), TensorDataset(test_pairs[0], test_pairs[1])]',
])
def test_explicit_dataset_constructors_are_accepted(replacement):
    source = operator_answer(1).replace('tuple(TensorDataset(*pairs) for pairs in (train_pairs, val_pairs, test_pairs))', replacement)
    assert all(check(source, 1)["checks"].values())


@pytest.mark.parametrize("old,new,component", [
    ('in_channels=1', 'in_channels=True', 'model'),
    ('in_channels=1', 'in_channels=1.0', 'model'),
    ('in_channels=1', 'in_channels=2', 'model'),
    ('FNO(', 'AFNO(', 'model'),
    ('**model_config', '**{}', 'model'),
    ('for pairs in (train_pairs, val_pairs, test_pairs)', 'for pairs in (test_pairs, val_pairs, train_pairs)', 'data_splits'),
    ('TensorDataset(*pairs)', 'TensorDataset(pairs[1], pairs[0])', 'data_splits'),
    ('TensorDataset(*pairs)', 'TensorDataset(pairs[0].zero_(), pairs[1])', 'data_splits'),
    ('TensorDataset(*pairs)', 'TensorDataset(pairs[0] * 2, pairs[1])', 'data_splits'),
    ('FNO(in_channels=1, out_channels=1, dimension=2, **model_config)', '__import__("os").system("id")', 'model'),
])
def test_wrong_or_unsupported_code_cannot_earn_full_credit(old, new, component):
    result = check(operator_answer(1).replace(old, new), 1)
    assert result["checks"][component] is False
    assert result["messages"][component]


def test_afno_checks_both_patch_dimensions():
    source = operator_answer(2).replace('if any(grid_size % size for size in patch):', 'if grid_size % patch[0]:')
    result = check(source, 2)
    assert result["checks"]["model"] is True
    assert result["checks"]["patch_compatibility"] is False
    for guard in ('if grid_size % patch[0] != 0 or grid_size % patch[1] != 0:',):
        assert all(check(operator_answer(2).replace('if any(grid_size % size for size in patch):', guard), 2)["checks"].values())


@pytest.mark.parametrize("residual", ['u + u.diff(x, 2) + u.diff(y, 2) - f', '0', 'u - u.diff(x, 2) - u.diff(y, 2)', '__import__("os").system("id")'])
def test_pino_wrong_equation_and_arbitrary_python_fail(residual):
    source = operator_answer(3).replace('u - u.diff(x, 2) - u.diff(y, 2) - f', residual)
    result = check(source, 3)
    assert result["checks"]["physics"] is False
    assert result["checks"]["model"] is True


def test_top_level_and_default_expressions_are_not_executed(tmp_path):
    marker = tmp_path / "MUST_NOT_EXIST"
    source = f'open({str(marker)!r}, "w").write("bad")\n' + operator_answer(1)
    source = source.replace('grid_size):', f'grid_size=open({str(marker)!r}, "w")):')
    assert all(check(source, 1)["checks"].values())
    assert not marker.exists()


def test_huge_model_description_never_constructs_a_model(monkeypatch):
    load_lesson("4", "fno_physicsnemo_l1.py")
    def forbidden(*args, **kwargs):
        raise AssertionError("Do not construct rejected model configurations")
    monkeypatch.setattr(sys.modules["operator_training"], "reference_fno", forbidden)
    source = operator_answer(1).replace('**model_config', 'latent_channels=9999')
    assert check(source, 1)["checks"]["model"] is False


@pytest.mark.parametrize("level", (1, 2, 3))
def test_export_matches_existing_exercises_and_contains_only_required_source(tmp_path, level):
    filename = f"fno_physicsnemo_l{level}.py"
    (tmp_path / filename).write_text(operator_answer(level) + '\nPRIVATE_VALUE="not exported"\n')
    path = export_submission("4", tmp_path, levels=(level,))
    payload = json.loads(path.read_text())
    assert payload["challenge"] == "4" and set(payload["sources"]) == {filename}
    assert "PRIVATE_VALUE" not in path.read_text()
    assert len(source_nodes(payload["sources"][filename], level)) == (3 if level == 3 else 2)
    assert all(check(payload["sources"][filename], level)["checks"].values())
    (tmp_path / filename).write_text(lesson_path("4", filename).read_text())
    with pytest.raises(SubmissionError, match="finish"):
        export_submission("4", tmp_path, levels=(level,))


def test_fourth_challenge_and_overall_ranks_use_the_same_completed_snapshot(tmp_path):
    store = Store.initialize(tmp_path / "judge", steps=2)
    token = store.add_participant("Operator learner")
    person = store.authenticate(token)["id"]
    store.submit(person, "4", {"fno_physicsnemo_l1.py": operator_answer(1)})
    job = store.claim()
    store.finish(job, {"kind":"pilot_not_official", "rubric":job["settings"]["rubric"], "challenge":"4", "score":80})
    board = store.board()
    row = board["participants"][0]
    assert board["rules"]["overall_max"] == 400
    assert row["scores"]["4"] == row["total"] == 80
    assert row["challenge_ranks"]["4"] == row["rank"] == 1
    assert "ReactionDiffusionPDE" in submission_html("4", False)


def test_real_fno_afno_pino_submission_training(tmp_path):
    store = Store.initialize(tmp_path / "judge", steps=2)
    token = store.add_participant("Operator CPU smoke")
    person = store.authenticate(token)["id"]
    sources = {f"fno_physicsnemo_l{level}.py": operator_answer(level) for level in (1, 2, 3)}
    store.submit(person, "4", sources)
    assert work_once(store)
    row = store.history(person)[0]
    logs = "\n".join(p.read_text()[-9000:] for p in store.directory.glob("runs/*/runner.log"))
    assert row["status"] == "completed", logs
    assert 50 < row["score"] <= 100
    assert all(level["status"] == "evaluated" for level in row["result"]["levels"].values())
    metrics_paths = list(store.directory.glob("runs/*/artifacts/fno_physicsnemo_l*/metrics.json"))
    assert len(metrics_paths) == 3
    for path in metrics_paths:
        metrics = json.loads(path.read_text())
        assert metrics["reference"] is False and metrics["steps"] == 2
        assert metrics["split_samples"] == {"train":64, "val":16, "test":16}
        assert metrics["dataset_exercise_checked"] is True
        assert metrics["evaluation_equations"] == "provided_reference_equations"
        if metrics["level"] == 3:
            assert abs(metrics["test"]["pde_rmse_fft"] - metrics["test"]["pde_rmse_physicsinformer"]) < 1e-3
