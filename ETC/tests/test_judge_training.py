"""Real CPU smoke: all eight Levels, through queue and child-process evaluator.

Two steps verify wiring and artifacts only, not convergence or event capacity.
Reference functions are test fixtures; source student exercises stay unfinished.
"""
import ast
import json
import pytest

from ETC.judge.catalog import CHALLENGES, lesson_path
from ETC.judge.store import Store
from ETC.judge.worker import work_once
from ETC.tests.test_judge import answer


@pytest.mark.parametrize("challenge", ("1", "2", "3"))
def test_real_submission_training_all_levels(challenge, tmp_path):
    store = Store.initialize(tmp_path / "judge", steps=2)
    token = store.add_participant("CPU smoke")
    identifier = store.authenticate(token)["id"]
    sources = {}
    for filename in CHALLENGES[challenge]["files"]:
        sources[filename] = answer(challenge, filename)
    store.submit(identifier, challenge, sources)
    assert work_once(store)
    row = store.history(identifier)[0]
    logs = "\n".join(p.read_text()[-8000:] for p in store.directory.glob("runs/*/runner.log"))
    assert row["status"] == "completed", logs
    assert row["score"] == 100
    assert all(level["status"] == "evaluated" for level in row["result"]["levels"].values())
    for metrics_path in store.directory.glob("runs/*/artifacts/*/metrics.json"):
        metrics = json.loads(metrics_path.read_text())
        assert metrics["reference_implementation"] is False
        assert metrics["steps"] == 2
    assert len(list(store.directory.glob("runs/*/artifacts/*/metrics.json"))) == len(sources)
    if challenge == "2":
        plots = list(store.directory.glob("runs/*/artifacts/*/openfoam_comparison.png"))
        assert len(plots) == 1 and plots[0].parent.name == "chip_2d_l1"
        assert plots[0].stat().st_size > 10000
        metrics = json.loads((plots[0].parent / "metrics.json").read_text())
        assert metrics["openfoam_evaluation_points"] == 56942
