"""Implementation scoring is complete before (and survives) optional feedback."""
import copy
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from ETC.judge.catalog import CHALLENGES, rules
from ETC.judge import evaluate as evaluator
from ETC.judge import result_checkpoint as checkpoints
from ETC.judge import worker
from ETC.judge.store import Store
from ETC.tests.test_judge import dummy_source
from ETC.tests.test_judge_operators import dummy_operator_source



def synthetic_assessment(challenge, sources, settings):
    """Transport/checkpoint fixture only: no equation grading or answer source."""
    levels = {}
    for filename in CHALLENGES[challenge]["files"]:
        if filename not in sources:
            levels[filename] = {"status": "not_submitted", "score": 0,
                                "feedback_status": "not_submitted"}
        else:
            levels[filename] = {"status": "implementation_checked", "score": 100,
                "feedback_status": "not_requested", "components": {"synthetic_check": True},
                "implementation_points": 100, "quality_points": 0,
                "evaluation_errors": {}}
    return {"kind": "pilot_not_official", "challenge": challenge,
        "rubric": settings["rubric"], "steps": settings["steps"], "seed": settings["seed"],
        "implementation_assessment_complete": True, "feedback_status": "not_requested",
        "score": round(sum(level["score"] for level in levels.values()) / len(levels), 2),
        "levels": levels}


def sources(challenge):
    return {name: dummy_operator_source(index) if challenge == "4" else dummy_source(challenge, name)
            for index, name in enumerate(CHALLENGES[challenge]["files"], 1)}


def payload(challenge="1"):
    return {"challenge": challenge, "sources": sources(challenge), "settings": rules(2),
            "run_nonce": "a" * 64}










def test_checkpoint_rejects_wrong_attempt_tampering_missing_levels_and_symlinks(tmp_path):
    request = payload()
    result = synthetic_assessment("1", request["sources"], request["settings"])
    path = tmp_path / "checkpoint.json"
    checkpoints.write_checkpoint(path, request, result)
    other = {**request, "run_nonce": "b" * 64}
    with pytest.raises(ValueError, match="different attempt"):
        checkpoints.recover_checkpoint(path, other, "failed")
    for change in (lambda r: r.update(score=99),
                   lambda r: r["levels"].pop("wave_l3.py"),
                   lambda r: r["levels"]["wave_l1.py"].update(quality_points=1),
                   lambda r: r["levels"]["wave_l1.py"]["components"].update(synthetic_check=1)):
        invalid = copy.deepcopy(result)
        change(invalid)
        checkpoints.write_checkpoint(path, request, invalid)
        with pytest.raises(ValueError):
            checkpoints.recover_checkpoint(path, request, "failed")
    link = tmp_path / "linked.json"
    link.symlink_to(path)
    with pytest.raises(ValueError):
        checkpoints.recover_checkpoint(link, request, "failed")


def test_checkpoint_is_atomic_if_replace_fails(tmp_path, monkeypatch):
    request = payload()
    result = synthetic_assessment("1", request["sources"], request["settings"])
    path = tmp_path / "checkpoint.json"
    checkpoints.write_checkpoint(path, request, result)
    before = path.read_bytes()
    monkeypatch.setattr(checkpoints.os, "replace", lambda *args: (_ for _ in ()).throw(OSError("disk error")))
    with pytest.raises(OSError):
        checkpoints.write_checkpoint(path, request, {**result, "score": 0})
    assert path.read_bytes() == before
    assert list(tmp_path.iterdir()) == [path]




def test_recovered_partial_attempt_never_replaces_a_higher_previous_best(tmp_path):
    store = Store.initialize(tmp_path / "judge", steps=2)
    participant = store.authenticate(store.add_participant("Previous best"))["id"]
    complete = payload()
    store.submit(participant, "1", complete["sources"])
    first = store.claim()
    score = synthetic_assessment("1", complete["sources"], complete["settings"])
    store.finish(first, result=score)
    partial = {**complete, "sources": {"wave_l1.py": complete["sources"]["wave_l1.py"]}, "run_nonce": "b" * 64}
    store.submit(participant, "1", partial["sources"], cooldown=0)
    second = store.claim()
    score = synthetic_assessment("1", partial["sources"], partial["settings"])
    score["feedback_status"] = "pending"
    score["levels"]["wave_l1.py"]["feedback_status"] = "pending"
    path = tmp_path / "partial-checkpoint.json"
    checkpoints.write_checkpoint(path, partial, score)
    recovered = checkpoints.recover_checkpoint(path, partial, "time_limit")
    store.finish(second, result=recovered)
    assert recovered["score"] == 33.33
    assert store.board()["participants"][0]["scores"]["1"] == 100
    assert len(store.history(participant)) == 2


@pytest.mark.parametrize("failure,checkpoint", [("crash", True), ("timeout", True),
                                               ("crash", False), ("timeout", False)])
def test_worker_recovers_only_completed_bound_implementation_assessment(tmp_path, monkeypatch, failure, checkpoint):
    store = Store.initialize(tmp_path / "judge", steps=2)
    participant = store.authenticate(store.add_participant("Resilience test"))["id"]
    store.submit(participant, "1", sources("1"))
    killed = []
    class Process:
        pid = 123456789
        returncode = 1
        def __init__(self, command, **kwargs):
            request = json.loads(Path(command[command.index("--input") + 1]).read_text())
            if checkpoint:
                result = synthetic_assessment("1", request["sources"], request["settings"])
                result["feedback_status"] = "running"
                for level in result["levels"].values():
                    level["feedback_status"] = "pending"
                checks_path = Path(command[command.index("--checkpoint") + 1])
                checkpoints.write_checkpoint(checks_path, request, result)
        def wait(self, timeout=None):
            if timeout is not None and failure == "timeout":
                raise subprocess.TimeoutExpired("trusted evaluator", timeout)
            return self.returncode
        def poll(self):
            return self.returncode
    monkeypatch.setattr(worker.subprocess, "Popen", Process)
    monkeypatch.setattr(worker.os, "killpg", lambda pid, sig: killed.append(pid))
    assert worker.work_once(store)
    row = store.history(participant)[0]
    if checkpoint:
        assert row["status"] == "completed" and row["score"] == 100
        expected = "time_limit" if failure == "timeout" else "failed"
        assert row["result"]["feedback_status"] == expected
        assert row["result"]["elapsed_seconds"] >= 0
        assert all("feedback_seconds" not in level for level in row["result"]["levels"].values())
        assert all(level["feedback_status"] == expected for level in row["result"]["levels"].values())
        assert store.board()["participants"][0]["scores"]["1"] == 100
    else:
        assert row["status"] == ("time_limit" if failure == "timeout" else "system_error")
        assert row["score"] is None
        assert store.board()["participants"][0]["scores"]["1"] is None
    assert bool(killed) == (failure == "timeout")


def test_current_recipe_budget_and_scoring_policy_are_unchanged():
    settings = rules()
    assert settings["steps"] == 200 and settings["timeout_seconds"] == 600
    assert settings["implementation_points"] == 100 and settings["quality_points"] == 0
