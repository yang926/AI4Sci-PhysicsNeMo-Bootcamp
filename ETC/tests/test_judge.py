"""Submission, score provenance, queue isolation and notebook hand-off tests."""
import ast
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from ETC.judge.catalog import CHALLENGES, lesson_path, quality_metrics, rules
from ETC.judge.evaluate import check_equations, level_points, load_lesson
from ETC.judge.expressions import SubmissionError
from ETC.judge.store import BusyError, Store
from ETC.judge.web import create_server
from ETC.runtime.submission import export_submission, submission_html


CASES = [(key, name) for key, spec in CHALLENGES.items() if key != "4" for name in spec["files"]]


def answer(challenge, filename):
    source = lesson_path(challenge, filename).read_text()
    function = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "reference_equations")
    return ast.get_source_segment(source, function).replace("def reference_equations(", "def student_equations(", 1)


@pytest.fixture
def store(tmp_path):
    return Store.initialize(tmp_path / "private", steps=2)


def person(store, name="Student"):
    token = store.add_participant(name)
    return store.authenticate(token)["id"], token


def completed(job, score):
    return {"kind": "pilot_not_official", "challenge": job["challenge"], "score": score, "rubric": job["settings"]["rubric"]}


@pytest.mark.parametrize("challenge,filename", CASES)
def test_every_equation_contract_accepts_reference_and_rejects_unfinished(challenge, filename):
    module = load_lesson(challenge, filename)
    fn, checks = check_equations(module, answer(challenge, filename), challenge)
    assert checks and all(checks.values())
    with pytest.raises(SubmissionError, match="unfinished"):
        check_equations(module, lesson_path(challenge, filename).read_text(), challenge)
    # Defaults containing Python floats must work during real training too.
    module.student_equations = fn
    pde_type = getattr(module, "WaveEquation2D", getattr(module, "NavierStokes2D", getattr(module, "ClimatePDE", None)))
    assert pde_type(reference=False).equations


@pytest.mark.parametrize("body", [
    'return __import__("os").system("id")',
    'return reference_equations(x,y,t,u,c)',
    'return {"wave": u.__class__}',
    'import os\n    return {"wave": u}',
    'while True: pass\n    return {"wave": u}',
    'return {"wave": open("/etc/passwd").read()}',
    'return {"wave": u.diff(t,9999)}',
    'return {"wave": 9999 ** 9999}',
    'return {"wave": u.diff(t,2).diff(t,2)}',
    'return {"wave": "x" * 9999}',
    'return {"wave": 1/0}',
    'return {"wave": u, "wave": 0}',
])
def test_unsupported_python_is_rejected_without_execution(body):
    with pytest.raises(SubmissionError):
        check_equations(load_lesson("1", "wave_l1.py"), "def student_equations(x,y,t,u,c=1.0):\n    " + body, "1")


def test_top_level_and_default_expressions_are_never_executed(tmp_path):
    marker = tmp_path / "MUST_NOT_EXIST"
    source = f'open({str(marker)!r}, "w").write("bad")\n' + answer("1", "wave_l1.py")
    source = source.replace("c=1.0", f'c=open({str(marker)!r}, "w")')
    _, checks = check_equations(load_lesson("1", "wave_l1.py"), source, "1")
    assert all(checks.values()) and not marker.exists()


def test_missing_climate_advection_is_detected_even_when_training_default_is_zero():
    source = answer("3", "climate_l1.py").replace('params["u0"]*T.diff(x)', '0')
    _, checks = check_equations(load_lesson("3", "climate_l1.py"), source, "3")
    assert checks == {"adr": False}


def test_equivalent_expanded_equation_is_accepted_and_zero_is_not():
    source = 'def student_equations(x,y,t,u,c=1.0):\n    return {"wave": u.diff(t,2)-c*c*u.diff(x,2)-c*c*u.diff(y,2)}'
    module = load_lesson("1", "wave_l1.py")
    assert all(check_equations(module, source, "1")[1].values())
    zero = 'def student_equations(x,y,t,u,c=1.0):\n    return {"wave": 0}'
    assert check_equations(module, zero, "1")[1] == {"wave": False}


def test_quality_scoring_requires_all_metrics_and_equation_gate():
    settings = rules(2)
    assert level_points({"pde": False}, {}, ["rmse"], settings)["score"] == 0
    assert level_points({"pde": True}, {"rmse": 0}, ["rmse"], settings)["score"] == 100
    assert level_points({"pde": True}, {"rmse": 1}, ["rmse"], settings)["score"] == 75
    for metrics in ({}, {"rmse": float("nan")}, {"rmse": -1}, {"rmse": True}):
        with pytest.raises(RuntimeError):
            level_points({"pde": True}, metrics, ["rmse"], settings)


def test_authentication_and_submission_validation(store):
    identifier, token = person(store)
    assert store.authenticate(token)["id"] == identifier
    assert store.authenticate("not-a-token") is None
    sources = {"wave_l1.py": answer("1", "wave_l1.py")}
    first = store.submit(identifier, "1", sources)
    assert store.submit(identifier, "1", sources) == first
    with pytest.raises(BusyError):
        store.submit(identifier, "2", {"chip_2d_l1.py": answer("2", "chip_2d_l1.py")})
    for challenge, files in (("4", sources), ("1", {}), ("1", {"../../evil.py": "bad"}), ("1", {"wave_l1.py": "x" * 65537})):
        with pytest.raises(SubmissionError):
            store.submit(identifier, challenge, files)


def test_atomic_queue_claims_each_submission_once(store):
    for index in range(8):
        identifier, _ = person(store, f"Student {index}")
        store.submit(identifier, "1", {"wave_l1.py": answer("1", "wave_l1.py")})
    with ThreadPoolExecutor(max_workers=8) as pool:
        jobs = list(pool.map(lambda _: store.claim(), range(8)))
    assert len({job["id"] for job in jobs}) == 8
    assert store.claim() is None


def test_best_scores_joint_ranks_and_failed_resubmission(store):
    people = [person(store, nickname)[0] for nickname in ("A", "B", "Not submitted")]
    for identifier in people[:2]:
        store.submit(identifier, "1", {"wave_l1.py": answer("1", "wave_l1.py")})
        job = store.claim()
        store.finish(job, completed(job, 75))
    identifier = people[0]
    store.submit(identifier, "1", {"wave_l1.py": answer("1", "wave_l1.py") + "\n# retry"}, cooldown=0)
    job = store.claim()
    store.finish(job, status="system_error", error="test failure")
    board = store.board()["participants"]
    assert [r["rank"] for r in board] == [1, 1, None]
    assert board[0]["total"] == 75 and board[0]["scores"]["2"] is None
    assert board[2]["scores"]["1"] is None
    assert "sources" not in str(board) and "token_hash" not in str(board)
    assert len(store.history(people[1])) == 1


def test_interrupted_worker_lease_cannot_overwrite_new_state(store):
    identifier, _ = person(store)
    source = {"wave_l1.py": answer("1", "wave_l1.py")}
    store.submit(identifier, "1", source)
    old = store.claim()
    with store.connect() as db:
        db.execute("UPDATE submissions SET started=0 WHERE id=?", (old["id"],))
    assert store.claim() is None
    store.finish(old, completed(old, 100))
    assert store.history(identifier)[0]["status"] == "system_error"
    assert store.board()["participants"][0]["rank"] is None
    assert store.submit(identifier, "1", source, cooldown=0) != old["id"]


def test_invalid_result_and_changed_rules_fail_closed(store):
    identifier, _ = person(store)
    store.submit(identifier, "1", {"wave_l1.py": answer("1", "wave_l1.py")})
    job = store.claim()
    for bad in (-1, 101, float("nan"), True):
        with pytest.raises(ValueError):
            store.finish(job, completed(job, bad))
    with store.connect() as db:
        db.execute("UPDATE settings SET fingerprint='changed'")
    with pytest.raises(ValueError, match="changed"):
        store.claim()


def test_export_is_explicit_student_only_and_contains_no_reference_code(tmp_path):
    path = tmp_path / "wave_l1.py"
    path.write_text(answer("1", "wave_l1.py") + '\nSECRET="do not export"\n')
    with pytest.raises(SubmissionError, match="Reference"):
        export_submission("1", tmp_path, reference=True)
    output = export_submission("1", tmp_path)
    payload = json.loads(output.read_text())
    assert set(payload) == {"challenge", "sources"}
    assert "SECRET" not in output.read_text()
    assert set(payload["sources"]) == {"wave_l1.py"}
    path.write_text(lesson_path("1", "wave_l1.py").read_text())
    with pytest.raises(SubmissionError, match="finish"):
        export_submission("1", tmp_path)


def test_submission_ui_does_not_invent_an_event_url():
    assert "not configured" in submission_html("1", False)
    assert "Instructor demonstration" in submission_html("2", True)
    assert "https://judge.example/" in submission_html("3", False, "https://judge.example/")
    for url in ("javascript:alert(1)", "https://user:secret@example.com", "http://localhost/?token=secret"):
        with pytest.raises(ValueError):
            submission_html("1", False, url)


def test_private_state_rejects_served_repository_and_shared_directory(tmp_path):
    from ETC.judge.catalog import ROOT
    with pytest.raises(ValueError, match="outside"):
        Store.initialize(ROOT / "ETC/judge/unsafe-state")
    shared = tmp_path / "shared"
    shared.mkdir(mode=0o755)
    with pytest.raises(ValueError, match="private"):
        Store.initialize(shared)
    assert not (shared / "judge.sqlite3").exists()


def test_http_submission_identity_csrf_and_privacy(store):
    identifier, token = person(store, "HTTP learner")
    server = create_server(store, 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        for route in ("/", "/display", "/display/", "/display?ranking=4&rotate=1"):
            with urlopen(base + route) as response:
                html = response.read().decode()
                assert 'data-view="display"' in html
                assert 'id="fullscreen"' in html
                assert 'id="token"' not in html and '<form' not in html
                assert "frame-ancestors 'none'" in response.headers["Content-Security-Policy"]
        with urlopen(base + "/api/board") as response:
            assert json.load(response)["rules"]["status"] == "pilot_not_official"
        with urlopen(base + "/api/board?challenge=3") as response:
            selected = json.load(response)
        assert selected["challenge"] == "3"
        assert set(selected["challenges"]) == {"3"}
        assert set(selected["participants"][0]) == {"nickname", "scores", "challenge_ranks"}
        assert selected["participants"][0]["nickname"] == "HTTP learner"
        for query in ("challenge=overall", "challenge=", "challenge=0", "challenge=1&challenge=2", "ranking=1"):
            with pytest.raises(HTTPError) as error:
                urlopen(base + "/api/board?" + query)
            assert error.value.code == 400
        with pytest.raises(HTTPError) as error:
            urlopen(base + "/api/me")
        assert error.value.code == 401
        payload = json.dumps({"challenge": "1", "sources": {"wave_l1.py": answer("1", "wave_l1.py")}}).encode()
        headers = {"Authorization": "Bearer " + token, "Content-Type": "application/json"}
        for authorization in ({}, {"Authorization": "Bearer unknown"}):
            with pytest.raises(HTTPError) as error:
                urlopen(Request(base + "/api/submissions", data=payload, headers={"Content-Type": "application/json", **authorization}))
            assert error.value.code == 401
        assert store.history(identifier) == []
        with pytest.raises(HTTPError) as error:
            urlopen(Request(base + "/api/submissions", data=payload, headers={**headers, "Origin": "https://evil.example"}))
        assert error.value.code == 403
        with urlopen(Request(base + "/api/submissions", data=payload, headers=headers)) as response:
            assert response.status == 202
        with urlopen(Request(base + "/api/me", headers=headers)) as response:
            own = json.load(response)
        assert len(own["submissions"]) == 1 and own["nickname"] == "HTTP learner"
        assert own["submissions"][0]["status"] == "queued"
        other, other_token = person(store, "Other")
        with urlopen(Request(base + "/api/me", headers={"Authorization": "Bearer " + other_token})) as response:
            assert json.load(response)["submissions"] == []
        assert store.history(identifier)[0]["score"] is None
    finally:
        server.shutdown(); server.server_close(); thread.join()


def test_challenge_board_uses_registered_name_and_only_that_challenges_scores(store):
    earlier_id, earlier_token = person(store, "Earlier leader")
    current_id, current_token = person(store, "Current leader")
    person(store, "No submissions yet")
    for identifier, challenge, filename, score in (
        (earlier_id, "1", "wave_l1.py", 100),
        (earlier_id, "2", "chip_2d_l1.py", 100),
        (earlier_id, "3", "climate_l1.py", 10),
        (current_id, "3", "climate_l1.py", 90),
    ):
        store.submit(identifier, challenge, {filename: answer(challenge, filename)}, cooldown=0)
        job = store.claim()
        store.finish(job, completed(job, score))
    store.submit(earlier_id, "1", {"wave_l1.py": answer("1", "wave_l1.py") + "\n# revised"}, cooldown=0)
    selected = store.board(challenge="3")
    assert selected["challenge"] == "3"
    assert set(selected["challenges"]) == {"3"}
    assert selected["queue"] == {"completed": 2}
    assert "overall_max" not in selected["rules"]
    rows = selected["participants"]
    assert [row["nickname"] for row in rows] == ["Current leader", "Earlier leader", "No submissions yet"]
    assert [row["challenge_ranks"]["3"] for row in rows] == [1, 2, None]
    assert [row["scores"]["3"] for row in rows] == [90, 10, None]
    for row in rows:
        assert set(row) == {"nickname", "scores", "challenge_ranks"}
        assert set(row["scores"]) == set(row["challenge_ranks"]) == {"3"}
    assert rows[0]["nickname"] == store.authenticate(current_token)["nickname"]
    assert rows[1]["nickname"] == store.authenticate(earlier_token)["nickname"]
    # Historical scores and overall totals stay available internally, unchanged.
    complete = store.board()
    assert complete["participants"][0]["nickname"] == "Earlier leader"
    assert complete["participants"][0]["total"] == 210
    assert complete["queue"] == {"completed": 4, "queued": 1}
    assert store.board(challenge="1")["queue"] == {"completed": 1, "queued": 1}
    with pytest.raises(ValueError):
        store.board(challenge="overall")
