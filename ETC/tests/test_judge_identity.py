"""Authenticated notebook nickname registration; never anonymous enrollment."""
from concurrent.futures import ThreadPoolExecutor
import json
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from ETC.judge.store import Store, NicknameConflict, NicknameRequired
from ETC.judge.web import create_server
from ETC.tests.test_judge import answer, completed


@pytest.fixture
def store(tmp_path):
    return Store.initialize(tmp_path / "state", steps=2)


def test_private_account_has_no_random_public_name_and_cannot_submit_until_named(store):
    token = store.add_participant()
    person = store.authenticate(token)
    assert person["nickname"] is None
    assert store.board()["participants"] == []
    sources = {"wave_l1.py": answer("1", "wave_l1.py")}
    with pytest.raises(NicknameRequired):
        store.submit(person["id"], "1", sources)
    assert store.history(person["id"]) == []
    assert store.set_nickname(person["id"], "  학생   이름  ") == "학생 이름"
    submission = store.submit(person["id"], "1", sources)
    job = store.claim()
    store.finish(job, completed(job, 75))
    store.set_nickname(person["id"], "New public name")
    assert store.authenticate(token) == {"id": person["id"], "nickname": "New public name"}
    assert store.history(person["id"])[0]["id"] == submission
    row = store.board("1")["participants"][0]
    assert row["nickname"] == "New public name" and row["scores"]["1"] == 75


@pytest.mark.parametrize("nickname", [None, "", "  ", "a" * 41, "x\ny", "x\ty", "x\u200by", "x\u202ey", 123, [], {}])
def test_invalid_names_are_rejected_without_modifying_identity(store, nickname):
    token = store.add_participant("Original")
    person = store.authenticate(token)
    with pytest.raises(ValueError):
        store.set_nickname(person["id"], nickname)
    assert store.authenticate(token) == person


def test_duplicate_names_are_atomic_and_normalized(store):
    people = [store.authenticate(store.add_participant())["id"] for _ in range(2)]

    def register(person):
        try:
            store.set_nickname(person, "Same name")
            return "saved"
        except NicknameConflict:
            return "duplicate"

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sorted(pool.map(register, people)) == ["duplicate", "saved"]
    rows = store.board()["participants"]
    assert len(rows) == 1 and rows[0]["nickname"] == "Same name"
    for name in ("same NAME", "  SAME   name  ", "Ｓａｍｅ ｎａｍｅ"):
        with pytest.raises(NicknameConflict):
            store.add_participant(name)


def test_profile_api_only_changes_authenticated_account(store):
    token = store.add_participant()
    other = store.authenticate(store.add_participant("Reserved"))
    person = store.authenticate(token)
    server = create_server(store, 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"

    def post(payload, headers=None):
        request = Request(base + "/api/me/nickname", data=json.dumps(payload).encode(),
                          headers={"Content-Type": "application/json", **(headers or {})})
        with urlopen(request) as response:
            assert response.status == 200
            return json.load(response)

    headers = {"Authorization": "Bearer " + token}
    try:
        for auth in ({}, {"Authorization": "Bearer unknown"}):
            with pytest.raises(HTTPError) as error:
                post({"nickname": "Learner"}, auth)
            assert error.value.code == 401
        with pytest.raises(HTTPError) as error:
            post({"nickname": "Learner"}, {**headers, "Origin": "https://evil.example"})
        assert error.value.code == 403
        with pytest.raises(HTTPError) as error:
            post({"nickname": "Learner", "participant": other["id"]}, headers)
        assert error.value.code == 400
        with pytest.raises(HTTPError) as error:
            post({"nickname": "reserved"}, headers)
        assert error.value.code == 409
        assert json.load(error.value)["error_code"] == "nickname_taken"
        assert post({"nickname": "Learner"}, headers) == {"nickname": "Learner", "submissions": []}
        assert post({"nickname": "Learner"}, headers)["nickname"] == "Learner"
        assert store.authenticate(token)["id"] == person["id"]
        with store.connect() as db:
            assert db.execute("SELECT nickname FROM participants WHERE id=?", (other["id"],)).fetchone()[0] == "Reserved"
        assert store.history(person["id"]) == []
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
