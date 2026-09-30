"""Operator credential revocation preserves results and cannot bypass the API."""
import hashlib
import json
import subprocess
import sys
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from ETC.judge import store as store_module
from ETC.judge.store import Store
from ETC.judge.web import create_server
from ETC.tests.test_judge import dummy_source, completed


@pytest.fixture
def store(tmp_path):
    return Store.initialize(tmp_path / "private-state", steps=2)


def scored_person(store):
    token = store.add_participant("Historical learner")
    identifier = store.authenticate(token)["id"]
    sources = {"wave_l1.py": dummy_source("1", "wave_l1.py")}
    store.submit(identifier, "1", sources)
    job = store.claim()
    store.finish(job, completed(job, 75))
    return identifier, token, sources


def test_revocation_is_permanent_idempotent_and_preserves_history(store):
    identifier, token, sources = scored_person(store)
    other = store.add_participant("Still active")
    history, settings = store.history(identifier), store.settings()
    board = store.board()["participants"]
    store.revoke_participant(identifier)
    first = store.list_participants()
    store.revoke_participant(identifier)
    assert store.list_participants() == first
    assert store.authenticate(token) is None
    assert store.authenticate(other)["nickname"] == "Still active"
    assert store.history(identifier) == history
    assert store.board()["participants"] == board
    assert store.require_current_version() == settings
    assert all(set(row) == {"id", "nickname", "revoked_at"} for row in first)
    assert token not in json.dumps(first)
    assert hashlib.sha256(token.encode()).hexdigest() not in json.dumps(first)
    # A stale in-process participant ID cannot evade reauthentication.
    with pytest.raises(ValueError, match="revoked"):
        store.submit(identifier, "1", sources, cooldown=0)
    with pytest.raises(ValueError, match="revoked"):
        store.set_nickname(identifier, "Must not rename")
    assert store.history(identifier) == history


def test_unknown_participant_does_not_create_revocation_record(store):
    with pytest.raises(ValueError, match="Unknown participant"):
        store.revoke_participant("not-a-real-participant")
    assert store.list_participants() == []
    with store.connect() as db:
        assert db.execute("SELECT count(*) FROM revoked_credentials").fetchone()[0] == 0


def test_revocation_does_not_relax_frozen_version_checks(store, monkeypatch):
    token = store.add_participant("Unchanged state")
    identifier = store.authenticate(token)["id"]
    monkeypatch.setattr(store_module, "fingerprint", lambda settings: "changed")
    with pytest.raises(ValueError, match="changed"):
        store.revoke_participant(identifier)
    assert store.authenticate(token)["id"] == identifier
    assert store.list_participants()[0]["revoked_at"] is None


def test_revoked_token_is_denied_by_every_authenticated_http_route(store):
    identifier, token, sources = scored_person(store)
    server = create_server(store, 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    headers = {"Authorization": "Bearer " + token, "Content-Type": "application/json"}
    try:
        with urlopen(Request(base + "/api/me", headers=headers), timeout=5) as response:
            assert json.load(response)["nickname"] == "Historical learner"
        store.revoke_participant(identifier)
        for route, payload in (
            ("/api/me", None),
            ("/api/me/nickname", {"nickname": "Must not rename"}),
            ("/api/submissions", {"challenge": "1", "sources": sources}),
        ):
            request = Request(base + route, headers=headers,
                              data=None if payload is None else json.dumps(payload).encode())
            with pytest.raises(HTTPError) as error:
                urlopen(request, timeout=5)
            assert error.value.code == 401
        # Published scores remain public; revoking access is not disqualification.
        with urlopen(base + "/api/board?challenge=1", timeout=5) as response:
            row = json.load(response)["participants"][0]
        assert row["nickname"] == "Historical learner"
        assert row["scores"]["1"] == 75
        assert len(store.history(identifier)) == 1
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_local_operator_cli_lists_metadata_and_revokes_without_printing_secret(store):
    token = store.add_participant("CLI learner")
    identifier = store.authenticate(token)["id"]
    command = [sys.executable, "-m", "ETC.judge", "--state", str(store.directory)]
    listing = subprocess.run(command + ["list-participants"], capture_output=True,
                             text=True, timeout=15, check=True)
    assert json.loads(listing.stdout) == [{"id": identifier, "nickname": "CLI learner", "revoked_at": None}]
    revoked = subprocess.run(command + ["revoke-participant", "--participant", identifier],
                             capture_output=True, text=True, timeout=15, check=True)
    assert "revoked" in revoked.stdout
    assert token not in listing.stdout + revoked.stdout + listing.stderr + revoked.stderr
    assert store.authenticate(token) is None
