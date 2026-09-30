"""A closed Challenge preserves notebook practice and read-only results access."""
import asyncio
from io import BytesIO
import json
from types import SimpleNamespace
from urllib.error import HTTPError

import pytest

from ETC.runtime.judge_client import JudgeChallengeClosed, JudgeClient, JudgeConnectionError
from ETC.runtime.submission_widgets import SubmissionPanel
from ETC.tests.test_judge import dummy_source
from ETC.tests.test_notebook_submission import FakeClient, TOKEN


@pytest.fixture(autouse=True)
def local_code_check(monkeypatch):
    # These tests cover transport and controls; numerical preflight has its own suite.
    monkeypatch.setattr("ETC.runtime.submission_widgets.check_saved_submission", lambda _payload: {
        "levels": {"wave_l1.py": {"status": "completed", "components": {"example": True}}},
    })


@pytest.mark.parametrize("status,body,route,closed", [
    (409, {"error_code": "challenge_closed"}, "/api/submissions", True),
    (400, {"error_code": "challenge_closed"}, "/api/submissions", False),
    (409, {"error_code": "challenge_closed_extra"}, "/api/submissions", False),
    (409, {"error_code": "nickname_taken"}, "/api/submissions", False),
    (409, {"error_code": "challenge_closed"}, "/api/me", False),
    (409, {"error_code": "challenge_closed"}, "/api/me/nickname", False),
    (409, "not a JSON object", "/api/submissions", False),
])
def test_transport_recognizes_only_exact_submission_closure(monkeypatch, status, body, route, closed):
    calls = []
    if isinstance(body, dict):
        body = {**body, "error": "Private response " + TOKEN}

    def reject(request, **_kwargs):
        calls.append(request.full_url)
        raise HTTPError(request.full_url, status, "Server response " + TOKEN, {}, BytesIO(json.dumps(body).encode()))

    monkeypatch.setattr("ETC.runtime.judge_client.build_opener", lambda *_args: SimpleNamespace(open=reject))
    with pytest.raises(JudgeConnectionError) as caught:
        JudgeClient("http://127.0.0.1:8090", TOKEN)._request(route, {})
    assert isinstance(caught.value, JudgeChallengeClosed) is closed
    assert TOKEN not in str(caught.value) and "Private response" not in str(caught.value)
    assert len(calls) == 1
    if closed:
        assert "No new submissions" in str(caught.value)
        assert "accepted before the deadline will still be graded" in str(caught.value)


class ClosingClient(FakeClient):
    def __init__(self):
        super().__init__()
        self.accepting = False
        self.reads = 0
        self.rows = [{"id": "earlier-receipt", "challenge": "1", "status": "completed", "score": 72.5}]

    def me(self):
        self.reads += 1
        return super().me()

    def submit(self, payload):
        if self.accepting:
            return super().submit(payload)
        self.sent.append(payload)
        raise JudgeChallengeClosed()


def test_closed_panel_keeps_practice_identity_and_results_without_retry(tmp_path):
    (tmp_path / "wave_l1.py").write_text(dummy_source("1", "wave_l1.py"))
    client = ClosingClient()
    panel = SubmissionPanel("1", tmp_path, client=client)
    other = SubmissionPanel("2", tmp_path, client=client)
    try:
        before_history, before_identity = panel.history.value, panel.identity.value
        panel.submit_button.click()
        assert len(client.sent) == 1
        assert panel.challenge_closed and panel.connected and not panel.closed
        assert panel.history.value == before_history and panel.identity.value == before_identity
        assert panel.submit_button.disabled and panel._poll_task is None
        for control in (panel.refresh_button, panel.check_button, panel.levels, panel.nickname_input):
            assert not control.disabled
        assert "unavailable" not in panel.status.value and "outdated" not in panel.status.value
        assert JudgeChallengeClosed.message in panel.status.value
        assert not other.challenge_closed and not other.submit_button.disabled

        # Even a programmatic click on a disabled widget must not send again.
        panel.submit_button.click()
        asyncio.run(panel._perform(submit=True))
        assert len(client.sent) == 1
        panel.refresh_button.click()
        panel.nickname_input.value = "Still connected"
        assert not panel.nickname_button.disabled
        panel.nickname_button.click()
        assert client.names_saved == ["Still connected"] and panel.connected
        panel.check_button.click()
        assert "Code check finished" in panel.status.value
        assert "can still be submitted" not in panel.status.value
        assert JudgeChallengeClosed.message in panel.status.value
        assert panel.challenge_closed and panel.submit_button.disabled
        assert "72.50 / 100" in panel.history.value and len(client.sent) == 1
    finally:
        panel.close()
        other.close()


def test_accepted_jobs_continue_polling_after_closure(monkeypatch, tmp_path):
    (tmp_path / "wave_l1.py").write_text(dummy_source("1", "wave_l1.py"))
    client = ClosingClient()
    monkeypatch.setattr(SubmissionPanel, "POLL_INTERVAL_SECONDS", 0.001)

    async def run():
        panel = SubmissionPanel("1", tmp_path, client=client)
        try:
            await panel._task
            panel.submit_button.click()
            await panel._task
            assert panel.challenge_closed and panel.connected
            client.rows[0].update(status="running", score=None)
            panel.refresh_button.click()
            await panel._task
            assert panel.pending and not panel._poll_task.done()
            client.rows[0].update(status="completed", score=82)
            await panel._poll_task
            assert not panel.pending and panel.connected and panel.challenge_closed
            assert "82.00 / 100" in panel.history.value
            assert JudgeChallengeClosed.message in panel.status.value
            assert panel.submit_button.disabled and len(client.sent) == 1
        finally:
            panel.close()

    asyncio.run(run())


def test_closure_survives_reconnect_and_only_new_panel_forgets_old_event(monkeypatch, tmp_path):
    (tmp_path / "wave_l1.py").write_text(dummy_source("1", "wave_l1.py"))
    client = ClosingClient()
    panel = SubmissionPanel("1", tmp_path, client=client)
    try:
        panel.submit_button.click()
        healthy_me = client.me
        def outage():
            raise JudgeConnectionError("Cannot reach the judge.")
        monkeypatch.setattr(client, "me", outage)
        panel.refresh_button.click()
        assert not panel.connected and panel.challenge_closed
        monkeypatch.setattr(client, "me", healthy_me)
        client.rows = []  # An operator's guarded event reset is not proof of reopening.
        client.accepting = True
        panel.refresh_button.click()
        assert panel.connected and panel.challenge_closed and panel.submit_button.disabled
        assert JudgeChallengeClosed.message in panel.status.value
        assert len(client.sent) == 1
    finally:
        panel.close()

    # Rerunning the controls cell creates a new panel for a new event.
    fresh = SubmissionPanel("1", tmp_path, client=client)
    try:
        assert not fresh.challenge_closed and not fresh.submit_button.disabled
        fresh.submit_button.click()
        assert len(client.sent) == 2
    finally:
        fresh.close()
