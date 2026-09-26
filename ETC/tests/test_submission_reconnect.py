"""Late Launchable provisioning and read-only notebook connection recovery."""
import asyncio
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading

import pytest

from ETC.runtime.judge_client import JudgeConnectionError
from ETC.runtime.submission_widgets import SubmissionPanel
from ETC.tests.test_judge import answer
from ETC.tests.test_notebook_submission import FakeClient, TOKEN, serving


@pytest.fixture(autouse=True)
def isolated_connection(monkeypatch, tmp_path):
    monkeypatch.delenv("AI4SCI_JUDGE_TOKEN", raising=False)
    monkeypatch.delenv("AI4SCI_JUDGE_URL", raising=False)
    monkeypatch.setenv("AI4SCI_JUDGE_CONFIG", str(tmp_path / "judge.json"))
    monkeypatch.setattr(SubmissionPanel, "POLL_INTERVAL_SECONDS", 0.001)
    monkeypatch.setattr(SubmissionPanel, "MAX_POLL_ATTEMPTS", 3)


def test_late_private_file_connects_without_rerun_or_post(tmp_path):
    methods = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_GET(self):
            methods.append("GET")
            assert self.path == "/api/me"
            body = json.dumps({"nickname": None, "submissions": []}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            methods.append("POST")
            self.send_response(500)
            self.end_headers()

    async def run(url):
        panel = SubmissionPanel("1", tmp_path)
        try:
            await panel._task
            assert panel.client is None and not panel.refresh_button.disabled
            assert "Local practice" in panel.status.value
            panel.nickname_input.value = "My unsaved nickname"
            config = tmp_path / "judge.json"
            config.write_text(json.dumps({"url": url, "token": TOKEN}))
            config.chmod(0o600)
            await panel._poll_task
            assert panel.connected and panel.client is not None
            assert panel.nickname_input.value == "My unsaved nickname"
            assert not panel.nickname_button.disabled
            assert panel.submit_button.disabled
            assert TOKEN not in panel.identity.value + panel.status.value + panel.history.value
        finally:
            panel.close()

    with serving(ThreadingHTTPServer(("127.0.0.1", 0), Handler)) as url:
        asyncio.run(run(url))
    assert methods == ["GET"]


def test_manual_refresh_can_connect_when_initial_configuration_was_missing(monkeypatch, tmp_path):
    client = FakeClient()
    client.nickname = None
    panel = SubmissionPanel("1", tmp_path)
    try:
        assert not panel.refresh_button.disabled
        assert not panel.nickname_input.disabled
        panel.nickname_input.value = "Keep this draft"
        monkeypatch.setattr("ETC.runtime.submission_widgets.JudgeClient.from_environment", lambda _: client)
        panel.refresh_button.click()
        assert panel.connected and panel.client is client
        assert panel.nickname_input.value == "Keep this draft"
        assert client.names_saved == [] and client.sent == []
    finally:
        panel.close()


def test_connection_retries_are_bounded_and_manual_refresh_rearms(monkeypatch, tmp_path):
    calls = []

    def missing(_):
        calls.append("configuration read")
        raise JudgeConnectionError("Judge connection not configured.")

    monkeypatch.setattr("ETC.runtime.submission_widgets.JudgeClient.from_environment", missing)

    async def run():
        panel = SubmissionPanel("1", tmp_path)
        try:
            await panel._task
            await panel._poll_task
            assert len(calls) == 4  # Initial attempt plus three automatic retries.
            assert "paused" in panel.identity.value
            assert "Local practice" in panel.status.value
            assert not panel.refresh_button.disabled and panel.submit_button.disabled
            await asyncio.sleep(0.01)
            assert len(calls) == 4
            panel.refresh_button.click()
            await panel._task
            await panel._poll_task
            assert len(calls) == 8
        finally:
            panel.close()

    asyncio.run(run())


def test_close_cancels_waiting_reconnect(monkeypatch, tmp_path):
    calls = []

    def missing(_):
        calls.append("configuration read")
        raise JudgeConnectionError("Not connected.")

    monkeypatch.setattr("ETC.runtime.submission_widgets.JudgeClient.from_environment", missing)
    monkeypatch.setattr(SubmissionPanel, "POLL_INTERVAL_SECONDS", 60)

    async def run():
        panel = SubmissionPanel("1", tmp_path)
        await panel._task
        polling = panel._poll_task
        panel.close()
        await asyncio.gather(polling, return_exceptions=True)
        assert polling.cancelled()
        assert calls == ["configuration read"]
        assert panel.refresh_button.disabled and panel.submit_button.disabled

    asyncio.run(run())


def test_close_during_private_config_read_does_not_contact_judge(monkeypatch, tmp_path):
    started, release = threading.Event(), threading.Event()
    client = FakeClient()
    calls = []

    def delayed(_):
        started.set()
        assert release.wait(timeout=5)
        return client

    client.me = lambda: calls.append("GET")
    monkeypatch.setattr("ETC.runtime.submission_widgets.JudgeClient.from_environment", delayed)

    async def run():
        panel = SubmissionPanel("1", tmp_path)
        try:
            assert await asyncio.to_thread(started.wait, 5)
            panel.close()
        finally:
            release.set()
        await asyncio.gather(panel._task, return_exceptions=True)
        assert panel._task.cancelled()
        assert panel._poll_task is None

    asyncio.run(run())
    assert calls == []


@pytest.mark.parametrize("draft", ["Draft during restart", ""])
def test_injected_client_recovers_after_server_restart_and_keeps_draft(draft, monkeypatch, tmp_path):
    client = FakeClient()
    original_me = client.me
    calls = []
    failing = False

    def me():
        calls.append("GET")
        if failing:
            raise JudgeConnectionError("Cannot reach the judge.")
        return original_me()

    def must_not_replace(_):
        pytest.fail("An explicitly supplied client must never be replaced.")

    client.me = me
    monkeypatch.setattr("ETC.runtime.submission_widgets.JudgeClient.from_environment", must_not_replace)

    async def run():
        nonlocal failing
        panel = SubmissionPanel("1", tmp_path, client=client)
        try:
            await panel._task
            panel.nickname_input.value = draft
            failing = True
            panel.refresh_button.click()
            await panel._task
            assert not panel.connected
            assert "unavailable" in panel.identity.value
            failing = False
            await panel._poll_task
            assert panel.connected and panel.client is client
            assert panel.nickname_input.value == draft
            assert client.sent == [] and client.names_saved == []
        finally:
            panel.close()

    asyncio.run(run())
    assert calls == ["GET", "GET", "GET"]


def test_reconnection_reloads_replaced_private_configuration(monkeypatch, tmp_path):
    old, replacement = FakeClient(), FakeClient()
    calls = []

    def old_me():
        raise JudgeConnectionError("Personal judge authentication failed.")

    old.me = old_me

    def configured(_):
        calls.append("configuration read")
        return old if len(calls) == 1 else replacement

    monkeypatch.setattr("ETC.runtime.submission_widgets.JudgeClient.from_environment", configured)

    async def run():
        panel = SubmissionPanel("1", tmp_path)
        try:
            await panel._task
            await panel._poll_task
            assert panel.connected and panel.client is replacement
            assert old.sent == replacement.sent == []
            assert old.names_saved == replacement.names_saved == []
        finally:
            panel.close()

    asyncio.run(run())
    assert len(calls) == 2


@pytest.mark.parametrize("action", ["submit", "nickname"])
def test_failed_post_is_not_replayed_during_reconnection(action, tmp_path):
    (tmp_path / "wave_l1.py").write_text(answer("1", "wave_l1.py"))
    client = FakeClient()

    def failed_submit(payload):
        client.sent.append(payload)
        raise JudgeConnectionError("Cannot reach the judge. A submission may already have arrived.")

    def failed_nickname(nickname):
        client.names_saved.append(nickname)
        raise JudgeConnectionError("Cannot reach the judge.")

    client.submit = failed_submit
    client.register_nickname = failed_nickname

    async def run():
        panel = SubmissionPanel("1", tmp_path, client=client)
        try:
            await panel._task
            if action == "submit":
                panel.submit_button.click()
            else:
                panel.nickname_input.value = "Unsaved draft"
                panel.nickname_button.click()
            await panel._task
            await panel._poll_task
            assert panel.connected
            assert len(client.sent) == (1 if action == "submit" else 0)
            assert len(client.names_saved) == (1 if action == "nickname" else 0)
            if action == "nickname":
                assert panel.nickname_input.value == "Unsaved draft"
        finally:
            panel.close()

    asyncio.run(run())
