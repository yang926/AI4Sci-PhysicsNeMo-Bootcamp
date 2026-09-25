"""Direct notebook submission, credential safety and no-submit-on-Run-All checks."""
import asyncio
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import threading

import pytest

from ETC.judge.store import Store
from ETC.judge.web import create_server
from ETC.runtime.judge_client import JudgeClient, JudgeConnectionError, configure_workspace, validate_url
from ETC.runtime.submission import collect_submission
from ETC.runtime.submission_widgets import SubmissionPanel, history_html
from ETC.tests.test_judge import answer, completed
from ETC.tests.test_judge_operators import operator_answer


TOKEN = "test_only_not_a_live_credential_123456789"


@pytest.fixture(autouse=True)
def private_environment(monkeypatch, tmp_path):
    for name in ("AI4SCI_JUDGE_TOKEN", "AI4SCI_JUDGE_URL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("AI4SCI_JUDGE_CONFIG", str(tmp_path / "missing.json"))


@contextmanager
def serving(server):
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.mark.parametrize("url", ["https://judge.example", "https://judge.example/api-prefix/", "http://localhost:8090", "http://127.0.0.1:8090", "http://[::1]:8090"])
def test_allowed_urls(url):
    assert validate_url(url) == url.rstrip("/")


@pytest.mark.parametrize("url", ["", "http://judge.example", "http://localhost.evil", "https://judge.example?token=secret", "https://user:password@judge.example", "https://judge.example#fragment", "https://judge.example:bad", "https://judge.example:0", "https://judge.example\n", "https://judge.example\\evil", "file:///tmp/test"])
def test_reject_unsafe_urls_without_echoing_secrets(url):
    with pytest.raises(JudgeConnectionError) as error:
        validate_url(url)
    assert "secret" not in str(error.value) and "password" not in str(error.value)


def test_private_configuration_and_redacted_repr(monkeypatch, tmp_path):
    config = tmp_path / "judge.json"
    config.write_text(json.dumps({"url": "https://judge.example", "token": TOKEN}))
    config.chmod(0o600)
    monkeypatch.setenv("AI4SCI_JUDGE_CONFIG", str(config))
    client = JudgeClient.from_environment()
    assert client.token == TOKEN and TOKEN not in repr(client)
    with pytest.raises(JudgeConnectionError, match="differ"):
        JudgeClient.from_environment("https://another.example")
    config.chmod(0o644)
    with pytest.raises(JudgeConnectionError, match="owner-only"):
        JudgeClient.from_environment()
    config.chmod(0o600)
    link = tmp_path / "symlink.json"
    link.symlink_to(config)
    monkeypatch.setenv("AI4SCI_JUDGE_CONFIG", str(link))
    with pytest.raises(JudgeConnectionError, match="owner-only"):
        JudgeClient.from_environment()


def test_launch_hook_saves_once_outside_course_without_printing_key(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "user")
    monkeypatch.setenv("AI4SCI_JUDGE_URL", "https://judge.example")
    monkeypatch.setenv("AI4SCI_JUDGE_TOKEN", TOKEN)
    configure_workspace()
    config = tmp_path / "user/.config/ai4sci/judge.json"
    assert config.stat().st_mode & 0o777 == 0o600
    assert config.parent.stat().st_mode & 0o777 == 0o700
    assert TOKEN not in capsys.readouterr().out
    monkeypatch.setenv("AI4SCI_JUDGE_CONFIG", str(config))
    monkeypatch.delenv("AI4SCI_JUDGE_TOKEN")
    monkeypatch.delenv("AI4SCI_JUDGE_URL")
    assert JudgeClient.from_environment().token == TOKEN
    monkeypatch.setenv("AI4SCI_JUDGE_URL", "https://judge.example")
    monkeypatch.setenv("AI4SCI_JUDGE_TOKEN", TOKEN)
    with pytest.raises(JudgeConnectionError, match="not overwritten"):
        configure_workspace()


@pytest.mark.parametrize("challenge,name", [("1", "wave_l1.py"), ("2", "chip_2d_l1.py"), ("3", "climate_l1.py"), ("4", "fno_physicsnemo_l1.py")])
def test_direct_transport_records_identity_and_deduplicates(challenge, name, tmp_path):
    store = Store.initialize(tmp_path / "state", steps=2)
    token = store.add_participant("Notebook learner")
    second = store.add_participant("Another learner")
    source = operator_answer(1) if challenge == "4" else answer(challenge, name)
    (tmp_path / name).write_text(source + '\nSECRET="not for submission"\n')
    payload = collect_submission(challenge, tmp_path)
    assert "SECRET" not in json.dumps(payload)
    assert not (tmp_path / "outputs").exists()
    with serving(create_server(store, 0)) as url:
        client = JudgeClient(url, token)
        assert client.me()["submissions"] == []
        identifier = client.submit(payload)
        assert client.submit(payload) == identifier
        mine = client.me()
        assert mine["nickname"] == "Notebook learner"
        assert mine["submissions"][0]["status"] == "queued"
        assert JudgeClient(url, second).me()["submissions"] == []
        job = store.claim()
        store.finish(job, completed(job, 82.5))
        assert client.me()["submissions"][0]["score"] == 82.5
        assert store.board()["participants"][0]["scores"][challenge] == 82.5
        with pytest.raises(JudgeConnectionError, match="authentication"):
            JudgeClient(url, TOKEN).me()


def test_no_redirect_or_server_error_secret_leak():
    seen = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_GET(self):
            seen.append(self.path)
            self.send_response(302)
            self.send_header("Location", "/login?secret=" + TOKEN)
            self.end_headers()
            self.wfile.write(TOKEN.encode())

    with serving(ThreadingHTTPServer(("127.0.0.1", 0), Handler)) as url:
        with pytest.raises(JudgeConnectionError, match="redirected") as error:
            JudgeClient(url, TOKEN).me()
    assert TOKEN not in str(error.value)
    assert seen == ["/api/me"]


@pytest.mark.parametrize("version", [None, 2, "3"])
def test_old_judge_cannot_silently_grade_only_the_pde(version):
    calls = []
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass
        def do_GET(self):
            calls.append(("GET", self.path))
            payload = {"nickname": "Legacy", "submissions": [], "submission_contract": version}
            body = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        def do_POST(self):
            calls.append(("POST", self.path))
            self.send_response(500)
            self.end_headers()
    with serving(ThreadingHTTPServer(("127.0.0.1", 0), Handler)) as url:
        with pytest.raises(JudgeConnectionError, match="update the judge and course together"):
            JudgeClient(url, TOKEN).submit({"challenge": "1", "sources": {"wave_l1.py": "unused"}})
    assert calls == [("GET", "/api/me")]


class FakeClient:
    def __init__(self):
        self.sent = []
        self.rows = []
        self.nickname = "<learner>"
        self.names_saved = []

    def me(self):
        return {"nickname": self.nickname, "submissions": self.rows}

    def register_nickname(self, nickname):
        if nickname == "taken":
            raise JudgeConnectionError("That nickname is already in use. Choose another nickname.")
        self.nickname = nickname.strip()
        self.names_saved.append(self.nickname)
        return self.me()

    def submit(self, payload):
        self.sent.append(payload)
        self.rows = [{"id": "receipt", "challenge": payload["challenge"], "status": "queued", "score": None}]
        return "receipt"


def test_panel_only_submits_on_click_and_refreshes_score(tmp_path):
    (tmp_path / "wave_l1.py").write_text(answer("1", "wave_l1.py"))
    client = FakeClient()
    panel = SubmissionPanel("1", tmp_path, client=client)
    try:
        assert client.sent == []
        assert "&lt;learner&gt;" in panel.identity.value
        assert not panel.submit_button.disabled
        panel.submit_button.click()
        assert len(client.sent) == 1 and "Queued" in panel.history.value
        assert panel.submit_button.disabled
        client.rows[0].update(status="completed", score=75)
        panel.refresh_button.click()
        assert "75.00 / 100" in panel.history.value
        assert not panel.submit_button.disabled
        assert len(client.sent) == 1
    finally:
        panel.close()


def test_async_polling_reads_results_without_resubmitting(monkeypatch, tmp_path):
    (tmp_path / "wave_l1.py").write_text(answer("1", "wave_l1.py"))
    client = FakeClient()
    original_sleep = asyncio.sleep

    async def fast_poll(seconds):
        await original_sleep(0)
        if seconds == 5 and client.rows:
            client.rows[0].update(status="completed", score=81)

    async def run():
        panel = SubmissionPanel("1", tmp_path, client=client)
        try:
            await panel._task
            assert not client.sent
            panel.submit_button.click()
            await panel._task
            await panel._poll_task
            assert "81.00 / 100" in panel.history.value
            assert len(client.sent) == 1
        finally:
            panel.close()

    monkeypatch.setattr("ETC.runtime.submission_widgets.asyncio.sleep", fast_poll)
    asyncio.run(run())


@pytest.mark.parametrize("reference", [True, "False"])
def test_reference_mode_and_missing_configuration_never_submit(reference, tmp_path):
    client = FakeClient()
    panel = SubmissionPanel("1", tmp_path, reference=reference, client=client)
    try:
        assert panel.submit_button.disabled
        panel.submit_button.click()  # Even a programmatically forced click must fail closed.
        assert client.sent == []
    finally:
        panel.close()
    missing = SubmissionPanel("2", tmp_path)
    try:
        assert missing.submit_button.disabled
        assert "not configured" in missing.identity.value
    finally:
        missing.close()


def test_history_is_escaped_and_challenge_specific():
    account = {"submissions": [
        {"id": "<img>", "challenge": "1", "status": "completed", "score": 0},
        {"id": "other", "challenge": "2", "status": "completed", "score": 99},
    ]}
    html = history_html(account, "1")
    assert "&lt;img&gt;" in html and "<img>" not in html
    assert "0.00 / 100" in html and "99.00" not in html


def test_nickname_is_registered_by_explicit_click_and_reused_in_all_challenges(tmp_path):
    client = FakeClient()
    client.nickname = None
    panel = SubmissionPanel("1", tmp_path, client=client)
    try:
        assert panel.nickname_input.value == "" and not panel.nickname_input.disabled
        assert panel.nickname_button.description == "Register nickname"
        assert panel.submit_button.disabled
        panel.submit_button.click()
        assert client.sent == []
        panel.nickname_input.value = "학습자"
        panel.refresh_button.click()
        assert panel.nickname_input.value == "학습자"
        assert client.names_saved == [] and client.sent == []
        panel.nickname_button.click()
        assert client.names_saved == ["학습자"] and client.sent == []
        assert not panel.submit_button.disabled
        assert "학습자" in panel.identity.value
    finally:
        panel.close()
    for challenge in "1234":
        restored = SubmissionPanel(challenge, tmp_path, client=client)
        try:
            assert restored.nickname_input.value == "학습자"
            assert not restored.submit_button.disabled
        finally:
            restored.close()
    assert client.names_saved == ["학습자"]


def test_nickname_conflict_draft_and_html_safety(tmp_path):
    client = FakeClient()
    panel = SubmissionPanel("1", tmp_path, client=client)
    try:
        panel.nickname_input.value = "taken"
        assert panel.submit_button.disabled
        panel.submit_button.click()
        assert client.sent == []
        panel.nickname_button.click()
        assert "already in use" in panel.status.value
        assert client.nickname == "<learner>"
        assert panel.nickname_input.value == "taken"
        assert not panel.nickname_button.disabled
        panel.nickname_input.value = "<img src=x>"
        panel.nickname_button.click()
        assert "&lt;img src=x&gt;" in panel.identity.value
        assert "<img" not in panel.identity.value
        assert not panel.submit_button.disabled
        panel.nickname_input.value = "Unsaved draft"
        panel.refresh_button.click()
        assert panel.nickname_input.value == "Unsaved draft"
        assert panel.submit_button.disabled
    finally:
        panel.close()


def test_actual_nickname_registration_transport_and_rename_preserves_score(tmp_path):
    store = Store.initialize(tmp_path / "state", steps=2)
    token = store.add_participant()
    store.add_participant("Taken")
    participant = store.authenticate(token)["id"]
    payload = {"challenge": "1", "sources": {"wave_l1.py": answer("1", "wave_l1.py")}}
    with serving(create_server(store, 0)) as url:
        client = JudgeClient(url, token)
        assert client.me()["nickname"] is None
        with pytest.raises(JudgeConnectionError, match="Register your nickname"):
            client.submit(payload)
        assert store.history(participant) == []
        with pytest.raises(JudgeConnectionError, match="already in use"):
            client.register_nickname("taken")
        account = client.register_nickname("학생 닉네임")
        assert account == {"nickname": "학생 닉네임", "submissions": []}
        receipt = client.submit(payload)
        job = store.claim()
        store.finish(job, completed(job, 82.5))
        client.register_nickname("수정한 이름")
        restored = JudgeClient(url, token).me()
        assert restored["nickname"] == "수정한 이름"
        assert restored["submissions"][0]["id"] == receipt
        assert restored["submissions"][0]["score"] == 82.5
        assert store.board("1")["participants"][0]["nickname"] == "수정한 이름"
        for value in ("", " " * 5, "a" * 41, "x\ny"):
            with pytest.raises(JudgeConnectionError, match="visible characters"):
                client.register_nickname(value)


def test_notebook_cells_open_controls_without_direct_post():
    root = Path(__file__).resolve().parents[2]
    notebooks = sorted((root / "02_challenges").glob("*/*.ipynb"))
    assert len(notebooks) == 4
    for notebook in notebooks:
        document = json.loads(notebook.read_text())
        sources = ["".join(cell["source"]) for cell in document["cells"]]
        combined = "\n".join(sources)
        assert "show_submission_controls(" in combined
        assert "**Nickname**" in combined and "**Register nickname**" in combined
        for removed in ("EXPORT_SUBMISSION", "show_export(", "Download the JSON", "upload it on"):
            assert removed not in combined
        final = next(cell for cell in document["cells"] if cell.get("id", "").startswith("judge-submit-"))
        assert "client.submit(" not in "".join(final["source"])


@pytest.mark.parametrize("challenge", ["1", "2", "3", "4"])
def test_actual_notebook_kernel_loads_connection_without_posting(challenge, tmp_path):
    import nbformat
    from nbclient import NotebookClient
    from ETC.judge.catalog import CHALLENGES

    root = Path(__file__).resolve().parents[2]
    notebook_path = next((root / "02_challenges" / CHALLENGES[challenge]["directory"]).glob("*.ipynb"))
    original = nbformat.read(notebook_path, as_version=4)
    setup = next(cell for cell in original.cells if cell.cell_type == "code")
    controls = next(cell for cell in original.cells if cell.get("id") == "judge-submit-" + challenge)
    check = '''import asyncio
from ETC.runtime.submission import _panels
panel = next(iter(_panels.values()))
if panel._task is not None:
    await panel._task
assert panel.nickname_input.value == ""
assert panel.submit_button.disabled
assert "No submissions" in panel.history.value
panel.nickname_input.value = "Notebook kernel fixture"
assert panel.submit_button.disabled
panel.nickname_button.click()
await panel._task
assert "Notebook kernel fixture" in panel.identity.value, panel.identity.value
assert not panel.submit_button.disabled, panel.status.value
assert "No submissions" in panel.history.value
print("Notebook connection checked; no submission was sent.")
'''
    rehearsal = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(setup.source),
        nbformat.v4.new_code_cell(controls.source), nbformat.v4.new_code_cell(check)])
    store = Store.initialize(tmp_path / "state", steps=2)
    token = store.add_participant()
    with serving(create_server(store, 0)) as url:
        config = tmp_path / "private.json"
        config.write_text(json.dumps({"url": url, "token": token}))
        config.chmod(0o600)
        env = {**os.environ, "AI4SCI_JUDGE_CONFIG": str(config), "AI4SCI_JUDGE_URL": "",
               "AI4SCI_JUDGE_TOKEN": "", "AI4SCI_DEVICE": "cpu"}
        executed = NotebookClient(rehearsal, timeout=45, kernel_name="python3").execute(cwd=str(root), env=env)
        assert token not in nbformat.writes(executed)
        assert store.history(store.authenticate(token)["id"]) == []
        assert not any(output.output_type == "error" for cell in executed.cells for output in cell.outputs)
