"""Notebook UI; authenticated requests originate in the Python kernel."""
import asyncio
from html import escape
import json
import math

from ETC.judge.catalog import CHALLENGES
from ETC.judge.expressions import SubmissionError
from .judge_client import JudgeClient, JudgeConnectionError
from .submission import collect_submission


def history_html(account, challenge):
    rows = []
    labels = {"queued": "Queued", "running": "Evaluating", "completed": "Completed",
              "system_error": "Server error", "time_limit": "Time limit"}
    for item in account["submissions"]:
        if not isinstance(item, dict) or str(item.get("challenge")) != str(challenge):
            continue
        value = item.get("score")
        score = f"{value:.2f} / 100" if type(value) in {int, float} and math.isfinite(value) else "—"
        status = labels.get(item.get("status"), "Unknown")
        feedback = ""
        if item.get("result") is not None or item.get("error"):
            details = json.dumps({"result": item.get("result"), "error": item.get("error")}, indent=2, ensure_ascii=False)
            feedback = f'<details><summary>Evaluation details</summary><pre>{escape(details)}</pre></details>'
        rows.append(f'<tr><td>{escape(str(item.get("id", ""))[:12])}{feedback}</td><td>{status}</td><td>{score}</td></tr>')
        if len(rows) == 10:
            break
    if not rows:
        return "<p>No submissions for this Challenge yet.</p>"
    return ('<table><thead><tr><th>Submission</th><th>Status</th><th>Pilot score</th></tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table><p>Latest 10 attempts. Your best complete attempt counts; omitted Levels score zero.</p>')


class SubmissionPanel:
    POLL_INTERVAL_SECONDS = 5
    MAX_POLL_ATTEMPTS = 180

    def __init__(self, challenge, lesson_dir, *, levels=(1,), reference=False, judge_url="", client=None):
        import ipywidgets as widgets
        self.challenge, self.lesson_dir, self.reference = str(challenge), lesson_dir, reference
        self.client = client
        self._environment_client = client is None
        self._judge_url = judge_url
        self.closed = self.busy = self.pending = False
        self.connected = False
        self.nickname = None
        self._account_loaded = False
        self._task = self._poll_task = None
        self.identity = widgets.HTML("<p>Checking judge connection…</p>")
        self.nickname_input = widgets.Text(
            description="Nickname", placeholder="Name shown on the scoreboard", disabled=True,
            style={"description_width": "initial"}, layout=widgets.Layout(width="320px", max_width="100%"),
        )
        self.nickname_button = widgets.Button(description="Register nickname", icon="user", disabled=True,
                                               layout=widgets.Layout(width="180px", margin="0 0 0 8px"))
        self.status = widgets.HTML()
        self.history = widgets.HTML()
        self.levels = widgets.SelectMultiple(
            options=[(f"Level {i}", i) for i in range(1, len(CHALLENGES[self.challenge]["files"]) + 1)],
            value=tuple(levels), description="Levels", rows=len(CHALLENGES[self.challenge]["files"]),
        )
        self.submit_button = widgets.Button(description="Submit code", button_style="success", icon="paper-plane", disabled=True)
        self.refresh_button = widgets.Button(description="Refresh results", icon="refresh")
        self.widget = widgets.VBox([
            widgets.HTML(f"<h3>Challenge {escape(self.challenge)} · Submit and results</h3><p>Save your .py files first. Only clicking Submit code sends your work.</p>"),
            self.identity,
            widgets.Box([self.nickname_input, self.nickname_button], layout=widgets.Layout(display="flex", flex_flow="row wrap")),
            widgets.HTML("<p>Register your nickname here before your first submission. The same name is used for Challenges 1-4 and the public scoreboard. Do not use an email address or other private information.</p>"),
            self.levels, widgets.HBox([self.submit_button, self.refresh_button]), self.status, self.history,
            widgets.HTML("<small>Scores are provisional. No file download, website upload or separate judge login is needed here.</small>"),
        ])
        self.submit_button.on_click(lambda _: self._dispatch(submit=True))
        self.refresh_button.on_click(lambda _: self._dispatch())
        self.nickname_button.on_click(lambda _: self._dispatch(save_nickname=True))
        self.nickname_input.observe(lambda _: self._update_controls(), names="value")
        self._dispatch()

    def _update_controls(self):
        unavailable = self.closed or self.busy
        dirty = self.nickname_input.value.strip() != self.nickname
        self.nickname_input.disabled = unavailable
        self.nickname_button.description = "Save nickname" if self.nickname else "Register nickname"
        self.nickname_button.disabled = (unavailable or self.client is None
                                         or not self.nickname_input.value.strip() or not dirty)
        self.refresh_button.disabled = unavailable
        self.submit_button.disabled = (unavailable or not self.connected or not self.nickname or dirty
                                       or self.reference is not False or self.pending)

    def _apply_account(self, account, *, saved_nickname=False):
        draft = self.nickname_input.value
        if (saved_nickname or (not self._account_loaded and not draft.strip())
                or (self._account_loaded and draft.strip() == (self.nickname or ""))):
            self.nickname_input.value = account["nickname"] or ""
        self.nickname = account["nickname"]
        self._account_loaded = True
        self.identity.value = (f'<p>Connected as <strong>{escape(self.nickname)}</strong></p>' if self.nickname
                               else "<p>Connected. Register your nickname below before submitting.</p>")
        self.history.value = history_html(account, self.challenge)
        self.pending = any(isinstance(item, dict) and item.get("status") in {"queued", "running"}
                           for item in account["submissions"])

    def _dispatch(self, *, submit=False, save_nickname=False):
        if self.closed or self.busy:
            return
        if self._task is not None and not self._task.done():
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(self._perform(submit=submit, save_nickname=save_nickname))
        else:
            self._task = loop.create_task(self._perform(submit=submit, save_nickname=save_nickname))

    async def _perform(self, *, submit=False, save_nickname=False):
        if self.closed or self.busy:
            return
        self.busy = True
        self._update_controls()
        healthy = False
        try:
            # Launchable provisioning may finish after this cell is opened. Only
            # read-only refreshes reload private configuration; never retry a POST.
            if not submit and not save_nickname and self._environment_client and not self.connected:
                self.client = await asyncio.to_thread(JudgeClient.from_environment, self._judge_url)
                if self.closed:
                    return
            if self.client is None:
                raise JudgeConnectionError("Judge connection not configured yet. Waiting for automatic workspace connection.")
            if submit:
                if not self.nickname or self.nickname_input.value.strip() != self.nickname:
                    raise JudgeConnectionError("Register or save your nickname before submitting code.")
                payload = collect_submission(self.challenge, self.lesson_dir, levels=self.levels.value, reference=self.reference)
                self.status.value = "<p>Sending saved exercise code…</p>"
                receipt = await asyncio.to_thread(self.client.submit, payload)
                if self.closed:
                    return
                self.status.value = f"<p>Accepted: {escape(receipt[:12])}. Waiting for evaluation.</p>"
            if save_nickname:
                self.status.value = "<p>Saving your nickname…</p>"
                account = await asyncio.to_thread(self.client.register_nickname, self.nickname_input.value)
            else:
                account = await asyncio.to_thread(self.client.me)
            if self.closed:
                return
            healthy = True
            self._apply_account(account, saved_nickname=save_nickname)
            if save_nickname:
                self.status.value = "<p>Nickname saved for all four Challenges and the scoreboard. No code was submitted.</p>"
            elif not self.nickname:
                self.status.value = "<p>Enter a nickname and click Register nickname to enable submission.</p>"
            elif not submit:
                self.status.value = "<p>Results refreshed. Updates every 5 seconds while evaluation is pending.</p>"
            if self.reference is not False:
                self.status.value = "<p>Instructor demonstration: submission disabled. Set USE_REFERENCE = False and rerun this cell.</p>"
        except (JudgeConnectionError, SubmissionError) as exc:
            self.status.value = f"<p>{escape(str(exc))}</p><p>Previously displayed results may be outdated.</p>"
        except Exception:
            # Never render exceptions containing request headers or exercise source.
            self.status.value = "<p>Could not update the judge connection. Refresh results or ask the instructor.</p>"
        finally:
            self.busy = False
            self.connected = healthy
            if not self.closed:
                if not healthy:
                    self.identity.value = ("<p>Judge connection not configured yet. Waiting for automatic workspace connection.</p>"
                                           if self.client is None else "<p>Judge connection unavailable. Checking again automatically.</p>")
                    self.status.value += "<p>Local practice is still available. Refresh results retries the connection; it never sends code.</p>"
                self._update_controls()
                self._start_polling()

    def _start_polling(self):
        if (self.pending or not self.connected) and (self._poll_task is None or self._poll_task.done()):
            self._poll_task = asyncio.create_task(self._poll())

    async def _poll(self):
        # Bounded read-only polling. Never automatically retry a POST.
        for _ in range(self.MAX_POLL_ATTEMPTS):
            await asyncio.sleep(self.POLL_INTERVAL_SECONDS)
            if self.closed or (self.connected and not self.pending):
                return
            if not self.busy:
                await self._perform()
            if self.connected and not self.pending:
                return
        if not self.closed and (self.pending or not self.connected):
            self.identity.value = (self.identity.value if self.connected else "<p>Judge connection unavailable. Automatic checks are paused.</p>")
            self.status.value = "<p>Automatic refresh paused. Click Refresh results to continue. Local practice is still available.</p>"

    def close(self):
        self.closed = True
        for task in (self._task, self._poll_task):
            if task is not None and not task.done():
                task.cancel()
        self._update_controls()
        self.widget.close()
