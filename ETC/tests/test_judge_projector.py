"""Optional real-browser checks. Set AI4SCI_TEST_CHROMIUM to a Chrome executable.

Only synthetic names/scores and disposable loopback state are used. No GPU runs.
"""
import base64
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import threading
import time
from urllib.request import urlopen

import pytest

from ETC.judge.store import Store, challenge_board
from ETC.judge.web import create_server
from ETC.tests.test_judge import answer


class Browser:
    def __init__(self, socket):
        self.socket = socket
        self.serial = 0
        self.exceptions = []
        self.requests = []

    def call(self, method, params=None):
        self.serial += 1
        self.socket.send(json.dumps({"id": self.serial, "method": method, "params": params or {}}))
        while True:
            message = json.loads(self.socket.recv())
            if message.get("method") == "Runtime.exceptionThrown":
                self.exceptions.append(message["params"])
            if message.get("method") == "Network.requestWillBeSent":
                self.requests.append(message["params"]["request"]["url"])
            if message.get("id") == self.serial:
                assert "error" not in message, message.get("error")
                return message.get("result", {})

    def js(self, expression):
        result = self.call("Runtime.evaluate", {"expression": expression, "returnByValue": True, "awaitPromise": True})
        assert "exceptionDetails" not in result, result.get("exceptionDetails")
        return result.get("result", {}).get("value")

    def until(self, expression, timeout=15):
        deadline = time.monotonic() + timeout
        while not self.js(expression):
            assert time.monotonic() < deadline, expression
            time.sleep(.1)

    def size(self, width, height):
        self.call("Emulation.setDeviceMetricsOverride", {"width": width, "height": height, "deviceScaleFactor": 1, "mobile": False})
        self.js("drawBoard()")

    def open(self, url):
        self.call("Page.navigate", {"url": url})
        self.until('document.body?.dataset.connection === "live"')

    def screenshot(self, name):
        directory = os.environ.get("AI4SCI_UI_SCREENSHOTS")
        if directory:
            self.js("document.fonts.ready.then(() => true)")
            path = Path(directory)
            path.mkdir(parents=True, exist_ok=True)
            (path / name).write_bytes(base64.b64decode(self.call("Page.captureScreenshot", {"format": "png"})["data"]))


def synthetic_board(template):
    """110 fictitious participants, including ties, zero, and unranked entries."""
    payload = deepcopy(template)
    payload["queue"] = {"completed": 278, "queued": 17, "running": 8}
    payload["queue_by_challenge"] = {key: {"completed": 60 + int(key), "queued": int(key), "running": 2} for key in "1234"}
    people = []
    for index in range(110):
        scores = {key: round(99 - (index // 2) * 1.5 - int(key), 2) if index < 89 else None for key in "1234"}
        if index == 89:
            scores["4"] = 0.0
        people.append({"nickname": f"Practice learner {index + 1:03}", "scores": scores, "total": sum(value for value in scores.values() if value is not None), "rank": None, "challenge_ranks": dict.fromkeys("1234")})
    people[0]["nickname"] = "Long participant alias for overflow test"
    people[1]["nickname"] = '<img src=x onerror=alert(1)>'
    for key in ["overall", *"1234"]:
        eligible = [row for row in people if any(v is not None for v in row["scores"].values())] if key == "overall" else [row for row in people if row["scores"][key] is not None]
        value = lambda row: row["total"] if key == "overall" else row["scores"][key]
        eligible.sort(key=lambda row: -value(row))
        previous, rank = None, None
        for index, row in enumerate(eligible, 1):
            if value(row) != previous:
                rank = index
            if key == "overall":
                row["rank"] = rank
            else:
                row["challenge_ranks"][key] = rank
            previous = value(row)
    payload["participants"] = people
    return payload


@pytest.fixture
def screen(tmp_path):
    chromium = os.environ.get("AI4SCI_TEST_CHROMIUM")
    if not chromium or not Path(chromium).is_file():
        pytest.skip("Set AI4SCI_TEST_CHROMIUM for the optional Chromium UI suite")
    websocket = pytest.importorskip("websocket")
    store = Store.initialize(tmp_path / "state", steps=2)

    class ScreenStore:
        payload = None

        def board(self, challenge=None):
            if self.payload is None:
                return store.board(challenge)
            return challenge_board(self.payload, challenge) if challenge is not None else self.payload

        def __getattr__(self, name):
            return getattr(store, name)

    data = ScreenStore()
    server = create_server(data, 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    profile = tmp_path / "chrome"
    process = subprocess.Popen([chromium, "--headless", "--no-sandbox", "--disable-gpu", "--no-proxy-server", "--remote-debugging-port=0", "--user-data-dir=" + str(profile), "about:blank"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    socket = None
    try:
        deadline = time.monotonic() + 20
        active = profile / "DevToolsActivePort"
        while not active.exists():
            assert time.monotonic() < deadline, "Chromium startup timeout"
            time.sleep(.1)
        port = int(active.read_text().splitlines()[0])
        while True:
            with urlopen(f"http://127.0.0.1:{port}/json") as response:
                pages = [page for page in json.load(response) if page["type"] == "page"]
            if pages:
                break
            assert time.monotonic() < deadline, "Chromium page timeout"
            time.sleep(.1)
        socket = websocket.create_connection(pages[0]["webSocketDebuggerUrl"], suppress_origin=True, timeout=15)
        browser = Browser(socket)
        for domain in ("Runtime", "Page", "Network"):
            browser.call(domain + ".enable")
        browser.call("Emulation.setDeviceMetricsOverride", {"width": 1920, "height": 1080, "deviceScaleFactor": 1, "mobile": False})
        yield browser, data, f"http://127.0.0.1:{server.server_port}"
        browser.js("true")
        assert not browser.exceptions
    finally:
        if socket:
            socket.close()
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_projection_readability_privacy_and_small_screens(screen):
    browser, data, url = screen
    data.payload = synthetic_board(data.board())
    browser.open(url + "/display?rotate=0")
    assert browser.js('document.getElementById("stat-people").textContent') == "110"
    assert browser.js('document.getElementById("stat-running").textContent') == "2"
    assert browser.js('document.querySelectorAll("thead th").length') == 3
    assert browser.js('document.getElementById("ranking").value') == "1"
    assert browser.js('[...document.querySelectorAll("#ranking option")].map(e=>e.value)') == list("1234")
    assert browser.js('document.getElementById("challenge-title").textContent.startsWith("Challenge 1")')
    assert browser.js('document.getElementById("stat-completed").textContent') == "61"
    assert browser.js('document.querySelectorAll("form, input[type=password], #history, #submit-tab").length') == 0
    assert browser.js('document.querySelectorAll("#standings img").length') == 0
    assert browser.js('document.querySelector("#standings").textContent.includes("<img src=x onerror=alert(1)>")')
    # Use ordinary, explicitly fictitious aliases in the presentation captures.
    data.payload["participants"][0]["nickname"] = "Practice learner 001"
    data.payload["participants"][1]["nickname"] = "Practice learner 002"
    browser.js('refreshBoard()')
    for width, height in ((1920, 1080), (1366, 768)):
        browser.size(width, height)
        assert browser.js('document.documentElement.scrollWidth <= innerWidth')
        assert browser.js('document.documentElement.scrollHeight <= innerHeight'), (width, height)
        assert browser.js('parseFloat(getComputedStyle(document.querySelector("td")).fontSize) >= 21')
        assert browser.js('document.querySelectorAll("#standings tr").length <= 10')
        browser.screenshot(f"projector-{width}.png")
    browser.size(390, 844)
    assert browser.js('document.documentElement.scrollWidth <= innerWidth')
    browser.screenshot("projector-mobile.png")
    assert not any("/api/me" in request or "/api/submissions" in request for request in browser.requests)


def test_pagination_rotation_challenge_ties_zero_and_empty_states(screen):
    browser, data, url = screen
    data.payload = synthetic_board(data.board())
    browser.open(url + "/display?ranking=4&rotate=0")
    assert browser.js('document.getElementById("ranking").value') == "4"
    assert browser.js('document.querySelectorAll("#standings tr:first-child .score-active")[0].cellIndex') == 2
    assert browser.js('[...document.querySelectorAll("#standings .rank")].slice(0,2).map(e=>e.textContent)') == ["1", "1"]
    browser.js('document.getElementById("next-page").click()')
    before = browser.js('document.getElementById("page-label").textContent')
    browser.js('document.getElementById("rotate-pages").click(); nextRotation=Date.now()-1;')
    browser.until('document.getElementById("page-label").textContent !== ' + json.dumps(before))
    browser.js('document.getElementById("rotate-pages").click(); page=pageCount-1; drawBoard()')
    assert browser.js('document.getElementById("next-page").disabled')
    assert browser.js('[...document.querySelectorAll("#standings .rank")].every(e=>e.textContent==="-")')
    browser.js('document.getElementById("ranking").value="3"; document.getElementById("ranking").dispatchEvent(new Event("change"))')
    browser.until('document.body.dataset.connection === "live" && board.challenge === "3"')
    assert browser.js('document.getElementById("page-label").textContent.startsWith("Page 1 /")')
    assert browser.js('location.search.includes("ranking=3")')
    data.payload["participants"] = [data.payload["participants"][89]]
    browser.js('document.getElementById("ranking").value="4"; document.getElementById("ranking").dispatchEvent(new Event("change"))')
    browser.until('document.body.dataset.connection === "live" && board.challenge === "4"')
    assert browser.js('document.querySelector("#standings tr").cells[2].textContent') == "0.00"
    data.payload["participants"] = []
    browser.js('refreshBoard()')
    assert browser.js('document.querySelector("#standings .empty").colSpan') == 3
    assert browser.js('document.getElementById("page-label").textContent') == "Page 1 / 1"


def test_connection_loss_retains_scores_pauses_rotation_and_recovers(screen):
    browser, data, url = screen
    data.payload = synthetic_board(data.board())
    browser.open(url + "/display?rotate=1")
    previous = browser.js('document.getElementById("standings").textContent')
    browser.call("Network.emulateNetworkConditions", {"offline": True, "latency": 0, "downloadThroughput": 0, "uploadThroughput": 0})
    browser.js('refreshBoard()')
    browser.js('nextRotation=Date.now()-1')
    browser.until('document.body.dataset.connection === "stale"')
    time.sleep(1.2)
    assert browser.js('document.getElementById("stale-warning").hidden') is False
    # A warning takes space, so the page may display fewer rows. Existing values remain.
    assert browser.js('document.querySelector("#standings tr").cells[2].textContent') in previous
    assert browser.js('page') == 0
    browser.call("Network.emulateNetworkConditions", {"offline": False, "latency": 0, "downloadThroughput": -1, "uploadThroughput": -1})
    browser.js('refreshBoard()')
    assert browser.js('document.body.dataset.connection') == "live"
    assert browser.js('document.getElementById("stale-warning").hidden') is True


def test_root_is_only_a_scoreboard_without_upload_or_login(screen):
    browser, data, url = screen
    token = data.add_participant("UI test learner")
    browser.open(url)
    assert browser.js('document.body.dataset.view') == "display"
    assert browser.js('document.querySelectorAll("form, input, #submit-tab, #history").length') == 0
    assert token not in browser.js('document.documentElement.outerHTML')
    assert not any("/api/me" in request or "/api/submissions" in request for request in browser.requests)
    browser.size(390, 844)
    assert browser.js('document.documentElement.scrollWidth <= innerWidth')
    browser.screenshot("scoreboard-root-mobile.png")


def test_reduced_motion_and_fullscreen_fallback(screen):
    browser, data, url = screen
    data.payload = synthetic_board(data.board())
    browser.call("Emulation.setEmulatedMedia", {"features": [{"name": "prefers-reduced-motion", "value": "reduce"}]})
    browser.open(url + "/display")
    assert browser.js('document.getElementById("rotate-pages").getAttribute("aria-pressed")') == "false"
    browser.js('document.documentElement.requestFullscreen=undefined; document.getElementById("fullscreen").click()')
    assert browser.js('document.getElementById("display-notice").textContent') == "Use your browser menu to enter full screen."


def test_challenge_selection_hides_other_scores_and_resets_ranking(screen):
    browser, data, url = screen
    data.payload = synthetic_board(data.board())
    data.payload["participants"] = [
        {"nickname": "Earlier leader", "scores": {"1": 100, "2": 100, "3": 10, "4": None}, "total": 210, "rank": 1, "challenge_ranks": {"1": 1, "2": 1, "3": 2, "4": None}},
        {"nickname": "Current leader", "scores": {"1": 5, "2": 5, "3": 90, "4": None}, "total": 100, "rank": 2, "challenge_ranks": {"1": 2, "2": 2, "3": 1, "4": None}},
    ]
    browser.open(url + "/display?ranking=3&rotate=0")
    assert browser.js('[...document.querySelector("#standings tr").cells].map(e=>e.textContent)') == ["1", "Current leader", "90.00"]
    assert browser.js('Object.keys(board.participants[0]).sort()') == ["challenge_ranks", "nickname", "scores"]
    assert browser.js('Object.keys(board.participants[0].scores)') == ["3"]
    assert browser.js('document.getElementById("stat-queued").textContent') == "3"
    assert not any("/api/board" in request and "?challenge=3" not in request for request in browser.requests)
    # An old overall bookmark now opens Challenge 1, not a cumulative score table.
    browser.open(url + "/display?ranking=overall&rotate=0")
    assert browser.js('document.getElementById("ranking").value') == "1"
    assert browser.js('[...document.querySelector("#standings tr").cells].map(e=>e.textContent)') == ["1", "Earlier leader", "100.00"]


def test_switching_challenge_offline_never_displays_previous_challenge_scores(screen):
    browser, data, url = screen
    data.payload = synthetic_board(data.board())
    browser.open(url + "/display?ranking=1&rotate=0")
    browser.call("Network.emulateNetworkConditions", {"offline": True, "latency": 0, "downloadThroughput": 0, "uploadThroughput": 0})
    browser.js('document.getElementById("ranking").value="3"; document.getElementById("ranking").dispatchEvent(new Event("change"))')
    browser.until('document.body.dataset.connection === "stale"')
    assert browser.js('document.getElementById("challenge-title").textContent') == "Challenge 3"
    assert browser.js('document.querySelectorAll("#standings .rank").length') == 0
    assert browser.js('document.getElementById("standings").textContent') == "Waiting for Challenge 3 results..."
    assert browser.js('document.getElementById("stat-completed").textContent') == "-"
    browser.call("Network.emulateNetworkConditions", {"offline": False, "latency": 0, "downloadThroughput": -1, "uploadThroughput": -1})
    browser.js('refreshBoard()')
    assert browser.js('board.challenge') == "3"
    assert browser.js('document.getElementById("score-heading").textContent') == "C3 score"
    assert browser.js('document.getElementById("stat-completed").textContent') == "63"
