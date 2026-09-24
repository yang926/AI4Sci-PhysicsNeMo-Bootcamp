"""Loopback-only pilot UI/API; deployment requires an authenticated TLS proxy."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from .store import BusyError, NicknameConflict, NicknameRequired

ASSETS = Path(__file__).parent / "static"


def create_server(store, port=8090):
    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(10)

        def log_message(self, *_):
            pass  # Never log submitted bodies or credentials.

        def send(self, status, value, content_type="application/json"):
            data = json.dumps(value, allow_nan=False).encode() if content_type == "application/json" else value
            self.send_response(status)
            self.send_header("Content-Type", content_type + "; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'")
            self.end_headers()
            self.wfile.write(data)

        def check_host(self):
            allowed = {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}
            host = self.headers.get("Host", "")
            origin = self.headers.get("Origin")
            if host not in allowed or (origin is not None and origin != "http://" + host):
                self.send(403, {"error": "This pilot accepts same-origin loopback requests only."})
                return False
            return True

        def participant(self):
            header = self.headers.get("Authorization", "")
            person = store.authenticate(header[7:]) if header.startswith("Bearer ") else None
            if person is None:
                self.send(401, {"error": "Workspace authentication failed. Ask the instructor to check its personal judge credential."})
            return person

        def do_GET(self):
            if not self.check_host():
                return
            path = urlsplit(self.path).path
            if path in {"/", "/display", "/display/", "/app.js", "/style.css"}:
                filename, mime = {"/": ("display.html", "text/html"), "/display": ("display.html", "text/html"),
                                  "/display/": ("display.html", "text/html"), "/app.js": ("app.js", "text/javascript"),
                                  "/style.css": ("style.css", "text/css")}[path]
                self.send(200, (ASSETS / filename).read_bytes(), mime)
            elif path == "/api/board":
                query = parse_qs(urlsplit(self.path).query, keep_blank_values=True)
                if not query:
                    self.send(200, store.board())
                elif set(query) != {"challenge"} or len(query["challenge"]) != 1:
                    self.send(400, {"error": "Choose exactly one Challenge."})
                else:
                    try:
                        self.send(200, store.board(challenge=query["challenge"][0]))
                    except ValueError:
                        self.send(400, {"error": "Choose Challenge 1, 2, 3 or 4."})
            elif path == "/api/me":
                person = self.participant()
                if person:
                    self.send(200, {"nickname": person["nickname"], "submissions": store.history(person["id"])})
            else:
                self.send(404, {"error": "Not found"})

        def do_POST(self):
            if not self.check_host():
                return
            if self.path not in {"/api/submissions", "/api/me/nickname"}:
                self.send(404, {"error": "Not found"})
                return
            person = self.participant()
            if person is None:
                return
            if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                self.send(415, {"error": "Send application/json"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 1024 * 1024:
                    raise ValueError("Submission body must be between 1 byte and 1 MiB.")
                payload = json.loads(self.rfile.read(length))
                if self.path == "/api/me/nickname":
                    if not isinstance(payload, dict) or set(payload) != {"nickname"}:
                        raise ValueError("Supply a nickname only; the account comes from the workspace credential.")
                    nickname = store.set_nickname(person["id"], payload["nickname"])
                    self.send(200, {"nickname": nickname, "submissions": store.history(person["id"])})
                    return
                if not isinstance(payload, dict) or set(payload) != {"challenge", "sources"}:
                    raise ValueError("Supply challenge and sources only; identity comes from your access code.")
                identifier = store.submit(person["id"], payload["challenge"], payload["sources"])
                self.send(202, {"submission_id": identifier})
            except NicknameConflict:
                self.send(409, {"error_code": "nickname_taken", "error": "That nickname is already in use."})
            except NicknameRequired:
                self.send(409, {"error_code": "nickname_required", "error": "Register your nickname before submitting."})
            except BusyError as exc:
                self.send(429, {"error": str(exc)})
            except (ValueError, TypeError, RecursionError) as exc:
                self.send(400, {"error": str(exc)})

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)
