"""Notebook-to-judge transport. Credentials stay outside notebook outputs."""
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import re
import stat
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener


class JudgeConnectionError(ValueError):
    """User-facing error without response bodies or credentials."""


def validate_url(value):
    try:
        if not isinstance(value, str) or not value or any(c.isspace() or ord(c) < 32 for c in value):
            raise ValueError
        parsed = urlsplit(value)
        if (not parsed.hostname or parsed.username is not None or parsed.password is not None
                or "?" in value or "#" in value or "\\" in value
                or (parsed.port is not None and not 1 <= parsed.port <= 65535)):
            raise ValueError
        if parsed.scheme != "https" and not (
                parsed.scheme == "http" and parsed.hostname in {"localhost", "127.0.0.1", "::1"}):
            raise ValueError
    except (ValueError, TypeError):
        raise JudgeConnectionError("Use an HTTPS judge URL without credentials or query parameters; HTTP is only for loopback rehearsal.") from None
    return value.rstrip("/")


def validate_token(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]{20,200}", value):
        raise JudgeConnectionError("The workspace has no valid personal judge credential. Ask the instructor to configure it.")
    return value


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Never forward Authorization to a login page or another host.
        return None


@dataclass(frozen=True)
class JudgeClient:
    url: str
    token: str = field(repr=False)
    timeout: float = 10

    def __post_init__(self):
        object.__setattr__(self, "url", validate_url(self.url))
        validate_token(self.token)

    @classmethod
    def from_environment(cls, judge_url=""):
        """Read private launch configuration, never notebook literals or URL tokens."""
        config = {}
        path = Path(os.environ.get("AI4SCI_JUDGE_CONFIG", "~/.config/ai4sci/judge.json")).expanduser()
        try:
            if path.exists() or path.is_symlink():
                info = path.lstat()
                if (not stat.S_ISREG(info.st_mode) or info.st_mode & 0o077
                        or (hasattr(os, "getuid") and info.st_uid != os.getuid()) or info.st_size > 8192):
                    raise JudgeConnectionError("Judge configuration must be an owner-only regular file (mode 600).")
                config = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(config, dict) or set(config) != {"url", "token"}:
                    raise JudgeConnectionError("Invalid judge configuration. Ask the instructor to replace it.")
        except (OSError, UnicodeError, json.JSONDecodeError):
            raise JudgeConnectionError("Cannot read the private judge configuration. Ask the instructor to check it.") from None
        configured_url = config.get("url") or os.environ.get("AI4SCI_JUDGE_URL", "")
        if judge_url and configured_url and validate_url(judge_url) != validate_url(configured_url):
            raise JudgeConnectionError("Notebook and workspace judge URLs differ. Credentials were not sent.")
        url = configured_url or judge_url
        token = config.get("token") or os.environ.get("AI4SCI_JUDGE_TOKEN", "")
        if not url or not token:
            raise JudgeConnectionError("Judge connection not configured. Practice is available; ask the instructor to connect this workspace.")
        return cls(url, token)

    def _request(self, route, payload=None, *, expected_status=None):
        data = None if payload is None else json.dumps(payload, allow_nan=False).encode()
        expected_status = expected_status if expected_status is not None else (200 if data is None else 202)
        if data is not None and len(data) > 1024 * 1024:
            raise JudgeConnectionError("Submission exceeds 1 MiB. Submit only the required exercise functions.")
        request = Request(self.url + route, data=data, headers={
            "Authorization": "Bearer " + self.token, "Accept": "application/json",
            "Content-Type": "application/json",
        })
        try:
            # Do not inherit unrelated proxy settings or follow authentication redirects.
            with build_opener(ProxyHandler({}), NoRedirect()).open(request, timeout=self.timeout) as response:
                if response.status != expected_status:
                    raise JudgeConnectionError("Unexpected judge response. Refresh status before trying again.")
                if response.headers.get_content_type() != "application/json":
                    raise JudgeConnectionError("The judge returned a login page or non-API response. Ask the instructor to check the API connection.")
                raw = response.read(2 * 1024 * 1024 + 1)
                if len(raw) > 2 * 1024 * 1024:
                    raise JudgeConnectionError("Judge response is too large. Ask the instructor to check the service.")
                result = json.loads(raw)
                if not isinstance(result, dict):
                    raise JudgeConnectionError("Invalid judge response.")
                return result
        except HTTPError as exc:
            code = exc.code
            error_code = None
            if code == 409:
                try:
                    body = json.loads(exc.read(4096))
                    error_code = body.get("error_code") if isinstance(body, dict) else None
                except (OSError, ValueError, UnicodeError):
                    pass
            exc.close()
            # Only recognize fixed error codes. Never display a raw response body.
            if error_code == "nickname_taken":
                raise JudgeConnectionError("That nickname is already in use. Choose another nickname and register again.") from None
            if error_code == "nickname_required":
                raise JudgeConnectionError("Register your nickname in the notebook before submitting.") from None
            messages = {
                401: "Personal judge authentication failed. Ask the instructor to reconnect this workspace.",
                403: "This workspace cannot access the judge API. Ask the instructor to check access.",
                404: "Judge API not found. The workspace may point to the display service instead.",
                400: ("Nickname could not be saved. Use 1 to 40 visible characters; if it still fails, ask the instructor to check the judge version."
                      if route == "/api/me/nickname" else "The judge rejected this submission. Check the selected Challenge and saved exercise functions."),
                429: "A submission is pending or a submission limit was reached. Refresh results before retrying.",
            }
            message = ("The judge redirected the request. Credentials were not forwarded; check the API address."
                       if 300 <= code < 400 else messages.get(code, "The judge is unavailable. Refresh status before retrying."))
            raise JudgeConnectionError(message) from None
        except (URLError, OSError, TimeoutError):
            raise JudgeConnectionError("Cannot reach the judge. A submission may already have arrived; refresh status before retrying.") from None
        except (UnicodeError, json.JSONDecodeError):
            raise JudgeConnectionError("Invalid judge response. Refresh status before retrying.") from None

    def me(self):
        return self._account(self._request("/api/me"))

    @staticmethod
    def _account(result):
        if ("nickname" not in result or (result["nickname"] is not None and
                (not isinstance(result["nickname"], str) or not result["nickname"].strip()))
                or not isinstance(result.get("submissions"), list)):
            raise JudgeConnectionError("Invalid account response from the judge.")
        return result

    def register_nickname(self, nickname):
        if not isinstance(nickname, str) or not 1 <= len(nickname.strip()) <= 40 or not nickname.isprintable():
            raise JudgeConnectionError("Use a nickname with 1 to 40 visible characters.")
        account = self._account(self._request("/api/me/nickname", {"nickname": nickname}, expected_status=200))
        if account["nickname"] is None:
            raise JudgeConnectionError("Nickname was not saved. Refresh results before trying again.")
        return account

    def submit(self, payload):
        if not isinstance(payload, dict) or set(payload) != {"challenge", "sources"}:
            raise JudgeConnectionError("Submit exercise code only; identity comes from the private credential.")
        # Old judges accepted PDE-only answers and ignored the added setup tasks.
        # Check capability before sending code, not after recording a false score.
        from ETC.judge.contracts import CONTRACT_VERSION
        if self.me().get("submission_contract") != CONTRACT_VERSION:
            raise JudgeConnectionError("Course/judge submission formats differ. Ask the instructor to update the judge and course together to format v3; no code was submitted.")
        result = self._request("/api/submissions", payload)
        if not isinstance(result.get("submission_id"), str) or not result["submission_id"]:
            raise JudgeConnectionError("No submission receipt was returned. Refresh status before retrying.")
        return result["submission_id"]


def configure_workspace():
    """Persist an ALREADY provisioned personal key outside the course.

    This does not discover a Brev identity, issue accounts, or grant GPU access.
    Never use a shared key in a public Launchable or container image.
    """
    client = JudgeClient(os.environ.get("AI4SCI_JUDGE_URL", ""), os.environ.get("AI4SCI_JUDGE_TOKEN", ""))
    directory = Path.home() / ".config" / "ai4sci"
    path = directory / "judge.json"
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    info = directory.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_mode & 0o077 or info.st_uid != os.getuid():
        raise JudgeConnectionError("Use an owner-only configuration directory (mode 700).")
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        raise JudgeConnectionError("Judge configuration already exists; it was not overwritten.") from None
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump({"url": client.url, "token": client.token}, stream)
    print("Private judge configuration saved. No account or network service was created.")


if __name__ == "__main__":
    try:
        configure_workspace()
    except (JudgeConnectionError, OSError) as exc:
        raise SystemExit(str(exc) if isinstance(exc, JudgeConnectionError) else "Could not save private judge configuration.") from None
