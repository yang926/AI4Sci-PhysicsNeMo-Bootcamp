"""Configure and restart only a verified, idle Brev ``jupyter.service``.

This module never starts another server, changes authentication/network options,
or searches for processes to kill. Unrecognized service layouts stop setup.
"""
from dataclasses import dataclass
import json
import os
from pathlib import Path
import pwd
import re
import subprocess
import tempfile
import time
from urllib.error import HTTPError
from urllib.parse import urljoin, urlsplit, urlunsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener


KERNEL_NAME = "ai4sci-physicsnemo-uv"
UNIT = "jupyter.service"
LANDING_PATH = "/lab/workspaces/ai4sci/tree/Start_Here.ipynb"


@dataclass
class Service:
    pid: int
    argv: list
    runtime: Path


def _command(*args):
    return subprocess.run([str(arg) for arg in args], check=True, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout


def _direct_jupyter(argv, startup):
    """Accept direct Jupyter entrypoints, including their Python shebang argv."""
    if not argv or not Path(startup).is_absolute():
        return False
    name = Path(startup).name
    if name in {"jupyter", "jupyter-lab", "jupyter-server"}:
        choices = {startup}
        if name == "jupyter":
            choices.update(str(Path(startup).with_name(item))
                           for item in ("jupyter-lab", "jupyter-server"))
        return argv[0] in choices or (
            len(argv) > 1 and Path(argv[0]).name.startswith("python") and argv[1] in choices)
    if name.startswith("python") and argv[0] == startup:
        return (len(argv) >= 3 and argv[1:3] in (["-m", "jupyterlab"], ["-m", "jupyter_server"])) or (
            len(argv) >= 2 and Path(argv[1]).name in {"jupyter-lab", "jupyter-server"}
            and Path(argv[1]).parent == Path(startup).parent)
    return False


def _check_cli(argv):
    overrides = {
        "--config", "--JupyterApp.config_file", "--ServerApp.config_file", "--LabApp.config_file",
        "--JupyterApp.config_file_name", "--ServerApp.config_file_name", "--LabApp.config_file_name",
        "--notebook-dir", "--notebook_dir", "--ServerApp.root_dir", "--NotebookApp.notebook_dir",
        "--ServerApp.notebook_dir", "--FileContentsManager.root_dir", "--ContentsManager.root_dir",
        "--ServerApp.default_url", "--NotebookApp.default_url", "--LabApp.default_url",
        "--KernelSpecManager.allowed_kernelspecs", "--KernelSpecManager.whitelist",
        "--KernelSpecManager.ensure_native_kernel", "--MultiKernelManager.default_kernel_name",
        "--MappingKernelManager.default_kernel_name",
        "--ServerApp.jpserver_extensions",
    }
    if any(arg.split("=", 1)[0] in overrides or arg == "--" for arg in argv):
        raise ValueError("jupyter.service has command-line course/config overrides. An administrator must remove those overrides before setup; the service was not changed.")


def inspect_service(home):
    properties = _command("systemctl", "show", UNIT, "--no-pager",
                          "--property=Id,User,MainPID,ActiveState,ExecStart")
    values = dict(line.split("=", 1) for line in properties.splitlines() if "=" in line)
    username = pwd.getpwuid(os.getuid()).pw_name
    if (values.get("Id") != UNIT or values.get("User") not in {username, str(os.getuid())}
            or values.get("ActiveState") != "active"):
        raise ValueError("Expected an active jupyter.service owned by the setup user; no service was changed.")
    pid = int(values.get("MainPID", "0"))
    if pid <= 0:
        raise ValueError("jupyter.service has no verifiable running process; no service was changed.")
    process = Path("/proc") / str(pid)
    if process.stat().st_uid != os.getuid():
        raise ValueError("The managed Jupyter process belongs to another user; no service was changed.")
    argv = [os.fsdecode(value) for value in (process / "cmdline").read_bytes().split(b"\0") if value]
    environment = dict(os.fsdecode(value).split("=", 1)
                       for value in (process / "environ").read_bytes().split(b"\0") if b"=" in value)
    startup = re.fullmatch(r"\{ path=([^ ;{}]+) ; argv\[\]=.* ; ignore_errors=no ; .*\}",
                           values.get("ExecStart", ""))
    if not startup or not _direct_jupyter(argv, startup[1]):
        raise ValueError("jupyter.service uses an unrecognized launcher; automatic replacement is unsafe.")
    _check_cli(argv)
    if Path(environment.get("HOME", "")) != home:
        raise ValueError("The managed Jupyter HOME differs from the setup user's home.")
    if Path(environment.get("JUPYTER_CONFIG_DIR", str(home / ".jupyter"))) != home / ".jupyter":
        raise ValueError("A custom JUPYTER_CONFIG_DIR requires administrator review; no service was changed.")
    if environment.get("JUPYTER_CONFIG_PATH"):
        raise ValueError("A custom JUPYTER_CONFIG_PATH requires administrator review; no service was changed.")
    runtime = Path(environment.get("JUPYTER_RUNTIME_DIR", str(
        Path(environment.get("XDG_DATA_HOME", str(home / ".local/share"))) / "jupyter/runtime")))
    if not runtime.is_absolute():
        raise ValueError("The managed Jupyter runtime directory is not absolute.")
    return Service(pid, argv, runtime)


def _server_info(service):
    for filename in service.runtime.glob("*server-*.json"):
        try:
            info = json.loads(filename.read_text())
        except (OSError, ValueError):
            continue
        if info.get("pid") == service.pid:
            return info
    raise ValueError("Cannot identify the managed Jupyter runtime record; no safe restart is possible.")


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError("Managed Jupyter redirected its local API; authentication could not be verified.")


def _local_request(info, resource):
    url = urlsplit(info["url"])
    port = info.get("port", url.port)
    base_url = info.get("base_url", url.path or "/")
    if (url.scheme not in {"http", "https"} or not url.hostname or url.username or url.password
            or url.query or url.fragment or type(port) is not int or not 0 < port < 65536
            or not isinstance(base_url, str) or not base_url.startswith("/")
            or any(character in base_url for character in "?#\r\n")):
        raise ValueError("The managed Jupyter runtime record has invalid local API settings.")
    # A server bound to 0.0.0.0 may advertise its VM hostname/private address.
    # Use only the PID-verified runtime record's port and base path: never resolve
    # or send its authentication token to that advertised network destination.
    host = "[::1]" if ":" in url.hostname else "127.0.0.1"
    netloc = host + ":" + str(port)
    endpoint = urlunsplit((url.scheme, netloc, base_url.rstrip("/") + "/" + resource, "", ""))
    headers = {"Authorization": "token " + info["token"]} if info.get("token") else {}
    return Request(endpoint, headers=headers)


def _api(info, resource):
    request = _local_request(info, "api/" + resource)
    # Neither environment proxies nor login redirects may forward the existing token.
    try:
        with build_opener(ProxyHandler({}), _NoRedirect()).open(
                request, timeout=5) as response:
            return json.load(response)
    except (OSError, ValueError) as exc:
        raise ValueError("Cannot authenticate to the managed Jupyter API; no safe restart is possible.") from exc


class _InspectRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # urllib exposes the redirect as HTTPError; its destination is never opened.
        return None


def _verify_landing(info):
    request = _local_request(info, "")
    expected = urlsplit(request.full_url.rstrip("/") + LANDING_PATH)
    try:
        with build_opener(ProxyHandler({}), _InspectRedirect()).open(request, timeout=5):
            pass
    except HTTPError as response:
        try:
            location = urlsplit(urljoin(request.full_url, response.headers.get("Location", "")))
            if response.code in {301, 302, 303, 307, 308} and location == expected:
                return
        finally:
            response.close()
    raise ValueError("The managed server root does not redirect to the course start notebook workspace.")


def _require_idle(info):
    sessions = _api(info, "sessions")
    kernels = _api(info, "kernels")
    if not isinstance(sessions, list) or not isinstance(kernels, list) or sessions or kernels:
        raise ValueError("Managed Jupyter has active notebook sessions or kernels. Save and shut them down before rerunning setup.")


def require_managed_idle(home):
    """Read-only preflight for changes to an existing managed environment."""
    _require_idle(_server_info(inspect_service(Path(home).resolve())))


def _check_kernel(info, prefix, *, exclusive=False):
    listing = _api(info, "kernelspecs")
    specs = listing.get("kernelspecs", {})
    argv = specs.get(KERNEL_NAME, {}).get("spec", {}).get("argv", [])
    if not argv or Path(argv[0]) != prefix / "bin/python":
        raise ValueError("The managed server cannot see the installed course kernel; no verified setup is possible.")
    if exclusive and (set(specs) != {KERNEL_NAME} or listing.get("default") != KERNEL_NAME):
        raise ValueError("The restarted service did not apply the course-only kernel selection.")


def _verify_course_runtime(info, course, prefix):
    """Verify effective settings, not merely the configuration saved on disk."""
    if Path(info.get("root_dir", "")) != course:
        raise ValueError("The managed server did not apply the course root.")
    _check_kernel(info, prefix, exclusive=True)
    notebook = _api(info, "contents/Start_Here.ipynb?content=0")
    if notebook.get("path") != "Start_Here.ipynb" or notebook.get("type") != "notebook":
        raise ValueError("The course start notebook is not available at the managed server root.")
    _verify_landing(info)
    _verify_proxy(info)


def _verify_proxy(info):
    request = _local_request(info, "server-proxy/servers-info")
    try:
        with build_opener(ProxyHandler({}), _NoRedirect()).open(request, timeout=5) as response:
            result = json.load(response)
        if not isinstance(result, dict) or not isinstance(result.get("server_processes"), list):
            raise ValueError("Unexpected Jupyter proxy response")
    except (OSError, ValueError) as exc:
        raise ValueError("The managed server has not enabled its authenticated TensorBoard proxy.") from exc


def course_config(existing, course):
    """Merge only course UI fields; retain existing authentication and other traits."""
    config = json.loads(existing) if existing is not None else {}
    if not isinstance(config, dict):
        raise ValueError("Existing Jupyter configuration is not a JSON object.")
    changes = {
        "ServerApp": {"root_dir": str(course), "default_url": LANDING_PATH},
        "LabApp": {"default_url": LANDING_PATH},
        "KernelSpecManager": {"allowed_kernelspecs": [KERNEL_NAME], "ensure_native_kernel": False},
        "MultiKernelManager": {"default_kernel_name": KERNEL_NAME},
    }
    for section, fields in changes.items():
        if not isinstance(config.setdefault(section, {}), dict):
            raise ValueError("Existing Jupyter configuration section is not an object: " + section)
        config[section].update(fields)
    extensions = config["ServerApp"].setdefault("jpserver_extensions", {})
    if not isinstance(extensions, dict):
        raise ValueError("Existing Jupyter server extension settings are not an object.")
    extensions["jupyter_server_proxy"] = True
    return (json.dumps(config, indent=2) + "\n").encode()


def _private_write(path, content):
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(content)


def configure_jupyter(course, prefix, home, *, allow_active_restart=False):
    """Leave matching servers untouched; otherwise require idle before reconfiguring.

    All service commands address only jupyter.service. sudo is noninteractive;
    unsupported service ownership, launchers, authentication, and busy servers
    stop setup rather than guessing. Backups include original bytes and are private.
    ``allow_active_restart`` is only for an explicit request to discard running
    notebook state; unattended Launchable setup must keep its default False.
    """
    course, prefix, home = (Path(path).resolve() for path in (course, prefix, home))
    if not (course / "Start_Here.ipynb").is_file() or not (prefix / "bin/python").is_file():
        raise ValueError("The course notebook and installed course Python must exist before Jupyter setup.")
    service = inspect_service(home)
    info = _server_info(service)
    config_dir = home / ".jupyter"
    config_file = config_dir / "jupyter_server_config.json"
    if config_dir.is_symlink() or config_file.is_symlink():
        raise ValueError("A symlinked Jupyter configuration requires review; nothing was overwritten.")
    original = config_file.read_bytes() if config_file.exists() else None
    updated = course_config(original, course)
    if original is not None and json.loads(original) == json.loads(updated):
        try:
            _verify_course_runtime(info, course, prefix)
        except (OSError, ValueError):
            # Saved configuration alone is not proof that the running process
            # loaded it. Drift still needs the normal guarded restart below.
            pass
        else:
            return {"status": "unchanged", "config": config_file, "backup": None, "service": UNIT}
    if not allow_active_restart:
        _require_idle(info)
    _check_kernel(info, prefix)
    backups = home / ".local/share/ai4sci-jupyter-backups"
    if backups.is_symlink():
        raise ValueError("The Jupyter backup directory must not be a symlink.")
    backups.mkdir(parents=True, exist_ok=True, mode=0o700)
    stage = Path(tempfile.mkdtemp(prefix="setup-", dir=backups))
    if original is not None:
        _private_write(stage / "jupyter_server_config.json.before", original)
    _private_write(stage / "jupyter_server_config.json", updated)
    _private_write(stage / "README.txt", (
        "The original config is jupyter_server_config.json.before. Absence means it did not exist.\n"
        "Config target: " + str(config_file) + "\n"
    ).encode())
    # Repeat the safety checks immediately before changing the known service.
    current = inspect_service(home)
    if current.pid != service.pid or current.argv != service.argv:
        raise ValueError("Managed Jupyter changed during setup; retry after it settles.")
    if not allow_active_restart:
        _require_idle(_server_info(current))
    config_dir.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".ai4sci-", dir=config_dir)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(updated)
    os.replace(temporary, config_file)
    if not allow_active_restart:
        _require_idle(_server_info(service))
    _command("sudo", "-n", "systemctl", "restart", UNIT)
    # Restart may return before Jupyter has written its new runtime record.
    last_error = None
    for attempt in range(20):
        try:
            restarted = inspect_service(home)
            running = _server_info(restarted)
            _verify_course_runtime(running, course, prefix)
            return {"status": "changed", "config": config_file, "backup": stage, "service": UNIT}
        except (OSError, ValueError, subprocess.CalledProcessError) as exc:
            last_error = exc
            if attempt < 19:
                time.sleep(0.5)
    raise ValueError("Managed Jupyter verification failed; retained private backups at " + str(stage)) from last_error
