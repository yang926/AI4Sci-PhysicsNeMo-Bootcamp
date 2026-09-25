"""Managed Jupyter configuration tests: no real service or notebook is changed."""
import io
import json
import os
from pathlib import Path
import pwd
import secrets
import stat
import subprocess
import sys
import tempfile
import time
from urllib.error import HTTPError

import pytest

from ETC.launchable import jupyter


def test_course_config_preserves_auth_network_and_unrelated_settings(tmp_path):
    original = {
        "IdentityProvider": {"token": "private-token"},
        "PasswordIdentityProvider": {"hashed_password": "private-hash"},
        "ServerApp": {"ip": "0.0.0.0", "port": 8888, "base_url": "/proxy/",
                      "allow_remote_access": True, "root_dir": "/old"},
        "LabApp": {"custom_css": True},
        "KernelSpecManager": {"kernel_dirs": ["/existing/kernels"]},
    }
    merged = json.loads(jupyter.course_config(json.dumps(original).encode(), tmp_path))
    for key in ("IdentityProvider", "PasswordIdentityProvider"):
        assert merged[key] == original[key]
    for key in ("ip", "port", "base_url", "allow_remote_access"):
        assert merged["ServerApp"][key] == original["ServerApp"][key]
    assert merged["ServerApp"]["root_dir"] == str(tmp_path)
    assert merged["ServerApp"]["default_url"] == "/lab/workspaces/ai4sci/tree/Start_Here.ipynb"
    assert merged["LabApp"] == {"custom_css": True, "default_url": jupyter.LANDING_PATH}
    assert merged["KernelSpecManager"]["kernel_dirs"] == ["/existing/kernels"]
    assert merged["KernelSpecManager"]["allowed_kernelspecs"] == [jupyter.KERNEL_NAME]
    assert merged["KernelSpecManager"]["ensure_native_kernel"] is False
    assert merged["MultiKernelManager"]["default_kernel_name"] == jupyter.KERNEL_NAME


@pytest.mark.parametrize("config", [b"[]", b"{broken", b'{"ServerApp":false}'])
def test_malformed_existing_config_is_not_replaced(config, tmp_path):
    with pytest.raises(ValueError):
        jupyter.course_config(config, tmp_path)


@pytest.mark.parametrize("argv,startup", [
    (["/venv/bin/python", "/venv/bin/jupyter-lab", "--ip=0.0.0.0"], "/venv/bin/jupyter-lab"),
    (["/venv/bin/python", "/venv/bin/jupyter-lab"], "/venv/bin/python"),
    (["/venv/bin/jupyter-lab"], "/venv/bin/jupyter-lab"),
    (["/venv/bin/python", "-m", "jupyterlab"], "/venv/bin/python"),
])
def test_known_direct_jupyter_launchers_are_supported(argv, startup):
    assert jupyter._direct_jupyter(argv, startup)


@pytest.mark.parametrize("argv,startup", [
    (["/bin/bash", "-c", "jupyter lab"], "/bin/bash"),
    (["/venv/bin/python", "/tmp/wrapper.py"], "/venv/bin/python"),
    (["/venv/bin/jupyter-lab"], "/usr/local/bin/custom-launcher"),
])
def test_unknown_launcher_is_not_rewritten(argv, startup):
    assert not jupyter._direct_jupyter(argv, startup)


@pytest.mark.parametrize("option", [
    "--ServerApp.root_dir=/old", "--notebook-dir", "--ServerApp.default_url=/tree",
    "--KernelSpecManager.allowed_kernelspecs=python3", "--config=/custom.py", "--",
])
def test_cli_overrides_are_rejected_without_rewriting_auth_options(option):
    with pytest.raises(ValueError, match="command-line"):
        jupyter._check_cli(["jupyter-lab", "--NotebookApp.token=private", option])


def test_auth_and_network_cli_options_are_accepted_unchanged():
    argv = ["/venv/bin/jupyter-lab", "--ip=0.0.0.0", "--no-browser",
            "--NotebookApp.token=private", "--NotebookApp.password=hash",
            "--ServerApp.allow_remote_access=True"]
    before = list(argv)
    jupyter._check_cli(argv)
    assert argv == before


@pytest.fixture
def managed_process(tmp_path, monkeypatch):
    home = tmp_path / "home"
    process = tmp_path / "proc/9722"
    process.mkdir(parents=True)
    process.joinpath("cmdline").write_bytes(
        b"/venv/bin/python\0/venv/bin/jupyter-lab\0--ip=0.0.0.0\0--NotebookApp.token=private\0")
    process.joinpath("environ").write_bytes(("HOME=" + str(home) + "\0").encode())
    values = {"Id": "jupyter.service", "User": pwd.getpwuid(os.getuid()).pw_name,
              "MainPID": "9722", "ActiveState": "active",
              "ExecStart": "{ path=/venv/bin/jupyter-lab ; argv[]=/venv/bin/jupyter-lab --ip=0.0.0.0 ; ignore_errors=no ; pid=9722 ; code=(null) ; status=0/0 }"}
    calls = []

    def command(*args):
        calls.append(args)
        return "\n".join(key + "=" + value for key, value in values.items())

    monkeypatch.setattr(jupyter, "_command", command)
    monkeypatch.setattr(jupyter, "Path", lambda *parts: (
        tmp_path / "proc" if parts == ("/proc",) else Path(*parts)))
    return home, process, values, calls


def test_inspect_only_known_service_and_current_user_process(managed_process):
    home, _, _, calls = managed_process
    service = jupyter.inspect_service(home)
    assert service.pid == 9722
    assert service.runtime == home / ".local/share/jupyter/runtime"
    assert len(calls) == 1 and calls[0][:3] == ("systemctl", "show", "jupyter.service")


@pytest.mark.parametrize("property,value", [("Id", "other.service"), ("User", "different-user"),
                                           ("ActiveState", "inactive"), ("MainPID", "0")])
def test_wrong_service_identity_or_state_stops_setup(managed_process, property, value):
    home, _, values, calls = managed_process
    values[property] = value
    with pytest.raises(ValueError):
        jupyter.inspect_service(home)
    assert len(calls) == 1


def test_custom_config_path_stops_setup(managed_process):
    home, process, _, _ = managed_process
    process.joinpath("environ").write_bytes(("HOME=" + str(home) + "\0JUPYTER_CONFIG_PATH=/custom\0").encode())
    with pytest.raises(ValueError, match="JUPYTER_CONFIG_PATH"):
        jupyter.inspect_service(home)


@pytest.fixture
def managed_server(tmp_path, monkeypatch):
    home, course, prefix = (tmp_path / name for name in ("home", "course", "prefix"))
    home.mkdir()
    course.mkdir()
    (course / "Start_Here.ipynb").write_text("{}")
    (prefix / "bin").mkdir(parents=True)
    (prefix / "bin/python").touch()
    service = jupyter.Service(9722, ["/venv/bin/python", "/venv/bin/jupyter-lab"], home / "runtime")
    state = {"restarted": False, "sessions": [], "kernels": [], "commands": [], "resources": []}
    monkeypatch.setattr(jupyter, "inspect_service", lambda _: service)
    monkeypatch.setattr(jupyter, "_server_info", lambda _: {
        "pid": service.pid, "root_dir": str(course if state["restarted"] else home),
        "url": "http://localhost:8888/", "token": "existing-token"})

    def api(info, resource):
        state["resources"].append(resource)
        if resource in {"sessions", "kernels"}:
            return state[resource]
        if resource == "kernelspecs":
            specs = {jupyter.KERNEL_NAME: {"spec": {"argv": [str(prefix / "bin/python"), "-m", "ipykernel_launcher"]}}}
            if not state["restarted"] or state.get("stale_kernels"):
                specs["python3"] = {"spec": {"argv": ["/managed/python"]}}
            return {"default": state.get("default_kernel", jupyter.KERNEL_NAME), "kernelspecs": specs}
        assert resource == "contents/Start_Here.ipynb?content=0"
        return {"path": "Start_Here.ipynb", "type": "notebook"}

    def command(*args):
        state["commands"].append(args)
        assert args == ("sudo", "-n", "systemctl", "restart", "jupyter.service")
        state["restarted"] = True
        return ""

    monkeypatch.setattr(jupyter, "_api", api)
    monkeypatch.setattr(jupyter, "_verify_landing", lambda _: state.update(landing_verified=True))
    monkeypatch.setattr(jupyter, "_command", command)
    monkeypatch.setattr(jupyter.time, "sleep", lambda _: None)
    return home, course, prefix, state


def test_configuration_is_private_backed_up_and_only_managed_unit_restarts(managed_server):
    home, course, prefix, state = managed_server
    config = home / ".jupyter/jupyter_server_config.json"
    config.parent.mkdir()
    original = b'{"ServerApp": {"token": "existing", "port": 8888}, "Other": {"x": 2}}\n'
    config.write_bytes(original)
    result = jupyter.configure_jupyter(course, prefix, home)
    assert result["config"] == config
    backup = result["backup"] / "jupyter_server_config.json.before"
    assert backup.read_bytes() == original
    for path in (config, backup):
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(result["backup"].stat().st_mode) == 0o700
    merged = json.loads(config.read_bytes())
    assert merged["ServerApp"]["token"] == "existing"
    assert merged["ServerApp"]["port"] == 8888
    assert merged["Other"] == {"x": 2}
    assert state["commands"] == [("sudo", "-n", "systemctl", "restart", "jupyter.service")]
    assert "contents/Start_Here.ipynb?content=0" in state["resources"]
    assert state["landing_verified"]
    assert not (home / ".jupyter/jupyter_server_config.d").exists()


@pytest.mark.parametrize("busy", ["sessions", "kernels"])
def test_active_notebook_state_blocks_all_config_and_service_changes(managed_server, busy):
    home, course, prefix, state = managed_server
    state[busy] = [{"id": "student-work"}]
    with pytest.raises(ValueError, match="active notebook"):
        jupyter.configure_jupyter(course, prefix, home)
    assert not (home / ".jupyter").exists()
    assert not state["commands"]


def test_matching_busy_server_is_noop_even_with_different_json_formatting(managed_server, monkeypatch):
    home, course, prefix, state = managed_server
    config = home / ".jupyter/jupyter_server_config.json"
    config.parent.mkdir()
    wanted = json.loads(jupyter.course_config(
        b'{"IdentityProvider":{"token":"keep-private"},"ServerApp":{"port":8888}}', course))
    config.write_text(json.dumps(wanted, separators=(",", ":")))
    original = config.read_bytes()
    stamp = config.stat().st_mtime_ns
    state.update(restarted=True, sessions=[{"id": "live-notebook"}], kernels=[{"id": "live-kernel"}])
    monkeypatch.setattr(jupyter, "_require_idle", lambda _: pytest.fail("Matching configuration must not require idle kernels."))
    monkeypatch.setattr(jupyter, "_private_write", lambda *_: pytest.fail("A no-op must not write backup/config files."))

    result = jupyter.configure_jupyter(course, prefix, home)

    assert result == {"status": "unchanged", "config": config, "backup": None, "service": "jupyter.service"}
    assert not state["commands"]
    assert "sessions" not in state["resources"] and "kernels" not in state["resources"]
    assert state["landing_verified"]
    assert config.read_bytes() == original
    assert config.stat().st_mtime_ns == stamp
    assert not (home / ".local").exists()


@pytest.mark.parametrize("drift", ["root", "kernels", "default", "landing", "config"])
def test_matching_disk_config_with_effective_drift_does_not_bypass_busy_guard(managed_server, monkeypatch, drift):
    home, course, prefix, state = managed_server
    config = home / ".jupyter/jupyter_server_config.json"
    config.parent.mkdir()
    config.write_bytes(jupyter.course_config(None, course))
    state.update(restarted=True, kernels=[{"id": "keep-this-kernel"}])
    if drift == "root":
        state["restarted"] = False
    elif drift == "kernels":
        state["stale_kernels"] = True
    elif drift == "default":
        state["default_kernel"] = "python3"
    elif drift == "landing":
        def wrong_landing(_):
            raise ValueError("Wrong landing path")
        monkeypatch.setattr(jupyter, "_verify_landing", wrong_landing)
    else:
        content = json.loads(config.read_bytes())
        content["ServerApp"]["default_url"] = "/lab"
        config.write_text(json.dumps(content))
    original = config.read_bytes()
    with pytest.raises(ValueError, match="active notebook"):
        jupyter.configure_jupyter(course, prefix, home)
    assert config.read_bytes() == original
    assert not (home / ".local").exists()
    assert not state["commands"]


def test_public_idle_preflight_never_changes_service(managed_server):
    home, _, _, state = managed_server
    jupyter.require_managed_idle(home)
    assert not state["commands"]
    state["sessions"] = [{"id": "learner"}]
    with pytest.raises(ValueError, match="active notebook"):
        jupyter.require_managed_idle(home)
    assert not state["commands"]


def test_explicit_discard_permission_allows_one_off_active_service_restart(managed_server):
    home, course, prefix, state = managed_server
    state["sessions"] = [{"id": "approved-test-session"}]
    state["kernels"] = [{"id": "approved-test-kernel"}]
    result = jupyter.configure_jupyter(course, prefix, home, allow_active_restart=True)
    assert result["service"] == "jupyter.service"
    assert state["commands"] == [("sudo", "-n", "systemctl", "restart", "jupyter.service")]
    assert "sessions" not in state["resources"] and "kernels" not in state["resources"]
    assert "kernelspecs" in state["resources"]


def test_symlinked_config_is_not_overwritten(managed_server, tmp_path):
    home, course, prefix, state = managed_server
    target = tmp_path / "unrelated.json"
    target.write_text('{"keep": true}')
    (home / ".jupyter").mkdir()
    (home / ".jupyter/jupyter_server_config.json").symlink_to(target)
    with pytest.raises(ValueError, match="symlinked"):
        jupyter.configure_jupyter(course, prefix, home)
    assert target.read_text() == '{"keep": true}'
    assert not state["commands"]


def test_verification_does_not_report_success_with_extra_visible_kernels(managed_server):
    home, course, prefix, state = managed_server
    state["stale_kernels"] = True
    with pytest.raises(ValueError, match="verification failed"):
        jupyter.configure_jupyter(course, prefix, home)
    assert len(state["commands"]) == 1


@pytest.mark.parametrize("advertised", ["0.0.0.0", "localhost", "ip-10-2-3-4", "10.2.3.4",
                                      "unrelated.example"])
def test_local_api_preserves_base_path_auth_and_avoids_proxy(monkeypatch, advertised):
    observed = {}

    class Opener:
        def open(self, request, timeout):
            observed["request"] = request
            assert timeout == 5
            return io.BytesIO(b"[]")

    def build(*handlers):
        assert handlers[0].proxies == {}
        assert isinstance(handlers[1], jupyter._NoRedirect)
        return Opener()

    monkeypatch.setattr(jupyter, "build_opener", build)
    assert jupyter._api({"url": "http://" + advertised + ":8888/proxy/", "port": 8888,
                        "base_url": "/proxy/", "token": "existing"}, "sessions") == []
    assert observed["request"].full_url == "http://127.0.0.1:8888/proxy/api/sessions"
    assert observed["request"].get_header("Authorization") == "token existing"


@pytest.mark.parametrize("settings", [
    {"url": "https://user:password@unrelated.example:8888/", "port": 8888},
    {"url": "file:///tmp/server", "port": 8888},
    {"url": "http://vm:8888/", "port": 0},
    {"url": "http://vm:8888/", "port": 65536},
    {"url": "http://vm:8888/", "port": "8888"},
    {"url": "http://vm:8888/?token=other", "port": 8888},
    {"url": "http://vm:8888/", "port": 8888, "base_url": "https://unrelated.example/"},
])
def test_invalid_runtime_record_is_rejected_before_request(monkeypatch, settings):
    monkeypatch.setattr(jupyter, "build_opener", lambda *_: pytest.fail("No remote request is permitted."))
    with pytest.raises(ValueError, match="invalid local API"):
        jupyter._api(dict(settings, token="private"), "sessions")


@pytest.mark.parametrize("location,accepted", [
    ("/proxy/lab/workspaces/ai4sci/tree/Start_Here.ipynb", True),
    ("/proxy/lab", False),
    ("/login", False),
    ("https://unrelated.example/proxy/lab/workspaces/ai4sci/tree/Start_Here.ipynb", False),
])
def test_actual_root_redirect_must_point_to_course_workspace(monkeypatch, location, accepted):
    class Opener:
        def open(self, request, timeout):
            assert request.full_url == "http://127.0.0.1:8888/proxy/"
            assert request.get_header("Authorization") == "token existing"
            raise HTTPError(request.full_url, 302, "Found", {"Location": location}, io.BytesIO())

    monkeypatch.setattr(jupyter, "build_opener", lambda *_: Opener())
    info = {"url": "http://vm:8888/proxy/", "token": "existing"}
    if accepted:
        jupyter._verify_landing(info)
    else:
        with pytest.raises(ValueError, match="does not redirect"):
            jupyter._verify_landing(info)


def test_real_jupyter_lab_loads_normal_config_and_course_landing(tmp_path):
    """Exercise JupyterLab's real default-url precedence on an isolated local server."""
    pytest.importorskip("jupyterlab")
    course, config, runtime, data = (tmp_path / name for name in ("course", "config", "runtime", "data"))
    for directory in (course, config, runtime, data):
        directory.mkdir(mode=0o700)
    (course / "Start_Here.ipynb").write_text(json.dumps({
        "cells": [], "metadata": {}, "nbformat": 4, "nbformat_minor": 5}))
    secret = secrets.token_urlsafe(32)
    existing = {"IdentityProvider": {"token": secret},
                "LabApp": {"workspaces_dir": str(tmp_path / "workspaces"),
                           "user_settings_dir": str(tmp_path / "settings")}}
    jupyter._private_write(config / "jupyter_server_config.json",
                           jupyter.course_config(json.dumps(existing).encode(), course))
    kernel = data / "kernels" / jupyter.KERNEL_NAME
    kernel.mkdir(parents=True)
    (kernel / "kernel.json").write_text(json.dumps({
        "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
        "display_name": "Isolated course test", "language": "python"}))
    environment = dict(os.environ, JUPYTER_CONFIG_DIR=str(config), JUPYTER_DATA_DIR=str(data),
                       JUPYTER_RUNTIME_DIR=str(runtime), JUPYTER_CONFIG_PATH="",
                       JUPYTER_PREFER_ENV_PATH="0")
    environment.pop("JUPYTER_TOKEN", None)
    environment.pop("JUPYTER_TOKEN_FILE", None)
    with tempfile.TemporaryFile() as log:
        process = subprocess.Popen([
            sys.executable, "-m", "jupyterlab", "--no-browser", "--ServerApp.ip=127.0.0.1",
            "--ServerApp.port=0", "--ServerApp.port_retries=0"], cwd=course,
            env=environment, stdout=log, stderr=subprocess.STDOUT)
        try:
            service = jupyter.Service(process.pid, [], runtime)
            for _ in range(200):
                assert process.poll() is None, "The isolated JupyterLab server exited during startup."
                try:
                    info = jupyter._server_info(service)
                    break
                except ValueError:
                    time.sleep(0.1)
            else:
                pytest.fail("The isolated JupyterLab server did not start within 20 seconds.")
            assert info["token"] == secret and info["port"] > 0
            assert Path(info["root_dir"]) == course
            jupyter._check_kernel(info, Path(sys.prefix), exclusive=True)
            assert jupyter._api(info, "contents/Start_Here.ipynb?content=0")["type"] == "notebook"
            jupyter._verify_landing(info)
        finally:
            # Only the temporary child created by this test is terminated.
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
