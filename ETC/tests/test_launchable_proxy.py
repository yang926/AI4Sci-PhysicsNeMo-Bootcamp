"""Do not replace server environments or restart sessions to show TensorBoard."""
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ETC.launchable import install, jupyter


@pytest.fixture
def managed(tmp_path, monkeypatch):
    home = tmp_path / "home"
    prefix = home / ".venv"
    (prefix / "bin").mkdir(parents=True)
    (prefix / "bin/python").touch()
    (prefix / "pyvenv.cfg").write_text("home = /python\n")
    service = SimpleNamespace(pid=42, argv=[str(prefix / "bin/python"), str(prefix / "bin/jupyter-lab")])
    state = {"installed": {name: None for name in install.PROXY_PACKAGES}, "events": []}
    module = SimpleNamespace(inspect_service=lambda _: service,
                             require_managed_idle=lambda _: state["events"].append("idle"))
    monkeypatch.setattr(install, "proxy_environment", lambda python: state["installed"].copy())
    monkeypatch.setattr(install, "ensure_uv", lambda _: "/private/uv")

    def run(*args):
        state["events"].append(args)
        state["installed"] = dict(install.PROXY_PACKAGES)

    monkeypatch.setattr(install, "run", run)
    return home, prefix, service, module, state


def test_only_pinned_proxy_packages_installed_into_known_server_environment(managed):
    home, prefix, _, module, state = managed
    assert install.ensure_server_proxy(home, module)
    assert state["events"][:2] == ["idle", "idle"]
    assert state["events"][2] == ("/private/uv", "pip", "install", "--python", prefix / "bin/python",
                                   "--no-deps", *[f"{key}=={value}" for key, value in install.PROXY_PACKAGES.items()])
    assert state["events"][3][:3] == (prefix / "bin/python", "-I", "-c")


def test_existing_compatible_server_dependencies_are_never_upgraded(managed):
    home, _, _, module, state = managed
    state["installed"].update({"attrs": "25.3.0", "idna": "3.20", "typing-extensions": "4.16.0"})
    install.ensure_server_proxy(home, module)
    command = state["events"][2]
    assert not any(str(arg).startswith(("attrs==", "idna==", "typing-extensions==")) for arg in command)


def test_matching_proxy_does_not_require_idle_or_modify_anything(managed):
    home, _, _, module, state = managed
    state["installed"] = dict(install.PROXY_PACKAGES)
    assert not install.ensure_server_proxy(home, module)
    assert not state["events"]


def test_busy_server_blocks_dependency_install(managed):
    home, _, _, module, state = managed

    def busy(_):
        raise ValueError("active notebook")

    module.require_managed_idle = busy
    with pytest.raises(ValueError, match="active notebook"):
        install.ensure_server_proxy(home, module)
    assert not state["events"]


def test_unrelated_installed_proxy_is_not_replaced(managed):
    home, _, _, module, state = managed
    state["installed"]["jupyter-server-proxy"] = "3.0.0"
    with pytest.raises(ValueError, match="review"):
        install.ensure_server_proxy(home, module)
    assert not state["events"]


def test_system_python_cannot_be_install_target(managed):
    home, _, service, _, _ = managed
    service.argv[0] = "/usr/bin/python3"
    with pytest.raises(ValueError, match="user-owned"):
        install.managed_server_python(service, home)


def test_proxy_extension_merge_preserves_other_extensions_and_auth(tmp_path):
    previous = {"ServerApp": {"jpserver_extensions": {"jupyter_lsp": True, "other": False}, "port": 8888},
                "IdentityProvider": {"token": "private"}, "ServerProxy": {"host_allowlist": ["localhost"]}}
    merged = json.loads(jupyter.course_config(json.dumps(previous), tmp_path))
    assert merged["ServerApp"]["jpserver_extensions"] == {
        "jupyter_lsp": True, "other": False, "jupyter_server_proxy": True}
    assert merged["IdentityProvider"] == previous["IdentityProvider"]
    assert merged["ServerProxy"] == previous["ServerProxy"]


def test_proxy_runtime_verification_stays_local_and_requires_correct_response(monkeypatch):
    class Opener:
        def open(self, request, timeout):
            assert request.full_url == "http://127.0.0.1:8888/prefix/server-proxy/servers-info"
            assert request.get_header("Authorization") == "token private"
            return io.BytesIO(b'{"server_processes": []}')

    monkeypatch.setattr(jupyter, "build_opener", lambda *_: Opener())
    jupyter._verify_proxy({"url": "http://remote-name:8888/", "port": 8888,
                           "base_url": "/prefix/", "token": "private"})


@pytest.mark.parametrize("bad", [None, [], True])
def test_malformed_extension_settings_are_not_overwritten(tmp_path, bad):
    with pytest.raises(ValueError, match="extension settings"):
        jupyter.course_config(json.dumps({"ServerApp": {"jpserver_extensions": bad}}), tmp_path)
