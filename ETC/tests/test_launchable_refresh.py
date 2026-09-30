"""Refresh orchestration: no live services, package downloads or GPU allocation."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ETC.launchable import bootstrap


@pytest.fixture
def managed_update(tmp_path, monkeypatch):
    home = tmp_path / "home"
    course = home / "course"
    course.mkdir(parents=True)
    (home / ".jupyter").mkdir()
    (home / ".jupyter/jupyter_server_config.json").write_text(json.dumps({"ServerApp": {"root_dir": str(course)}}))
    monkeypatch.setattr(bootstrap.Path, "home", classmethod(lambda cls: home))
    state = {"sessions": [{"path": "lesson.ipynb", "kernel": {"id": "k"}}],
             "kernels": [{"id": "k", "execution_state": "idle", "connections": 0}],
             "dependency_change": False, "changed": b"lesson.ipynb\0"}
    jupyter = SimpleNamespace(inspect_service=lambda _: object(), _server_info=lambda _: {},
                              _api=lambda _, resource: state[resource])
    monkeypatch.setattr(bootstrap, "release_module", lambda *args: jupyter)
    monkeypatch.setattr(bootstrap, "runtime_signature", lambda _, rev: ("3.12", b"new" if state["dependency_change"] and rev == "new" else b"same"))
    monkeypatch.setattr(bootstrap, "git_bytes", lambda *args: state["changed"])
    return course, state


def test_closed_tabs_with_idle_kernels_do_not_block_source_update(managed_update):
    course, _ = managed_update
    bootstrap.preflight_managed_update(course, "old", "new")


def test_executing_notebook_blocks_before_any_source_change(managed_update):
    course, state = managed_update
    state["kernels"][0]["execution_state"] = "busy"
    with pytest.raises(ValueError, match="still executing"):
        bootstrap.preflight_managed_update(course, "old", "new")


def test_changed_dependencies_need_idle_server_before_source_update(managed_update):
    course, state = managed_update
    state["dependency_change"] = True
    with pytest.raises(ValueError, match="changes Python packages"):
        bootstrap.preflight_managed_update(course, "old", "new")
    state["sessions"] = state["kernels"] = []
    bootstrap.preflight_managed_update(course, "old", "new")


def test_connected_changed_notebook_names_tab_to_close(managed_update):
    course, state = managed_update
    state["kernels"][0]["connections"] = 1
    with pytest.raises(ValueError, match=r"lesson.ipynb.*Idle kernels can stay"):
        bootstrap.preflight_managed_update(course, "old", "new")


def test_unchanged_open_notebook_does_not_block_other_files(managed_update):
    course, state = managed_update
    state["kernels"][0]["connections"] = 1
    state["changed"] = b"other.ipynb\0"
    bootstrap.preflight_managed_update(course, "old", "new")


def test_unmanaged_checkout_does_not_inspect_an_unrelated_server(tmp_path, monkeypatch):
    monkeypatch.setattr(bootstrap.Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setattr(bootstrap, "release_module", lambda *args: pytest.fail("Unrelated server inspected"))
    bootstrap.preflight_managed_update(tmp_path / "course", "old", "new")


@pytest.mark.parametrize("modern", [False, True])
@pytest.mark.parametrize("mode", ["--launchable", "--refresh"])
def test_refresh_flag_only_passed_to_installers_that_support_it(tmp_path, monkeypatch, modern, mode):
    course = tmp_path / "course"
    installer = course / "ETC/launchable/install.py"
    installer.parent.mkdir(parents=True)
    installer.write_text('parser.add_argument("--refresh")\n' if modern else "# Old installer\n")
    calls = []
    monkeypatch.setattr(bootstrap.os, "geteuid", lambda: 1000)
    monkeypatch.setattr(bootstrap.sys, "argv", ["bootstrap.py", mode, "--destination", str(course)])
    monkeypatch.setattr(bootstrap, "prepare_launchable_checkout", lambda *args: course)
    monkeypatch.setattr(bootstrap.subprocess, "run", lambda command, **kwargs: calls.append(command))
    bootstrap.main()
    assert len(calls) == 1
    assert "--configure-jupyter" in calls[0]
    assert ("--refresh" in calls[0]) is modern


def test_git_environment_excludes_routing_and_index_overrides(monkeypatch):
    for name in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_CONFIG_COUNT"):
        monkeypatch.setenv(name, "must-not-leak")
    seen = []
    monkeypatch.setattr(bootstrap.subprocess, "check_output", lambda command, **kwargs: seen.append(kwargs["env"]) or b"ok\n")
    assert bootstrap.git(Path("/fixture"), "rev-parse", "HEAD") == "ok"
    assert not any(key.startswith("GIT_") for key in seen[0])


def test_runtime_signature_compares_python_and_lock_not_installer_text(monkeypatch):
    code = {"old": b'PYTHON_VERSION = "3.12.11"\n',
            "new": b'# Other implementation changes\nPYTHON_VERSION = "3.12.11"\n'}
    def blob(directory, command, ref):
        revision, path = ref.split(":", 1)
        return b"locked-packages" if path.endswith(".txt") else code[revision]
    monkeypatch.setattr(bootstrap, "git_bytes", blob)
    assert bootstrap.runtime_signature(Path("/fixture"), "old") == bootstrap.runtime_signature(Path("/fixture"), "new")
    code["new"] = b'PYTHON_VERSION = "3.13.1"\n'
    assert bootstrap.runtime_signature(Path("/fixture"), "old") != bootstrap.runtime_signature(Path("/fixture"), "new")
