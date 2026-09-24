"""GitHub Launchable setup preserves learner files and managed Jupyter.

Git operations use a temporary local repository. Installation tests mock commands:
they never download packages, allocate a GPU, or start a notebook server.
"""
from pathlib import Path
import json
import os
import subprocess
import sys

import pytest

from ETC.launchable import bootstrap, install


REPOSITORY_URL = "https://github.com/example/physicsnemo-course.git"


def test_documented_paste_script_matches_brev_form_and_valid_shell_syntax():
    readme = Path(install.__file__).with_name("README.md").read_text()
    script = readme.split("```bash\n", 1)[1].split("```", 1)[0]
    assert script.splitlines()[0] == "#!/bin/bash"
    assert script == Path(install.__file__).with_name("setup.sh").read_text()
    subprocess.run(["bash", "-n"], input=script, text=True, check=True)


@pytest.mark.parametrize("download_exit", [0, 22])
def test_shared_launchable_fetches_fresh_course_without_student_repair(tmp_path, download_exit):
    """Execute the actual shell template with fake network/Python commands."""
    binaries = tmp_path / "bin"
    binaries.mkdir()
    calls_file = tmp_path / "calls.jsonl"
    for name in ("curl", "python3"):
        command = binaries / name
        command.write_text(
            f"#!{sys.executable}\n"
            "import json, os, pathlib, sys\n"
            "with open(os.environ['AI4SCI_TEST_CALLS'], 'a') as stream:\n"
            "    stream.write(json.dumps([pathlib.Path(sys.argv[0]).name, *sys.argv[1:]]) + '\\n')\n"
            "if pathlib.Path(sys.argv[0]).name == 'curl':\n"
            "    sys.exit(int(os.environ['AI4SCI_TEST_DOWNLOAD_EXIT']))\n"
        )
        command.chmod(0o755)
    script = Path(install.__file__).with_name("setup.sh")
    environment = dict(os.environ, PATH=str(binaries) + os.pathsep + os.environ["PATH"],
                       TMPDIR=str(tmp_path), AI4SCI_TEST_CALLS=str(calls_file),
                       AI4SCI_TEST_DOWNLOAD_EXIT=str(download_exit))
    result = subprocess.run(["bash", str(script)], env=environment, capture_output=True, text=True)
    calls = [json.loads(line) for line in calls_file.read_text().splitlines()]
    assert calls[0][0] == "curl"
    assert "https://raw.githubusercontent.com/yang926/AI4Sci-PhysicsNeMo-Bootcamp/main/ETC/launchable/bootstrap.py" in calls[0]
    if download_exit:
        assert result.returncode != 0 and len(calls) == 1
    else:
        assert result.returncode == 0, result.stderr
        assert len(calls) == 2
        assert calls[1][0] == "python3" and calls[1][-1] == "--launchable"
        assert calls[1][1] == calls[0][calls[0].index("--output") + 1]


@pytest.fixture
def course_remote(tmp_path, monkeypatch):
    """Serve a public-looking GitHub URL from a private, temporary Git fixture."""
    real_run = subprocess.run
    remote = tmp_path / "fixture-remote"
    remote.mkdir()

    def git(*args, cwd=remote):
        return real_run(["git", *args], cwd=cwd, check=True, text=True,
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    git("init", "-b", "main")
    git("config", "user.name", "Launchable fixture")
    git("config", "user.email", "fixture@example.invalid")
    (remote / "Start_Here.ipynb").write_text("{\"fixture\": 1}\n")
    (remote / "lesson.py").write_text("STEPS = 200\n")
    (remote / "ETC/launchable").mkdir(parents=True)
    (remote / "ETC/launchable/install.py").write_text("# Installer fixture\n")
    git("add", ".")
    git("commit", "-m", "Initial course fixture")
    calls = []

    def local_run(command, *args, **kwargs):
        command = list(command)
        calls.append(command)
        if "git" in Path(str(command[0])).name:
            assert not any(word in command for word in ("pull", "reset", "clean")), command
            if "clone" in command:
                assert REPOSITORY_URL in command, "Tests must never contact a real Git server."
                destination = Path(command[-1])
                mapped = [str(remote) if arg == REPOSITORY_URL else arg for arg in command]
                result = real_run(mapped, *args, **kwargs)
                if result.returncode == 0:
                    git("remote", "set-url", "origin", REPOSITORY_URL, cwd=destination)
                return result
            if "fetch" in command:
                assert "origin" in command
                return real_run([str(remote) if arg == "origin" else arg for arg in command],
                                *args, **kwargs)
        return real_run(command, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", local_run)
    return remote, git, calls


def test_first_launch_clones_selected_course_without_external_network(course_remote, tmp_path):
    _, git, calls = course_remote
    destination = tmp_path / "workspace" / "PhysicsNeMo"
    checkout = bootstrap.prepare_checkout(REPOSITORY_URL, "main", destination)
    assert checkout == destination
    assert (checkout / "Start_Here.ipynb").is_file()
    assert git("remote", "get-url", "origin", cwd=checkout).stdout.strip() == REPOSITORY_URL
    assert sum("clone" in command for command in calls) == 1


def test_repeat_launch_preserves_unsaved_course_changes_and_outputs(course_remote, tmp_path):
    _, git, calls = course_remote
    destination = tmp_path / "PhysicsNeMo"
    bootstrap.prepare_checkout(REPOSITORY_URL, "main", destination)
    (destination / "lesson.py").write_text("STEPS = 20000\n# Learner changes\n")
    output = destination / "outputs" / "my-training" / "metrics.json"
    output.parent.mkdir(parents=True)
    output.write_text('{"loss": 0.125}\n')
    notebook = destination / "Start_Here.ipynb"
    notebook.write_text('{"fixture": 1, "learner_notes": "Keep these"}\n')
    previous_status = git("status", "--porcelain", cwd=destination).stdout
    previous_head = git("rev-parse", "HEAD", cwd=destination).stdout
    calls.clear()

    checkout = bootstrap.prepare_checkout(REPOSITORY_URL, "main", destination)

    assert checkout == destination
    assert "20000" in (destination / "lesson.py").read_text()
    assert "Keep these" in notebook.read_text()
    assert output.read_text() == '{"loss": 0.125}\n'
    assert git("status", "--porcelain", cwd=destination).stdout == previous_status
    assert git("rev-parse", "HEAD", cwd=destination).stdout == previous_head
    assert not any("clone" in command or "fetch" in command for command in calls)


def test_update_creates_new_checkout_and_preserves_old_work(course_remote, tmp_path):
    remote, git, _ = course_remote
    destination = tmp_path / "PhysicsNeMo"
    bootstrap.prepare_checkout(REPOSITORY_URL, "main", destination)
    learner_file = destination / "lesson.py"
    learner_file.write_text("STEPS = 20000\n")
    output = destination / "outputs" / "result.json"
    output.parent.mkdir()
    output.write_text('{"learner": true}\n')
    (remote / "lesson.py").write_text("STEPS = 300\n")
    git("add", "lesson.py")
    git("commit", "-m", "Updated instructor lesson")

    updated = bootstrap.prepare_checkout(REPOSITORY_URL, "main", destination, update=True)

    assert updated != destination
    assert updated.parent == destination.with_name(destination.name + "-updates")
    assert (updated / "lesson.py").read_text() == "STEPS = 300\n"
    assert learner_file.read_text() == "STEPS = 20000\n"
    assert output.read_text() == '{"learner": true}\n'
    assert not (updated / "outputs").exists()


@pytest.mark.parametrize("url", [
    "http://github.com/example/course.git",
    "git@github.com:example/course.git",
    "file:///tmp/course",
    "--upload-pack=bad",
    "https://user:secret@github.com/example/course.git",
    "https://github.com/example/course.git?token=secret",
    "https://github.com/example/course.git#main",
    "https://github.com.evil.example/example/course.git",
    "https://github.com/example/course.git\n",
    "https://github.com/example/course.git extra",
    "https://github.com/example/course.git/../../elsewhere",
])
def test_unsafe_repository_rejected_before_commands(monkeypatch, tmp_path, url):
    def reject_commands(*_args, **_kwargs):
        pytest.fail("Invalid repository input reached a subprocess.")

    monkeypatch.setattr(subprocess, "run", reject_commands)
    with pytest.raises(ValueError):
        bootstrap.prepare_checkout(url, "main", tmp_path / "course")


@pytest.mark.parametrize("reference", ["--upload-pack=bad", "main\nnext", "main branch", "../outside"])
def test_unsafe_ref_rejected_before_commands(monkeypatch, tmp_path, reference):
    def reject_commands(*_args, **_kwargs):
        pytest.fail("Invalid ref input reached a subprocess.")

    monkeypatch.setattr(subprocess, "run", reject_commands)
    with pytest.raises(ValueError):
        bootstrap.prepare_checkout(REPOSITORY_URL, reference, tmp_path / "course")


def test_non_repository_directory_is_not_replaced(tmp_path):
    destination = tmp_path / "course"
    destination.mkdir()
    existing = destination / "learner.ipynb"
    existing.write_text("Saved work\n")
    with pytest.raises((ValueError, RuntimeError)):
        bootstrap.prepare_checkout(REPOSITORY_URL, "main", destination)
    assert existing.read_text() == "Saved work\n"


def test_mismatched_existing_origin_is_not_repurposed(course_remote, tmp_path):
    _, git, calls = course_remote
    destination = tmp_path / "PhysicsNeMo"
    bootstrap.prepare_checkout(REPOSITORY_URL, "main", destination)
    other_url = "https://github.com/another-owner/another-course.git"
    git("remote", "set-url", "origin", other_url, cwd=destination)
    calls.clear()
    with pytest.raises((ValueError, RuntimeError)):
        bootstrap.prepare_checkout(REPOSITORY_URL, "main", destination)
    assert git("remote", "get-url", "origin", cwd=destination).stdout.strip() == other_url
    assert not any("clone" in command for command in calls)


def test_requested_revision_can_be_pinned_and_updates_do_not_collide(course_remote, tmp_path):
    remote, git, _ = course_remote
    initial_revision = git("rev-parse", "HEAD").stdout.strip()
    (remote / "lesson.py").write_text("STEPS = 500\n")
    git("add", "lesson.py")
    git("commit", "-m", "Newer main fixture")
    destination = tmp_path / "PhysicsNeMo"
    checkout = bootstrap.prepare_checkout(REPOSITORY_URL, initial_revision, destination)
    assert git("rev-parse", "HEAD", cwd=checkout).stdout.strip() == initial_revision
    assert (checkout / "lesson.py").read_text() == "STEPS = 200\n"
    first = bootstrap.prepare_checkout(REPOSITORY_URL, "main", destination, update=True)
    second = bootstrap.prepare_checkout(REPOSITORY_URL, "main", destination, update=True)
    assert first != second
    assert (first / "lesson.py").read_text() == (second / "lesson.py").read_text() == "STEPS = 500\n"


def test_symlink_destination_is_not_followed(tmp_path, monkeypatch):
    original = tmp_path / "existing-course"
    original.mkdir()
    (original / "notes.txt").write_text("Keep notes\n")
    destination = tmp_path / "course-link"
    destination.symlink_to(original, target_is_directory=True)
    monkeypatch.setattr(subprocess, "run", lambda *_args, **_kwargs: pytest.fail("Must not follow a destination symlink."))
    with pytest.raises(ValueError, match="symlink"):
        bootstrap.prepare_checkout(REPOSITORY_URL, "main", destination)
    assert (original / "notes.txt").read_text() == "Keep notes\n"


@pytest.mark.parametrize("reference,update", [("main", False), ("event-2026", True), ("abcd1234", True)])
def test_bootstrap_forces_separate_checkout_for_non_main_ref(tmp_path, monkeypatch, reference, update):
    destination = tmp_path / "course"
    calls = []
    commands = []

    def prepare(repo, ref, directory, create_update):
        calls.append((repo, ref, directory, create_update))
        return directory

    monkeypatch.setattr(bootstrap, "prepare_checkout", prepare)
    monkeypatch.setattr(bootstrap.os, "geteuid", lambda: 1000)
    monkeypatch.setattr(bootstrap.sys, "argv", ["bootstrap.py", "--repo", REPOSITORY_URL,
                                               "--ref", reference, "--destination", str(destination)])
    monkeypatch.setattr(subprocess, "run", lambda command, **kwargs: commands.append(command))
    bootstrap.main()
    assert calls == [(REPOSITORY_URL, reference, destination, update)]
    assert len(commands) == 1
    assert commands[0][1:] == [str(destination / "ETC/launchable/install.py"), "--course-dir", str(destination)]


@pytest.fixture
def installer_workspace(tmp_path):
    course = tmp_path / "course"
    lock = course / "ETC/launchable" / install.LOCK_NAME
    lock.parent.mkdir(parents=True)
    lock.write_text("nvidia-physicsnemo==2.2.2\ntorch==2.10.0+cu128\n")
    home = tmp_path / "learner"
    home.mkdir()
    return course, home, lock


def test_environment_key_depends_on_locked_packages_and_python(installer_workspace, monkeypatch):
    _, _, lock = installer_workspace
    original = install.environment_key(lock)
    assert original == install.environment_key(lock)
    lock.write_text(lock.read_text() + "ipywidgets==8.1.9\n")
    assert install.environment_key(lock) != original
    changed_lock = install.environment_key(lock)
    monkeypatch.setattr(install, "PYTHON_VERSION", "3.12.12")
    assert install.environment_key(lock) != changed_lock


def test_uv_installs_only_to_its_versioned_private_target(tmp_path, monkeypatch):
    commands = []
    environments = []

    def standalone_run(*args, **kwargs):
        command = [str(arg) for arg in args]
        commands.append(command)
        assert "pip" not in command and install.sys.executable not in command
        if command[0] == "sh":
            environment = kwargs["env"]
            environments.append(environment)
            binary = Path(environment["UV_UNMANAGED_INSTALL"]) / "uv"
            binary.parent.mkdir(parents=True)
            binary.write_text("# Standalone uv fixture\n")

    # Even an inherited installation directory must not redirect the install.
    monkeypatch.setenv("UV_INSTALL_DIR", str(tmp_path / "unrelated-bin"))
    monkeypatch.setattr(install, "run", standalone_run)
    monkeypatch.setattr(subprocess, "check_output", lambda *args, **kwargs: "uv 0.8.17 (fixture)\n")
    binary = install.ensure_uv(tmp_path)
    assert binary == tmp_path / ".local/share/ai4sci-tools/uv-0.8.17/bin/uv"
    assert len(commands) == 2
    assert commands[0][0] == "curl" and commands[1][0] == "sh"
    assert "https://astral.sh/uv/0.8.17/install.sh" in commands[0]
    assert environments[0]["UV_INSTALL_DIR"] == environments[0]["UV_UNMANAGED_INSTALL"]
    assert environments[0]["UV_NO_MODIFY_PATH"] == "1"
    assert not (tmp_path / "unrelated-bin").exists()
    assert binary.read_text() == "# Standalone uv fixture\n"
    assert not list(binary.parent.parent.parent.glob(".uv-install-*"))
    commands.clear()
    assert install.ensure_uv(tmp_path) == binary
    assert commands == []


@pytest.mark.parametrize("failed_step", ["curl", "sh", "version"])
def test_failed_standalone_uv_install_can_retry_without_partial_target(tmp_path, monkeypatch, failed_step):
    def failing_run(*args, **kwargs):
        if args[0] == "sh":
            binary = Path(kwargs["env"]["UV_UNMANAGED_INSTALL"]) / "uv"
            binary.parent.mkdir(parents=True)
            binary.write_text("# Partially installed fixture\n")
        if args[0] == failed_step:
            raise subprocess.CalledProcessError(1, list(args))

    monkeypatch.setattr(install, "run", failing_run)
    monkeypatch.setattr(subprocess, "check_output", lambda *args, **kwargs: "uv 0.1.0\n")
    with pytest.raises((subprocess.CalledProcessError, ValueError)):
        install.ensure_uv(tmp_path)
    tools = tmp_path / ".local/share/ai4sci-tools"
    assert not (tools / "uv-0.8.17").exists()
    assert not list(tools.glob(".uv-install-*"))


def test_unrecognized_partial_uv_directory_is_preserved(tmp_path, monkeypatch):
    tools = tmp_path / ".local/share/ai4sci-tools/uv-0.8.17"
    tools.mkdir(parents=True)
    note = tools / "notes.txt"
    note.write_text("Keep me\n")
    monkeypatch.setattr(install, "run", lambda *_args, **_kwargs: pytest.fail("Unknown tool directory must not be overwritten."))
    with pytest.raises(ValueError, match="incomplete tool directory"):
        install.ensure_uv(tmp_path)
    assert note.read_text() == "Keep me\n"


def test_uv_existing_wrong_version_is_rejected(tmp_path, monkeypatch):
    binary = tmp_path / ".local/share/ai4sci-tools/uv-0.8.17/bin/uv"
    binary.parent.mkdir(parents=True)
    binary.write_text("# Existing tool\n")
    monkeypatch.setattr(install, "run", lambda *_args, **_kwargs: pytest.fail("Existing private uv must not be overwritten."))
    monkeypatch.setattr(subprocess, "check_output", lambda *args, **kwargs: "uv 0.1.0\n")
    with pytest.raises(ValueError, match="version"):
        install.ensure_uv(tmp_path)
    assert binary.read_text() == "# Existing tool\n"


def test_new_environment_uses_private_prefix_pinned_cuda_and_import_check(installer_workspace, monkeypatch):
    course, home, lock = installer_workspace
    private_uv = home / ".local/share/ai4sci-tools/uv-test/bin/uv"
    monkeypatch.setattr(install, "ensure_uv", lambda actual_home: private_uv)
    commands = []

    def record_run(*args, **kwargs):
        args = [str(arg) for arg in args]
        commands.append(args)
        assert not kwargs.get("shell")
        if "venv" in args:
            python = Path(args[-1]) / "bin/python"
            python.parent.mkdir(parents=True)
            python.write_text("# Mock Python; never executed\n")

    monkeypatch.setattr(install, "run", record_run)
    prefix = install.ensure_environment(course, home)
    python = prefix / "bin/python"
    assert prefix.parent == home / ".venvs"
    assert prefix.name == "ai4sci-brev-" + install.environment_key(lock)
    assert [str(private_uv), "venv", "--python", "3.12.11", str(prefix)] in commands
    cuda_install = next(command for command in commands if "torch==2.10.0+cu128" in command)
    assert "torchvision==0.25.0+cu128" in cuda_install
    assert "https://download.pytorch.org/whl/cu128" in cuda_install
    locked_install = next(command for command in commands if "-r" in command)
    assert "--no-deps" in locked_install and str(lock) in locked_install
    for command in commands:
        if "pip" in command:
            assert command[command.index("--python") + 1] == str(python)
        assert not any(arg in command for arg in ("jupyter", "lab", "notebook", "--system", "sudo"))
    assert [str(python), str(course / "ETC/launchable/verify.py")] in commands
    marker = json.loads((prefix / ".ai4sci-environment.json").read_text())
    assert marker["key"] == install.environment_key(lock)
    assert marker["python"] == "3.12.11"
    assert marker["status"] == "ready"


def test_interrupted_owned_install_resumes_packages_without_recreating_venv(installer_workspace, monkeypatch):
    course, home, lock = installer_workspace
    private_uv = home / ".local/share/ai4sci-tools/uv-test/bin/uv"
    monkeypatch.setattr(install, "ensure_uv", lambda actual_home: private_uv)
    prefix = home / ".venvs" / ("ai4sci-brev-" + install.environment_key(lock))
    marker = prefix / ".ai4sci-environment.json"
    commands = []
    fail_download = True

    def record_run(*args, **kwargs):
        command = [str(arg) for arg in args]
        commands.append(command)
        if "venv" in command:
            python = Path(command[-1]) / "bin/python"
            python.parent.mkdir(parents=True)
            python.write_text("# Preserve this interpreter across retry\n")
        if "pip" in command and "install" in command:
            assert json.loads(marker.read_text())["status"] == "installing"
            if fail_download:
                raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(install, "run", record_run)
    with pytest.raises(subprocess.CalledProcessError):
        install.ensure_environment(course, home)
    assert json.loads(marker.read_text())["status"] == "installing"
    assert (prefix / "bin/python").read_text() == "# Preserve this interpreter across retry\n"
    commands.clear()
    fail_download = False

    assert install.ensure_environment(course, home) == prefix
    assert not any("venv" in command for command in commands)
    assert sum("install" in command and "pip" in command for command in commands) == 2
    assert json.loads(marker.read_text())["status"] == "ready"
    assert (prefix / "bin/python").read_text() == "# Preserve this interpreter across retry\n"


def test_matching_environment_with_unrecognized_state_is_not_resumed(installer_workspace, monkeypatch):
    course, home, lock = installer_workspace
    prefix = home / ".venvs" / ("ai4sci-brev-" + install.environment_key(lock))
    (prefix / "bin").mkdir(parents=True)
    python = prefix / "bin/python"
    python.write_text("# Not owned by this installer\n")
    marker = prefix / ".ai4sci-environment.json"
    original_marker = json.dumps({"key": install.environment_key(lock), "status": "other-tool"})
    marker.write_text(original_marker)
    monkeypatch.setattr(install, "run", lambda *_args, **_kwargs: pytest.fail("Unknown environment status must not trigger installation."))
    with pytest.raises(ValueError, match="not overwritten"):
        install.ensure_environment(course, home)
    assert marker.read_text() == original_marker
    assert python.read_text() == "# Not owned by this installer\n"


def test_verified_environment_reused_without_installs(installer_workspace, monkeypatch):
    course, home, lock = installer_workspace
    prefix = home / ".venvs" / ("ai4sci-brev-" + install.environment_key(lock))
    (prefix / "bin").mkdir(parents=True)
    (prefix / "bin/python").write_text("# Fixture\n")
    (prefix / ".ai4sci-environment.json").write_text(json.dumps({"key": install.environment_key(lock)}))
    commands = []
    monkeypatch.setattr(install, "run", lambda *args, **kwargs: commands.append([str(arg) for arg in args]))
    monkeypatch.setattr(install, "ensure_uv", lambda _: pytest.fail("A valid environment must not install uv."))

    assert install.ensure_environment(course, home) == prefix
    assert commands == [[str(prefix / "bin/python"), str(course / "ETC/launchable/verify.py")]]


@pytest.mark.parametrize("marker", [None, {"key": "wrong"}])
def test_incomplete_environment_not_overwritten(installer_workspace, monkeypatch, marker):
    course, home, lock = installer_workspace
    prefix = home / ".venvs" / ("ai4sci-brev-" + install.environment_key(lock))
    prefix.mkdir(parents=True)
    note = prefix / "existing-work.txt"
    note.write_text("Keep me\n")
    if marker is not None:
        (prefix / ".ai4sci-environment.json").write_text(json.dumps(marker))
    monkeypatch.setattr(install, "run", lambda *_args, **_kwargs: pytest.fail("Must not repair by replacing an existing environment."))
    with pytest.raises(ValueError, match="not overwritten"):
        install.ensure_environment(course, home)
    assert note.read_text() == "Keep me\n"


def test_kernel_has_unique_name_and_preserves_managed_python_kernel(tmp_path, monkeypatch):
    data = tmp_path / "jupyter-data"
    managed = data / "kernels/python3/kernel.json"
    managed.parent.mkdir(parents=True)
    original_managed = '{"argv": ["/managed/python", "-m", "ipykernel_launcher"], "display_name": "Managed Python"}\n'
    managed.write_text(original_managed)
    prefix = tmp_path / ".venvs/ai4sci-brev-fixture"
    extension = prefix / "share/jupyter/labextensions/@jupyter-widgets/jupyterlab-manager"
    extension.mkdir(parents=True)
    (extension / "package.json").write_text('{"version": "5.0.15"}\n')
    monkeypatch.setattr(subprocess, "check_output", lambda *args, **kwargs: str(data) + "\n")
    commands = []
    monkeypatch.setattr(install, "run", lambda *args, **kwargs: commands.append([str(arg) for arg in args]))

    install.connect_kernel(prefix)

    assert len(commands) == 1
    command = commands[0]
    assert command[:4] == [str(prefix / "bin/python"), "-m", "ipykernel", "install"]
    assert command[command.index("--name") + 1] == "ai4sci-physicsnemo-uv"
    assert command[command.index("--env") + 1:command.index("--env") + 3] == ["AI4SCI_DEVICE", "cuda"]
    assert "--user" in command and "python3" not in command
    assert managed.read_text() == original_managed
    target = data / "labextensions/@jupyter-widgets/jupyterlab-manager"
    assert target.is_symlink() and target.resolve() == extension


def test_unrelated_kernel_with_same_name_is_not_replaced(tmp_path, monkeypatch):
    data = tmp_path / "jupyter-data"
    spec = data / "kernels" / install.KERNEL_NAME / "kernel.json"
    spec.parent.mkdir(parents=True)
    original = '{"argv": ["/other/environment/bin/python"]}\n'
    spec.write_text(original)
    monkeypatch.setattr(subprocess, "check_output", lambda *args, **kwargs: str(data) + "\n")
    monkeypatch.setattr(install, "run", lambda *_args, **_kwargs: pytest.fail("Existing unrelated kernel must not be replaced."))
    with pytest.raises(ValueError, match="unrelated"):
        install.connect_kernel(tmp_path / ".venvs/ai4sci-brev-fixture")
    assert spec.read_text() == original


def test_existing_widgets_frontend_conflict_is_not_overwritten(tmp_path, monkeypatch):
    data = tmp_path / "jupyter-data"
    target = data / "labextensions/@jupyter-widgets/jupyterlab-manager"
    target.mkdir(parents=True)
    (target / "package.json").write_text('{"version": "old"}\n')
    prefix = tmp_path / ".venvs/ai4sci-brev-fixture"
    extension = prefix / "share/jupyter/labextensions/@jupyter-widgets/jupyterlab-manager"
    extension.mkdir(parents=True)
    (extension / "package.json").write_text('{"version": "new"}\n')
    monkeypatch.setattr(subprocess, "check_output", lambda *args, **kwargs: str(data) + "\n")
    monkeypatch.setattr(install, "run", lambda *_args, **_kwargs: pytest.fail("Widget preflight must finish before kernel installation."))
    with pytest.raises(ValueError, match="different version"):
        install.connect_kernel(prefix)
    assert (target / "package.json").read_text() == '{"version": "old"}\n'
    assert not target.is_symlink()


def test_missing_widgets_frontend_prevents_kernel_installation(tmp_path, monkeypatch):
    data = tmp_path / "jupyter-data"
    monkeypatch.setattr(subprocess, "check_output", lambda *args, **kwargs: str(data) + "\n")
    monkeypatch.setattr(install, "run", lambda *_args, **_kwargs: pytest.fail("Missing widgets must be detected before changing kernels."))
    with pytest.raises(ValueError, match="missing"):
        install.connect_kernel(tmp_path / ".venvs/ai4sci-brev-fixture")
    assert not data.exists()


def test_committed_lock_has_core_cuda_and_widget_pins():
    lock = Path(install.__file__).with_name(install.LOCK_NAME)
    requirements = [line for line in lock.read_text().splitlines() if line and not line.startswith("#")]
    assert requirements and all("==" in line and " @ " not in line for line in requirements)
    assert len(requirements) == len(set(line.split("==")[0] for line in requirements))
    for requirement in ("torch==2.10.0+cu128", "torchvision==0.25.0+cu128",
                        "nvidia-physicsnemo==2.2.2", "ipywidgets==8.1.9"):
        assert requirement in requirements
