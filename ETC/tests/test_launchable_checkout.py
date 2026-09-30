"""The shared Launchable safely maintains one canonical course checkout.

All Git traffic is redirected to a temporary local repository. These tests never
download packages, allocate a GPU, or start a notebook server.
"""
from pathlib import Path
import json
import subprocess

import pytest

from ETC.launchable import bootstrap
from ETC.launchable import update as course_update


REPOSITORY_URL = "https://github.com/example/physicsnemo-course.git"
CHECKOUT_ERRORS = (ValueError, RuntimeError, subprocess.CalledProcessError)


def test_existing_setup_loads_safe_updater_from_fetched_release(launchable_remote, tmp_path, monkeypatch):
    remote, git, _ = launchable_remote
    destination = tmp_path / "workspace/PhysicsNeMo"
    notebook = {"nbformat": 4, "nbformat_minor": 5, "metadata": {}, "cells": [
        {"cell_type": "code", "metadata": {}, "id": "c1", "source": ["print(1)\n"],
         "outputs": [], "execution_count": None}]}
    (remote / "Start_Here.ipynb").write_text(json.dumps(notebook))
    git("add", "Start_Here.ipynb")
    git("commit", "-m", "Real notebook fixture")
    bootstrap.prepare_launchable_checkout(REPOSITORY_URL, "main", destination)
    saved = json.loads(json.dumps(notebook))
    saved["cells"][0]["execution_count"] = 1
    saved["cells"][0]["outputs"] = [{"output_type": "stream", "name": "stdout", "text": "1\n"}]
    saved_bytes = json.dumps(saved).encode()
    (destination / "Start_Here.ipynb").write_bytes(saved_bytes)
    (destination / "my_answers.py").write_text("answer = 42\n")
    (remote / "ETC/launchable/update.py").write_bytes(Path(course_update.__file__).read_bytes())
    notebook["cells"][0]["source"] = ["print(2)\n"]
    (remote / "Start_Here.ipynb").write_text(json.dumps(notebook))
    git("add", ".")
    git("commit", "-m", "Release with safe updater")
    target = git("rev-parse", "HEAD").stdout.strip()
    monkeypatch.setattr(bootstrap, "preflight_managed_update", lambda *args: None)

    bootstrap.prepare_launchable_checkout(REPOSITORY_URL, "main", destination)

    assert bootstrap.git(destination, "rev-parse", "HEAD") == target
    assert json.loads((destination / "Start_Here.ipynb").read_text()) == notebook
    assert (destination / "my_answers.py").read_text() == "answer = 42\n"
    backups = list((destination.parent / ".ai4sci-course-backups").glob("*/files/Start_Here.ipynb"))
    assert len(backups) == 1 and backups[0].read_bytes() == saved_bytes


@pytest.fixture
def launchable_remote(tmp_path, monkeypatch):
    """Use file transport so shallow clones and ancestry checks are real."""
    real_run = subprocess.run
    remote = tmp_path / "fixture-remote"
    remote.mkdir()

    def git(*args, cwd=remote):
        return real_run(["git", *args], cwd=cwd, check=True, text=True,
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    git("init", "-b", "main")
    git("config", "user.name", "Launchable fixture")
    git("config", "user.email", "fixture@example.invalid")
    (remote / ".gitignore").write_text("outputs/\n")
    (remote / "Start_Here.ipynb").write_text('{"fixture": 1}\n')
    (remote / "lesson.py").write_text("STEPS = 200\n")
    (remote / "ETC/launchable").mkdir(parents=True)
    (remote / "ETC/launchable/install.py").write_text("# Installer fixture\n")
    git("add", ".")
    git("commit", "-m", "Initial course fixture")
    calls = []

    def local_run(command, *args, **kwargs):
        command = [str(argument) for argument in command]
        calls.append(command)
        if Path(command[0]).name == "git":
            assert not any(word in command for word in ("pull", "reset", "clean")), command
            if "clone" in command:
                assert REPOSITORY_URL in command, "Tests must never contact a real Git server."
                mapped = [remote.as_uri() if word == REPOSITORY_URL else word for word in command]
                result = real_run(mapped, *args, **kwargs)
                if result.returncode == 0:
                    git("remote", "set-url", "origin", REPOSITORY_URL, cwd=Path(command[-1]))
                return result
            if "fetch" in command:
                assert "origin" in command, "All fetches must use the verified origin."
                mapped = [remote.as_uri() if word == "origin" else word for word in command]
                return real_run(mapped, *args, **kwargs)
        return real_run(command, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", local_run)
    return remote, git, calls


def update_lesson(remote, git):
    (remote / "lesson.py").write_text("STEPS = 300\n")
    git("add", "lesson.py")
    git("commit", "-m", "Updated instructor lesson")
    return git("rev-parse", "HEAD").stdout.strip()


def worktree_files(destination):
    return {path.relative_to(destination): path.read_bytes()
            for path in destination.rglob("*")
            if path.is_file() and ".git" not in path.relative_to(destination).parts}


def test_first_launch_publishes_selected_revision_at_canonical_path(launchable_remote, tmp_path):
    remote, git, calls = launchable_remote
    selected = git("rev-parse", "HEAD").stdout.strip()
    update_lesson(remote, git)
    destination = tmp_path / "workspace" / "PhysicsNeMo"

    checkout = bootstrap.prepare_launchable_checkout(REPOSITORY_URL, selected, destination)

    assert checkout == destination
    assert git("rev-parse", "HEAD", cwd=checkout).stdout.strip() == selected
    assert git("remote", "get-url", "origin", cwd=checkout).stdout.strip() == REPOSITORY_URL
    assert (checkout / "Start_Here.ipynb").is_file()
    assert (checkout / "lesson.py").read_text() == "STEPS = 200\n"
    clone = next(command for command in calls if "clone" in command)
    assert Path(clone[-1]) != destination, "First installs must be prepared before publication."
    assert set(destination.parent.iterdir()) == {destination}


def test_clean_checkout_advances_in_place_and_retains_ignored_outputs(launchable_remote, tmp_path):
    remote, git, calls = launchable_remote
    destination = tmp_path / "workspace" / "PhysicsNeMo"
    bootstrap.prepare_launchable_checkout(REPOSITORY_URL, "main", destination)
    output = destination / "outputs" / "training" / "metrics.json"
    output.parent.mkdir(parents=True)
    output.write_text('{"loss": 0.125}\n')
    selected = update_lesson(remote, git)
    calls.clear()

    assert bootstrap.prepare_launchable_checkout(REPOSITORY_URL, "main", destination) == destination
    assert git("rev-parse", "HEAD", cwd=destination).stdout.strip() == selected
    assert (destination / "lesson.py").read_text() == "STEPS = 300\n"
    assert output.read_text() == '{"loss": 0.125}\n'
    checkouts = [command for command in calls if "checkout" in command]
    assert len(checkouts) == 1
    assert all(flag in checkouts[0] for flag in ("--detach", "--no-overwrite-ignore"))
    assert selected in checkouts[0] or "FETCH_HEAD" in checkouts[0]
    assert not any(flag in checkouts[0] for flag in ("--force", "-f"))
    assert not any("clone" in command for command in calls)
    assert set(destination.parent.iterdir()) == {destination}


def test_repeated_launch_does_not_create_additional_course_folders(launchable_remote, tmp_path):
    _, git, calls = launchable_remote
    destination = tmp_path / "workspace" / "PhysicsNeMo"
    bootstrap.prepare_launchable_checkout(REPOSITORY_URL, "main", destination)
    original_head = git("rev-parse", "HEAD", cwd=destination).stdout
    calls.clear()

    for _ in range(3):
        assert bootstrap.prepare_launchable_checkout(REPOSITORY_URL, "main", destination) == destination

    assert git("rev-parse", "HEAD", cwd=destination).stdout == original_head
    assert set(destination.parent.iterdir()) == {destination}
    assert not any("clone" in command or "checkout" in command for command in calls)
    assert sum("fetch" in command for command in calls) >= 3


def test_unchanged_source_preserves_student_answers_and_outputs(launchable_remote, tmp_path):
    _, git, calls = launchable_remote
    destination = tmp_path / "PhysicsNeMo"
    bootstrap.prepare_launchable_checkout(REPOSITORY_URL, "main", destination)
    (destination / "lesson.py").write_text("STEPS = 20000\n# Student answer\n")
    (destination / "Start_Here.ipynb").write_text('{"student_notes": "Keep these"}\n')
    (destination / "my_answers.py").write_text("ANSWER = 42\n")
    (destination / "outputs").mkdir()
    (destination / "outputs" / "result.json").write_text('{"student": true}\n')
    git("add", "lesson.py", cwd=destination)
    before = worktree_files(destination)
    status = git("status", "--porcelain", cwd=destination).stdout
    original_head = git("rev-parse", "HEAD", cwd=destination).stdout
    calls.clear()

    assert bootstrap.prepare_launchable_checkout(REPOSITORY_URL, "main", destination) == destination

    assert worktree_files(destination) == before
    assert git("status", "--porcelain", cwd=destination).stdout == status
    assert git("rev-parse", "HEAD", cwd=destination).stdout == original_head
    assert any("fetch" in command and "main" in command for command in calls)
    assert not any("clone" in command or "checkout" in command for command in calls)


def test_annotated_tag_at_current_commit_preserves_student_answers(launchable_remote, tmp_path):
    _, git, calls = launchable_remote
    git("tag", "-a", "event-release", "-m", "Pinned course release")
    destination = tmp_path / "PhysicsNeMo"
    bootstrap.prepare_launchable_checkout(REPOSITORY_URL, "event-release", destination)
    (destination / "lesson.py").write_text("# Student answer\n")
    before = worktree_files(destination)
    original_head = git("rev-parse", "HEAD", cwd=destination).stdout
    calls.clear()

    assert bootstrap.prepare_launchable_checkout(REPOSITORY_URL, "event-release", destination) == destination

    assert worktree_files(destination) == before
    assert git("rev-parse", "HEAD", cwd=destination).stdout == original_head
    assert not any("checkout" in command for command in calls)


@pytest.mark.parametrize("change_kind", ["tracked", "staged", "untracked"])
@pytest.mark.parametrize("reference", ["main", "event-release"])
def test_changed_source_refuses_dirty_work_before_checkout(launchable_remote, tmp_path, change_kind, reference):
    remote, git, calls = launchable_remote
    destination = tmp_path / "PhysicsNeMo"
    bootstrap.prepare_launchable_checkout(REPOSITORY_URL, "main", destination)
    learner_file = destination / ("my_answers.py" if change_kind == "untracked" else "lesson.py")
    learner_file.write_text("# Irreplaceable student work\n")
    if change_kind == "staged":
        git("add", "lesson.py", cwd=destination)
    update_lesson(remote, git)
    git("tag", "event-release")
    before = worktree_files(destination)
    status = git("status", "--porcelain", cwd=destination).stdout
    original_head = git("rev-parse", "HEAD", cwd=destination).stdout
    calls.clear()

    with pytest.raises(CHECKOUT_ERRORS):
        bootstrap.prepare_launchable_checkout(REPOSITORY_URL, reference, destination)

    assert worktree_files(destination) == before
    assert git("status", "--porcelain", cwd=destination).stdout == status
    assert git("rev-parse", "HEAD", cwd=destination).stdout == original_head
    assert not any("checkout" in command or "clone" in command for command in calls)


@pytest.mark.parametrize("advance_remote", [False, True])
def test_clean_local_commit_is_not_discarded(launchable_remote, tmp_path, advance_remote):
    remote, git, calls = launchable_remote
    destination = tmp_path / "PhysicsNeMo"
    bootstrap.prepare_launchable_checkout(REPOSITORY_URL, "main", destination)
    git("config", "user.name", "Student", cwd=destination)
    git("config", "user.email", "student@example.invalid", cwd=destination)
    (destination / "my_answers.py").write_text("# Committed student answer\n")
    git("add", "my_answers.py", cwd=destination)
    git("commit", "-m", "Student solution", cwd=destination)
    if advance_remote:
        update_lesson(remote, git)
    original_head = git("rev-parse", "HEAD", cwd=destination).stdout
    before = worktree_files(destination)
    assert not git("status", "--porcelain", cwd=destination).stdout
    calls.clear()

    with pytest.raises(CHECKOUT_ERRORS):
        bootstrap.prepare_launchable_checkout(REPOSITORY_URL, "main", destination)

    assert git("rev-parse", "HEAD", cwd=destination).stdout == original_head
    assert worktree_files(destination) == before
    assert not any("checkout" in command for command in calls)


def test_ignored_output_collision_cannot_be_overwritten(launchable_remote, tmp_path):
    remote, git, calls = launchable_remote
    destination = tmp_path / "PhysicsNeMo"
    bootstrap.prepare_launchable_checkout(REPOSITORY_URL, "main", destination)
    (destination / "outputs").mkdir()
    (destination / "outputs" / "result.json").write_text('{"student": true}\n')
    (remote / "outputs").mkdir()
    (remote / "outputs" / "result.json").write_text('{"instructor": true}\n')
    git("add", "--force", "outputs/result.json")
    git("commit", "-m", "Track a path previously used for student output")
    before = worktree_files(destination)
    original_head = git("rev-parse", "HEAD", cwd=destination).stdout
    assert not git("status", "--porcelain", cwd=destination).stdout
    calls.clear()

    with pytest.raises(CHECKOUT_ERRORS):
        bootstrap.prepare_launchable_checkout(REPOSITORY_URL, "main", destination)

    assert worktree_files(destination) == before
    assert git("rev-parse", "HEAD", cwd=destination).stdout == original_head
    checkouts = [command for command in calls if "checkout" in command]
    assert all("--no-overwrite-ignore" in command for command in checkouts)


@pytest.mark.parametrize("failed_step", ["clone", "fetch", "checkout"])
def test_failed_first_install_does_not_poison_canonical_directory(launchable_remote, tmp_path, monkeypatch, failed_step):
    _, _, calls = launchable_remote
    destination = tmp_path / "workspace" / "PhysicsNeMo"
    local_run = subprocess.run

    def interrupted_run(command, *args, **kwargs):
        result = local_run(command, *args, **kwargs)
        if failed_step in command:
            raise subprocess.CalledProcessError(1, command)
        return result

    with monkeypatch.context() as interruption:
        interruption.setattr(subprocess, "run", interrupted_run)
        with pytest.raises(CHECKOUT_ERRORS):
            bootstrap.prepare_launchable_checkout(REPOSITORY_URL, "main", destination)

    assert not destination.exists()
    assert not destination.parent.exists() or not list(destination.parent.iterdir())
    assert any(failed_step in command for command in calls)
    assert bootstrap.prepare_launchable_checkout(REPOSITORY_URL, "main", destination) == destination
    assert (destination / "Start_Here.ipynb").is_file()


def test_invalid_source_checkout_is_not_published(launchable_remote, tmp_path):
    remote, git, _ = launchable_remote
    git("rm", "ETC/launchable/install.py")
    git("commit", "-m", "Fixture without the installer")
    destination = tmp_path / "workspace" / "PhysicsNeMo"

    with pytest.raises(CHECKOUT_ERRORS):
        bootstrap.prepare_launchable_checkout(REPOSITORY_URL, "main", destination)

    assert not destination.exists()
    assert not destination.parent.exists() or not list(destination.parent.iterdir())
    assert not (remote / "ETC/launchable/install.py").exists()


def test_existing_other_origin_is_preserved_without_fetch(launchable_remote, tmp_path):
    _, git, calls = launchable_remote
    destination = tmp_path / "PhysicsNeMo"
    bootstrap.prepare_launchable_checkout(REPOSITORY_URL, "main", destination)
    other_url = "https://github.com/another-owner/another-course.git"
    git("remote", "set-url", "origin", other_url, cwd=destination)
    before = worktree_files(destination)
    calls.clear()

    with pytest.raises(CHECKOUT_ERRORS):
        bootstrap.prepare_launchable_checkout(REPOSITORY_URL, "main", destination)

    assert git("remote", "get-url", "origin", cwd=destination).stdout.strip() == other_url
    assert worktree_files(destination) == before
    assert not any("fetch" in command or "checkout" in command or "clone" in command for command in calls)


def test_existing_non_repository_directory_is_preserved(tmp_path, monkeypatch):
    destination = tmp_path / "PhysicsNeMo"
    destination.mkdir()
    (destination / "notes.txt").write_text("Existing student work\n")
    monkeypatch.setattr(subprocess, "run", lambda *_args, **_kwargs: pytest.fail("Must not modify a non-repository directory."))

    with pytest.raises(CHECKOUT_ERRORS):
        bootstrap.prepare_launchable_checkout(REPOSITORY_URL, "main", destination)

    assert (destination / "notes.txt").read_text() == "Existing student work\n"
    assert set(destination.iterdir()) == {destination / "notes.txt"}


def test_symlink_destination_is_rejected_before_commands(tmp_path, monkeypatch):
    original = tmp_path / "student-work"
    original.mkdir()
    (original / "notes.txt").write_text("Keep notes\n")
    destination = tmp_path / "PhysicsNeMo"
    destination.symlink_to(original, target_is_directory=True)
    monkeypatch.setattr(subprocess, "run", lambda *_args, **_kwargs: pytest.fail("Must not follow a destination symlink."))

    with pytest.raises(ValueError, match="symlink"):
        bootstrap.prepare_launchable_checkout(REPOSITORY_URL, "main", destination)

    assert (original / "notes.txt").read_text() == "Keep notes\n"


@pytest.mark.parametrize("repo,reference", [
    ("file:///tmp/course", "main"),
    ("https://user:secret@github.com/example/course.git", "main"),
    ("https://github.com/example/course.git?token=secret", "main"),
    (REPOSITORY_URL, "--upload-pack=bad"),
    (REPOSITORY_URL, "main\nnext"),
    (REPOSITORY_URL, "../outside"),
])
def test_invalid_source_is_rejected_before_commands(tmp_path, monkeypatch, repo, reference):
    monkeypatch.setattr(subprocess, "run", lambda *_args, **_kwargs: pytest.fail("Invalid source reached a command."))

    with pytest.raises(ValueError):
        bootstrap.prepare_launchable_checkout(repo, reference, tmp_path / "PhysicsNeMo")
