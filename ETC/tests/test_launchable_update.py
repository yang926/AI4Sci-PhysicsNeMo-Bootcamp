"""Local-only Git regression tests for live course source updates."""
import fcntl
import json
import os
from pathlib import Path
import stat
import subprocess

import pytest

from ETC.launchable import update


def notebook(source="print('lesson')", output=None):
    return {"nbformat": 4, "nbformat_minor": 5, "metadata": {}, "cells": [
        {"id": "teaching-cell", "cell_type": "code", "metadata": {},
         "source": source, "execution_count": 1 if output else None,
         "outputs": [{"output_type": "stream", "name": "stdout", "text": output}] if output else []}]}


@pytest.fixture
def course(tmp_path):
    remote = tmp_path / "upstream"
    local = tmp_path / "course"
    remote.mkdir()

    def git(*arguments, cwd=remote, check=True):
        environment = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
        return subprocess.run(["git", "-C", str(cwd), *arguments], capture_output=True,
                              text=True, check=check, env=environment)

    git("init", "-b", "main")
    git("config", "user.name", "Course fixture")
    git("config", "user.email", "test@example.invalid")
    (remote / "lesson.ipynb").write_text(json.dumps(notebook()))
    (remote / "lesson.py").write_text("VALUE = 1\n")
    (remote / "answer.py").write_text("# Write your answer\n")
    (remote / ".gitignore").write_text("outputs/\n.ipynb_checkpoints/\n")
    git("add", ".")
    git("commit", "-m", "Initial course")
    git("clone", str(remote), str(local))
    git("config", "user.name", "Course fixture", cwd=local)
    git("config", "user.email", "test@example.invalid", cwd=local)
    before = git("rev-parse", "HEAD", cwd=local).stdout.strip()

    def release(changes):
        for name, content in changes.items():
            path = remote / name
            if content is None:
                path.unlink()
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content)
        git("add", "--all")
        git("commit", "-m", "Release", "--allow-empty")
        git("fetch", "origin", cwd=local)
        return git("rev-parse", "HEAD").stdout.strip()

    return local, git, release, before


def saved_notebook(local, *, source="print('lesson')", output="student result"):
    content = json.dumps(notebook(source, output), indent=1).encode()
    (local / "lesson.ipynb").write_bytes(content)
    return content


def snapshot(local, git):
    return (git("rev-parse", "HEAD", cwd=local).stdout,
            git("status", "--porcelain", "--untracked-files=all", cwd=local).stdout,
            {str(path.relative_to(local)): path.read_bytes() for path in local.rglob("*")
             if path.is_file() and ".git" not in path.relative_to(local).parts})


def test_output_only_save_is_backed_up_then_replaced(course):
    local, git, release, before = course
    saved = saved_notebook(local)
    target_content = json.dumps(notebook("print('new lesson')"))
    target = release({"lesson.ipynb": target_content})

    result = update.update_checkout(local, target)

    assert result["before"] == before and result["after"] == target
    assert result["updated"] == ["lesson.ipynb"] and result["preserved"] == []
    assert (local / "lesson.ipynb").read_text() == target_content
    backup = Path(result["backup"])
    assert backup.is_relative_to(local.parent / ".ai4sci-course-backups")
    assert (backup / "files/lesson.ipynb").read_bytes() == saved
    manifest = json.loads((backup / "manifest.json").read_text())
    assert manifest["before"] == before and manifest["after"] == target
    assert manifest["status"] == "applied"
    assert manifest["files"][0]["path"] == "lesson.ipynb"
    assert stat.S_IMODE(backup.stat().st_mode) == 0o700
    assert stat.S_IMODE((backup / "files/lesson.ipynb").stat().st_mode) == 0o600
    assert not git("status", "--porcelain", cwd=local).stdout


@pytest.mark.parametrize("edit", ["source", "markdown", "attachments", "unknown_metadata", "cell_id", "staged_output"])
def test_substantive_or_staged_notebook_change_is_not_overwritten(course, edit):
    local, git, release, _ = course
    saved = notebook(output="result")
    if edit == "source":
        saved["cells"][0]["source"] = "print('my answer')"
    elif edit == "markdown":
        saved["cells"].append({"id": "notes", "cell_type": "markdown", "metadata": {}, "source": "My notes"})
    elif edit == "attachments":
        saved["cells"][0]["attachments"] = {"important.png": {"image/png": "image"}}
    elif edit == "unknown_metadata":
        saved["metadata"]["student"] = {"important": "notes"}
    elif edit == "cell_id":
        saved["cells"][0]["id"] = "my-own-cell"
    (local / "lesson.ipynb").write_text(json.dumps(saved))
    if edit == "staged_output":
        git("add", "lesson.ipynb", cwd=local)
    target = release({"lesson.ipynb": json.dumps(notebook("print('new lesson')"))})
    old = snapshot(local, git)

    with pytest.raises(update.UpdateError, match="lesson.ipynb"):
        update.update_checkout(local, target)

    assert snapshot(local, git) == old
    assert not (local.parent / ".ai4sci-course-backups").exists()


@pytest.mark.parametrize("staged", [False, True])
def test_unrelated_student_source_and_untracked_files_remain_in_place(course, staged):
    local, git, release, _ = course
    (local / "answer.py").write_text("my_answer = 42\n")
    (local / "personal.py").write_text("my_private_notes = True\n")
    if staged:
        git("add", "answer.py", "personal.py", cwd=local)
    target = release({"lesson.py": "VALUE = 2\n"})
    old_status = git("status", "--porcelain", cwd=local).stdout

    result = update.update_checkout(local, target)

    assert (local / "answer.py").read_text() == "my_answer = 42\n"
    assert (local / "personal.py").read_text() == "my_private_notes = True\n"
    assert result["preserved"] == ["answer.py", "personal.py"]
    assert result["backup"] is None
    assert git("status", "--porcelain", cwd=local).stdout == old_status


def test_disjoint_notebook_output_is_not_cleared(course):
    local, _, release, _ = course
    saved = saved_notebook(local)
    target = release({"lesson.py": "VALUE = 2\n"})
    result = update.update_checkout(local, target)
    assert result["preserved"] == ["lesson.ipynb"]
    assert result["backup"] is None
    assert (local / "lesson.ipynb").read_bytes() == saved


@pytest.mark.parametrize("name", ["personal.txt", "outputs/result.txt"])
@pytest.mark.parametrize("equal", [False, True])
def test_untracked_or_ignored_collision_never_adopts_or_overwrites_learner_file(course, name, equal):
    local, git, release, _ = course
    path = local / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("upstream content" if equal else "private student result")
    # Force an ignored upstream asset into its release.
    target = release({".gitignore": "", name: "upstream content"})
    old = snapshot(local, git)
    with pytest.raises(update.UpdateError, match=name):
        update.update_checkout(local, target)
    assert snapshot(local, git) == old


def test_ignored_outputs_not_in_release_are_preserved(course):
    local, _, release, _ = course
    (local / "outputs").mkdir()
    (local / "outputs/result.txt").write_text("my run")
    target = release({"lesson.py": "VALUE = 2\n"})
    update.update_checkout(local, target)
    assert (local / "outputs/result.txt").read_text() == "my run"


def test_same_target_is_noop_even_with_saved_work(course):
    local, git, _, before = course
    saved_notebook(local, source="my_answer = 42")
    git("add", "lesson.ipynb", cwd=local)
    old = snapshot(local, git)
    result = update.update_checkout(local, before)
    assert result == {"before": before, "after": before, "backup": None,
                      "preserved": ["lesson.ipynb"], "updated": []}
    assert snapshot(local, git) == old


def test_deleted_upstream_notebook_keeps_output_backup(course):
    local, _, release, _ = course
    saved = saved_notebook(local)
    target = release({"lesson.ipynb": None})
    result = update.update_checkout(local, target)
    assert not (local / "lesson.ipynb").exists()
    assert (Path(result["backup"]) / "files/lesson.ipynb").read_bytes() == saved


def test_checkout_failure_restores_original_notebook_and_index(course, monkeypatch):
    local, git, release, _ = course
    saved_notebook(local)
    target = release({"lesson.ipynb": json.dumps(notebook("new lesson"))})
    original = snapshot(local, git)
    real_git = update._git

    def fail_checkout(destination, *args, **kwargs):
        if args[0] == "checkout":
            return subprocess.CompletedProcess(args, 1, b"", b"simulated checkout failure")
        return real_git(destination, *args, **kwargs)

    monkeypatch.setattr(update, "_git", fail_checkout)
    with pytest.raises(update.UpdateError, match="Original notebook backup"):
        update.update_checkout(local, target)
    assert snapshot(local, git) == original
    manifests = list((local.parent / ".ai4sci-course-backups").glob("*/manifest.json"))
    assert json.loads(manifests[0].read_text())["status"] == "rolled-back"


def test_editor_save_while_backing_up_aborts_without_overwriting_new_work(course, monkeypatch):
    local, git, release, before = course
    saved_notebook(local)
    target = release({"lesson.ipynb": json.dumps(notebook("new lesson"))})
    real_backup = update._backup
    newest = json.dumps(notebook("student edit while updater prepares"))

    def concurrent_save(*args, **kwargs):
        result = real_backup(*args, **kwargs)
        (local / "lesson.ipynb").write_text(newest)
        return result

    monkeypatch.setattr(update, "_backup", concurrent_save)
    with pytest.raises(update.UpdateError, match="changed while preparing"):
        update.update_checkout(local, target)
    assert (local / "lesson.ipynb").read_text() == newest
    assert git("rev-parse", "HEAD", cwd=local).stdout.strip() == before


@pytest.mark.parametrize("flag", ["--assume-unchanged", "--skip-worktree"])
def test_hidden_index_changes_are_not_overwritten(course, flag):
    local, git, release, _ = course
    git("update-index", flag, "lesson.py", cwd=local)
    (local / "lesson.py").write_text("hidden_student_answer = 42\n")
    target = release({"lesson.py": "VALUE = 2\n"})
    old = snapshot(local, git)
    with pytest.raises(update.UpdateError, match="index flags"):
        update.update_checkout(local, target)
    assert snapshot(local, git) == old


def test_existing_index_lock_blocks_all_file_changes(course):
    local, git, release, _ = course
    saved_notebook(local)
    target = release({"lesson.ipynb": json.dumps(notebook("new lesson"))})
    (local / ".git/index.lock").write_text("busy")
    original = snapshot(local, git)
    with pytest.raises(update.UpdateError, match="index.lock"):
        update.update_checkout(local, target)
    assert snapshot(local, git) == original


def test_concurrent_updater_lock_blocks_second_invocation(course):
    local, git, release, _ = course
    target = release({"lesson.py": "VALUE = 2\n"})
    original = snapshot(local, git)
    with (local / ".git/ai4sci-update.lock").open("w") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(update.UpdateError, match="Another course update"):
            update.update_checkout(local, target)
    assert snapshot(local, git) == original


def test_private_commit_cannot_be_dropped(course):
    local, git, release, _ = course
    (local / "answer.py").write_text("my_answer = 42\n")
    git("add", "answer.py", cwd=local)
    git("commit", "-m", "Student commit", cwd=local)
    target = release({"lesson.py": "VALUE = 2\n"})
    original = snapshot(local, git)
    with pytest.raises(update.UpdateError, match="private work was preserved"):
        update.update_checkout(local, target)
    assert snapshot(local, git) == original


def test_staged_rename_overlapping_release_is_preserved(course):
    local, git, release, _ = course
    git("mv", "lesson.py", "renamed.py", cwd=local)
    target = release({"lesson.py": "VALUE = 2\n"})
    original = snapshot(local, git)
    with pytest.raises(update.UpdateError, match="lesson.py"):
        update.update_checkout(local, target)
    assert snapshot(local, git) == original


def test_symlinked_notebook_is_not_followed(course, tmp_path):
    local, git, release, _ = course
    external = tmp_path / "irreplaceable.ipynb"
    external.write_text(json.dumps(notebook("private notebook")))
    (local / "lesson.ipynb").unlink()
    (local / "lesson.ipynb").symlink_to(external)
    target = release({"lesson.ipynb": json.dumps(notebook("new lesson"))})
    old = external.read_bytes()
    with pytest.raises(update.UpdateError, match="Symlinks"):
        update.update_checkout(local, target)
    assert external.read_bytes() == old
    assert (local / "lesson.ipynb").is_symlink()


@pytest.mark.parametrize("metadata", [{"execution": {"iopub.status.busy": "time"}},
                                     {"trusted": True}, {"jupyter": {"outputs_hidden": True}}])
def test_known_transient_cell_metadata_is_not_a_source_edit(metadata):
    baseline = notebook()
    saved = notebook(output="result")
    saved["cells"][0]["metadata"] = metadata
    assert update._output_only(json.dumps(baseline).encode(), json.dumps(saved).encode())


def test_invalid_or_unknown_notebook_metadata_is_never_treated_as_output_only():
    baseline = json.dumps(notebook()).encode()
    assert not update._output_only(baseline, b"not json")
    edited = notebook()
    edited["cells"][0]["metadata"]["important_answer"] = "student text"
    assert not update._output_only(baseline, json.dumps(edited).encode())


def test_partial_notebook_preparation_failure_restores_already_rewritten_file(course, monkeypatch):
    local, git, release, _ = course
    saved_notebook(local)
    target = release({"lesson.ipynb": json.dumps(notebook("new lesson"))})
    original = snapshot(local, git)
    real_write = update._atomic_write
    failed = False

    def fail_after_write(path, content, mode=0o600):
        nonlocal failed
        real_write(path, content, mode)
        if path == local / "lesson.ipynb" and not failed:
            failed = True
            raise OSError("simulated fsync failure after replace")

    monkeypatch.setattr(update, "_atomic_write", fail_after_write)
    with pytest.raises(update.UpdateError, match="Original notebook backup"):
        update.update_checkout(local, target)
    assert snapshot(local, git) == original


def test_backup_failure_never_replaces_the_notebook(course, monkeypatch):
    local, git, release, _ = course
    saved_notebook(local)
    target = release({"lesson.ipynb": json.dumps(notebook("new lesson"))})
    original = snapshot(local, git)

    def full_disk(*args, **kwargs):
        raise OSError("No space left on device")

    monkeypatch.setattr(update, "_write_manifest", full_disk)
    with pytest.raises(OSError, match="No space left"):
        update.update_checkout(local, target)
    assert snapshot(local, git) == original


def test_public_or_inside_checkout_backup_locations_are_refused(course, tmp_path):
    local, git, release, _ = course
    saved_notebook(local)
    target = release({"lesson.ipynb": json.dumps(notebook("new lesson"))})
    original = snapshot(local, git)
    public = tmp_path / "public-backups"
    public.mkdir(mode=0o755)
    with pytest.raises(update.UpdateError, match="private"):
        update.update_checkout(local, target, backup_root=public)
    with pytest.raises(update.UpdateError, match="outside"):
        update.update_checkout(local, target, backup_root=local / "backups")
    assert snapshot(local, git) == original


def test_inherited_git_routing_cannot_redirect_updater(course, monkeypatch, tmp_path):
    local, git, release, _ = course
    target = release({"lesson.py": "VALUE = 2\n"})
    monkeypatch.setenv("GIT_DIR", str(tmp_path / "unrelated"))
    monkeypatch.setenv("GIT_WORK_TREE", str(tmp_path / "unrelated-worktree"))
    monkeypatch.setenv("GIT_INDEX_FILE", str(tmp_path / "unrelated-index"))
    result = update.update_checkout(local, target)
    assert result["after"] == target
    assert (local / "lesson.py").read_text() == "VALUE = 2\n"
    assert not (tmp_path / "unrelated-index").exists()


def course_runtime_metadata(*, expanded=False, display_name="AI4Sci PhysicsNeMo 2.2.2 (uv / CUDA)"):
    """Representative metadata from the real saved Lab 3/4 and Challenge books."""
    language = {"name": "python"}
    if expanded:
        language.update(codemirror_mode={"name": "ipython", "version": 3},
                        file_extension=".py", mimetype="text/x-python",
                        nbconvert_exporter="python", pygments_lexer="ipython3", version="3.12.3")
    return {"kernelspec": {"display_name": display_name, "language": "python",
                           "name": "ai4sci-physicsnemo-uv"}, "language_info": language}


@pytest.mark.parametrize("display_name", ["Python 3", "Python 3 (ipykernel)"])
@pytest.mark.parametrize("initially_expanded", [False, True])
def test_actual_jupyter_language_expansion_and_kernel_label_are_output_only(display_name, initially_expanded):
    baseline = notebook()
    baseline["metadata"] = course_runtime_metadata(expanded=initially_expanded)
    saved = notebook(output="learner run")
    saved["metadata"] = course_runtime_metadata(expanded=True, display_name=display_name)
    assert update._output_only(json.dumps(baseline).encode(), json.dumps(saved).encode())


@pytest.mark.parametrize("change", ["kernel_name", "kernel_language", "language_name", "unknown_language", "unknown_kernel"])
def test_runtime_normalization_does_not_hide_kernel_identity_or_unknown_metadata(change):
    baseline = notebook()
    baseline["metadata"] = course_runtime_metadata()
    saved = notebook(output="learner run")
    saved["metadata"] = course_runtime_metadata(expanded=True, display_name="Python 3")
    if change == "kernel_name":
        saved["metadata"]["kernelspec"]["name"] = "student-custom-environment"
    elif change == "kernel_language":
        saved["metadata"]["kernelspec"]["language"] = "julia"
    elif change == "language_name":
        saved["metadata"]["language_info"]["name"] = "julia"
    elif change == "unknown_language":
        saved["metadata"]["language_info"]["student_notes"] = "keep this"
    else:
        saved["metadata"]["kernelspec"]["student_notes"] = "keep this"
    assert not update._output_only(json.dumps(baseline).encode(), json.dumps(saved).encode())


def test_missing_metadata_and_empty_containers_after_transient_removal_are_equivalent():
    baseline = notebook()
    del baseline["metadata"]
    del baseline["cells"][0]["metadata"]
    saved = notebook(output="learner run")
    saved["metadata"] = {"widgets": {"state": "output"}, "language_info": {"version": "3.12"}}
    saved["cells"][0]["metadata"] = {"execution": {"busy": "time"}, "jupyter": {"outputs_hidden": True}}
    assert update._output_only(json.dumps(baseline).encode(), json.dumps(saved).encode())
