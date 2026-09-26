"""Restricted event enrollment, without real SSH or live judge credentials."""
import io
import json
import os
from pathlib import Path
import pwd
import threading
import time

import pytest

from ETC.launchable import bootstrap, enrollment, install


CONFIG = {"url": enrollment.JUDGE_URL, "token": "test_fixture_" + "x" * 40}


def private_file(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.write_bytes(payload)
    path.chmod(0o600)
    return path


def read_wire(payload):
    reader, writer = os.pipe()
    try:
        os.write(writer, payload)
        os.close(writer)
        writer = None
        return enrollment.read_message(reader, timeout=0.1)
    finally:
        os.close(reader)
        if writer is not None:
            os.close(writer)


def test_receiver_installs_private_and_preserves_existing_ssh_keys(tmp_path, capsys):
    original = b'ssh-ed25519 EXISTING_FIXTURE_KEY personal access without trailing newline'
    keys = private_file(tmp_path / ".ssh/authorized_keys", original)
    receiver = enrollment.install_receiver(enrollment.EVENT_ID, home=tmp_path)
    assert receiver == tmp_path / ".local/lib/ai4sci-enrollment/receiver.py"
    assert receiver.read_bytes() == Path(enrollment.__file__).read_bytes()
    assert receiver.stat().st_mode & 0o777 == 0o600
    assert receiver.parent.stat().st_mode & 0o777 == 0o700
    expected = enrollment.enrollment_key_line(receiver).encode()
    assert keys.read_bytes() == original + b"\n" + expected + b"\n"
    assert keys.stat().st_mode & 0o777 == 0o600
    for restriction in ('restrict,port-forwarding', 'permitopen="127.0.0.1:8090"',
                        'permitlisten="127.0.0.1:8090"',
                        'command="/usr/bin/python3 \'' + str(receiver) + "' --receive\""):
        assert restriction.encode() in expected
    enrollment.install_receiver(enrollment.EVENT_ID, home=tmp_path)
    assert keys.read_bytes() == original + b"\n" + expected + b"\n"
    assert not (tmp_path / ".config/ai4sci/judge.json").exists()
    assert "PRIVATE KEY" not in receiver.read_text()
    assert CONFIG["token"] not in capsys.readouterr().out


@pytest.mark.parametrize("with_lib", [False, True])
def test_brev_group_writable_defaults_are_hardened_only_at_opt_in(tmp_path, with_lib):
    home = tmp_path / "ubuntu"
    home.mkdir()
    paths = [home, home / ".local", home / ".config"]
    for path in paths[1:]:
        path.mkdir()
    if with_lib:
        paths.append(home / ".local/lib")
        paths[-1].mkdir()
    for path in paths:
        path.chmod(0o775)
    unrelated = home / "student-work"
    unrelated.mkdir(mode=0o775)
    unrelated.chmod(0o775)
    learner_file = unrelated / "exercise.py"
    learner_file.write_text("preserve learner work\n")
    learner_file.chmod(0o664)
    original = b"ssh-ed25519 STUDENT_KEY preserved\n"
    keys = private_file(home / ".ssh/authorized_keys", original)

    receiver = enrollment.install_receiver(enrollment.EVENT_ID, home=home)
    assert all(path.stat().st_mode & 0o777 == 0o755 for path in paths)
    assert unrelated.stat().st_mode & 0o777 == 0o775
    assert learner_file.stat().st_mode & 0o777 == 0o664
    assert learner_file.read_text() == "preserve learner work\n"
    assert receiver.stat().st_mode & 0o777 == 0o600
    assert (home / ".config/ai4sci").stat().st_mode & 0o777 == 0o700
    assert not (home / ".config/ai4sci/judge.json").exists()
    first_keys = keys.read_bytes()
    enrollment.install_receiver(enrollment.EVENT_ID, home=home)
    assert keys.read_bytes() == first_keys and first_keys.startswith(original)
    assert all(path.stat().st_mode & 0o777 == 0o755 for path in paths)
    enrollment.save_configuration(CONFIG, home)
    assert json.loads((home / ".config/ai4sci/judge.json").read_text()) == CONFIG


@pytest.mark.parametrize("location", ["", ".local", ".local/lib", ".config"])
@pytest.mark.parametrize("unsafe", ["world_write", "foreign_owner", "foreign_group"])
def test_install_hardening_rejects_unsafe_set_before_any_chmod(tmp_path, monkeypatch, location, unsafe):
    home = tmp_path / "ubuntu"
    home.mkdir()
    (home / ".local/lib").mkdir(parents=True)
    (home / ".config").mkdir()
    paths = [home, home / ".local", home / ".local/lib", home / ".config"]
    for path in paths:
        path.chmod(0o775)
    target = home / location
    if unsafe == "world_write":
        target.chmod(0o777)
    else:
        original_lstat = Path.lstat

        def foreign_stat(path, *args, **kwargs):
            result = original_lstat(path, *args, **kwargs)
            if path == target:
                values = list(result)
                values[4 if unsafe == "foreign_owner" else 5] += 1
                return os.stat_result(values)
            return result

        monkeypatch.setattr(Path, "lstat", foreign_stat)
    before = {path: path.stat().st_mode for path in paths}
    with pytest.raises(ValueError, match="student-owned"):
        enrollment.install_receiver(enrollment.EVENT_ID, home=home)
    assert {path: path.stat().st_mode for path in paths} == before
    assert not (home / ".config/ai4sci").exists()
    assert not (home / ".ssh").exists()


def test_receiver_never_repairs_group_writable_home(tmp_path):
    tmp_path.chmod(0o775)
    with pytest.raises(ValueError, match="safe permissions"):
        enrollment.save_configuration(CONFIG, tmp_path)
    assert tmp_path.stat().st_mode & 0o777 == 0o775
    assert not (tmp_path / ".config").exists()


def test_install_hardening_preserves_non_group_write_permission_bits(tmp_path):
    (tmp_path / ".local/lib").mkdir(parents=True)
    (tmp_path / ".config").mkdir()
    targets = [tmp_path, tmp_path / ".local", tmp_path / ".local/lib", tmp_path / ".config"]
    modes = [0o775, 0o2775, 0o750, 0o770]
    for path, mode in zip(targets, modes):
        path.chmod(mode)
    enrollment.prepare_install_directories(tmp_path)
    assert [path.stat().st_mode & 0o7777 for path in targets] == [mode & ~0o020 for mode in modes]


@pytest.mark.parametrize("location", [".local", ".local/lib", ".local/lib/ai4sci-enrollment", ".ssh", ".config"])
def test_install_rejects_symlinked_directories(tmp_path, location):
    home, outside = tmp_path / "student", tmp_path / "outside"
    home.mkdir(mode=0o700)
    outside.mkdir(mode=0o700)
    path = home / location
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.symlink_to(outside, target_is_directory=True)
    with pytest.raises((ValueError, OSError)):
        enrollment.install_receiver(enrollment.EVENT_ID, home=home)
    assert list(outside.iterdir()) == []


@pytest.mark.parametrize("location", [".ssh/authorized_keys", ".local/lib/ai4sci-enrollment/receiver.py"])
def test_install_rejects_symlinked_files_without_changing_target(tmp_path, location):
    original = private_file(tmp_path / "outside.txt", b"preserve me")
    path = tmp_path / location
    path.parent.mkdir(parents=True, mode=0o700)
    path.symlink_to(original)
    with pytest.raises((ValueError, OSError)):
        enrollment.install_receiver(enrollment.EVENT_ID, home=tmp_path)
    assert original.read_bytes() == b"preserve me"


@pytest.mark.parametrize("original", [
    enrollment.PUBLIC_KEY + " unrestricted duplicate\n",
    'restrict,command="somewhere else" ' + enrollment.PUBLIC_KEY + "\n",
    "ssh-ed25519 OTHER_KEY ai4sci-enrollment:" + enrollment.EVENT_ID + "\n",
])
def test_incompatible_existing_key_is_preserved(tmp_path, original):
    keys = private_file(tmp_path / ".ssh/authorized_keys", original.encode())
    with pytest.raises(ValueError, match="incompatible"):
        enrollment.install_receiver(enrollment.EVENT_ID, home=tmp_path)
    assert keys.read_text() == original


@pytest.mark.parametrize("mode", [0o644, 0o666])
def test_unsafe_existing_keys_are_not_rewritten(tmp_path, mode):
    keys = private_file(tmp_path / ".ssh/authorized_keys", b"existing\n")
    keys.chmod(mode)
    with pytest.raises(ValueError, match="unsafe"):
        enrollment.install_receiver(enrollment.EVENT_ID, home=tmp_path)
    assert keys.read_bytes() == b"existing\n"
    assert keys.stat().st_mode & 0o777 == mode


def test_unknown_event_rejected_before_writing(tmp_path):
    with pytest.raises(ValueError, match="approved"):
        enrollment.install_receiver("other-event", home=tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_message_accepts_only_fixed_endpoint_and_personal_token():
    assert read_wire(json.dumps(CONFIG).encode() + b"\n") == CONFIG


@pytest.mark.parametrize("bad", [
    {**CONFIG, "url": "https://elsewhere.invalid"},
    {**CONFIG, "url": "http://localhost:8090"},
    {**CONFIG, "url": "http://127.0.0.1:8091"},
    {**CONFIG, "token": "x" * 42},
    {**CONFIG, "token": "x" * 201},
    {**CONFIG, "token": "x" * 43 + "\n"},
    {**CONFIG, "token": 123},
    {**CONFIG, "command": "run a shell"},
    [CONFIG],
])
def test_invalid_message_rejected_without_echoing_secret(bad):
    with pytest.raises(ValueError) as error:
        read_wire(json.dumps(bad).encode() + b"\n")
    assert CONFIG["token"] not in str(error.value)


@pytest.mark.parametrize("payload", [
    b"{}\n{}\n", b"not json\n", b"\xff\n", b"{}", b"x" * 4097,
    b'{"url":"one","url":"two","token":"duplicate"}\n',
])
def test_malformed_duplicate_oversized_and_multiple_messages_fail(payload):
    with pytest.raises(ValueError):
        read_wire(payload)


@pytest.mark.parametrize("partial", [b"", b'{"url":'])
def test_timeout_bounds_no_newline_read_with_open_writer(partial):
    reader, writer = os.pipe()
    try:
        if partial:
            os.write(writer, partial)
        started = time.monotonic()
        with pytest.raises(ValueError, match="timed out"):
            enrollment.read_message(reader, timeout=0.02)
        assert time.monotonic() - started < 0.5
    finally:
        os.close(reader)
        os.close(writer)


def test_configuration_is_private_idempotent_and_conflicts_are_preserved(tmp_path):
    enrollment.save_configuration(CONFIG, tmp_path)
    target = tmp_path / ".config/ai4sci/judge.json"
    original = target.read_bytes()
    assert target.stat().st_mode & 0o777 == 0o600
    assert target.stat().st_nlink == 1
    assert target.parent.stat().st_mode & 0o777 == 0o700
    enrollment.save_configuration(CONFIG, tmp_path)
    assert target.read_bytes() == original
    with pytest.raises(ValueError, match="preserved"):
        enrollment.save_configuration({**CONFIG, "token": "y" * 43}, tmp_path)
    assert target.read_bytes() == original
    assert not list(target.parent.glob(".enrollment-*"))


@pytest.mark.parametrize("unsafe", ["symlink", "hardlink", "permissions", "parent_symlink", "parent_permissions"])
def test_configuration_rejects_unsafe_destinations(tmp_path, unsafe):
    target = tmp_path / ".config/ai4sci/judge.json"
    outside = private_file(tmp_path / "preserve.json", json.dumps(CONFIG).encode())
    target.parent.mkdir(parents=True, mode=0o700)
    if unsafe == "symlink":
        target.symlink_to(outside)
    elif unsafe == "hardlink":
        os.link(outside, target)
    elif unsafe == "permissions":
        private_file(target, json.dumps(CONFIG).encode()).chmod(0o644)
    elif unsafe == "parent_symlink":
        target.parent.rmdir()
        target.parent.symlink_to(tmp_path, target_is_directory=True)
    else:
        target.parent.chmod(0o755)
    with pytest.raises((ValueError, OSError)):
        enrollment.save_configuration(CONFIG, tmp_path)
    assert json.loads(outside.read_text()) == CONFIG


def test_ack_follows_private_save_and_session_holds_until_eof(tmp_path, monkeypatch):
    acked = threading.Event()
    failures = []

    class Output(io.StringIO):
        def flush(self):
            acked.set()

    output = Output()
    reader, writer = os.pipe()
    monkeypatch.setenv("SSH_ORIGINAL_COMMAND", "touch never-executed")

    def run():
        try:
            enrollment.receive(reader, output, home=tmp_path, timeout=1)
        except Exception as error:
            failures.append(error)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    try:
        os.write(writer, json.dumps(CONFIG).encode() + b"\n")
        assert acked.wait(timeout=2)
        assert json.loads((tmp_path / ".config/ai4sci/judge.json").read_text()) == CONFIG
        assert output.getvalue() == enrollment.ACK + "\n"
        assert CONFIG["token"] not in output.getvalue()
        assert thread.is_alive()
        os.write(writer, b"arbitrary input is not executed")
    finally:
        os.close(writer)
        thread.join(timeout=2)
        os.close(reader)
    assert not thread.is_alive() and failures == []
    assert not (tmp_path / "never-executed").exists()


def test_receiver_home_is_account_home_not_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    assert enrollment.user_home() == Path(pwd.getpwuid(os.getuid()).pw_dir)


@pytest.mark.parametrize("opt_in", [False, True])
def test_bootstrap_enrollment_is_explicit_and_forwarded(monkeypatch, tmp_path, opt_in):
    calls = []
    argv = ["bootstrap.py", "--launchable", "--destination", str(tmp_path / "course")]
    if opt_in:
        argv += ["--enroll-event", enrollment.EVENT_ID]
    monkeypatch.setattr(bootstrap.sys, "argv", argv)
    monkeypatch.setattr(bootstrap.os, "geteuid", lambda: 1000)
    monkeypatch.setattr(bootstrap, "prepare_launchable_checkout", lambda *args: tmp_path / "course")
    monkeypatch.setattr(bootstrap.subprocess, "run", lambda command, **kwargs: calls.append(command))
    bootstrap.main()
    assert len(calls) == 1
    assert ("--enroll-event" in calls[0]) == opt_in
    if opt_in:
        assert calls[0][calls[0].index("--enroll-event") + 1] == enrollment.EVENT_ID


@pytest.mark.parametrize("opt_in", [False, True])
def test_installer_enrollment_is_explicit(monkeypatch, tmp_path, opt_in):
    calls = []
    argv = ["install.py", "--course-dir", str(tmp_path / "course")]
    if opt_in:
        argv += ["--enroll-event", enrollment.EVENT_ID]
    monkeypatch.setattr(install.sys, "argv", argv)
    monkeypatch.setattr(install.os, "geteuid", lambda: 1000)
    monkeypatch.setattr(install.shutil, "which", lambda _: "/fixture/nvidia-smi")
    monkeypatch.setattr(install, "ensure_environment", lambda *args, **kwargs: tmp_path / "venv")
    monkeypatch.setattr(install, "connect_kernel", lambda *args, **kwargs: False)
    monkeypatch.setattr(install, "configure_event_enrollment", calls.append)
    install.main()
    assert calls == ([enrollment.EVENT_ID] if opt_in else [])
