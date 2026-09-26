#!/usr/bin/env python3
"""Opt-in event enrollment: a forced SSH receiver, never an interactive shell."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import pwd
import re
import select
import stat
import sys
import tempfile
import time


EVENT_ID = "ai4science-korea-2026"
PUBLIC_KEY = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAILqIXnE50wZlKtU/rniaSvdL4B57BUOdK/FAA+Gu0K93"
JUDGE_URL = "http://127.0.0.1:8090"
ACK = "AI4SCI_ENROLLMENT_READY_V1"
MAX_MESSAGE_BYTES = 4096
INITIAL_TIMEOUT = 30.0


def user_home():
    # The forced receiver takes neither a destination nor a command from SSH.
    # Do not let an environment variable redirect a credential write.
    if os.geteuid() == 0:
        raise ValueError("Enrollment must run as the student, not root.")
    return Path(pwd.getpwuid(os.getuid()).pw_dir)


def check_directory(path, *, private=False):
    info = path.lstat()
    if (not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid()
            or info.st_mode & (0o077 if private else 0o022)):
        raise ValueError("Enrollment requires real, user-owned directories with safe permissions.")


def directory_below(home, *parts):
    home = Path(home).absolute()
    # Check every existing ancestor for symlinks, including the home itself.
    for ancestor in (*reversed(home.parents), home):
        if ancestor.is_symlink():
            raise ValueError("Enrollment paths must not contain symlinks.")
    check_directory(home)
    path = home
    for index, part in enumerate(parts):
        path = path / part
        try:
            path.mkdir(mode=0o700)
        except FileExistsError:
            pass
        check_directory(path, private=index == len(parts) - 1)
    return path


def read_private(path, *, maximum=MAX_MESSAGE_BYTES):
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as stream:
        info = os.fstat(stream.fileno())
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                or info.st_mode & 0o777 != 0o600 or info.st_nlink != 1 or info.st_size > maximum):
            raise ValueError("Enrollment files must be user-owned regular files with mode 600.")
        result = stream.read(maximum + 1)
        if len(result) > maximum:
            raise ValueError("Enrollment file exceeds its size limit.")
        return result


def temporary_payload(directory, payload):
    descriptor, name = tempfile.mkstemp(prefix=".enrollment-", dir=directory)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        os.unlink(name)
        raise
    return Path(name)


def enrollment_key_line(receiver):
    # This path is installer-selected, not sent by the enrollment server. Keep
    # shell and authorized_keys quoting deliberately narrow and auditable.
    if not re.fullmatch(r"/[A-Za-z0-9_./-]+", str(receiver)):
        raise ValueError("The enrollment receiver requires a simple absolute home path.")
    command = "/usr/bin/python3 '" + str(receiver) + "' --receive"
    # OpenSSH authorized_keys requires host:port for permitopen. Restrict both
    # forwarding directions to the same loopback judge API, never a shell or
    # another local service.
    return ('restrict,port-forwarding,permitopen="127.0.0.1:8090",permitlisten="127.0.0.1:8090",'
            'command="' + command + '" ' + PUBLIC_KEY + " ai4sci-enrollment:" + EVENT_ID)


def install_receiver(event_id, *, home=None):
    if event_id != EVENT_ID:
        raise ValueError("This installer supports only the approved Ai4Science Korea event.")
    home = user_home() if home is None else Path(home)
    receiver_dir = directory_below(home, ".local", "lib", "ai4sci-enrollment")
    receiver = receiver_dir / "receiver.py"
    line = enrollment_key_line(receiver).encode()
    if receiver.exists() or receiver.is_symlink():
        read_private(receiver, maximum=1024 * 1024)
    staged = temporary_payload(receiver_dir, Path(__file__).read_bytes())
    try:
        os.replace(staged, receiver)
    finally:
        staged.unlink(missing_ok=True)
    ssh = directory_below(home, ".ssh")
    keys = ssh / "authorized_keys"
    descriptor = os.open(keys, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600)
    with os.fdopen(descriptor, "r+b") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        info = os.fstat(stream.fileno())
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                or info.st_mode & 0o777 != 0o600 or info.st_nlink != 1 or info.st_size > 1024 * 1024):
            raise ValueError("Existing authorized_keys is unsafe; it was not changed.")
        previous = stream.read(1024 * 1024 + 1)
        if len(previous) > 1024 * 1024:
            raise ValueError("Existing authorized_keys is too large; it was not changed.")
        existing = previous.splitlines()
        matches = [entry for entry in existing if PUBLIC_KEY.split()[1].encode() in entry.split()
                   or ("ai4sci-enrollment:" + EVENT_ID).encode() in entry]
        if matches:
            if matches != [line]:
                raise ValueError("An incompatible enrollment key already exists; it was not changed.")
        else:
            stream.seek(0, os.SEEK_END)
            stream.write((b"\n" if previous and not previous.endswith(b"\n") else b"") + line + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
    print("Event enrollment receiver installed. No personal judge credential is embedded in the Launchable.")
    return receiver


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate enrollment fields are not allowed.")
        result[key] = value
    return result


def validate_message(payload):
    try:
        config = json.loads(payload, object_pairs_hook=unique_object)
    except (ValueError, UnicodeError):
        raise ValueError("Invalid enrollment message.") from None
    if (not isinstance(config, dict) or set(config) != {"url", "token"}
            or config["url"] != JUDGE_URL or not isinstance(config["token"], str)
            or re.fullmatch(r"[A-Za-z0-9_-]{43,200}", config["token"]) is None):
        raise ValueError("Enrollment requires the fixed local judge endpoint and a personal credential.")
    return config


def read_message(descriptor, *, timeout=INITIAL_TIMEOUT):
    deadline = time.monotonic() + timeout
    payload = bytearray()
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0 or not select.select([descriptor], [], [], max(0, remaining))[0]:
            raise ValueError("Enrollment message timed out.")
        chunk = os.read(descriptor, MAX_MESSAGE_BYTES + 1 - len(payload))
        if not chunk:
            raise ValueError("Enrollment ended before the complete message arrived.")
        payload.extend(chunk)
        if len(payload) > MAX_MESSAGE_BYTES:
            raise ValueError("Enrollment message exceeds its size limit.")
        if b"\n" in payload:
            message, trailing = payload.split(b"\n", 1)
            if trailing:
                raise ValueError("Send exactly one enrollment message.")
            return validate_message(message)


def save_configuration(config, home):
    directory = directory_below(home, ".config", "ai4sci")
    target = directory / "judge.json"
    if target.exists() or target.is_symlink():
        if validate_message(read_private(target)) != config:
            raise ValueError("This workspace has a different judge credential; it was preserved.")
        return
    payload = (json.dumps(config, sort_keys=True) + "\n").encode()
    staged = temporary_payload(directory, payload)
    try:
        try:
            # Publish a complete mode-600 file without ever replacing a winner
            # from a simultaneous connection or following a newly placed link.
            os.link(staged, target, follow_symlinks=False)
        except FileExistsError:
            if validate_message(read_private(target)) != config:
                raise ValueError("This workspace has a different judge credential; it was preserved.") from None
    finally:
        staged.unlink(missing_ok=True)


def receive(descriptor, output, *, home=None, timeout=INITIAL_TIMEOUT):
    config = read_message(descriptor, timeout=timeout)
    save_configuration(config, user_home() if home is None else Path(home))
    output.write(ACK + "\n")
    output.flush()
    # The SSH session carries the restricted reverse tunnel. No additional
    # message, SSH_ORIGINAL_COMMAND or stdin content is ever executed.
    while os.read(descriptor, 4096):
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receive", action="store_true", required=True)
    parser.parse_args()
    receive(sys.stdin.fileno(), sys.stdout)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError):
        # Credentials and untrusted message contents must never enter logs.
        raise SystemExit("Event enrollment failed. Existing configuration was preserved; ask the organizer to check the connection.") from None
