"""Update course source without restarting Jupyter or discarding learner work.

The caller fetches and verifies the release first. This module operates only on
local Git objects. Notebook output-only saves may be backed up automatically;
source changes that overlap a release require a person to review them.
"""
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import tempfile


class UpdateError(ValueError):
    """An update was stopped without intentionally discarding learner files."""


def _git(destination, *arguments, check=True):
    # Inherited Git routing variables must never redirect commands to another
    # repository or an alternate index (including when running the test suite).
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith("GIT_")}
    environment["GIT_OPTIONAL_LOCKS"] = "0"
    return subprocess.run(["git", "-C", str(destination), *arguments],
                          capture_output=True, check=check, env=environment)


def _text(destination, *arguments):
    return _git(destination, *arguments).stdout.decode("utf-8").strip()


def _path(name):
    path = PurePosixPath(name)
    if (not name or path.is_absolute() or any(part in ("..", ".git") for part in path.parts)
            or str(path) != name or "\\" in name or any(ord(c) < 32 for c in name)):
        raise UpdateError("Unsafe course path: " + repr(name))
    return path


def _no_symlink(path):
    for component in (path, *path.parents):
        if component.is_symlink():
            raise UpdateError("Symlinks are not supported during an update: " + str(component))


def _safe_file(destination, name):
    path = destination.joinpath(*_path(name).parts)
    _no_symlink(path)
    return path


@contextmanager
def _lock(git_dir):
    lock_file = git_dir / "ai4sci-update.lock"
    fd = os.open(lock_file, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise UpdateError("Another course update is running; retry when it finishes.") from None
        if (git_dir / "index.lock").exists():
            raise UpdateError("Git is busy (index.lock exists); finish the other Git operation first.")
        yield
    finally:
        os.close(fd)


def _tree(destination, revision):
    entries = {}
    for record in _git(destination, "ls-tree", "-r", "-z", revision).stdout.split(b"\0"):
        if not record:
            continue
        info, raw_name = record.split(b"\t", 1)
        mode, kind, object_id = info.decode("ascii").split()
        name = raw_name.decode("utf-8")
        _path(name)
        if kind != "blob" or mode not in ("100644", "100755"):
            raise UpdateError("Symlinks and submodules require manual review: " + name)
        entries[name] = (mode, object_id)
    return entries


def _status(destination):
    records = _git(destination, "status", "--porcelain=v1", "-z", "--no-renames",
                   "--untracked-files=all").stdout
    entries = {}
    for record in records.split(b"\0"):
        if record:
            name = record[3:].decode("utf-8")
            _path(name)
            entries[name] = record[:2].decode("ascii")
    return entries


def _related(left, right):
    return left == right or left.startswith(right + "/") or right.startswith(left + "/")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate notebook key")
        result[key] = value
    return result


def _notebook_content(content):
    """Remove only explicitly known execution and view-state fields.

    Unknown metadata, attachments, cell IDs, ordering and all student text remain
    significant. This intentionally errs on the side of preserving user work.
    """
    notebook = json.loads(content, object_pairs_hook=_unique_object,
                          parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    if (not isinstance(notebook, dict) or notebook.get("nbformat") != 4
            or not isinstance(notebook.get("cells"), list)):
        raise ValueError("Not a supported notebook")
    notebook = deepcopy(notebook)
    metadata = notebook.setdefault("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("Invalid notebook metadata")
    metadata.pop("widgets", None)
    language = metadata.get("language_info")
    if isinstance(language, dict):
        # A kernel expands {"name": "python"} to its kernel_info response on
        # execution. Keep language identity and any unknown/student metadata.
        for key in ("version", "codemirror_mode", "file_extension", "mimetype",
                    "nbconvert_exporter", "pygments_lexer"):
            language.pop(key, None)
        if not language:
            metadata.pop("language_info")
    kernel = metadata.get("kernelspec")
    if (isinstance(kernel, dict) and isinstance(kernel.get("name"), str)
            and kernel["name"] and isinstance(kernel.get("language"), str) and kernel["language"]):
        # Jupyter can relabel the same kernel as "Python 3 (ipykernel)". Its
        # actual name/language still participate in the comparison below.
        kernel.pop("display_name", None)
    for cell in notebook["cells"]:
        if not isinstance(cell, dict) or cell.get("cell_type") not in ("code", "markdown", "raw"):
            raise ValueError("Invalid notebook cell")
        if isinstance(cell.get("source"), list):
            cell["source"] = "".join(cell["source"])
        cell_metadata = cell.setdefault("metadata", {})
        if not isinstance(cell_metadata, dict):
            raise ValueError("Invalid cell metadata")
        for key in ("execution", "ExecuteTime", "trusted", "collapsed", "scrolled"):
            cell_metadata.pop(key, None)
        view = cell_metadata.get("jupyter")
        if isinstance(view, dict):
            for key in ("outputs_hidden", "source_hidden"):
                view.pop(key, None)
            if not view:
                cell_metadata.pop("jupyter")
        if cell["cell_type"] == "code":
            cell.pop("outputs", None)
            cell.pop("execution_count", None)
    return notebook


def _output_only(original, saved):
    try:
        return _notebook_content(original) == _notebook_content(saved)
    except (ValueError, TypeError, UnicodeDecodeError):
        return False


def _snapshot(path):
    try:
        information = path.lstat()
    except FileNotFoundError:
        return None
    if not stat.S_ISREG(information.st_mode):
        raise UpdateError("Not a regular course file: " + str(path))
    content = path.read_bytes()
    return (information.st_dev, information.st_ino, information.st_mode,
            information.st_size, information.st_mtime_ns, hashlib.sha256(content).hexdigest())


def _fsync_directory(directory):
    descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write(path, content, mode=0o600):
    descriptor, name = tempfile.mkstemp(prefix=".ai4sci-write-", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            os.fchmod(stream.fileno(), mode)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_manifest(backup, manifest):
    _atomic_write(backup / "manifest.json", (json.dumps(manifest, indent=2) + "\n").encode())


def _backup(destination, root, before, target, originals, modes):
    root = Path(root).expanduser().absolute()
    _no_symlink(root)
    if root == destination or destination in root.parents:
        raise UpdateError("Update backups must be outside the course checkout.")
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    information = root.stat()
    if information.st_uid != os.getuid() or stat.S_IMODE(information.st_mode) & 0o077:
        raise UpdateError("Backup directory must be owned by you and private (mode 700): " + str(root))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    backup = Path(tempfile.mkdtemp(prefix=destination.name + "-" + stamp + "-", dir=root))
    manifest = {"version": 1, "course": str(destination), "before": before, "after": target,
                "status": "prepared", "files": []}
    for name, content in originals.items():
        saved = backup / "files" / name
        saved.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        _atomic_write(saved, content)
        manifest["files"].append({"path": name, "mode": modes[name],
                                   "sha256": hashlib.sha256(content).hexdigest()})
    _write_manifest(backup, manifest)
    _fsync_directory(root)
    return backup, manifest


def update_checkout(destination: Path, target: str, *, backup_root: Path | None = None) -> dict:
    """Safely advance to a locally available descendant commit.

    Return ``before``, ``after``, ``backup`` (path or None), ``preserved`` and
    ``updated``. The target must be a complete commit SHA, already fetched and
    validated by the caller. A conflict raises UpdateError with exact paths.
    """
    destination = Path(destination).expanduser().absolute()
    _no_symlink(destination)
    git_dir = destination / ".git"
    _no_symlink(git_dir)
    if not git_dir.is_dir() or _text(destination, "rev-parse", "--show-toplevel") != str(destination):
        raise UpdateError("Use the course Git checkout, not a subdirectory or linked worktree.")
    if not re.fullmatch(r"[0-9a-fA-F]{40}|[0-9a-fA-F]{64}", target):
        raise UpdateError("The update target must be a verified full commit SHA.")
    with _lock(git_dir):
        before = _text(destination, "rev-parse", "HEAD")
        target = _text(destination, "rev-parse", "--verify", target + "^{commit}")
        status = _status(destination)
        result = {"before": before, "after": before, "backup": None,
                  "preserved": sorted(status), "updated": []}
        if before == target:
            return result
        if _git(destination, "merge-base", "--is-ancestor", before, target, check=False).returncode:
            raise UpdateError("The release does not contain your current commits; private work was preserved.")
        for entry in _git(destination, "ls-files", "-v", "-z").stdout.split(b"\0"):
            if entry and entry[:1] != b"H":
                raise UpdateError("Resolve index flags or conflicts before updating: " + entry[2:].decode("utf-8"))
        old_tree, new_tree = _tree(destination, before), _tree(destination, target)
        changed = {name for name in old_tree.keys() | new_tree.keys()
                   if old_tree.get(name) != new_tree.get(name)}
        # Git sometimes permits an equal-content untracked file to become
        # tracked. The updater must not silently adopt even those learner files.
        others = set()
        for extra in ([], ["--ignored"]):
            for raw in _git(destination, "ls-files", "--others", "--exclude-standard", "-z", *extra).stdout.split(b"\0"):
                if raw:
                    others.add(raw.decode("utf-8"))
        collisions = {name for name in others if any(_related(name, new) for new in new_tree if new not in old_tree)}
        conflicts = set(collisions)
        prepared, originals, modes = {}, {}, {}
        tracked_paths = changed | set(status)
        paths = {name: _safe_file(destination, name) for name in tracked_paths}
        snapshots = {name: _snapshot(path) for name, path in paths.items()}
        for name, state in status.items():
            if not any(_related(name, upstream) for upstream in changed):
                continue
            if (state == " M" and name.endswith(".ipynb") and name in old_tree
                    and snapshots[name] is not None):
                original = _git(destination, "cat-file", "blob", old_tree[name][1]).stdout
                saved = paths[name].read_bytes()
                mode = stat.S_IMODE(paths[name].stat().st_mode)
                executable = old_tree[name][0] == "100755"
                if bool(mode & 0o111) == executable and _output_only(original, saved):
                    prepared[name], originals[name], modes[name] = original, saved, mode
                    continue
            conflicts.add(name)
        if conflicts:
            raise UpdateError("Course update conflicts with learner files; nothing was replaced:\n  "
                              + "\n  ".join(sorted(conflicts)))
        index_snapshot = _snapshot(git_dir / "index")
        backup, manifest = None, None
        if originals:
            backup, manifest = _backup(destination, backup_root or destination.parent / ".ai4sci-course-backups",
                                        before, target, originals, modes)
            result["backup"] = str(backup)
        rewritten = []
        try:
            if (_text(destination, "rev-parse", "HEAD") != before or _status(destination) != status
                    or _snapshot(git_dir / "index") != index_snapshot
                    or any(_snapshot(paths[name]) != snapshot for name, snapshot in snapshots.items())):
                raise UpdateError("Course files changed while preparing the update. Save notebooks and retry.")
            if (git_dir / "index.lock").exists():
                raise UpdateError("Git became busy; retry after the other operation finishes.")
            for name, original in prepared.items():
                if _snapshot(paths[name]) != snapshots[name]:
                    raise UpdateError("Notebook changed while preparing the update: " + name)
                # Track first: even a filesystem error after replace must trigger
                # restoration using the already durable backup.
                rewritten.append(name)
                _atomic_write(paths[name], original, modes[name])
            checkout = _git(destination, "checkout", "--detach", "--no-overwrite-ignore", target, check=False)
            if checkout.returncode:
                raise UpdateError("Git refused the update: " + checkout.stderr.decode("utf-8", errors="replace").strip())
        except (Exception, KeyboardInterrupt) as error:
            recovery = []
            if _text(destination, "rev-parse", "HEAD") == before:
                for name in rewritten:
                    try:
                        path = _safe_file(destination, name)
                        if path.read_bytes() == prepared[name]:
                            _atomic_write(path, originals[name], modes[name])
                        elif path.read_bytes() != originals[name]:
                            recovery.append(name)
                    except (OSError, UpdateError):
                        recovery.append(name)
            else:
                recovery = rewritten
            if backup:
                manifest["status"] = "needs-review" if recovery else "rolled-back"
                manifest["recovery_paths"] = recovery
                try:
                    _write_manifest(backup, manifest)
                except OSError:
                    pass  # Prepared manifest and original bytes are durable.
            detail = str(error)
            if backup:
                detail += "\nOriginal notebook backup: " + str(backup)
            if recovery:
                detail += "\nConcurrent changes were not overwritten; review: " + ", ".join(recovery)
            raise UpdateError(detail) from error
        result.update(after=target, updated=sorted(changed),
                      preserved=sorted(set(status) - set(prepared)))
        if backup:
            manifest["status"] = "applied"
            try:
                _write_manifest(backup, manifest)
            except OSError as error:
                raise UpdateError("Source updated, but recording completion failed. Original notebooks remain at "
                                  + str(backup)) from error
        return result
