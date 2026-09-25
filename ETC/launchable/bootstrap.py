#!/usr/bin/env python3
"""Brev VM entry point. Download this file or run it from a course checkout."""
import argparse
import ast
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import types
from urllib.parse import urlsplit


DEFAULT_REPO = "https://github.com/yang926/AI4Sci-PhysicsNeMo-Bootcamp.git"


def validate_source(repo, ref):
    if not isinstance(repo, str) or any(c.isspace() or ord(c) < 32 for c in repo):
        raise ValueError("Use a public GitHub URL without whitespace or control characters.")
    url = urlsplit(repo)
    if (url.scheme != "https" or url.hostname != "github.com" or url.netloc != "github.com"
            or url.query or url.fragment or not re.fullmatch(r"/[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*", url.path)):
        raise ValueError("Use a public https://github.com/owner/repository URL without credentials.")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/-]*", ref) or ".." in ref or ref.endswith("/"):
        raise ValueError("Use a branch, tag or commit SHA as the course ref.")


def git_environment():
    # Shell/test Git routing must never redirect the student's update elsewhere.
    return {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}


def git_bytes(directory, *args):
    return subprocess.check_output(["git", "-C", str(directory), *args], env=git_environment())


def git(directory, *args):
    return git_bytes(directory, *args).decode().strip()


def release_module(directory, revision, relative):
    """Load an installer helper from the selected, verified repository revision.

    Old deployed checkouts do not have the update helper. Read the fetched
    commit, not the learner's potentially edited working files.
    """
    source = git_bytes(directory, "show", revision + ":" + relative)
    name = "_ai4sci_release_" + relative.replace("/", "_").replace(".", "_")
    module = types.ModuleType(name)
    module.__file__ = str(Path(directory) / relative)
    sys.modules[name] = module
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module


def runtime_signature(directory, revision):
    """Compare dependency inputs without importing or running either installer."""
    lock = git_bytes(directory, "show", revision + ":ETC/launchable/requirements-linux-cu128.lock.txt")
    installer = git_bytes(directory, "show", revision + ":ETC/launchable/install.py")
    for node in ast.parse(installer).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "PYTHON_VERSION"
                                               for t in node.targets):
            return ast.literal_eval(node.value), lock
    raise ValueError("Cannot identify this release's Python version; no course files were updated.")


def supports_refresh(course):
    """Older pinned releases remain installable with their original interface."""
    installer = Path(course) / "ETC/launchable/install.py"
    if not installer.is_file():
        return False
    return any(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
               and node.func.attr == "add_argument" and any(isinstance(arg, ast.Constant)
               and arg.value == "--refresh" for arg in node.args)
               for node in ast.walk(ast.parse(installer.read_text())))


def preflight_managed_update(directory, before, target):
    """Protect running training and connected, stale notebook editors.

    Closed tabs can leave idle kernels alive; these do not block source-only
    updates. This function never stops a kernel or changes server settings.
    """
    home = Path.home()
    config = home / ".jupyter/jupyter_server_config.json"
    if not config.is_file():
        return
    settings = json.loads(config.read_text())
    if settings.get("ServerApp", {}).get("root_dir") != str(directory):
        return
    jupyter = release_module(directory, target, "ETC/launchable/jupyter.py")
    info = jupyter._server_info(jupyter.inspect_service(home))
    sessions = jupyter._api(info, "sessions")
    kernels = jupyter._api(info, "kernels")
    if not isinstance(sessions, list) or not isinstance(kernels, list):
        raise ValueError("Cannot inspect running notebooks; no course files were updated.")
    if any(k.get("execution_state") != "idle" for k in kernels):
        raise ValueError("A notebook is still executing. Wait for training to finish, then rerun the same command. "
                         "No kernels were stopped and no course files were updated.")
    if runtime_signature(directory, before) != runtime_signature(directory, target) and (sessions or kernels):
        raise ValueError("This release changes Python packages. Save your work and shut down notebook kernels "
                         "before rerunning the same command. No course files were updated.")
    changed = set(git_bytes(directory, "diff", "--name-only", "-z", before, target).decode().rstrip("\0").split("\0"))
    live = {k.get("id"): k for k in kernels}
    open_paths = sorted({s.get("path") for s in sessions if s.get("path") in changed
                         and live.get(s.get("kernel", {}).get("id"), {}).get("connections", 0) > 0})
    if open_paths:
        raise ValueError("Save and close these notebook tabs (or the Jupyter browser tab), then rerun the same command: "
                         + ", ".join(open_paths) + ". Idle kernels can stay running. "
                         "This prevents an old browser buffer from overwriting the updated notebook.")


def prepare_launchable_checkout(repo, ref, destination):
    """Use one course folder and preserve learner work during source updates."""
    validate_source(repo, ref)
    destination = Path(destination).expanduser().absolute()
    if destination.is_symlink():
        raise ValueError("The course destination must not be a symlink.")
    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".ai4sci-source-", dir=destination.parent) as temporary:
            staged = Path(temporary) / "course"
            prepare_checkout(repo, ref, staged)
            if destination.exists() or destination.is_symlink():
                raise ValueError("The course destination appeared during setup; it was not overwritten.")
            staged.rename(destination)
        return destination
    if not (destination / ".git").is_dir():
        raise ValueError("Destination already exists but is not a course checkout; it was not changed.")
    origin = git(destination, "remote", "get-url", "origin")
    if origin.removesuffix(".git") != repo.removesuffix(".git"):
        raise ValueError("Destination belongs to another repository; it was not changed.")
    before = git(destination, "rev-parse", "HEAD")
    subprocess.run(["git", "-C", str(destination), "fetch", "origin", ref], check=True, env=git_environment())
    target = git(destination, "rev-parse", "FETCH_HEAD^{commit}")
    for required in ("Start_Here.ipynb", "ETC/launchable/install.py"):
        git(destination, "cat-file", "-e", target + ":" + required)
    if before != target:
        if git(destination, "rev-parse", "--is-shallow-repository") == "true":
            subprocess.run(["git", "-C", str(destination), "fetch", "--unshallow", "origin", ref], check=True, env=git_environment())
        ancestor = subprocess.run(["git", "-C", str(destination), "merge-base", "--is-ancestor", before, target], check=False, env=git_environment())
        if ancestor.returncode != 0:
            raise ValueError("Existing course commits are not an ancestor of the requested release. They were preserved; no checkout was replaced.")
        helper = subprocess.run(["git", "-C", str(destination), "cat-file", "-e",
                                 target + ":ETC/launchable/update.py"], capture_output=True, env=git_environment())
        if helper.returncode == 0:
            preflight_managed_update(destination, before, target)
            result = release_module(destination, target, "ETC/launchable/update.py").update_checkout(destination, target)
            if result.get("backup"):
                print("Saved notebook outputs backed up to:", result["backup"], flush=True)
            if result.get("preserved"):
                print("Preserved local work:", ", ".join(result["preserved"]), flush=True)
        else:
            # Compatibility with pinned releases predating the safe updater.
            if git(destination, "status", "--porcelain", "--untracked-files=all"):
                raise ValueError("This older release cannot safely update learner changes. They were preserved.")
            subprocess.run(["git", "-C", str(destination), "checkout", "--detach", "--no-overwrite-ignore", target], check=True, env=git_environment())
    if not (destination / "Start_Here.ipynb").is_file() or not (destination / "ETC/launchable/install.py").is_file():
        raise ValueError("The selected source does not contain the course installer.")
    print("Course folder:", destination, flush=True)
    print("Course revision:", git(destination, "rev-parse", "HEAD"), flush=True)
    return destination


def prepare_checkout(repo, ref, destination, update=False):
    validate_source(repo, ref)
    destination = Path(destination).expanduser().absolute()
    if destination.is_symlink():
        raise ValueError("The course destination must not be a symlink.")
    if update:
        # No checkout, pull, stash or reset touches the student's working directory.
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        destination = destination.with_name(destination.name + "-updates") / stamp
    if destination.exists():
        if not (destination / ".git").exists():
            raise ValueError("Destination already exists but is not a course checkout; choose another directory.")
        origin = git(destination, "remote", "get-url", "origin")
        if origin.removesuffix(".git") != repo.removesuffix(".git"):
            raise ValueError("Destination belongs to another repository; it was not changed.")
        print("Reusing the existing checkout unchanged. Use --update for a separate new copy.")
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--depth=1", "--no-checkout", "--", repo, str(destination)], check=True, env=git_environment())
        subprocess.run(["git", "-C", str(destination), "fetch", "--depth=1", "origin", ref], check=True, env=git_environment())
        subprocess.run(["git", "-C", str(destination), "checkout", "--detach", "FETCH_HEAD"], check=True, env=git_environment())
    if not (destination / "Start_Here.ipynb").is_file() or not (destination / "ETC/launchable/install.py").is_file():
        raise ValueError("This checkout predates the Launchable installer. Preserve it and use --update.")
    print("Course revision:", git(destination, "rev-parse", "HEAD"))
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=os.environ.get("AI4SCI_COURSE_REPO", DEFAULT_REPO))
    parser.add_argument("--ref", default=os.environ.get("AI4SCI_COURSE_REF", "main"))
    parser.add_argument("--destination", type=Path, default=Path.home() / "AI4Sci-PhysicsNeMo-Bootcamp")
    parser.add_argument("--update", action="store_true", help="Create a separate updated checkout; retain all earlier student work")
    parser.add_argument("--launchable", action="store_true", help="Use the canonical course folder and configure managed Jupyter for students")
    parser.add_argument("--refresh", action="store_true", help="Update the existing course, preserving learner work and reusing its environment")
    args = parser.parse_args()
    if os.geteuid() == 0:
        parser.error("Run as the Brev/Jupyter user, not root or sudo.")
    if sum((args.launchable, args.refresh, args.update)) > 1:
        parser.error("--launchable, --refresh and --update are separate workflows; choose one.")
    existing = args.destination.expanduser().exists()
    if args.launchable or args.refresh:
        course = prepare_launchable_checkout(args.repo, args.ref, args.destination)
    else:
        course = prepare_checkout(args.repo, args.ref, args.destination, args.update or args.ref != "main")
    command = [sys.executable, str(course / "ETC/launchable/install.py"), "--course-dir", str(course)]
    if args.launchable or args.refresh:
        command.append("--configure-jupyter")
        if (existing or args.refresh) and supports_refresh(course):
            command.append("--refresh")
    subprocess.run(command, check=True)
    print("Open Jupyter in Brev, then open:", course / "Start_Here.ipynb")
    print("Learner changes, outputs and Jupyter authentication settings are preserved.")
    if existing and (args.launchable or args.refresh):
        print("Reopen updated notebooks from disk. Restart only a notebook kernel that has already imported changed Python code.")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        raise SystemExit("Launchable setup stopped: " + str(exc)) from None
