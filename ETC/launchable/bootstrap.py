#!/usr/bin/env python3
"""Brev VM entry point. Download this file or run it from a course checkout."""
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
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


def git(directory, *args):
    return subprocess.check_output(["git", "-C", str(directory), *args], text=True).strip()


def prepare_launchable_checkout(repo, ref, destination):
    """Use one course folder; only advance a clean, upstream-owned checkout."""
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
    subprocess.run(["git", "-C", str(destination), "fetch", "origin", ref], check=True)
    target = git(destination, "rev-parse", "FETCH_HEAD^{commit}")
    for required in ("Start_Here.ipynb", "ETC/launchable/install.py"):
        git(destination, "cat-file", "-e", target + ":" + required)
    if before != target:
        if git(destination, "status", "--porcelain", "--untracked-files=all"):
            raise ValueError("This course has learner changes. They were preserved; an organizer must review before upgrading this workspace.")
        if git(destination, "rev-parse", "--is-shallow-repository") == "true":
            subprocess.run(["git", "-C", str(destination), "fetch", "--unshallow", "origin", ref], check=True)
        ancestor = subprocess.run(["git", "-C", str(destination), "merge-base", "--is-ancestor", before, target], check=False)
        if ancestor.returncode != 0:
            raise ValueError("Existing course commits are not an ancestor of the requested release. They were preserved; no checkout was replaced.")
        # Git rejects untracked/ignored collisions; never use reset, force, or clean.
        subprocess.run(["git", "-C", str(destination), "checkout", "--detach", "--no-overwrite-ignore", target], check=True)
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
        subprocess.run(["git", "clone", "--depth=1", "--no-checkout", "--", repo, str(destination)], check=True)
        subprocess.run(["git", "-C", str(destination), "fetch", "--depth=1", "origin", ref], check=True)
        subprocess.run(["git", "-C", str(destination), "checkout", "--detach", "FETCH_HEAD"], check=True)
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
    args = parser.parse_args()
    if os.geteuid() == 0:
        parser.error("Run as the Brev/Jupyter user, not root or sudo.")
    if args.launchable and args.update:
        parser.error("--launchable and --update are separate workflows; choose one.")
    if args.launchable:
        course = prepare_launchable_checkout(args.repo, args.ref, args.destination)
    else:
        course = prepare_checkout(args.repo, args.ref, args.destination, args.update or args.ref != "main")
    command = [sys.executable, str(course / "ETC/launchable/install.py"), "--course-dir", str(course)]
    if args.launchable:
        command.append("--configure-jupyter")
    subprocess.run(command, check=True)
    print("Open Jupyter in Brev, then open:", course / "Start_Here.ipynb")
    print("Learner changes, outputs and Jupyter authentication settings are preserved.")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        raise SystemExit("Launchable setup stopped: " + str(exc)) from None
