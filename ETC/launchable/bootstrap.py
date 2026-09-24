#!/usr/bin/env python3
"""Brev VM entry point. Download this file or run it from a course checkout."""
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import re
import subprocess
import sys
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
    args = parser.parse_args()
    if os.geteuid() == 0:
        parser.error("Run as the Brev/Jupyter user, not root or sudo.")
    # Brev's Source clone normally follows main. An explicit release/tag must
    # get its own checkout instead of silently reusing that clone.
    course = prepare_checkout(args.repo, args.ref, args.destination, args.update or args.ref != "main")
    subprocess.run([sys.executable, str(course / "ETC/launchable/install.py"), "--course-dir", str(course)], check=True)
    print("Open Jupyter in Brev, then open:", course / "Start_Here.ipynb")
    print("No existing notebook, answer file, output, Jupyter server or authentication setting was replaced.")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        raise SystemExit("Launchable setup stopped: " + str(exc)) from None
