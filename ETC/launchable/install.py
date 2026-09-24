#!/usr/bin/env python3
"""Install a course kernel for Brev-managed Jupyter, without starting a server."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


PYTHON_VERSION = "3.12.11"
UV_VERSION = "0.8.17"
KERNEL_NAME = "ai4sci-physicsnemo-uv"
LOCK_NAME = "requirements-linux-cu128.lock.txt"


def run(*args, **kwargs):
    subprocess.run([str(value) for value in args], check=True, **kwargs)


def environment_key(lock):
    return hashlib.sha256((PYTHON_VERSION + "\n").encode() + Path(lock).read_bytes()).hexdigest()[:16]


def check_uv_version(binary):
    version = subprocess.check_output([str(binary), "--version"], text=True).split()
    if version[:2] != ["uv", UV_VERSION]:
        raise ValueError("The private uv tool version does not match the installer.")


def ensure_uv(home):
    tools = home / ".local/share/ai4sci-tools" / ("uv-" + UV_VERSION)
    binary = tools / "bin/uv"
    if tools.is_symlink():
        raise ValueError("The private uv tool directory must not be a symlink.")
    if not binary.exists():
        if tools.exists():
            raise ValueError("An incomplete tool directory exists. Inspect it or select a clean workspace; nothing was deleted.")
        tools.parent.mkdir(parents=True, exist_ok=True)
        # Brev's system Python may have neither pip nor ensurepip. Use Astral's
        # pinned standalone installer, without touching the host Python or PATH.
        # Stage it first so a failed download does not poison the retry path.
        with tempfile.TemporaryDirectory(prefix=".uv-install-", dir=tools.parent) as temporary:
            stage = Path(temporary)
            payload = stage / "tool"
            installer = stage / "install.sh"
            run("curl", "--fail", "--silent", "--show-error", "--location", "--retry", "3",
                "--connect-timeout", "20", "--max-time", "120",
                "https://astral.sh/uv/" + UV_VERSION + "/install.sh", "--output", installer)
            environment = dict(os.environ, UV_INSTALL_DIR=str(payload / "bin"),
                               UV_UNMANAGED_INSTALL=str(payload / "bin"), UV_NO_MODIFY_PATH="1")
            run("sh", installer, env=environment)
            check_uv_version(payload / "bin/uv")
            if tools.exists() or tools.is_symlink():
                raise ValueError("The private uv destination appeared during setup; it was not overwritten.")
            payload.rename(tools)
    check_uv_version(binary)
    return binary


def ensure_environment(course, home):
    lock = course / "ETC/launchable" / LOCK_NAME
    key = environment_key(lock)
    prefix = home / ".venvs" / ("ai4sci-brev-" + key)
    marker = prefix / ".ai4sci-environment.json"
    python = prefix / "bin/python"
    if prefix.is_symlink():
        raise ValueError("The managed course environment must not be a symlink.")
    metadata = None
    if prefix.exists():
        metadata = json.loads(marker.read_text()) if marker.is_file() else None
        if (not isinstance(metadata, dict) or metadata.get("key") != key or not python.is_file()
                or metadata.get("status", "ready") not in {"installing", "ready"}):
            raise ValueError("An unrecognized or incomplete environment exists. It was not overwritten.")
    else:
        uv = ensure_uv(home)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        run(uv, "venv", "--python", PYTHON_VERSION, prefix)
        metadata = {"key": key, "python": PYTHON_VERSION, "status": "installing"}
        marker.write_text(json.dumps(metadata) + "\n")
    if metadata.get("status", "ready") == "installing":
        # Resume only our own, matching environment after a download failure.
        # Never recreate a venv or touch an unrelated pre-existing directory.
        uv = ensure_uv(home)
        run(uv, "pip", "install", "--python", python, "torch==2.10.0+cu128", "torchvision==0.25.0+cu128",
            "--default-index", "https://download.pytorch.org/whl/cu128")
        # Every dependency is pinned. Keep the already installed CUDA wheels;
        # install remaining distributions from PyPI, not a mixed-priority index.
        run(uv, "pip", "install", "--python", python, "--no-deps", "-r", lock)
        run(uv, "pip", "check", "--python", python)
        # This is an import/device check, not a training job or dataset generation.
        run(python, course / "ETC/launchable/verify.py")
        marker.write_text(json.dumps({"key": key, "python": PYTHON_VERSION, "status": "ready"}) + "\n")
    else:
        print("Reusing the verified course environment:", prefix)
    run(python, course / "ETC/launchable/verify.py")
    return prefix


def connect_kernel(prefix):
    python = prefix / "bin/python"
    data = Path(subprocess.check_output([str(python), "-c", "from jupyter_core.paths import jupyter_data_dir; print(jupyter_data_dir())"], text=True).strip())
    spec = data / "kernels" / KERNEL_NAME / "kernel.json"
    if spec.exists():
        existing = json.loads(spec.read_text())
        previous = Path(existing["argv"][0])
        if previous != python and (previous.parent.parent.parent != prefix.parent or not previous.parent.parent.name.startswith("ai4sci-brev-")):
            raise ValueError("The course kernel name is already used by an unrelated environment; it was not replaced.")
    # Managed Jupyter and the kernel may run from different Python environments.
    # Expose the prebuilt widgets frontend in Jupyter's per-user data path.
    extension = prefix / "share/jupyter/labextensions/@jupyter-widgets/jupyterlab-manager"
    if not (extension / "package.json").is_file():
        raise ValueError("The pinned Jupyter widgets frontend is missing.")
    target = data / "labextensions/@jupyter-widgets/jupyterlab-manager"
    exists = target.exists() or target.is_symlink()
    if exists:
        if not (target / "package.json").is_file():
            raise ValueError("The existing widgets frontend is incomplete; it was not overwritten.")
        installed = json.loads((target / "package.json").read_text()).get("version")
        expected = json.loads((extension / "package.json").read_text()).get("version")
        if installed != expected:
            raise ValueError("An existing widgets frontend has a different version. Review it before updating; it was not overwritten.")
    run(python, "-m", "ipykernel", "install", "--user", "--name", KERNEL_NAME,
        "--display-name", "AI4Sci PhysicsNeMo 2.2.2 (uv / CUDA)", "--env", "AI4SCI_DEVICE", "cuda")
    if not exists:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(extension, target_is_directory=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--course-dir", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    if sys.platform != "linux" or os.geteuid() == 0:
        parser.error("Use a Linux NVIDIA GPU VM as the same non-root user as Brev Jupyter.")
    if not shutil.which("nvidia-smi"):
        parser.error("NVIDIA drivers are required. Select a GPU VM; do not install drivers from this script.")
    course = args.course_dir.resolve()
    prefix = ensure_environment(course, Path.home())
    connect_kernel(prefix)
    # No judge credentials are embedded in this public template. Personal
    # provisioning remains separate and local practice works without a judge.
    print("Course kernel ready. Use Brev's managed Jupyter; no second server was started.")
    print("Kernel:", KERNEL_NAME)
    print("Judge connection is separate; never share one participant token in a Launchable.")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        raise SystemExit("Course environment setup stopped: " + str(exc)) from None
