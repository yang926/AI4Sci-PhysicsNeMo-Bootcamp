#!/usr/bin/env python3
"""Install a course kernel for Brev-managed Jupyter, without starting a server."""
import argparse
import hashlib
import importlib.util
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
KERNEL_DISPLAY_NAME = "AI4Sci PhysicsNeMo 2.2.2 (uv / CUDA)"
LOCK_NAME = "requirements-linux-cu128.lock.txt"
# The managed Jupyter environment is separate from the CUDA course kernel.
# Add missing packages only; compatible existing server dependencies stay put.
PROXY_PACKAGES = {
    "jupyter-server-proxy": "4.6.0", "simpervisor": "1.0.0", "aiohttp": "3.14.3",
    "aiohappyeyeballs": "2.7.1", "aiosignal": "1.4.0", "attrs": "26.1.0",
    "frozenlist": "1.8.0", "multidict": "6.9.0", "propcache": "0.5.4",
    "yarl": "1.25.1", "idna": "3.20", "typing-extensions": "4.16.0",
}
PROXY_COMPATIBILITY = {
    "jupyter-server-proxy": "==4.6.0", "simpervisor": "==1.0.0", "aiohttp": ">=3.14.3,<4",
    "aiohappyeyeballs": ">=2.5.0", "aiosignal": ">=1.4.0", "attrs": ">=17.3.0",
    "frozenlist": ">=1.1.1", "multidict": ">=4.5,<7", "propcache": ">=0.2.1",
    "yarl": ">=1.17.0,<2", "idna": ">=2.0", "typing-extensions": ">=4.4",
}


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


def verify_environment_metadata(python, prefix, lock):
    """Check the interpreter and pinned packages without importing CUDA.

    A ready marker alone is insufficient: the interpreter or packages may have
    been removed since installation. Distribution metadata does not import
    torch, PhysicsNeMo, or start GPU contexts.
    """
    run(python, "-I", "-c", "\n".join([
        "import importlib.metadata as metadata, pathlib, sys",
        "assert pathlib.Path(sys.prefix) == pathlib.Path(sys.argv[1]), 'Wrong environment prefix'",
        "assert '.'.join(map(str, sys.version_info[:3])) == sys.argv[2], 'Wrong Python version'",
        "for line in pathlib.Path(sys.argv[3]).read_text().splitlines():",
        "    if line.strip() and not line.lstrip().startswith('#'):",
        "        name, expected = line.strip().split('==', 1)",
        "        assert metadata.version(name) == expected, 'Package differs from course lock: ' + name",
    ]), prefix, PYTHON_VERSION, lock)


def ensure_environment(course, home, *, refresh=False):
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
                or metadata.get("python", PYTHON_VERSION) != PYTHON_VERSION
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
        if refresh:
            verify_environment_metadata(python, prefix, lock)
        else:
            run(python, course / "ETC/launchable/verify.py")
    return prefix


def connect_kernel(prefix, *, check_only=False):
    """Connect missing integrations, leaving matching kernel/widget files intact.

    Return whether a change is needed/made. ``check_only`` permits the caller to
    verify that the managed server is idle before replacing a kernel definition.
    """
    python = prefix / "bin/python"
    data = Path(subprocess.check_output([str(python), "-c", "from jupyter_core.paths import jupyter_data_dir; print(jupyter_data_dir())"], text=True).strip())
    spec = data / "kernels" / KERNEL_NAME / "kernel.json"
    existing = None
    if spec.is_symlink() or spec.parent.is_symlink():
        raise ValueError("A symlinked course kernelspec requires review; it was not replaced.")
    if spec.exists():
        existing = json.loads(spec.read_text())
        if not isinstance(existing, dict) or not existing.get("argv"):
            raise ValueError("The existing course kernelspec is incomplete; it was not replaced.")
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
    argv = [str(python), "-m", "ipykernel_launcher", "-f", "{connection_file}"]
    frozen_argv = [str(python), "-Xfrozen_modules=off", *argv[1:]]
    kernel_matches = (existing is not None and existing.get("argv") in (argv, frozen_argv)
                      and existing.get("display_name") == KERNEL_DISPLAY_NAME
                      and existing.get("language") == "python"
                      and isinstance(existing.get("env"), dict)
                      and existing["env"].get("AI4SCI_DEVICE") == "cuda")
    changed = not kernel_matches or not exists
    if check_only:
        return changed
    if not kernel_matches:
        run(python, "-m", "ipykernel", "install", "--user", "--name", KERNEL_NAME,
            "--display-name", KERNEL_DISPLAY_NAME, "--env", "AI4SCI_DEVICE", "cuda")
    if not exists:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(extension, target_is_directory=True)
    return changed


def load_jupyter_setup():
    # Loading the sibling by path works outside the checkout or when the root
    # package is absent from sys.path.
    spec = importlib.util.spec_from_file_location("ai4sci_jupyter_setup", Path(__file__).with_name("jupyter.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def managed_server_python(service, home):
    """Accept only the verified user's virtualenv, never mutate system Python."""
    argv = service.argv
    if argv and Path(argv[0]).name.startswith("python"):
        python = Path(argv[0])
    elif argv and Path(argv[0]).name in {"jupyter", "jupyter-lab", "jupyter-server"}:
        first = Path(argv[0]).open(encoding="utf-8").readline().strip()
        if not first.startswith("#!/") or " " in first:
            raise ValueError("The Jupyter entrypoint has an unsupported Python shebang.")
        python = Path(first[2:])
    else:
        raise ValueError("Cannot identify the managed Jupyter Python interpreter.")
    if (not python.is_absolute() or not python.is_relative_to(home)
            or python.parent.name != "bin" or not (python.parent.parent / "pyvenv.cfg").is_file()
            or not python.is_file()):
        raise ValueError("Jupyter proxy setup requires the existing user-owned server virtualenv; system Python was not changed.")
    return python


def proxy_environment(python):
    script = "\n".join([
        "import importlib.metadata as m, json, pathlib, sys",
        "from packaging.version import Version",
        "from packaging.specifiers import SpecifierSet",
        "assert pathlib.Path(sys.prefix) == pathlib.Path(sys.argv[1]), 'Unexpected server environment'",
        "assert sys.version_info >= (3, 11), 'Server Python 3.11 or later required'",
        "required = {'jupyter-server': '1.24.0', 'tornado': '6.1.0', 'traitlets': '5.1.0'}",
        "for name, minimum in required.items():",
        "    assert Version(m.version(name)) >= Version(minimum), 'Incompatible existing server dependency: ' + name",
        "versions = {}",
        "constraints = json.loads(sys.argv[2])",
        "for name in constraints:",
        "    try: versions[name] = m.version(name)",
        "    except m.PackageNotFoundError: versions[name] = None",
        "    if versions[name] is not None:",
        "        assert Version(versions[name]) in SpecifierSet(constraints[name]), 'Incompatible existing server dependency: ' + name",
        "print(json.dumps(versions))",
    ])
    return json.loads(subprocess.check_output([str(python), "-I", "-c", script,
                                               str(python.parent.parent), json.dumps(PROXY_COMPATIBILITY)], text=True))


def ensure_server_proxy(home, module):
    """Install only missing proxy pins, after checking the known server is idle."""
    home = Path(home).resolve()
    service = module.inspect_service(home)
    python = managed_server_python(service, home)
    installed = proxy_environment(python)
    wrong = {name: installed.get(name) for name in ("jupyter-server-proxy", "simpervisor")
             if installed.get(name) is not None and installed[name] != PROXY_PACKAGES[name]}
    if wrong:
        raise ValueError("Existing Jupyter proxy packages differ from the tested pins; review before replacing: " + str(wrong))
    missing = [f"{name}=={version}" for name, version in PROXY_PACKAGES.items() if installed.get(name) is None]
    if missing:
        module.require_managed_idle(home)
        uv = ensure_uv(home)
        current = module.inspect_service(home)
        if current.pid != service.pid or current.argv != service.argv:
            raise ValueError("Managed Jupyter changed during proxy setup; no packages were installed.")
        module.require_managed_idle(home)
        run(uv, "pip", "install", "--python", python, "--no-deps", *missing)
        verified = proxy_environment(python)
        if any(verified.get(name) is None for name in PROXY_PACKAGES):
            raise ValueError("Managed Jupyter proxy package verification failed.")
        run(python, "-I", "-c", "import aiohttp, jupyter_server_proxy, simpervisor")
    return bool(missing)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--course-dir", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--configure-jupyter", action="store_true", help="Configure the existing Brev Jupyter service for this course")
    parser.add_argument("--refresh", action="store_true", help="Reuse matching verified dependencies without importing CUDA or reinstalling the kernel")
    args = parser.parse_args()
    if sys.platform != "linux" or os.geteuid() == 0:
        parser.error("Use a Linux NVIDIA GPU VM as the same non-root user as Brev Jupyter.")
    if not shutil.which("nvidia-smi"):
        parser.error("NVIDIA drivers are required. Select a GPU VM; do not install drivers from this script.")
    course = args.course_dir.resolve()
    prefix = ensure_environment(course, Path.home(), refresh=args.refresh)
    module = None
    if args.configure_jupyter or args.refresh:
        module = load_jupyter_setup()
    if args.configure_jupyter:
        ensure_server_proxy(Path.home(), module)
    if connect_kernel(prefix, check_only=True):
        if module is not None:
            module.require_managed_idle(Path.home())
        connect_kernel(prefix)
    if args.configure_jupyter:
        result = module.configure_jupyter(course, prefix, Path.home())
        if result["status"] == "unchanged":
            print("Managed Jupyter already matches the course. Running kernels were left untouched.")
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
