"""Fresh-VM entrypoint guards; never allocate a VM or start systemd services."""
from pathlib import Path
import json
import os
import subprocess

import pytest

from ETC.launchable import jupyter


SCRIPT = Path(__file__).parents[1] / "launchable/fresh_vm.sh"


def embedded_python(source):
    current = None
    for line in source.splitlines():
        if current is None and line.endswith("<<'PY'"):
            current = []
        elif current is not None and line == "PY":
            yield "\n".join(current)
            current = None
        elif current is not None:
            current.append(line)
    assert current is None, "Unterminated Python heredoc"


def test_fresh_vm_shell_and_embedded_python_syntax():
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)
    blocks = list(embedded_python(SCRIPT.read_text()))
    assert len(blocks) == 7
    for number, source in enumerate(blocks, 1):
        compile(source, f"fresh-vm-block-{number}", "exec")


@pytest.fixture
def safe_environment(tmp_path):
    binaries = tmp_path / "bin"
    binaries.mkdir()
    hostname = binaries / "hostname"
    hostname.write_text("#!/bin/sh\nprintf '%s\\n' brev-fixture123\n")
    hostname.chmod(0o755)
    sentinel = tmp_path / "unsafe-command-called"
    for name in ("curl", "sudo", "systemctl", "nvidia-smi"):
        command = binaries / name
        command.write_text(
            "#!/bin/sh\nprintf '%s\\n' unexpected > \"$AI4SCI_TEST_SENTINEL\"\nexit 99\n"
        )
        command.chmod(0o755)
    environment = dict(os.environ, PATH=str(binaries) + os.pathsep + os.environ["PATH"],
                       AI4SCI_TEST_SENTINEL=str(sentinel))
    environment.pop("AI4SCI_SECURE_LINK_HOST", None)
    return environment, sentinel


@pytest.mark.parametrize("arguments, expected", [
    ([], "exact provisioned Secure Link hostname"),
    (["--unknown"], "Unknown argument"),
    (["--secure-link-host"], "Missing value"),
    (["--secure-link-host", "jupyter-fixture123.gobrev.dev", "--expected-host", "brev-other"],
     "expected Brev VM hostname"),
    (["--secure-link-host", "jupyter-fixture123.gobrev.dev", "--enroll-event", "unknown"],
     "Unsupported event"),
    (["--secure-link-host", "https://jupyter-fixture123.gobrev.dev"], "exact lowercase DNS hostname"),
    (["--secure-link-host", "*.gobrev.dev"], "exact lowercase DNS hostname"),
    (["--secure-link-host", "good.gobrev.dev\nother.invalid"], "exact lowercase DNS hostname"),
])
def test_invalid_inputs_stop_before_install_or_privileged_commands(safe_environment, arguments, expected):
    environment, sentinel = safe_environment
    result = subprocess.run(["bash", str(SCRIPT), *arguments], env=environment, capture_output=True, text=True)
    assert result.returncode != 0
    assert expected in result.stderr
    assert not sentinel.exists()


def test_help_has_no_installation_side_effects(safe_environment):
    environment, sentinel = safe_environment
    result = subprocess.run(["bash", str(SCRIPT), "--help"], env=environment, capture_output=True, text=True)
    assert result.returncode == 0
    assert "--secure-link-host EXACT_BREV_DNS" in result.stdout
    assert not sentinel.exists()


def test_security_and_existing_install_paths_are_preserved():
    source = SCRIPT.read_text()
    assert "--ServerApp.allow_remote_access=False" in source
    assert "'allow_remote_access': False" in source
    assert "'local_hostnames': ['localhost', hostname]" in source
    assert "os.O_EXCL | os.O_NOFOLLOW" in source
    assert "unit.read_bytes() != content" in source
    assert "if not info.get('token')" in source
    assert "unapproved.ai4sci.invalid" in source
    assert "--enroll-event \"$task_event\"" in source
    assert "--course-dir \"$task_course\" --configure-jupyter" in source
    assert "--repo https://github.com/yang926/AI4Sci-PhysicsNeMo-Bootcamp.git" in source
    assert "--ref main --destination \"$task_course\"" in source
    for unsafe in ("--public", "disable_check_xsrf", "allow_origin", "IdentityProvider.token="):
        assert unsafe not in source


def test_existing_course_configuration_keeps_fresh_vm_host_restrictions(tmp_path):
    server = {"ip": "0.0.0.0", "allow_remote_access": False,
              "local_hostnames": ["localhost", "jupyter-fixture123.gobrev.dev"]}
    initial = json.dumps({"ServerApp": server}).encode()
    configured = json.loads(jupyter.course_config(initial, tmp_path))
    for key, value in server.items():
        assert configured["ServerApp"][key] == value
    assert "IdentityProvider" not in configured  # Random-token authentication remains the default.
    assert configured["MultiKernelManager"]["default_kernel_name"] == jupyter.KERNEL_NAME
