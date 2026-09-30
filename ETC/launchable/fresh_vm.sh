#!/bin/bash
# Fresh Brev VM path when platform-managed Jupyter installation is disabled.
# This creates only our authenticated course service, never a foreign service.
set -euo pipefail
umask 077

task_origin="${AI4SCI_SECURE_LINK_HOST:-}"
task_event=""
task_expected_host=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --secure-link-host|--enroll-event|--expected-host)
      [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 2; }
      case "$1" in
        --secure-link-host) task_origin="$2" ;;
        --enroll-event) task_event="$2" ;;
        --expected-host) task_expected_host="$2" ;;
      esac
      shift 2
      ;;
    --help)
      printf '%s\n' 'Usage: bash fresh_vm.sh --secure-link-host EXACT_BREV_DNS [--enroll-event ai4science-korea-2026] [--expected-host brev-INSTANCE_ID]'
      exit 0
      ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
task_vm_host="$(hostname)"
if [[ ! "$task_vm_host" =~ ^brev-[a-z0-9]+$ || ( -n "$task_expected_host" && "$task_vm_host" != "$task_expected_host" ) ]]; then
  echo "Refusing setup: this is not the expected Brev VM hostname." >&2
  exit 2
fi
if [[ -z "$task_origin" ]]; then
  echo "An exact provisioned Secure Link hostname is required; no hostname is guessed." >&2
  exit 2
fi
if [[ -n "$task_event" && "$task_event" != ai4science-korea-2026 ]]; then
  echo "Unsupported event enrollment." >&2
  exit 2
fi
python3 -I - "$task_origin" <<'PY'
import re, sys
hostname = sys.argv[1]
label = r'[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?'
if len(hostname) > 253 or not re.fullmatch(r'(?:' + label + r'\.)+' + label, hostname):
    raise SystemExit('Pass only the exact lowercase DNS hostname returned by Brev, not a URL or wildcard.')
PY
if [[ "$(id -u)" == 0 || "$(uname -s)" != Linux ]]; then
  echo "Run as the VM's non-root Linux course user, not sudo/root." >&2
  exit 2
fi
for task_command in python3 curl git sudo systemctl nvidia-smi flock; do
  command -v "$task_command" >/dev/null || { echo "Missing prerequisite: $task_command" >&2; exit 2; }
done
sudo -n true
task_user="$(id -un)"
task_home="$(python3 -I -c 'import os,pwd; print(pwd.getpwuid(os.getuid()).pw_dir)')"
if [[ "$HOME" != "$task_home" || -L "$task_home" ]]; then
  echo "The login user's real home must match HOME and must not be a symlink." >&2
  exit 2
fi
task_course="$task_home/AI4Sci-PhysicsNeMo-Bootcamp"
task_state="$task_home/.local/state/ai4sci-nojup-20260930"
python3 -I - "$task_home" "$task_state" <<'PY'
import os, pathlib, stat, sys
home, target = map(pathlib.Path, sys.argv[1:])
for path in (home, home / '.local', home / '.local/state', target):
    if path.is_symlink():
        raise SystemExit('Refusing symlinked setup state: ' + str(path))
    if path.exists():
        info = path.stat()
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o002:
            raise SystemExit('Unsafe setup state directory: ' + str(path))
    else:
        path.mkdir(mode=0o700)
PY
exec 9>"$task_state/setup.lock"
flock -n 9 || { echo 'Another setup is running; leaving it untouched.' >&2; exit 2; }
task_log="$(mktemp "$task_state/setup-XXXXXXXX.log")"
exec > >(tee -a "$task_log") 2>&1
echo "Setup host: $(hostname); user: $task_user; allowed Secure Link host: $task_origin; log: $task_log"

# Reject a foreign service before the download/install phase. A rerun may see
# only this helper's own service; full byte-for-byte checking happens below.
python3 -I - "$task_home" "$task_state" "$task_origin" <<'PY'
import json, os, pathlib, socket, stat, subprocess, sys
home = pathlib.Path(sys.argv[1])
state = pathlib.Path(sys.argv[2])
hostname = sys.argv[3]
unit = pathlib.Path('/etc/systemd/system/jupyter.service')
result = subprocess.run(['systemctl', 'show', 'jupyter.service', '--property=LoadState,DropInPaths', '--no-pager'], text=True, capture_output=True)
props = dict(line.split('=', 1) for line in result.stdout.splitlines() if '=' in line)
if props.get('DropInPaths'):
    raise SystemExit('Existing Jupyter service drop-ins require review; nothing was changed.')
if unit.is_symlink():
    raise SystemExit('Existing symlinked Jupyter service requires review.')
if unit.exists():
    if not unit.read_bytes().startswith(b'# AI4SCI_NOJUP_MANAGED_20260930_V1\n'):
        raise SystemExit('Existing Jupyter service is not owned by this helper; leaving it untouched.')
elif props.get('LoadState') != 'not-found':
    raise SystemExit('Another Jupyter unit already exists or its state is unknown; leaving it untouched.')
else:
    with socket.socket() as sock:
        sock.bind(('0.0.0.0', 8888))
config_dir = home / '.jupyter'
config = config_dir / 'jupyter_server_config.json'
if config_dir.is_symlink() or config.is_symlink():
    raise SystemExit('Symlinked Jupyter configuration requires review.')
others = [path for path in config_dir.glob('*config*') if path != config]
if others:
    raise SystemExit('Other Jupyter configuration requires review: ' + ', '.join(map(str, others)))
if config.exists():
    marker = state / 'jupyter-origin.json'
    if marker.is_symlink() or not marker.is_file():
        raise SystemExit('Existing Jupyter configuration is not owned by this helper.')
    expected = {'home': str(home), 'secure_link_hostname': hostname, 'format': 1}
    if json.loads(marker.read_text()) != expected:
        raise SystemExit('Existing helper configuration belongs to a different Secure Link.')
    info = config.stat()
    if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise SystemExit('Existing Jupyter configuration has unsafe ownership/permissions.')
    settings = json.loads(config.read_text())
    if set(settings) - {'ServerApp', 'LabApp', 'KernelSpecManager', 'MultiKernelManager'}:
        raise SystemExit('Unexpected Jupyter configuration sections require review.')
    server = settings.get('ServerApp', {})
    if (server.get('ip') != '0.0.0.0' or server.get('allow_remote_access') is not False
            or server.get('local_hostnames') != ['localhost', hostname]
            or set(server) - {'ip', 'allow_remote_access', 'local_hostnames', 'root_dir', 'default_url', 'jpserver_extensions'}):
        raise SystemExit('Existing Jupyter security configuration differs; it was not changed.')
print('Preflight passed: no foreign Jupyter service will be replaced.')
PY
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader

task_bootstrap="$(mktemp "$task_state/bootstrap-XXXXXXXX.py")"
curl --fail --silent --show-error --location --retry 3 \
  --connect-timeout 20 --max-time 120 \
  https://raw.githubusercontent.com/yang926/AI4Sci-PhysicsNeMo-Bootcamp/main/ETC/launchable/bootstrap.py \
  --output "$task_bootstrap"
# Deliberately omit --launchable/--refresh: those require an existing server.
python3 -I "$task_bootstrap" \
  --repo https://github.com/yang926/AI4Sci-PhysicsNeMo-Bootcamp.git \
  --ref main --destination "$task_course"

task_python="$(python3 -I - "$task_home" <<'PY'
import json, os, pathlib, sys
home = pathlib.Path(sys.argv[1])
spec = home / '.local/share/jupyter/kernels/ai4sci-physicsnemo-uv/kernel.json'
python = pathlib.Path(json.loads(spec.read_text())['argv'][0])
prefix = python.parent.parent
if (not python.is_absolute() or python.parent.name != 'bin'
        or prefix.parent != home / '.venvs' or not prefix.name.startswith('ai4sci-brev-')
        or prefix.is_symlink() or not python.is_file() or not (prefix / 'pyvenv.cfg').is_file()):
    raise SystemExit('Unexpected course Python path; refusing service setup.')
if json.loads((prefix / '.ai4sci-environment.json').read_text()).get('status') != 'ready':
    raise SystemExit('Course environment has not completed verification.')
print(python)
PY
)"

# Brev's authenticated HTTP proxy targets the VM network address, not loopback.
# Keep token authentication ON and explicitly disallow arbitrary Host headers.
# An exclusive create and private provenance file make reruns non-destructive.
python3 -I - "$task_home" "$task_state" "$task_origin" <<'PY'
import json, os, pathlib, sys
home, state = map(pathlib.Path, sys.argv[1:3])
hostname = sys.argv[3]
directory = home / '.jupyter'
directory.mkdir(mode=0o700, exist_ok=True)
if directory.is_symlink():
    raise SystemExit('Jupyter config directory became a symlink; no configuration written.')
marker = state / 'jupyter-origin.json'
expected = {'home': str(home), 'secure_link_hostname': hostname, 'format': 1}
created_marker = False
def exclusive_write(path, value):
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, 'w') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
if marker.exists() or marker.is_symlink():
    if marker.is_symlink() or json.loads(marker.read_text()) != expected:
        raise SystemExit('Secure Link provenance changed; no configuration written.')
else:
    exclusive_write(marker, expected)
    created_marker = True
config = directory / 'jupyter_server_config.json'
if not config.exists() and not config.is_symlink():
    exclusive_write(config, {'ServerApp': {
        'ip': '0.0.0.0', 'allow_remote_access': False,
        'local_hostnames': ['localhost', hostname],
    }})
    print('Created private Jupyter config: token defaults retained; only localhost and the exact Secure Link hostname allowed.')
else:
    if created_marker or config.is_symlink():
        raise SystemExit('An unexpected Jupyter config appeared during setup; it was not changed.')
    settings = json.loads(config.read_text())
    server = settings.get('ServerApp', {})
    if (server.get('ip') != '0.0.0.0' or server.get('allow_remote_access') is not False
            or server.get('local_hostnames') != ['localhost', hostname]
            or set(settings) - {'ServerApp', 'LabApp', 'KernelSpecManager', 'MultiKernelManager'}
            or set(server) - {'ip', 'allow_remote_access', 'local_hostnames', 'root_dir', 'default_url', 'jpserver_extensions'}):
        raise SystemExit('Jupyter config changed during setup; it was not overwritten.')
    print('Preserving the previously verified helper-owned Jupyter config.')
PY

# This narrow privileged helper creates only an absent exact unit. It never
# overwrites one; reruns require exact content and root-owned safe permissions.
sudo -n python3 -I - "$task_user" "$task_home" "$task_python" <<'PY'
import os, pathlib, pwd, re, socket, stat, subprocess, sys
user, home, python = sys.argv[1:]
record = pwd.getpwnam(user)
if record.pw_uid == 0 or record.pw_dir != home:
    raise SystemExit('Unexpected service account.')
if not all(re.fullmatch(r'[A-Za-z0-9_./-]+', value) for value in (user, home, python)):
    raise SystemExit('Service account/path contains unsupported characters.')
unit = pathlib.Path('/etc/systemd/system/jupyter.service')
content = f'''# AI4SCI_NOJUP_MANAGED_20260930_V1
[Unit]
Description=AI4Science authenticated course Jupyter
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User={user}
WorkingDirectory={home}
Environment=HOME={home}
UMask=0077
ExecStart={python} -m jupyterlab --no-browser --ServerApp.ip=0.0.0.0 --ServerApp.allow_remote_access=False --ServerApp.port=8888 --ServerApp.port_retries=0
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
'''.encode()
properties = subprocess.run(['systemctl', 'show', 'jupyter.service', '--property=LoadState,DropInPaths', '--no-pager'], capture_output=True, text=True)
props = dict(line.split('=', 1) for line in properties.stdout.splitlines() if '=' in line)
if props.get('DropInPaths'):
    raise SystemExit('Jupyter service drop-ins appeared; no unit was written.')
try:
    info = unit.lstat()
except FileNotFoundError:
    if props.get('LoadState') != 'not-found':
        raise SystemExit('Another Jupyter service appeared; no unit was written.')
    with socket.socket() as sock:
        sock.bind(('0.0.0.0', 8888))
    descriptor = os.open(unit, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644)
    with os.fdopen(descriptor, 'wb') as stream:
        os.fchmod(stream.fileno(), 0o644)
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    print('Created the new course-owned jupyter.service; default token authentication retained.')
else:
    if not stat.S_ISREG(info.st_mode) or info.st_uid != 0 or info.st_mode & 0o022 or unit.read_bytes() != content:
        raise SystemExit('Existing Jupyter service differs from this helper; it was not overwritten.')
    print('Reusing the identical course-owned jupyter.service without rewriting it.')
PY
sudo -n systemctl daemon-reload
sudo -n systemctl enable --now jupyter.service

# Wait without printing authentication tokens or server log contents. Verify
# authentication, not just a listening port, before configuring the course UI.
if ! "$task_python" -I - "$task_course" "$task_home" "$task_origin" <<'PY'
import importlib.util, pathlib, subprocess, sys, time, urllib.error, urllib.request
course, home = map(pathlib.Path, sys.argv[1:3])
hostname = sys.argv[3]
spec = importlib.util.spec_from_file_location('ai4sci_nojup_verify', course / 'ETC/launchable/jupyter.py')
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
last = None
for attempt in range(60):
    try:
        service = module.inspect_service(home)
        info = module._server_info(service)
        if not info.get('token'):
            raise RuntimeError('Jupyter did not generate a nonempty authentication token.')
        module._api(info, 'sessions')
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), module._NoRedirect())
        endpoint = 'http://127.0.0.1:8888/api/sessions'
        request = urllib.request.Request(endpoint, headers={'Host': hostname, 'Authorization': 'token ' + info['token']})
        with opener.open(request, timeout=5) as response:
            if response.status != 200:
                raise RuntimeError('Authenticated exact Secure Link Host was not accepted.')
        request = urllib.request.Request(endpoint, headers={'Host': hostname})
        try:
            with opener.open(request, timeout=5):
                raise RuntimeError('Jupyter API unexpectedly permits unauthenticated access.')
        except urllib.error.HTTPError as exc:
            if exc.code not in (401, 403):
                raise
        request = urllib.request.Request(endpoint, headers={'Host': 'unapproved.ai4sci.invalid', 'Authorization': 'token ' + info['token']})
        try:
            with opener.open(request, timeout=5):
                raise RuntimeError('Jupyter unexpectedly accepts an unapproved Host header.')
        except urllib.error.HTTPError as exc:
            if exc.code != 403:
                raise
        print('Jupyter ready: authenticated exact-host API works; anonymous access and unapproved Host headers are denied.')
        break
    except RuntimeError:
        raise
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        last = exc
        if attempt == 59:
            raise SystemExit('Jupyter readiness verification failed: ' + str(last))
        time.sleep(1)
PY
then
  echo 'Jupyter verification failed; stopping only the exact course-owned service created/reused by this helper.' >&2
  sudo -n systemctl stop jupyter.service
  exit 1
fi

task_install_args=(--course-dir "$task_course" --configure-jupyter)
if [[ -n "$task_event" ]]; then
  task_install_args+=(--enroll-event "$task_event")
fi
python3 -I "$task_course/ETC/launchable/install.py" "${task_install_args[@]}"
echo 'SUCCESS: course environment and authenticated exact-host Jupyter verified.'
echo 'No firewall or public route was changed. Verify the protected Brev route separately.'
echo 'Event opt-in installs only the restricted enrollment receiver; no shared judge or Brev credential is embedded.'
echo "Course revision: $(git -C "$task_course" rev-parse HEAD)"
echo "Setup log: $task_log"
