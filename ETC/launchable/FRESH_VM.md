# Fresh event VM without Brev's Jupyter installer

This deployment path keeps the course, GPU environment and exercises unchanged.
It separates the course Jupyter installation from Brev's optional host Jupyter
installation. It is not a workaround for unavailable GPUs or every possible
NetBird registration failure.

## Organizer configuration

- Use VM mode and the existing event hardware/source settings.
- Turn **Install Jupyter on the host** off.
- Add a protected HTTP Secure Link named `jupyter`, destination port `8888`.
  Do not add a public raw TCP rule or unauthenticated access.
- Use the startup script below. Keep the existing event Launchable ID so the
  judge's server-side Launchable provenance check continues to recognize it.

```bash
#!/bin/bash
set -euo pipefail
umask 077
task_vm_host="$(hostname)"
if [[ ! "$task_vm_host" =~ ^brev-([a-z0-9]+)$ ]]; then
  echo 'Unexpected Brev VM hostname.' >&2
  exit 2
fi
# Expected hostname for this event's named Brev Secure Link.
# Verify the actual returned hostname when rehearsing a new template/provider.
task_secure_host="jupyter-${BASH_REMATCH[1]}.gobrev.dev"
task_setup="$(mktemp -t ai4sci-fresh-vm.XXXXXX)"
curl --fail --silent --show-error --location --retry 3 \
  --connect-timeout 20 --max-time 120 \
  https://raw.githubusercontent.com/yang926/AI4Sci-PhysicsNeMo-Bootcamp/main/ETC/launchable/fresh_vm.sh \
  --output "$task_setup"
bash "$task_setup" --secure-link-host "$task_secure_host" \
  --expected-host "$task_vm_host" --enroll-event ai4science-korea-2026
```

The helper accepts the exact Secure Link hostname; it does not call the Brev API
or carry an organizer credential. The named route must be created by the
Launchable itself. If Brev returns a different domain, update the template's
hostname argument before distributing it; do not allow arbitrary Host headers.

## Access and enrollment

Students open their instance's **Open course** Secure Link. No local tunnel or
manual port opening is required. Brev access control and Jupyter's own token
authentication remain enabled. This script does not disable authentication or
publish the token; a bare link can still show Jupyter's login page. The instance
owner can obtain their private token using the registered course environment's
`jupyter server list`. Do not share a token or place it in the Launchable.

After the VM build completes, the running event enrollment service discovers
matching Launchable instances on a 20-second inventory cycle. It installs a
personal credential through the restricted receiver and maintains the private
loopback connection. Network or service failures can delay registration; the
20-second cycle is not an availability guarantee.

In a Challenge notebook, run the setup and submission-control cells. The widget
shows connection status, then offers **Nickname → Register nickname**. The same
nickname is used across all four Challenges. Submission still requires an
explicit button click. Local practice does not require the judge.

## Scope and verification

The helper creates only an absent course-owned `jupyter.service`. It refuses
foreign services and conflicting configuration. It checks an authenticated API
request, rejects anonymous API access and unknown Host headers, and configures
Start Here and the course CUDA kernel. Existing learner VMs are not changed by
editing this template. Use the existing `update.sh` workflow for their materials.

Before directing learners to a modified template, verify a fresh deployment's
actual Secure Link hostname and access policy, notebook CUDA execution and
personal judge connection. Local unit tests alone do not validate Brev routing
or a 110-student launch.
