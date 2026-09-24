# Student Brev Launchable

Use the existing course repository. Brev provides the VM and managed Jupyter;
the setup below installs an isolated course kernel and configures the existing
managed Jupyter service to open the course. It does not start a second server,
change authentication, create GPU instances or create judge accounts.

## Brev builder settings

| Setting | Value |
| --- | --- |
| Source | `https://github.com/yang926/AI4Sci-PhysicsNeMo-Bootcamp.git` |
| Software | **VM Mode** |
| Install Jupyter on the host | **On** |
| Hardware | One NVIDIA GPU per learner. Verify the actual L4/L40 selection before deployment. |
| Disk | Start rehearsal with 100 GiB; measure package/data/output usage before the event. |
| Network | Keep the managed **Jupyter Secure Link**. Do not expose raw Jupyter TCP ports publicly. |
| Access | Prefer **Only my organization** for the workshop. Visibility does not limit shared-credit spending. |

Hardware and disk values are rehearsal recommendations, not measured capacity
or a confirmed allocation. This repository does not change the existing Launchable.

The organizer configures this once in the shared Launchable. Students deploy
that template and open Jupyter; they do not install pip or run recovery commands.
Use the script below for a new Launchable, or replace the earlier script in the
existing Launchable before sharing it. GitHub changes do not edit a saved Brev
form automatically. This repository cannot save the Brev console form for you.

In **Setup script → Paste Script**, use this small bootstrap. Its URL stays the
same when the implementation in GitHub changes. If the builder offers a URL
field, do not give it an HTML `github.com/.../blob/...` page as executable code.
The pasted bootstrap works without relying on that optional field.
The first line must be exactly `#!/bin/bash`, with no blank line before it.
Brev's form rejects `#!/usr/bin/env bash` even though it is a valid shell shebang.

```bash
#!/bin/bash
set -euo pipefail
bootstrap_file="$(mktemp -t ai4sci-bootstrap.XXXXXX)"
curl --fail --silent --show-error --location --retry 3 \
  https://raw.githubusercontent.com/yang926/AI4Sci-PhysicsNeMo-Bootcamp/main/ETC/launchable/bootstrap.py \
  --output "$bootstrap_file"
python3 "$bootstrap_file" --launchable
```

The same script is available as [setup.sh](setup.sh) for the builder's File Upload
option. Use `--launchable`, not the earlier `--update` flag. It uses the existing
Source checkout at `~/AI4Sci-PhysicsNeMo-Bootcamp` and checks the requested GitHub
revision. A clean upstream checkout can advance in place; local edits or private
commits block an upgrade rather than being overwritten. A fresh clone is staged
until complete so a failed download can be retried without a partial course folder.

The script runs as Brev's default non-root user, which must also own the Jupyter
session. It handles a setup working directory outside the checkout. Repeated
setup uses the same canonical folder and does not create timestamped copies.
Previous `*-updates` folders are left intact outside the new course browser root;
they are not silently deleted, merged or selected as the active course.

After setup finishes, open **Jupyter** in Brev. The file browser starts at the
course root, **Start_Here.ipynb** is the landing page, and the available/default
kernel is **AI4Sci PhysicsNeMo 2.2.2 (uv / CUDA)**. Reload a browser that was open
before setup completed. Do not start `jupyter lab` again in a terminal.

Setup configures the normal per-user Jupyter server configuration, preserving
existing authentication and network settings. It identifies only Brev's existing
`jupyter.service`, checks that it belongs to the same user and has no active
notebook sessions or kernels, then restarts that unit and verifies the effective
root, notebook URL and kernel list. Unknown service wrappers, conflicting startup
arguments or active notebooks stop setup with an organizer-facing error. No other
Jupyter service is stopped. This is an installation step, not a recurring restart.

The kernel is installed in the user's Jupyter data directory under the unique
name `ai4sci-physicsnemo-uv`; the host's `python3` kernel is not overwritten.
The native kernel is retained on disk but hidden from this course server's kernel
list. This GPU kernel supplies `AI4SCI_DEVICE=cuda`, including for the morning Labs.
The prebuilt widgets frontend is made available in the user data directory
without installing Python packages into Brev's managed Jupyter environment.
A conflicting frontend/kernel is reported, never silently replaced.

The host needs Python 3, Git, curl and NVIDIA drivers, but does not need pip,
ensurepip or a preinstalled uv. The pinned uv 0.8.17 standalone installer uses
a private directory without modifying shell profiles or the host Python.

The earlier installer at revision `b9e4b48` assumed the host had pip. The corrected
shared setup above is the student deployment path, not per-student terminal repair.
After changing the template, the organizer rehearses a fresh deployment before
distributing it. Existing failed VMs are not changed merely by editing the template.

## Updating the course

The checked-in CUDA lock pins all 181 packages, including widgets. Environments
are named by that lock's fingerprint. A new lock gets a separate environment;
the previous one is retained. This lock reproduces the local WSL package set,
not a claim that a clean Brev VM has already passed rehearsal.
An interrupted package download can be retried in the matching installer-owned
environment. Unrecognized existing environments are never repaired by replacement.

- **New deployments:** the source/bootstrap uses GitHub `main` by default.
- **Existing learners:** their running workspace does not change when GitHub
  changes. A deliberate setup rerun uses the canonical folder. Uncommitted work
  or private commits prevent source upgrades; ignored training outputs remain.
- **Manual bootstrap without `--update`:** retains its original behavior of
  reusing the existing checkout. This is not the shared Launchable's setup command.
- **Instructor copies:** the manual `--update` option still creates a dated copy
  without changing earlier work. It is not used by the student Launchable and
  does not switch the active Jupyter root to that copy.
- **Event freeze:** set an optional Launch parameter `AI4SCI_COURSE_REF` to a
  tested commit SHA or tag before deployment. It must include this Launchable
  installer. An existing checkout is never silently downgraded or moved off local
  commits; use a fresh destination for a different release. To freeze the bootstrap
  as well, replace `main` in its raw URL with that commit SHA. Do not move the tag.
- **Different repository:** change both Brev Source and optional
  `AI4SCI_COURSE_REPO`. Use a public GitHub repository with this same installer
  layout and an unused destination. Do not reuse a working folder from another repo.

Updating GitHub or editing a Launchable does not update a running student's
files. VM setup does not automatically run again when the VM restarts.

## Judge connection

Local Labs and Challenges work without the judge. Nickname registration and
submission are enabled only after the student's workspace has its own private
judge connection. The projector remains a separate, read-only page.

Do not put a common participant token, administrator token or private roster in
this script, the repository, a shared image or Launch parameter defaults.
Personal credential provisioning and the event HTTPS API are still separate
deployment work. A browser-authenticated Jupyter Secure Link is not a notebook
kernel's judge API login. See [judge connection](../environment/JUDGE_CONNECTION.md).

## Before sharing with 110 learners

Deploy **one** rehearsal GPU VM after approving its cost, then check:

1. Setup completes on the selected GPU, with sufficient disk and no dependency conflict.
2. The managed Jupyter Secure Link opens Start Here at the course root. No source
   or timestamp-folder choice is needed, and only the course kernel is offered.
3. Run **00_Setup** in the course kernel; confirm CUDA and the selected GPU.
4. In each Challenge, run only its setup and submission-control cells first.
   The nickname input/buttons must render as widgets, not plain text. With no
   judge provisioned, registration/submission must remain disabled and practice
   must still work. Do not enable Run All against reference solutions.
5. Re-run setup before student work: no duplicate folder is created. Confirm
   learner edits block an attempted source upgrade and remain unchanged.
6. Stop/start that same VM and verify managed Jupyter, storage and course kernel
   persist. Measure cold-start time before issuing 110 deployment links.

Do not present static tests or this checklist as a completed Brev deployment.
Record the actual template revision and rehearsal results before event distribution.

Official references: [Launchables](https://docs.nvidia.com/brev/concepts/launchables),
[setup scripts](https://docs.nvidia.com/brev/cli/instance-management),
[Brev Jupyter port conflict](https://docs.nvidia.com/datascience/deployment/stable/cloud/nvidia/brev/),
[widgets across separate environments](https://ipywidgets.readthedocs.io/en/stable/user_install.html),
[uv standalone installation](https://docs.astral.sh/uv/getting-started/installation/),
[uv unmanaged installer options](https://docs.astral.sh/uv/reference/installer/).
