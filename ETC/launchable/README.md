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
revision. Existing checkouts use the safe update path described below; private
commits and overlapping source edits are never overwritten. A fresh clone is staged
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
`jupyter.service` and checks its owner, effective root, notebook URL and kernel
list. If the configuration already matches, it does not rewrite it or restart
the server, even with idle notebook kernels. A configuration change requires an
idle server before that unit is restarted. Unknown service wrappers and conflicting
startup arguments stop setup. No other Jupyter service is stopped.

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

Save your work, let training finish, and close notebook/text-editor tabs that
you are editing. Keep idle kernels and the Jupyter server running. In the
instance terminal, run:

```bash
bash ~/AI4Sci-PhysicsNeMo-Bootcamp/ETC/launchable/update.sh
```

On an older instance that does not have `update.sh` yet, the existing command
downloads the new bootstrap and uses the same safe update path:

```bash
bash ~/AI4Sci-PhysicsNeMo-Bootcamp/ETC/launchable/setup.sh
```

For ordinary notebook/Python changes, **do not manually stash files, reinstall
packages, or restart the Jupyter server**. The update command:

1. Checks the selected GitHub revision and the files that would change.
2. Preserves local answers, notes and untracked files that do not overlap the
   release. If learner source and the release both changed the same file, it
   stops and names the conflict before changing course files.
3. Backs up execution-only notebook changes before replacing an updated notebook.
   Backups are private, outside the course folder, under
   `~/.ai4sci-course-backups/`. The output prints the exact directory; each backup
   contains the original notebook bytes and a revision/file manifest. Unknown
   metadata, cell source, notes and attachments are treated as learner content,
   not disposable output.
4. Reuses the existing ready Python environment and kernel. Unchanged Jupyter
   configuration is checked without restarting the server or stopping kernels.

If a notebook that will be replaced is still connected in your browser, the
command asks you to save and close that tab (or the Jupyter browser tab), then
rerun the **same command**. You do not need to shut down idle kernels. This keeps
an old browser editor from saving over the newly downloaded notebook. A running
training cell must finish before source files are updated. Jupyter's API cannot
detect every browser text editor or a notebook that never started a kernel;
closing editing tabs first is still necessary even if the command detects none.

After success, reopen changed notebooks from disk. Python modules already
imported by a kernel are not hot-reloaded; restart only that notebook's kernel
before using the changed module. Saved results and learner answers stay on disk.
An update does not automatically restore old Git stashes over the new files.

If the release changes Python or locked packages, the command says so **before
updating source** and asks you to save and shut down kernels. Only that kind of
environment change needs the installation path. If installation subsequently
fails, the log reports the source revision separately; do not interpret a
downloaded revision as a completed environment installation.

The checked-in CUDA lock pins the course packages, including widgets. Environments
are named by that lock's fingerprint. A new lock gets a separate environment;
the previous one is retained. This lock reproduces the local WSL package set,
not a claim that a clean Brev VM has already passed rehearsal.
An interrupted package download can be retried in the matching installer-owned
environment. Unrecognized existing environments are never repaired by replacement.

- **New deployments:** the source/bootstrap uses GitHub `main` by default.
- **Existing learners:** GitHub changes do not silently alter their files. The
  explicit update above uses the canonical folder and preserves local work.
  Ignored training outputs remain; colliding files or private commits stop updates.
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
5. Re-run setup: no duplicate folder or unnecessary server restart occurs.
   Check an output-only saved notebook, an unrelated learner answer, an overlapping
   answer, and a connected notebook tab. Backups and conflict messages must match
   the cases above; learner work must remain recoverable.
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
