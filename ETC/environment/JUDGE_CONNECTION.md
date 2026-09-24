# Connect the notebook to the judge

Students submit inside the Challenge notebook. They save their exercise files,
run the last cell and enter a **Nickname**, then click **Register nickname**.
This name is saved on their personal judge account and reused across Challenges
1-4 and the public scoreboard. Select Levels and click **Submit code**. The panel shows their queue status, score and evaluation details.
While a job is pending it refreshes every five seconds, for up to 15 minutes.
**Refresh results** resumes checks. Rerunning the cell closes its previous panel.
Run All never registers a name or submits work. There is no separate judge login
or website upload. Nicknames must contain 1-40 visible characters and be unique;
case, full-width characters and extra spaces do not create distinct names.
**Save nickname** changes the public name without changing the account or scores.
Submission is disabled until the name is saved. Do not use private information
as a public nickname. If the judge is not connected, both registration and
submission are unavailable, but local practice remains usable.

The projector only shows standings. Closing the projector does not stop the
judge. The notebook's Python kernel sends code to the API; the student's browser
does not need an SSH tunnel to the judge.

## Workspace configuration (instructor)

Provision a different judge credential for each participant. On the judge,
`add-participant` without a name creates an unnamed private account; it does not
assign a random public nickname. The student chooses the name in Jupyter.
Optional instructor-assigned names are still supported. The notebook reads
an owner-only JSON file at `~/.config/ai4sci/judge.json`, outside the course and
Jupyter's served directory. Its two fields are `url` and `token`. An explicit
`AI4SCI_JUDGE_CONFIG` can select another owner-only file. Do not put this file in
the course checkout, a shared image, GitHub, slide decks or notebook outputs.
The configuration belongs to the same OS user that runs Jupyter.

A launch setup can receive `AI4SCI_JUDGE_URL` and a **personal**
`AI4SCI_JUDGE_TOKEN` through private provisioning, then run this command from the
course root using the course Python environment:

```bash
python -m ETC.runtime.judge_client
```

This saves the two values with file mode 600 in a mode-700 directory. It refuses
to overwrite existing configuration. It does not print credentials, create a
judge account, discover a Brev identity, contact the server or allocate GPUs.
Do not enable shell tracing or print the setup environment. For a local rehearsal,
the client can also read these environment variables directly; a notebook URL
override must match the configured URL before credentials are sent.

Persisting configuration is needed for VM-mode Launchables: their launch
parameters reach the setup process, not automatically every future Jupyter
kernel. See [Brev launch parameters](https://docs.nvidia.com/brev/concepts/launchables).
The setup hook must run as the Jupyter user, before starting Jupyter. Preparing
the environment and assigning a personal credential are different operations.

## What is not deployed yet

The client and notebook controls are implemented. The separate
[judge repository](https://github.com/yang926/ai4sci-physicsnemo-judge) has the
submission API and a read-only projector listener. Its server still binds to
loopback for local development. The production API must be reachable by student
**instances**, use HTTPS and accept scoped personal credentials without browser
login redirects. A browser-only Brev Secure Link is not by itself an API login.
Do not disable server Host/Origin checks, publish the pilot HTTP listener or
give learners administrative SSH access to make the connection work.

An HTTPS endpoint, explicit deployment origin policy, restricted workers,
credential expiry/revocation and verified mapping from a Brev account to a
participant remain event deployment work. The shared Launchable must not contain
one common participant token or a judge/admin API key. A typed name, instance
name or email is not proof of identity. We have not verified a Brev identity
handoff that would make per-person provisioning automatic.

The client only allows plain HTTP for literal loopback addresses in rehearsal.
It verifies HTTPS normally, rejects redirects and does not use ambient proxy
settings. A disconnected panel keeps previous results with a warning and never
automatically retries a submission. An identical retry is deduplicated by the
judge while the job is queued, running or completed.
