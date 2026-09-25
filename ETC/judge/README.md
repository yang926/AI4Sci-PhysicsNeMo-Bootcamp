# Challenge 1-4 scoring pilot

This is a local development system for this course, not a deployed event service.
It accepts individual submissions, queues fixed-budget evaluations and displays
standings for the selected Challenge. All eleven Levels are included: Wave,
Fluid, Climate, and Neural Operators (FNO, AFNO and PINO).

## Start locally

Use the course Python environment and run these commands from the repository
root. Keep judge state outside the repository and outside Jupyter's served
directory. The following path is an example private directory, not a URL.

```bash
python -m ETC.judge --state "$HOME/.local/share/ai4sci-judge-pilot-v3" init --steps 2 --device cpu
python -m ETC.judge --state "$HOME/.local/share/ai4sci-judge-pilot-v3" add-participant
```

The second command prints a randomly generated participant access code once.
Provision each personal code privately into its workspace, outside the course and Jupyter root. It is not a Brev credential.
Without an optional name, `add-participant` creates an unnamed private account.
The student enters **Nickname** and clicks **Register nickname** in the notebook
submission panel. No random public nickname is assigned. The name is saved to
the authenticated account and reused in all four Challenges and the scoreboard.
Brev account names are not fetched automatically. Without a valid personal
credential, submission is rejected, not recorded anonymously.
Only its hash is stored in the database. Do not put codes in URLs, notebooks,
Git, screenshots or shell command arguments.

Start the web page in one terminal and the worker in another:

```bash
python -m ETC.judge --state "$HOME/.local/share/ai4sci-judge-pilot-v3" serve --port 8090
python -m ETC.judge --state "$HOME/.local/share/ai4sci-judge-pilot-v3" worker --device cpu
```

The page is at `http://127.0.0.1:8090/` **on the computer running the server**.
To use it on a Mac connected to another computer, forward remote port 8090 to
Mac port 8090 first, just as for Jupyter's port 8888. No forward is created by
these commands. The notebook API URL must be reachable from its Python kernel; the Mac display URL is not the student API address.
Do not expose the local development HTTP server to the public internet.

## Projector and notebook

Both `/` and `/display` show the read-only scoreboard. There is no web upload
form. For the example port above, open `http://127.0.0.1:8090/display` on the
host or on the Mac after forwarding port 8090. These are local addresses, not
deployed event links. Students submit and read their personal results inside
the Challenge notebook, using a privately configured workspace credential.

The projector page shows three columns: rank, registered participant name and
score for the current Challenge. It opens on Challenge 1; the instructor selects
Challenges 1-4. Earlier scores and the overall total are not shown. Queue counts
are for the selected Challenge; the registered count covers all participants.
Large, high-contrast text and a full-screen button support projection. The page
requests only the selected Challenge from the public board API. It has no login
form, access code, source upload, or personal submission history.
Use suitable public aliases when issuing accounts; do not register email
addresses or other private information as nicknames.

Scores refresh every five seconds. Pages advance every 15 seconds, showing up
to ten people at a time, fewer on a short projector viewport. Auto paging can
be paused; arrow buttons or left/right keys navigate manually. Reduced-motion
preferences disable automatic paging initially. Ties retain the server's joint
ranks; zero is distinct from an ungraded dash. All 110 participants are reachable
through the pages, including participants without a completed score.

`/display?ranking=4&rotate=0` opens Challenge 4 with automatic paging paused.
Choose `ranking=1` through `4`, and `rotate=1` to enable paging. Old overall
bookmarks fall back to Challenge 1. All scores and totals remain on the server.
The URL saves only these public display settings, never an access code.

A lost connection keeps the last received scores, shows a warning and receipt
time, and pauses automatic paging until connection recovery. "Completed submissions"
counts completed submissions for this Challenge, not distinct participants.
Switching Challenges clears the old scores until the new results arrive. No fictitious scores
are inserted by the application, and the **NOT OFFICIAL** label remains visible
until a separately reviewed event rubric is implemented.

Two training steps verify the plumbing only. Create a fresh state with an
agreed training budget for meaningful rehearsal. `init` refuses to overwrite
an existing database. Trusted code, configuration or dependency changes require
a new pilot state, so results from different versions are never mixed.
The rubric is `bootcamp-task-completion-pilot-v3`. Preserve existing v1/v2 databases;
start a fresh v3 state rather than mixing earlier PDE-only scores with complete-task scores.

## Student workflow

1. Leave `USE_REFERENCE = False` and complete the linked `.py` exercises:
   equations/speed/conditions for Wave; equations/conditions/geometry for Fluid;
   equations/parameters/conditions/analytic solutions for Climate; dataset/model
   functions and PINO physics for Operators. See [the task mapping](../course_materials/CHALLENGE_CONTRACTS.md).
2. Run the final notebook cell. Before the first submission, enter **Nickname**
   and click **Register nickname**. Names must be unique and contain 1-40 visible
   characters. The same name is reused across all four Challenges. Select the
   completed Levels. Running this cell or Run All does not register or submit.
3. Click **Submit code**. The notebook sends saved exercise functions directly
   to the judge API; there is no separate login or website upload.
4. Read queue status, points and evaluation details in the same panel. Pending
   jobs refresh every five seconds. Use **Refresh results** to check again.

**Save nickname** can correct a name without changing the account or its scores.
An unnamed account cannot submit and is not shown on the public scoreboard.
The projector has no nickname or account-editing controls.

The server uses the authenticated participant identity, never a name supplied
inside the uploaded file. A submission is one immutable source snapshot.
Omitted Levels score zero. For a later attempt, include the earlier completed
Levels again if you want them counted. Best *submission* scores are retained;
Levels from different attempts are not silently stitched together.

## Accepted code

The submitted file is **not imported or executed as Python**. For Challenges
1-3, the evaluator extracts all required `student_*` functions and interprets a
restricted language: local assignments, tuple unpacking, bounded literal
dictionaries/tuples, supplied parameter dictionaries, arithmetic, `.diff()`,
`sin()`, `cos()` and `exp()`. Geometry is bounded rectangle data, not executable
CSG code. Old PDE-only submissions receive an explicit update/completion error.
The notebook checks `submission_contract: 3` from the authenticated account API
before sending code. A new course cannot silently use an older PDE-only judge.
Keep the original argument names and order. Numerical defaults from uploaded
code are not evaluated; the trusted lesson supplies its own defaults.

Imports, loops, branches, helper-function calls, decorators, file/network access,
arbitrary attributes and higher derivatives are not supported. Reference
function calls are not available in the interpreter. Uploads are limited to
64 KiB per lesson and bounded syntax/expression complexity. Other parts of a
full uploaded lesson file are ignored, including `main()` and imports.

For Challenge 4, the interpreter extracts `build_datasets`, `build_model` and,
for PINO, `ReactionDiffusionPDE.__init__`. The accepted constructor patterns are
`TensorDataset(*pairs)`, `TensorDataset(pairs[0], pairs[1])`, `FNO(...)` and
`AFNO(...)`, with the original keyword settings and `**model_config`. Local
assignments, tuple unpacking and a bounded comprehension over the three supplied
pairs are supported. AFNO grid checks may use `%`, comparisons, `any`, `all`,
`assert` or an `if` containing a single `raise`, without an `else` branch.
Keep model channels, dimensions and every configuration value unchanged.

Model calls first produce inert descriptions. Only after comparison with the
frozen configuration does trusted code construct the specified PhysicsNeMo model.
An oversized or different model is never instantiated from a submission. Dataset
checks reject swapped splits, changed targets or repeated normalization. AFNO
must explicitly reject a grid incompatible with either patch dimension. PINO
uses the existing `Symbol`/`Function` declarations and `self.equations` assignment;
its residual must match `u - u_xx - u_yy - f`. The validated submitted expression
is passed to PhysicsInformer and cross-checked against an independent FFT residual.

The server owns `build_physics`, normalization, loaders and training. Those
functions and the local YAML files are not accepted as student overrides.

This is a deliberate course-specific contract, **not** a sandbox for arbitrary
Python submissions. Replacing it with `exec`, imports of uploads or unrestricted
subprocess commands would require a new isolation design.

## Provisional scoring

Rules live in [catalog.py](catalog.py); the board shows the active version,
training budget and source/configuration fingerprint.

Each Level has 100 pilot points:

- 100 implementation points, split equally over required checks. Challenges 1-3
  include PDE residuals, conditions, wave speed, chip geometry, climate coefficients
  and baseline analytic expressions where requested. Challenge 4 checks data,
  specified model construction and PINO physics. Symbolic coefficients catch
  missing terms hidden by zero training defaults. Use the documented residual
  form; arbitrary rescaling is not accepted.
- Zero numerical-quality points. After **all** components match, trusted code
  trains with the fixed budget/seed and returns independent numerical diagnostics.
  These errors are feedback, not optimizer-tuning competition points. Fully
  correct implementations tie; there is no hidden runtime or submission-time
  tiebreaker.

Wrong, well-formed implementations retain credit for their passed components
and skip training. Missing, unfinished or unsupported functions invalidate the
Level and receive an explicit diagnostic; no instructor fallback is used.
Missing/nonfinite trusted metrics fail the job instead of becoming a good score.

Level scores are averaged within each Challenge: 3 Wave, 3 Fluid, 2 Climate,
3 Neural Operators.
The board retains each participant's highest completed submission per Challenge.
The pilot total is **400** (100 per Challenge), still not an approved event rubric.
Ranking uses the displayed two-decimal scores and assigns joint ranks.
Participants without a completed submission are unranked. Server failures and
timeouts do not replace an existing best score with zero.

For Fluid, feedback reports unweighted PDE residuals, boundary and flux errors.
Level 1 also compares all supplied OpenFOAM points and exports matched-scale
reference/prediction/error plots. None of these numerical errors contributes points. Wave Levels 2-3 and
Fluid Levels 2-3 have no supplied independent solution truth. Residual checks
alone do not establish complete physical accuracy.

Challenge 4 numerical feedback uses test relative L2, physical-unit RMSE and independent
FFT PDE RMSE, not training loss. All three models use the same server-generated
periodic reaction-diffusion dataset and the original 64x64 course model configs.
The pilot uses **64/16/16 train/validation/test samples**, max Fourier mode 6,
and fixed seed 1729 (distinct split streams). This is smaller than the lesson's
8,000/1,000/1,000 dataset and is explicitly a throughput/development pilot.
Change and calibrate these frozen settings before claiming event readiness;
doing so requires a new scoring state. Data are generated once per submission,
then reused across its three models. Normalization uses training data only.

Reference answers remain visible in the teaching repository. This system does
not claim to detect copying or prove independent work. With fixed training and
equivalent equations many participants can tie, which is appropriate for a
completion-oriented workshop. A model-optimization competition needs a different
submission contract and rubric.

## Workers and recovery

The current queue uses SQLite on **one host/local disk**, with atomic claims.
For a single host with eight GPUs, create a separate CUDA pilot state using
`init --device cuda`, then run eight worker processes with `--device cuda --gpu 0`
through `--gpu 7`. Each process receives only its assigned GPU through
`CUDA_VISIBLE_DEVICES`. Do not mix CPU and CUDA runs in one scoring session.
Use identical GPU models for comparable scores; GPU determinism and runtime
have not yet been validated on the event hardware.

Eight separate Brev VMs are **not** supported by sharing the SQLite database over
a network mount. That topology needs a central queue/database and remote worker
transport. The single-host design must not be presented as already supporting it.

Each participant may have one queued/running submission at a time. There is a
15-second cooldown, 30 attempts per Challenge, duplicate-source reuse, and a
600-second whole-submission limit. A worker recovering an expired lease marks
the interrupted job as a server error; the participant can resubmit. Completed
scores survive process restarts. A worker that finishes an expired lease cannot
overwrite the recovered state. If all workers stop, start a worker to recover
expired jobs. Inspect private `runs/<submission-id>-*/runner.log` for failures.
Run records are retained; there is no automatic cleanup or retention policy yet.

## Before the event

- Agree on and calibrate the rubric, accepted changes, learning budget,
  submission limit, deadline and whether the board is formative or competitive.
- Benchmark all Levels on the actual GPU, including incorrect, slow and
  nonfinite cases. Rehearse a 110-participant burst and resubmissions.
- Add production serving/TLS, identity issuance/revocation, operator access,
  process supervision, backups, quotas, deadlines and data-retention controls.
  Current HTTP serving is loopback-only, with same-origin/Host checks and no CORS.
- Keep the control database and secrets out of student-accessible machines.
  The restricted interpreter is not a general security boundary. Deploy workers
  in disposable, restricted environments with resource/network limits and review
  the deployment before accepting externally supplied submissions.
- Do not run untrusted student submissions directly on a company workstation.
  Local development should use instructor-owned test fixtures only.

## Verification

```bash
python -m pytest ETC/tests/test_judge.py ETC/tests/test_judge_training.py ETC/tests/test_judge_operators.py -q
```

The training tests submit all eleven Levels through the queue and child-process
runner using completed exercise implementations as **test fixtures**, for two CPU steps each.
They do not alter student exercise files and do not certify convergence, GPU
capacity, production security, or a public event deployment.

Local verification on 2026-09-23 also executed all four modified notebooks as
separate reference-mode copies, including their plots and submission panels.
Those checks covered the earlier upload UI. The current workflow replaces it
with notebook controls; browser regressions now require a display-only root.
Existing learner notebook outputs were preserved, not cleared or certified.

Optional real-browser regression tests (requires Chromium and websocket-client):

```bash
AI4SCI_TEST_CHROMIUM=/absolute/path/to/chrome python -m pytest ETC/tests/test_judge_projector.py -q
```

These use disposable private state and synthetic 110-person data, never event
results. They check 1920x1080/1366x768 projection layouts, narrow-screen overflow,
pagination, joint ranks, zero/empty states, loss and recovery of connection,
public-page privacy, unsafe nickname escaping, and the absence of upload/login controls.
Without the browser executable setting the optional tests are skipped.
The page uses local system fonts, including Apple SD Gothic Neo and Malgun
Gothic fallbacks. A Linux screenshot host needs an installed Korean font; it
does not download fonts or analytics from an external service.
