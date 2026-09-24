# Validation record: PhysicsNeMo 2.2.2

## 2026-09-25: FP32 Lab accuracy

The current Lab settings and measured CPU/L4 results are recorded in
[Labs 1–4 FP32 validation](LABS_FP32_VALIDATION.md). This adds absolute solution,
initial/boundary/interface and PDE checks to the historical execution and
relative-improvement checks below. It does not certify weather forecasting or
classroom capacity. Historical counts and settings below retain their dates.

## 2026-09-23: publication checks

- **339 tests and 22 subtests passed** on CPU. Fifteen existing dependency/test warnings remain. Reproduce with `python -m pytest ETC/tests ETC/course_materials/wave/test_reference.py -q`.
- The clean publication snapshot passed the strict static check: 50 Python files, 12 notebooks, 265 local links, all 18 execution modes, all 11 Challenge levels and all 45 original data/media assets. No saved-output exemption was used; the static report had no errors or warnings.
- Start Here combines the event schedule and all ten Setup/lesson links in one course table. HTML export and browser checks verified the combined table, original logo images and absence of a private agenda link.
- Private planning URLs, local environment details, generated datasets and run artifacts are excluded from publication. Executed notebook outputs remain in the instructor's working files; the published copies contain the same cell source and IDs without saved execution output.

This pass checks publication integrity and the CPU test suite. It does not add a GPU, L40 performance, full-convergence or classroom-capacity claim. Historical execution evidence below retains its original scope and date.

## 2026-09-21: course refactor and end-to-end checks

The refactor keeps the same four Labs, eleven Challenge levels and student exercises. Notebook result cells share a small presentation module; the equations and training loops remain in the lesson programs. A common staging helper protects existing results and removes this call's temporary files after normal write failures. Publishing multiple files is not an atomic transaction or crash-recovery mechanism.

- **330 tests and 22 subtests passed.** Coverage includes configuration errors, derivatives, dataset normalization, spectral residuals, parameter-family evaluation, run identity, mode switching, incomplete artifacts and failed-write recovery. Fifteen existing dependency/test warnings remain.
- **All 12 notebooks passed** on CPU in reference mode, with two optimizer steps per training run. All **19 result sets**, **19 HTML result tables** and **20 embedded plots** were checked. The runner now verifies model checkpoints, finite predictions, loss histories and provenance, not only notebook execution. Every notebook hash matches its executed version; cell IDs and order match the preceding validated edition.
- **Static checks passed:** 50 Python files, 12 notebooks, 260 local links, all 18 course execution modes and all 45 original data/media assets. Two pre-existing Setup outputs remain preserved but are not certified by the static check. `git diff --check` passed.
- **GUI checks** cover actual executed HTML, table captions/headers/metric rows, expandable details, HTML escaping and correct per-problem result selection. No browser-automation or direct screenshot inspection was available; this is not a pixel-level browser audit.

The parameterized introductory PINN now evaluates five lengths from 1 to 2; its preview still shows length 1.5. Climate 2 plots both atmosphere and ocean fields against their analytical references. Operator initial/validation evaluation accumulates metrics without retaining unused field arrays, and uses float64 error sums with correct final-minibatch weighting.

### Measured learning improvement

The following separate CPU checks used 500 optimizer steps and required at least 1% improvement on the same evaluation data before and after training. These results are not official points or full convergence certification.

| Problem | Metric | Before | After |
|---|---|---:|---:|
| Projectile | Held-out solution RMSE | 54.8227 | 0.759793 |
| Composite bar | Held-out solution RMSE | 43.7926 | 0.064095 |
| Wave 1 | Mid-time analytical relative L2 | 1.68427 | 0.644638 |
| FNO | Test relative L2 | 1.00689 | 0.016634 |
| AFNO | Test relative L2 | 1.02159 | 0.310477 |
| PINO | Test relative L2 | 1.00689 | 0.012145 |

Operator improvement checks use a 16x16 grid, reduced models and 128/32/32 train/validation/test samples. The notebook checks separately execute the full 64x64, 8,000/1,000/1,000 data path, but only for two training steps. Wave and AFNO still have substantial remaining error; passing improvement does not mean they are sufficiently trained for a demonstration. No new learning-quality claim is made for advanced levels without an independent solution reference.

Local evidence, under `ETC/validation-runs/`: `refactor-20260921-unit.xml`, `refactor-20260921-static.json`, `refactor-20260921-notebooks/report.json`, and `refactor-20260921-improvement/report.json`. Notebook and training reports include source hashes; generated reports are excluded from Git.

All model checks used the private CPU namespace described below to avoid this WSL session's unresolved GPU-driver wait. No Jupyter/WSL restart, GPU training, cloud provisioning, scoreboard implementation, external slide edit or GitHub publication was performed. L40 execution, classroom timing and 110-person capacity still need event-environment rehearsal.

## 2026-09-21: lesson correctness and practice evaluation

This revision keeps all four Labs and eleven Challenge levels. It corrects the Wave 3 initial boundary compatibility, Fluid 3 startup conditions, and Climate 2's default coupling and analytical reference. Evaluation now uses the provided equations independently of student equation implementations, with multi-time analytical comparisons and canonical Operator validation/test datasets. These are local practice diagnostics, not official competition scores or a secure judge.

- **171 tests and 22 subtests passed** on CPU, including equation derivatives, initial/boundary conditions, coupled temperature references, dataset integrity and evaluation independence. Fifteen warnings remain from dependency deprecations and an existing test's tensor-to-scalar conversion.
- **All 12 notebooks passed** in reference mode on CPU, with two optimizer steps per training run and 20 embedded plots. The main run covers all 18 course modes plus the supplementary Wave reference. Its 19 artifact sets passed finite-value, checkpoint provenance, loss-history and PNG checks. Three notebooks were rerun after final wording changes; every final notebook's source hash matches its latest passing execution record.
- **Static checks passed:** 43 Python files, 12 notebooks, 252 local links and all 45 preserved original assets. The check retains the two pre-existing environment-check outputs without certifying them; newly executed copies are saved separately. `git diff --check` also passed.
- The current schedule follows the instructor's 09:30–17:30 agenda: 330 minutes of teaching/Q&A, 90 minutes of lunch and 60 minutes of breaks. Participation is individual; official point weights and ranking rules remain undecided.

Local evidence: `ETC/validation-runs/quality-20260921-unit.xml`, `quality-20260921-static.json`, and `report.json` in `quality-20260921-notebooks`, `quality-20260921-navigation-recheck` and `quality-20260921-challenge-recheck`, all under `ETC/validation-runs/`. Generated evidence is excluded from Git and will not appear in a fresh clone. The Operator notebooks use the full 64x64 grid and 8,000/1,000/1,000 samples; two optimizer steps check execution, not learning quality.

**Environment caveat:** ordinary PhysicsNeMo imports in this WSL session stalled in Warp's CUDA initialization at `dxgglobal_acquire_process_adapter_lock`, even with CUDA hidden through environment variables. The CPU checks ran in a private child mount namespace with `/dev/dxg` masked; the real library, model and derivative code executed without mocks. This does not repair or validate normal GPU access. No host mount, Jupyter server, WSL instance or cloud resource was restarted or changed. Termination was requested for the earlier blocked test processes; they remained in uninterruptible driver wait at the last check.

These checks do not establish full convergence, acceptable lesson duration, L40 performance, or capacity for 110 simultaneous participants. The scoreboard and remote judge are outside this revision.

## 2026-09-20: numbered lesson folders

After moving the lessons into `01_labs` and `02_challenges`, the following checks passed:

- 121 tests and 8 subtests, including navigation, equations, and notebook run controls.
- All 12 notebooks executed in reference mode on CPU with two optimizer steps per run; 20 plots were embedded. This checks execution, not convergence or lesson duration.
- Python syntax, notebook structure, local links, and all 45 original figures/data assets.
- Saved outputs, execution counts, cell IDs, and metadata in all 12 source notebooks remained unchanged.

Local reports are in `ETC/validation-runs/layout-20260920-unit.xml`, `ETC/validation-runs/layout-20260920-static.json`, and `ETC/validation-runs/layout-20260920-notebooks/report.json`. Generated reports are excluded from Git. No GPU run or server restart was needed.

Earlier reports below retain their original source paths and results. The [path map](layout.json) relates those paths to the numbered layout. See the [installation guide](../environment/SETUP.md) for reproduction commands.

The remaining sections record the earlier **2026-09-15 CPU validation**, **2026-09-19 local WSL/CUDA checks**, and **2026-09-20 notebook-workflow regressions**. [Historical machine-readable record](validation-2026-09-15.json).

## Notebook-workflow regressions: 2026-09-20

The notebook-aligned course update was checked on the existing WSL environment with **CPU execution only** (`CUDA_VISIBLE_DEVICES=''`). No cloud instance, new environment, paid resource, GPU training or Jupyter-server restart was used.

- **120 tests and 8 subtests passed** in the final unit/regression run. The suite includes actual analytical/gradient tests and 37 mocked notebook-workflow tests covering repeated runs, student/reference selection, cross-level result lookup, failed attempts and mixed-mode Operator comparisons. Mocked subprocess tests do not establish model training or student-solution correctness.
- **All 12 notebooks passed**, with **20 embedded plots**, using two actual optimizer steps per training mode in reference mode. The initial complete run was followed by rerunning the four notebooks changed during QA. Every final notebook's source hash matches its latest passing record.
- **Static checks passed:** 39 Python files, 12 notebooks, 251 local links, all 18 executable modes and all 11 Challenge levels. Original data/media integrity checks remain enabled. The working-copy check explicitly preserved two existing environment-check cell outputs and warned that those saved outputs were not validated; this is not the strict clean-output publication check.
- `git diff --check` passed. Existing lesson outputs and user edits were retained; the notebook runner wrote executed copies and artifacts to separate local validation directories.

Local evidence: `ETC/validation-runs/course-ready-20260920-notebooks/report.json`, `ETC/validation-runs/course-ready-20260920-recheck/report.json`, and `ETC/validation-runs/course-ready-20260920-static-final.json`. These generated reports are not included in a fresh GitHub clone. Reproduce the unit run with `python -m pytest ETC/tests ETC/course_materials/wave/test_reference.py -q` and the notebook checks with the commands in the deployment guide.

This update verifies notebook execution and teaching workflow, not full convergence, complete student solutions, L40 performance, fresh installation, Docker execution, or a timed classroom rehearsal. The parameter-family and pressure-gauge evaluation limitations are recorded in the [instructor guide](INSTRUCTOR.md#mathematical-and-data-interpretation). External slides, the shared spreadsheet, and GitHub publication were not changed by this local update.

## Local WSL / RTX 3080 checks: 2026-09-19

Stored local reports identify Python **3.12.11**, PhysicsNeMo **2.2.2**, PyTorch **2.10.0+cu128**, and WSL2 Linux x86_64. The executed environment-check notebook identifies **NVIDIA GeForce RTX 3080**. The environment is a uv-managed virtual environment outside Conda; the local checkout's `.venv` points to it.

| Check | Recorded result | Scope |
|---|---|---|
| Unit tests | 36 tests and 8 subtests passed; 15 warnings | CPU tests in the CUDA-enabled environment; pytest reported 35.32 seconds |
| GPU smoke | All 18 modes passed, 20 steps each, seed 42 | Execution, finite artifacts, and checkpoint checks; instructor references where needed |
| GPU notebooks | All 12 notebooks passed, 2 steps per training mode, 20 embedded plots | Reference-mode execution and plotting, including the full-size Operator data path |

Local evidence paths are `ETC/validation-runs/local-unit/report.json`, `ETC/validation-runs/local-unit/unit.log`, `ETC/validation-runs/local-gpu-smoke/report.json`, `ETC/validation-runs/local-gpu-notebooks/report.json`, and `ETC/validation-runs/local-gpu-notebooks/preflight/executed.ipynb`. These generated files are **not distributed in a fresh GitHub clone**. The reports record the tested source snapshot; edits to the teaching files require new checks and must not inherit the old pass result automatically.

The GPU smoke report spans **472.785 seconds** from start to finish. Summing the notebook report's per-case execution times gives **316.254 seconds**; this is not a measurement of an entire interactive class. Process startup, imports, evaluation, and artifact work contribute to these measurements. They cannot be multiplied by step count or participant count to predict full training, L40 performance, or class duration.

Operator smoke checks use a small 16x16 benchmark with 24/8/8 train/validation/test samples and reduced models. The Operator notebook generated the 64x64 lesson data with 8,000/1,000/1,000 samples, but trained each method for only two steps. The default Navier–Stokes checks use the synthetic Taylor–Green fixture. The local GPU records do **not** establish original-weather-data accuracy, full convergence, student exercise completion, or concurrent-user capacity.

The planned Brev **L40** environment has not been installed or measured by these records. Confirm its actual GPU and driver, run the environment check and short CUDA checks there, then measure the intended lesson workflow separately. No L40 timings or participant-capacity claims are supplied here.

## Historical CPU environments and scope: 2026-09-15

| Environment | Runtime | Checks |
|---|---|---|
| Linux x86_64 CPU, isolated Python 3.12 venv | PhysicsNeMo 2.2.2, PyTorch 2.14.0+cpu | Full unit, smoke, improvement and notebook execution suites |
| macOS arm64 CPU, isolated Python 3.12 venv | PhysicsNeMo 2.2.2, PyTorch 2.14.0 | Unit and smoke tests, improvement experiments, original-data Navier–Stokes run |

The installed Linux dependencies passed `pip check`. Exact package sets are recorded in [Linux CPU lock](../environment/requirements-linux-cpu.lock.txt) and [macOS lock](../environment/requirements-macos.lock.txt). No new server or GPU instance was started for validation.

- **36 tests and 8 subtests passed:** analytical equations, derivatives, initial/boundary conditions, composite-bar flux, independent operator data splits, batched PINO residuals, finite gradients, protected output directories and checkpoint provenance.
- **18 training modes passed at 20 steps each:** all Lab modes and all eleven Challenge levels, using explicit instructor reference mode where required.
- Each run produced finite metrics, a valid loss history, a safely reloadable checkpoint, finite prediction arrays and a readable PNG. Recorded seed, device, steps and framework version must match the requested run.
- **12 notebooks executed successfully with 20 embedded plots** at two training steps per mode. The final English edition at that time passed; recorded source hashes identify that validated edition, not later edits.
- All **45 original data/media assets** are byte-identical to the upstream Git blobs. Notebook structure, code syntax, local links and the seven-hour schedule are checked automatically.

## Historical measured learning improvement: CPU

Each experiment below used 500 training steps and the same fixed held-out evaluation before and after training. The acceptance criterion was at least 1% error reduction. It is an improvement check, not a claim of full convergence.

| Experiment | Metric | Before | After |
|---|---|---:|---:|
| Projectile | RMSE | 54.823 | 0.75979 |
| Composite-bar diffusion | RMSE | 43.793 | 0.064103 |
| Wave Level 1 (mid-time slice) | Relative L2 | 1.6843 | 0.64464 |
| FNO (16×16 benchmark) | Relative L2 | 1.0069 | 0.016634 |
| AFNO (16×16 benchmark) | Relative L2 | 1.074 | 0.31391 |
| PINO (16×16 benchmark) | Relative L2 | 1.0069 | 0.012145 |

Operator smoke and improvement tests use a fresh 16×16 benchmark with small models. Notebook execution separately covers the lesson's 64×64 grid and 8,000/1,000/1,000 train/validation/test split, but only two optimizer steps. These are distinct experiment settings.

A separate macOS Wave run at 5,000 steps reduced mid-time spatial-slice relative L2 to **0.3383**, with PDE RMSE **0.03843**. The prediction still underestimates amplitude. AFNO also retains substantial error at 500 steps. These results do not establish teaching-quality convergence for every problem or the entire time domain.

PINO's independent FFT and PhysicsInformer residual calculations agree within the numerical tolerance enforced by the tests. Training/test normalization uses training data only; held-out samples are not used as optimizer targets.

## Historical original-data Navier–Stokes check: macOS CPU

The preserved original array was run on macOS CPU for two steps without `--smoke-data`. All five artifacts were produced; predictions were finite with shape `[11, 1024, 3]`. Fixed evaluation objective decreased from **13.8636 to 10.0646**. This confirms execution of that data path, not weather-forecast accuracy. The [data provenance note](../../01_labs/04_navier_stokes/DATA_PROVENANCE.md) records unresolved source metadata and the preserved pressure normalization.

## Remaining validation

- The 2026-09-19 RTX 3080 checks establish the short local CUDA paths described above. They do not validate Brev L40 execution. The 26.08-based Docker image has not been built/run in these recorded checks; AMD GPU and Apple MPS execution also remain unvalidated.
- Full convergence of every advanced problem, practical lesson duration, participant editing time and the current eight-hour event require rehearsal on the event environment.
- Multi-user GPU sharing, storage persistence, reconnect/recovery behavior, and support workload require event-environment checks. They are not measured by sequential single-user smoke runs.
- There is no held-out future-weather dataset in this repository, so weather-forecast skill is not claimed.

The historical migration corrected issues found during validation, including the removed predefined Navier–Stokes API, checkpoint version serialization, inline notebook plotting, Fluid Level 3 viscosity, and shared prediction/reference plot color limits. The final English Linux suite passed after those corrections. The historical English-only scan found no Hangul in 75 text/JSON files or their filenames; original illustrations and the PDF were also checked. Those counts describe the recorded snapshot, not a guarantee about subsequent repository changes.
