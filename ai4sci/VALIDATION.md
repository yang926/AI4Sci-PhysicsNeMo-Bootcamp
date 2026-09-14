# Validation record — PhysicsNeMo 2.2.2

Validation date: **2026-09-15**. The tests use actual PhysicsNeMo models, symbolic residuals and training; no execution results are simulated. [Machine-readable record](validation-2026-09-15.json) · [Reproduction commands](../Deployment_Guide.MD).

## Verified environments and scope

| Environment | Runtime | Checks |
|---|---|---|
| Linux x86_64 CPU, isolated Python 3.12 venv | PhysicsNeMo 2.2.2, PyTorch 2.14.0+cpu | Full unit, smoke, improvement and notebook execution suites |
| macOS arm64 CPU, isolated Python 3.12 venv | PhysicsNeMo 2.2.2, PyTorch 2.14.0 | Unit and smoke tests, improvement experiments, original-data Navier–Stokes run |

The installed Linux dependencies passed `pip check`. Exact package sets are recorded in [Linux CPU lock](../requirements-linux-cpu.lock.txt) and [macOS lock](../requirements-macos.lock.txt). No new server or GPU instance was started for validation.

- **36 tests and 8 subtests passed:** analytical equations, derivatives, initial/boundary conditions, composite-bar flux, independent operator data splits, batched PINO residuals, finite gradients, protected output directories and checkpoint provenance.
- **18 training modes passed at 20 steps each:** all Lab modes and all eleven Challenge levels, using explicit instructor reference mode where required.
- Each run produced finite metrics, a valid loss history, a safely reloadable checkpoint, finite prediction arrays and a readable PNG. Recorded seed, device, steps and framework version must match the requested run.
- **12 notebooks executed successfully with 20 embedded plots** at two training steps per mode. The final English edition passed; recorded source hashes match the current files.
- All **45 original data/media assets** are byte-identical to the upstream Git blobs. Notebook structure, code syntax, local links and the seven-hour schedule are checked automatically.

## Measured learning improvement

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

## Original-data Navier–Stokes check

The preserved original array was run on macOS CPU for two steps without `--smoke-data`. All five artifacts were produced; predictions were finite with shape `[11, 1024, 3]`. Fixed evaluation objective decreased from **13.8636 to 10.0646**. This confirms execution of that data path, not weather-forecast accuracy. The [data provenance note](../tutorial/navier_stokes/DATA_PROVENANCE.md) records unresolved source metadata and the preserved pressure normalization.

## Remaining validation

- NVIDIA GPU execution and the 26.08-based Docker image have **not** been built/run in these checks. CPU results do not establish CUDA, AMD GPU or Apple MPS compatibility.
- Full convergence of every advanced problem, practical lesson duration, participant editing time and the complete six- or seven-hour event require rehearsal on the event environment.
- There is no held-out future-weather dataset in this repository, so weather-forecast skill is not claimed.

The migration corrected issues found during validation, including the removed predefined Navier–Stokes API, checkpoint version serialization, inline notebook plotting, Fluid Level 3 viscosity, and shared prediction/reference plot color limits. The final English Linux suite passed after those corrections. The English-only scan found no Hangul in 75 text/JSON files or their filenames; original illustrations and the PDF were also checked.
