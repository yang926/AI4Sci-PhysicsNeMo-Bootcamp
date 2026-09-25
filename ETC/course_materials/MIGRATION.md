# Migration to PhysicsNeMo 2.2.2

Reference date: 2026-09-15. The source is OpenHackathons commit `9cae27f8303268cdaf7528fe963ce12ba439377f`. Git history and original data/images are retained. The introduction, Labs 1–4 and all eleven Challenge levels remain in the course. All teaching and instructor materials are in English.

## API changes

The changes follow the [official v2 migration guide](https://github.com/NVIDIA/physicsnemo/blob/v2.2.2/v2.0-MIGRATION-GUIDE.md). Symbolic PDE functionality is used through the current `physicsnemo.sym` module.

| Previous implementation | Current implementation |
|---|---|
| Separate Sym package and mixed installation instructions | `nvidia-physicsnemo[sym]==2.2.2` |
| `Key`, `instantiate_arch`, Sym-specific models | `physicsnemo.models.mlp.FullyConnected` and tensor inputs/outputs |
| `Domain`, `PointwiseConstraint`, `Solver` | Sampling functions, explicit constraint losses and a PyTorch training loop |
| PhysicsNeMo Hydra presets | Plain YAML plus `--steps`, `--device`, `--seed`, `--output-dir` |
| Legacy FNO/AFNO Arch classes | Current `physicsnemo.models.fno.FNO` and `physicsnemo.models.afno.AFNO` |
| Legacy framework output folders | `metrics.json`, `loss.csv`, `model.pt`, `predictions.npz`, `preview.png` |

Every PDE declares `dim`. PhysicsInformer evaluates spatial derivatives; programs supply time derivatives such as `u__t` through autograd. PINO uses spectral derivatives on a periodic grid. Its wrapper calls PhysicsInformer per sample to accommodate the current spectral adapter's single-sample handling and checks the result against independent batched FFT residuals.

## Equation and data consistency

- **Wave Level 1:** Both initial displacement and initial velocity are `sin(x) sin(y)`, with speed 1. The corrected reference is `sin(x)sin(y)[cos(sqrt(2)t)+sin(sqrt(2)t)/sqrt(2)]`. Tests independently check the PDE and initial/boundary conditions.
- **Neural Operators:** The lesson's `u − Δu = f` was inconsistent with the old Poisson data generator. New data uses exact Fourier solutions on periodic `[0,1)^2` with independent train/validation/test draws. Historical Poisson HDF5 files remain but are rejected as current training data. All three levels use the same constant-coefficient reaction–diffusion benchmark.
- **Diffusion:** Composite-bar interfaces enforce physical heat-flux continuity. Parameter definitions, analytical solutions and losses are checked together.
- **Navier–Stokes:** The original array and its numerical normalization are used in the student Lab. The unresolved pressure conversion and source metadata are described in [data provenance](../../01_labs/04_navier_stokes/DATA_PROVENANCE.md). Output now includes 11 frames from 0 to 60 hours, notebook playback and ParaView VTI/PVD export. The command-line `--smoke-data` option selects a separate analytical regression fixture for tests only; the student notebook has no such switch.
- **Fluid and Climate:** Original startup and baseline coefficients were restored on 2026-09-25 as listed below. Fluid Levels 1/2 use viscosity 0.02; Level 3 retains the original 0.01. Obstacles are fixed; two-way fluid–structure coupling is not implemented.

See [tutorial migration details](../legacy/tutorial/MIGRATION.md) for Lab-specific changes.

## Local changes to the original problems

These are changes to the exercises, not just API or presentation updates. The current course must not be described as an unchanged copy of the original bootcamp.

| Exercise | Original source at the reference commit | Current local adaptation |
|---|---|---|
| Wave Level 3 | Initial displacement is the sum of two Gaussian bumps. | Restored the original two Gaussians; removed the added envelope. The original initial Robin compatibility is not exact and is now stated explicitly. |
| Fluid Level 3 | Time-independent parabolic inlet and unit section flux, with zero initial velocity. | Restored the original sudden startup; removed the added ramp. The discontinuity at the initial inlet corner is documented. |
| Climate Level 2 | The notebook's default validation case is decoupled: `gamma0=0`. | Restored `gamma0=0`. The coupled analytical reference remains available for a separate `gamma0=0.5` experiment. |

The Neural Operators data correction above follows the original notebook's reaction–diffusion PDE, not its inconsistent Poisson generator. It is not the same dataset or experiment as the historical HDF5 files.

On 2026-09-25, Lab 4's unintended synthetic default was removed. The original `data_lat.npy` path is again the student lesson, with original-input visualization and full time-series playback. The original six-layer, 256-unit SiLU/weight-normalized network settings, 50,000-update Adam budget, learning-rate decay and constraint scaling are restored through the current tensor API. Sampling uses a direct loop rather than the retired Solver, so execution is not bitwise identical. No pretrained checkpoint is bundled. Taylor–Green remains a separate test fixture; its convergence results must not be attributed to the original-data Lab.

The subsequent measured Lab 4 class preset uses 3,000 updates, periodic feature frequencies 1/2/4/8 and output mean/std from the training observations only. It keeps the original array, PDE, six-by-256 SiLU network and constraint weights. This representation change improves initial-fit efficiency but does not establish convergence; the near-initial-time PDE residual can increase. A 3,000-update production run took 202.69 seconds on L4. The restored original representation and 50,000-update configuration are still available with `--recipe upstream`. See [the measurements and limitations](LAB4_EFFICIENCY.md).

The earlier decision to remove unrelated advertised topics did not authorize replacing the exercises or shrinking their tasks. On 2026-09-25 the condition, wave-speed, fluid-geometry, climate-coefficient and analytic-solution exercises were restored as named `student_*` functions. These functions now reach local training and the bounded submission interpreter. Trusted evaluation never substitutes an instructor implementation for a missing learner function. See [the task mapping](CHALLENGE_CONTRACTS.md).

The tensor-loop adaptation still supplies framework plumbing such as model input wiring and sample/constraint assembly; it is not a literal restoration of the old `Domain`/`Solver` API blanks. Mathematical setup tasks are now explicit and graded, including Robin and chip geometry. This distinction must remain visible in instructor explanations.

The class training budgets are also explicit adaptations, not a replay of the
legacy Hydra settings. Wave 1 uses 20,000 Adam updates and 512/256/256 collocation
points; the other PINN Challenge levels use 10,000 updates with their documented
current models and sampling. All three Operators use 3,000 updates on the full
64-by-64, 8,000/1,000/1,000 dataset. AFNO's original 8-by-8 patches and 256-wide
embedding are restored. See [PINN measurements](CHALLENGE_EFFICIENCY.md) and
[Operator measurements](OPERATOR_EFFICIENCY.md), including remaining errors.

## Original titles and linked files

Historical record only: the upstream README advertised different topics from the exercises linked by its start notebook. This table is not the current course agenda.

| Original README Challenge title | Actual files linked by the original start notebook |
|---|---|
| Advanced Wave Dynamics | Wave: three levels |
| Solving the Darcy-Flow problem using AFNO | Fluid-Structure Interaction: three levels |
| Forecasting weather using FourCastNet | Multi-Physics Climate Modeling: two levels |
| Modeling Magnetohydrodynamics with Physics Informed Neural Operators | Advanced Neural Operators: FNO, AFNO, PINO |

This mismatch is present in the [upstream bootcamp](https://github.com/openhackathons-org/AI-Powered-Physics-Bootcamp); the authors' reason is unknown. The agreed course follows the actual Wave, Fluid, Climate and Neural Operators exercises. The advertised alternatives are not course requirements or planned extensions. The shared event sheet and external slides have not been edited.

## Validation

Participants complete the `student_*` functions. `--reference` runs instructor answers for reproducible checks. Execution success, held-out error improvement, full convergence and GPU event readiness are separate claims. See the [validation record](VALIDATION.md) for measured results and remaining work.
