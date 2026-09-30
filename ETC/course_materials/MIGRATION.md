# Migration to PhysicsNeMo 2.2.2

Reference date: 2026-09-15. The source is OpenHackathons commit `9cae27f8303268cdaf7528fe963ce12ba439377f`. Original data/images and attribution are retained. The public development history was replaced with release snapshots on 2026-10-01; earlier commit identifiers below are historical references, not commits available in this release history. The course has an Introduction, four Labs and all eleven Challenge levels. On 2026-09-29, the instructor approved replacing the Navier–Stokes Lab 4 teaching slot with pretrained FourCastNet weather inference. The previous PINN Lab is archived, not deleted. All teaching and instructor materials are in English.

## 2026-09-29: student-only Challenges

All four Challenges now run only saved learner implementations. Completed exercise answer functions, notebook answer blocks and the instructor-mode switch are removed from the student workflow. Local preflight checks syntax and completeness; the judge checks correctness. PINN local residuals use learner equations/conditions/geometry, and Climate local analytical plots use the learner's `student_solution` without claiming independent correctness. Wave 1 analytical verification and Fluid 1 CFD data remain numerical comparisons.

All eight Wave/Fluid/Climate levels now default to 5,000 Adam updates. FNO/AFNO/PINO remain at 3,000. Learning rates, model sizes, sampling, problem statements and the judge's fixed evaluation budget are unchanged. Notebook subprocesses use `-u` for visible progress. Lab 4's pretrained weather model and ERA5 verification are unaffected. Historical measurements below retain their original scope and do not establish current 5,000-update accuracy.

## 2026-09-29: approved weather Lab replacement

The active [Lab 4: AI Weather Forecasting with FourCastNet](../../01_labs/04_weather_forecasting/Lab_4_Weather_Forecasting.ipynb) uses NVIDIA's pretrained 26-channel FourCastNet1 AFNO checkpoint, with a global ERA5 initial state at 00 UTC on 1 September 2022. Eight six-hour autoregressive steps produce a 48-hour forecast. Later ERA5 reanalysis is used only for verification, with a persistence baseline and physical-unit errors. The Lab performs inference; it does not train from scratch, fine-tune or minimize PDE residuals. Its single-case validation does not establish general forecast skill.

This is a deliberate change in educational objective, not an API-only migration or a reproduction of the upstream Navier–Stokes lesson. The previous [Lab 4 notebook and programs](../reference_labs/04_navier_stokes/Lab_4_Navier_Stokes.ipynb), original array and teaching media remain under `ETC/reference_labs/04_navier_stokes`. The active course navigation, Introduction and instructor guide now distinguish the weather inference workflow from PINN training. Lab 1–3 problem settings, Challenge tasks and scoring contracts are unchanged by this replacement. The published event session title and times are retained; external slides have not been edited here.

Model and data artifacts are downloaded to a verified cache outside the checkout, not committed to Git. See the [weather validation record](LAB4_WEATHER_VALIDATION.md) for sources, measurements and limitations. The dated Navier–Stokes records below describe archived implementations, not the active weather Lab.

## API changes

The changes follow the [official v2 migration guide](https://github.com/NVIDIA/physicsnemo/blob/v2.2.2/v2.0-MIGRATION-GUIDE.md). Symbolic PDE functionality is used through the current `physicsnemo.sym` module.

| Previous implementation | Current implementation |
|---|---|
| Separate Sym package and mixed installation instructions | `nvidia-physicsnemo[sym]==2.2.2` |
| `Key`, `instantiate_arch`, Sym-specific models | `physicsnemo.models.mlp.FullyConnected` and tensor inputs/outputs |
| `Domain`, `PointwiseConstraint`, `Solver` | Sampling functions, explicit constraint losses and a PyTorch training loop |
| PhysicsNeMo Hydra presets | Plain YAML plus `--steps`, `--device`, `--seed`, `--output-dir` |
| Legacy FNO/AFNO Arch classes | Current `physicsnemo.models.fno.FNO` and `physicsnemo.models.afno.AFNO` |
| Legacy framework output folders | Training exercises: `metrics.json`, `loss.csv`, `model.pt`, `predictions.npz`, `preview.png`; weather inference: `forecast.npz`, `runtime.json` and a separate evaluation directory |

Every PDE declares `dim`. PhysicsInformer evaluates spatial derivatives; programs supply time derivatives such as `u__t` through autograd. PINO uses spectral derivatives on a periodic grid. Its wrapper calls PhysicsInformer per sample to accommodate the current spectral adapter's single-sample handling and checks the result against independent batched FFT residuals.

## Equation and data consistency

- **Wave Level 1:** Both initial displacement and initial velocity are `sin(x) sin(y)`, with speed 1. The corrected reference is `sin(x)sin(y)[cos(sqrt(2)t)+sin(sqrt(2)t)/sqrt(2)]`. Tests independently check the PDE and initial/boundary conditions.
- **Neural Operators:** The lesson's `u − Δu = f` was inconsistent with the old Poisson data generator. New data uses exact Fourier solutions on periodic `[0,1)^2` with independent train/validation/test draws. Historical Poisson HDF5 files remain but are rejected as current training data. All three levels use the same constant-coefficient reaction–diffusion benchmark.
- **Diffusion:** Composite-bar interfaces enforce physical heat-flux continuity. Parameter definitions, analytical solutions and losses are checked together.
- **Archived Navier–Stokes:** The original array and numerical normalization are preserved. The archived notebook explicitly selects `--recipe six_hour`, deriving incompressible wind and compatible pressure for 11 frames from 0 to 6 hours, notebook playback and ParaView VTI/PVD export. This is a qualitative flow demonstration, not a weather forecast. The raw-input 60-hour recipes remain available. See [data provenance](../reference_labs/04_navier_stokes/DATA_PROVENANCE.md) and [six-hour validation](LAB4_SIX_HOUR.md). The command-line `--smoke-data` option selects a separate analytical regression fixture for tests only; the archived notebook has no such switch.
- **Fluid and Climate:** Original startup and baseline coefficients were restored on 2026-09-25 as listed below. Fluid Levels 1/2 use viscosity 0.02; Level 3 retains the original 0.01. Obstacles are fixed; two-way fluid–structure coupling is not implemented.

See [tutorial migration details](../legacy/tutorial/MIGRATION.md) for Lab-specific changes.

## Local changes to the original problems

These are changes to the exercises, not just API or presentation updates. The current course must not be described as an unchanged copy of the original bootcamp.

| Exercise | Original source at the reference commit | Current local adaptation |
|---|---|---|
| Wave Level 3 | Initial displacement is the sum of two Gaussian bumps. | Restored the original two Gaussians; removed the added envelope. The original initial Robin compatibility is not exact and is now stated explicitly. |
| Fluid Level 3 | Time-independent parabolic inlet and unit section flux, with zero initial velocity. | Restored the original sudden startup; removed the added ramp. The discontinuity at the initial inlet corner is documented. |
| Climate Level 2 | The notebook's default validation case is decoupled: `gamma0=0`. | Restored `gamma0=0`. A separate `gamma0=0.5` experiment remains possible; the current student-derived baseline comparison is disabled for nonzero coupling. |
| Lab 4 | Supplied wind and pressure, with a 60-hour time interval. | With the instructor's explicit approval, the active Lab now performs pretrained FourCastNet/AFNO weather inference and independent ERA5 verification. The prior PINN implementations, including the derived-input six-hour experiment and original-input 60-hour recipes, are preserved as reference material. |

The Neural Operators data correction above follows the original notebook's reaction–diffusion PDE, not its inconsistent Poisson generator. It is not the same dataset or experiment as the historical HDF5 files.

On 2026-09-25, Lab 4's unintended synthetic default was removed. The original `data_lat.npy` path was restored as the student lesson, with original-input visualization and full time-series playback. The original six-layer, 256-unit SiLU/weight-normalized network settings, 50,000-update Adam budget, learning-rate decay and constraint scaling were restored through the current tensor API. Sampling uses a direct loop rather than the retired Solver, so execution is not bitwise identical. No pretrained checkpoint is bundled. Taylor–Green remains a separate test fixture; its convergence results must not be attributed to the original-data Lab.

The subsequent raw-input Lab 4 preset uses 3,000 updates, periodic feature frequencies 1/2/4/8 and output mean/std from the training observations only. It keeps the original array, PDE, six-by-256 SiLU network and constraint weights. This representation change improves initial-fit efficiency but does not establish convergence; the near-initial-time PDE residual can increase. A 3,000-update production run took 202.69 seconds on L4. These historical measurements describe `--recipe efficient`, which remains the command-line default; the original representation and 50,000-update configuration remain available with `--recipe upstream`. See [the measurements and limitations](LAB4_EFFICIENCY.md).

On 2026-09-29, the notebook was changed to explicitly request `--recipe six_hour` for 3,000 Adam updates. The original array stays unchanged, but its periodic Helmholtz-projected wind and recalculated pressure form a separately labeled initial condition. One normalized time unit still means 60 hours; the training interval is `[0, 0.1]` and the network receives `t / 0.1` with derivatives taken through the chain rule. Integrated kinetic-energy and mean-momentum identities supplement the original sample-sum and area-weighted losses. Independent numerical comparisons support a qualitative large-scale six-hour demonstration, with remaining damping and fine-scale errors. No future reference labels train the model, and no weather-forecast claim follows. See [the six-hour validation record](LAB4_SIX_HOUR.md).

The earlier decision to remove unrelated advertised topics did not authorize replacing the exercises or shrinking their tasks. On 2026-09-25 the condition, wave-speed, fluid-geometry, climate-coefficient and analytic-solution exercises were restored as named `student_*` functions. These functions now reach local training and the bounded submission interpreter. Trusted evaluation never substitutes an instructor implementation for a missing learner function. See [the task mapping](CHALLENGE_CONTRACTS.md).

The tensor-loop adaptation still supplies framework plumbing such as model input wiring and sample/constraint assembly; it is not a literal restoration of the old `Domain`/`Solver` API blanks. Mathematical setup tasks are now explicit and graded, including Robin and chip geometry. This distinction must remain visible in instructor explanations.

The class training budgets are also explicit adaptations, not a replay of the
legacy Hydra settings. All eight PINN Challenge levels now use 5,000 Adam updates. Wave 1 retains
512/256/256 collocation points; the other levels retain their documented
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

This mismatch is present in the [upstream bootcamp](https://github.com/openhackathons-org/AI-Powered-Physics-Bootcamp); the authors' reason is unknown. The agreed Challenges follow the actual Wave, Fluid, Climate and Neural Operators exercises, not the advertised alternatives. The separately approved weather Lab added on 2026-09-29 does not imply that FourCastNet was implemented in the original Climate Challenge. The shared event sheet and external slides have not been edited.

## Validation

Participants complete the `student_*` functions. The student distribution has no completed answer mode; correctness is checked by the separate judge. Execution success, held-out error improvement, full convergence and GPU event readiness are separate claims. See the [validation record](VALIDATION.md) for measured results and remaining work.
