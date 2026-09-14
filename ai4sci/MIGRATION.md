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
- **Neural Operators:** The lesson's `u − Δu = f` was inconsistent with the old Poisson data generator. New data uses exact Fourier solutions on periodic `[0,1)^2` with independent train/validation/test draws. Historical Poisson HDF5 files remain but are rejected as current training data. This constant-coefficient teaching benchmark is not a variable-permeability Darcy or MHD implementation.
- **Diffusion:** Composite-bar interfaces enforce physical heat-flux continuity. Parameter definitions, analytical solutions and losses are checked together.
- **Navier–Stokes:** The original array path and output-frame spacing are corrected. Original data and its numerical normalization are preserved; the unresolved pressure conversion and source metadata are described in [data provenance](../tutorial/navier_stokes/DATA_PROVENANCE.md). `--smoke-data` explicitly selects the separate Taylor–Green test fixture.
- **Fluid and Climate:** Original equations, domains and level-specific conditions are retained with executable student functions, reference implementations and residual checks. Fluid Levels 1/2 use viscosity 0.02; Level 3 retains the original 0.01. Obstacles are fixed; two-way fluid–structure coupling is not implemented.

See [tutorial migration details](../tutorial/MIGRATION.md) for Lab-specific changes.

## Original titles and linked files

| Original README Challenge title | Actual files linked by the original start notebook |
|---|---|
| Advanced Wave Dynamics | Wave: three levels |
| Solving the Darcy-Flow problem using AFNO | Fluid-Structure Interaction: three levels |
| Forecasting weather using FourCastNet | Multi-Physics Climate Modeling: two levels |
| Modeling Magnetohydrodynamics with Physics Informed Neural Operators | Advanced Neural Operators: FNO, AFNO, PINO |

This difference already existed upstream; the authors' reason is unknown. The current index follows the actual files. The shared event sheet has not been edited. The course does not claim to implement FourCastNet or MHD.

## Validation

Participants complete the `student_*` functions. `--reference` runs instructor answers for reproducible checks. Execution success, held-out error improvement, full convergence and GPU event readiness are separate claims. See the [validation record](VALIDATION.md) for measured results and remaining work.
