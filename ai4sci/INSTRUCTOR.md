# Instructor guide

Teach the introduction, Labs 1–4 and all eleven levels of Challenges 1–4 in the [course guide](README.md). Use the [schedule](course-plan.md) for event timing. Slides should follow the actual equations and program sequence in this repository.

## Preparation

1. Create the PhysicsNeMo 2.2.2 environment using the [deployment guide](../Deployment_Guide.MD), then run the environment-check notebook.
2. Review the hardware, versions and scope in the [validation record](VALIDATION.md). Run the reference implementations and notebooks on the event hardware and measure their duration.
3. Ask participants to complete the `student_*` functions. Use the corresponding `reference_*` functions and `--reference` flag for explanation, recovery and comparison.
4. Inspect constraint-specific losses in `loss.csv`, before/after evaluation in `metrics.json`, and plots in `preview.png`. Adjust training length based on the actual errors.
5. Agree on teaching and support responsibilities between Mingyu Yang and the co-instructor, then rehearse the full six- or seven-hour event.

## Explain the implementation

For a PINN, `FullyConnected` predicts the solution, `PDE` expresses the equations, and `PhysicsInformer` evaluates residuals. The explicit PyTorch loop combines those residuals with initial-condition, boundary-condition and data losses. FNO and AFNO learn an operator from paired data; PINO adds a physics residual.

Each program shows its equations and loss terms directly. Repeated sampling and output utilities live in shared files. Notebooks connect the explanation with the actual program and plots. Demonstrate **edit → save → run → inspect new results** with the notebook and `.py` file open side by side. Editing a Markdown example does not modify the program.

## Mathematical and data interpretation

| Topic | Teaching point |
|---|---|
| Wave Level 1 | Initial velocity is nonzero, so the exact solution includes a sine term in time. |
| Composite bar | Verify continuity of both temperature and physical heat flux. |
| Fluid | These are flows around fixed obstacles; two-way structural deformation is not implemented. |
| Climate | These are educational coupled PDEs, not FourCastNet. |
| Neural Operators | Compare FNO, AFNO and PINO on the same periodic reaction–diffusion problem; generate fresh data rather than using the historical Poisson HDF5 files. |
| Navier–Stokes | Distinguish the Taylor–Green fixture from the original array. Review unresolved source and pressure-unit questions in [data provenance](../tutorial/navier_stokes/DATA_PROVENANCE.md). |

The [migration notes](MIGRATION.md) explain the original Darcy/FourCastNet/MHD title mismatch and all API, equation and data corrections. The shared event sheet retains its original labels and needs to be aligned with the actual files before teaching.

## Evaluation and event operations

A short successful run is not a convergence result. Compare fixed held-out errors and analytical references, and inspect residuals, prediction fields and runtime for advanced problems. Full-course completion within the event window requires rehearsal. The seven-hour plan contains 320 minutes of teaching, 60 minutes of lunch and 40 minutes of breaks.

Use a new output directory for each experiment. Require `--device cuda` for GPU checks. Verify successful container execution and GitHub publication independently.
