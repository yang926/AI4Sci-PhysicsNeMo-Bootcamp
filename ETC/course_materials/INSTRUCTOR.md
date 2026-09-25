# Instructor guide

Teach the introduction, Labs 1–4 and all eleven levels of Challenges 1–4 in the [course guide](README.md). Use the [schedule](course-plan.md) for event timing. Slides should follow the actual equations and program sequence in this repository.

## Preparation

1. Create the PhysicsNeMo 2.2.2 environment using the [deployment guide](../environment/SETUP.md), then run the environment-check notebook.
2. Review the hardware, versions and scope in the [validation record](VALIDATION.md). Run the reference implementations and notebooks on the event hardware and measure their duration.
3. Ask participants to complete the marked exercise code: `student_equations` in Challenges 1–3, and `build_datasets`, `build_model`, plus Level 3's `ReactionDiffusionPDE` in Challenge 4. Use the instructor implementations and `--reference` flag for explanation, recovery and comparison.
4. Inspect constraint-specific losses in `loss.csv`, before/after evaluation in `metrics.json`, and plots in `preview.png`. Adjust training length based on the actual errors.
5. Agree on teaching and support responsibilities between Mingyu Yang and the co-instructor, then rehearse the 09:30–17:30 agenda with participant editing and questions included.

## Student and instructor modes

Each Challenge notebook exposes `USE_REFERENCE` in its setup cell and prints the active mode before training. `False` executes the learner's exercise code; `True` executes the completed instructor implementation. All four Challenges default explicitly to `False` and ignore `AI4SCI_REFERENCE`. The notebook validation tool changes only its executed copies to reference mode, never the student source notebooks.

For a student exercise, explicitly set `USE_REFERENCE = False` in the setup cell and run it (or execute that assignment in a new code cell after setup). Check the mode printed immediately before execution, then edit and save the actual `.py` file. For an instructor comparison, explicitly select `True`. Switching back to `False` must be part of the demonstration, not an invisible server setting. Do not overwrite the student's implementation with the reference answer.

Each training-cell execution chooses a new result directory. Rerun the training cell, then its plot/evaluation cell; a failed run is not a completed result. Old files remain available but must not be presented as the latest attempt. All Challenge result cells check the saved student/reference mode. The Neural Operators comparison requires successful results for every compared level in the current mode.

## Explain the implementation

For a PINN, `FullyConnected` predicts the solution, `PDE` expresses the equations, and `PhysicsInformer` evaluates residuals. The explicit PyTorch loop combines those residuals with initial-condition, boundary-condition and data losses. FNO and AFNO learn an operator from paired data; PINO adds a physics residual.

Use the [learning checkpoints](course-plan.md#learning-checkpoints-and-transitions) to connect lessons. Before the first Challenge, demonstrate an edit, save it, and run the changed program. Before Neural Operators, explain that the input changes from coordinates to an entire forcing field, and that training uses many input/output field pairs. Older `Domain`/`Solver` slides need a separate explanation because these lessons use explicit PyTorch loops.

Keep the notebook and `.py` file open side by side, with JupyterLab's table of contents available. Show how each equation becomes a loss term, then read the before/after table and plot. Expand **Full metrics and run settings** only when discussing details. Editing a Markdown example does not modify the program.

## Mathematical and data interpretation

| Topic | Teaching point |
|---|---|
| Wave Level 1 | Initial velocity is nonzero, so the exact solution includes a sine term in time. |
| Wave Levels 2–3 | Level 2 also changes initial velocity to zero. Level 3 tapers both initial bumps by `(1-x*x-y*y)**2`, satisfying the initial Robin condition on the circle. |
| Parameterized PINN | Lab 1 evaluates lengths 1, 1.25, 1.5, 1.75 and 2, with aggregate and `per_length` errors. The preview and legacy `validation_rmse` still show only length 1.5. Five checks do not prove accuracy at every length. |
| Composite bar | Verify continuity of both temperature and physical heat flux. |
| Fluid | Obstacles are fixed. Level 3 returns to one block, uses viscosity 0.01, and starts inlet velocity and section flux with the same smooth ramp from rest. |
| Climate | These are educational temperature PDEs. Level 2 defaults to `gamma0=0.5`; its coupled sine-mode analytical solution remains valid with heat exchange. Nonzero advection, sources or relaxation still require a different reference. |
| Neural Operators | Compare FNO, AFNO and PINO on the same periodic reaction–diffusion problem; generate fresh data rather than using the historical Poisson HDF5 files. |
| Navier–Stokes | Use the original ERA5-derived array as the initial field. Show input wind, predicted evolution and ParaView playback. Review the source and pressure-unit limitations in [data provenance](../../01_labs/04_navier_stokes/DATA_PROVENANCE.md). |

For Lab 4, first show the original input wind and the embedded original ParaView recording. Label the recording as historical, not the current run's output. Prepare the original 50,000-update FP32 training run before the live demonstration and measure its duration on the event GPU. The source settings use six 256-unit hidden layers, SiLU, weight normalization and Adam with learning-rate decay. No pretrained checkpoint is bundled. A short override is an execution check, not evidence that the detailed wind field has converged.

Compare the original input and prediction at time zero, then play the 11 frames from 0 to 60 hours. The initial-data fit and held-out PDE/periodicity residuals describe model consistency; no future weather observations are bundled. Do not call a low PDE residual a verified forecast. Pressure gradients drive the flow, and the equations do not fix a spatially constant pressure offset at later times.

For the ParaView demonstration, run the export cell, download and extract its ZIP on the presentation Mac, then open `flow.pvd` and press **Apply**. Select **Surface**, color by **speed**, reset the camera and look along **+Z**. Rescale over all timesteps once, then press **Play**. The included README describes velocity glyphs and a prepared-view script. Rehearse the local ParaView view before class; exporting and replaying an existing result does not need another training run.

Taylor–Green remains an internal analytical regression fixture under `ETC`, not an alternative student lesson. Its accuracy gates do not validate the original-data experiment. The student notebook rejects stale fixture results rather than silently relabeling them.

The [migration notes](MIGRATION.md) record the API, equation and data changes, including Wave Level 3's initial displacement, Fluid Level 3's inlet/flux schedule and Climate Level 2's default coupling. Teach the conditions shown in the current notebooks and distinguish synthetic analytical checks from real-world validation.

## Evaluation and event operations

Students submit from the Challenge notebook and inspect their results there.
The projector is a read-only standings display, not an upload site. Configure
each workspace's [private judge connection](../environment/JUDGE_CONNECTION.md)
before the session; the notebook does not collect Brev passwords or issue judge
accounts. The connection hook is implemented, but verified Brev identity handoff
and the event HTTPS deployment are still outstanding.

Follow the [evaluation guide](ASSESSMENT.md). Challenge evaluation uses the provided PDE rather than the student's equation builder. Training still uses the student's implementation. This distinction makes a wrong equation visible as a poor physical result even when its training loss is small. It is a local diagnostic, not a tamper-proof judge.

Wave and Climate add analytical comparisons over five times when a reference exists. For Climate, read the per-field errors as well as the combined value. Operator validation/test data is reconstructed independently of the student dataset function. A missing analytical reference is not zero error, and raw losses from different problems must not be added into a ranking.

A short successful run is not a convergence result. Full-course completion requires rehearsal. The current eight-hour plan contains 330 minutes of teaching/Q&A, 90 minutes of lunch and 60 minutes of breaks.

Use a new output directory for each experiment. Require `--device cuda` for GPU checks. Verify successful container execution and GitHub publication independently.

The expected audience is 110 individual participants, each with a planned personal Brev GPU environment. Give each learner a separate writable checkout so edits and results do not conflict. A [local judge and scoreboard pilot](../judge/README.md) supports all four Challenges; it is not yet deployed to the planned eight-GPU Brev judge. Confirm the actual GPU model and single-host versus multi-host topology before provisioning. The pilot's point weights and error scales are provisional, not official event rules. Challenge 4's scoring dataset is intentionally smaller than the full lesson dataset; calibrate it on event hardware before freezing the rubric.

Before ranking participants, decide what they may change and how learning resources are made comparable. Reference implementations are visible in this teaching repository; merely selecting student mode does not prove an independently completed exercise. Preserve all eleven levels, and calibrate any eventual points against measured reference performance rather than assigning thresholds from a short smoke run.
