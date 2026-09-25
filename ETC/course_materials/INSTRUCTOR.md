# Instructor guide

Teach the introduction, Labs 1–4 and all eleven levels of Challenges 1–4 in the [course guide](README.md). Use the [schedule](course-plan.md) for event timing. Slides should follow the actual equations and program sequence in this repository.

## Preparation

1. Create the PhysicsNeMo 2.2.2 environment using the [deployment guide](../environment/SETUP.md), then run the environment-check notebook.
2. Review the hardware, versions and scope in the [validation record](VALIDATION.md). Run the reference implementations and notebooks on the event hardware and measure their duration.
3. Ask participants to complete the marked exercise code: equations, wave speed and conditions in Wave; equations, conditions and chip geometry in Fluid; equations, coefficients, conditions and analytic solutions in Climate; dataset/model functions and PINO's PDE in Operators. See the [task mapping](CHALLENGE_CONTRACTS.md). `--reference` supplies every instructor component, not just the PDE.
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
| Wave Levels 2–3 | Level 2 changes speed and initial velocity. Level 3 uses the original Gaussian pair without tapering; its tails are not exactly Robin-compatible at the initial boundary. Students implement the Robin residual, not a supplied replacement. |
| Parameterized PINN | Lab 1 evaluates lengths 1, 1.25, 1.5, 1.75 and 2, with aggregate and `per_length` errors. The preview and legacy `validation_rmse` still show only length 1.5. Five checks do not prove accuracy at every length. |
| Composite bar | Verify continuity of both temperature and physical heat flux. |
| Fluid | Obstacles are fixed. Students construct the chip cutouts. Level 3 returns to one block and viscosity 0.01, with the original abrupt inlet and rest initial state; discuss the initial inlet-corner incompatibility. |
| Climate | These are educational temperature PDEs. Level 2 restores the original `gamma0=0` baseline. Derive its analytic solutions before a separate nonzero-coupling experiment. The independent coupled sine-mode reference supports that extension. |
| Neural Operators | Compare FNO, AFNO and PINO on the same periodic reaction–diffusion problem; generate fresh data rather than using the historical Poisson HDF5 files. |
| Navier–Stokes | Use the original ERA5-derived array as the initial field. Show input wind, predicted evolution and ParaView playback. Review the source and pressure-unit limitations in [data provenance](../../01_labs/04_navier_stokes/DATA_PROVENANCE.md). |

For Lab 4, first show the original input wind and the embedded original ParaView recording. Label the recording as historical, not the current run's output. The measured class preset uses 3,000 FP32 Adam updates, six 256-unit hidden layers, SiLU and weight normalization; the complete command took 202.69 seconds on an NVIDIA L4. Integer periodic feature frequencies 1, 2, 4 and 8 and training-field output scaling improve initial-field fitting efficiency without changing the input array or equations. The original 50,000-update representation and settings remain available with `--recipe upstream`; that full run has not been validated here. No pretrained checkpoint is bundled. See the [efficiency comparison](LAB4_EFFICIENCY.md) for the exact changes and measurements.

Do not present Lab 4 as a converged weather forecast. Better initial fitting increases the near-zero-time continuity residual because the supplied planar field has substantial discrete divergence; later predictions weaken strongly. Compare the reserved initial fit and the per-time PDE errors separately. The instructor recording demonstrates the historical workflow, not proof that the current learned flow is physically accurate.

Compare the original input and prediction at time zero, then play the 11 frames from 0 to 60 hours. The initial-data fit and held-out PDE/periodicity residuals describe model consistency; no future weather observations are bundled. Do not call a low PDE residual a verified forecast. Pressure gradients drive the flow, and the equations do not fix a spatially constant pressure offset at later times.

For the ParaView demonstration, run the export cell, download and extract its ZIP on the presentation Mac, then open `flow.pvd` and press **Apply**. Select **Surface**, color by **speed**, reset the camera and look along **+Z**. Rescale over all timesteps once, then press **Play**. The included README describes velocity glyphs and a prepared-view script. Rehearse the local ParaView view before class; exporting and replaying an existing result does not need another training run.

Taylor–Green remains an internal analytical regression fixture under `ETC`, not an alternative student lesson. Its accuracy gates do not validate the original-data experiment. The student notebook rejects stale fixture results rather than silently relabeling them.

The [migration notes](MIGRATION.md) record both the earlier unintended adaptations and the restored Wave initial displacement, Fluid startup and Climate baseline. Teach the conditions shown in the current notebooks and distinguish analytical checks from real-world validation.

## Evaluation and event operations

Students submit from the Challenge notebook and inspect their results there.
The projector is a read-only standings display, not an upload site. Configure
each workspace's [private judge connection](../environment/JUDGE_CONNECTION.md)
before the session; the notebook does not collect Brev passwords or issue judge
accounts. The connection hook is implemented, but verified Brev identity handoff
and the event HTTPS deployment are still outstanding.

Follow the [evaluation guide](ASSESSMENT.md). Training consumes the learner's setup; held-out evaluation uses the stated reference PDE, conditions and geometry. The v3 judge parses all required functions and checks individual components. An unchanged PDE cannot complete Wave 3 without Robin or Fluid 2 without its three chips. Local metrics remain editable diagnostics, not trusted submissions.

Wave and Climate add analytical comparisons over five times when a reference exists. For Climate, read the per-field errors as well as the combined value. Operator validation/test data is reconstructed independently of the student dataset function. A missing analytical reference is not zero error, and raw losses from different problems must not be added into a ranking.

A short successful run is not a convergence result. Full-course completion requires rehearsal. The current eight-hour plan contains 330 minutes of teaching/Q&A, 90 minutes of lunch and 60 minutes of breaks.

Use a new output directory for each experiment. Require `--device cuda` for GPU checks. Verify successful container execution and GitHub publication independently.

The expected audience is 110 individual participants, each with a planned personal Brev GPU environment. Give each learner a separate writable checkout so edits and results do not conflict. A [local judge and scoreboard pilot](../judge/README.md) supports all four Challenges; it is not yet deployed to the planned eight-GPU Brev judge. Confirm the actual GPU model and single-host versus multi-host topology before provisioning. The pilot's point weights and error scales are provisional, not official event rules. Challenge 4's scoring dataset is intentionally smaller than the full lesson dataset; calibrate it on event hardware before freezing the rubric.

The agreed pilot ranks original-task completion, not optimizer or model tuning. Fully correct submissions share a rank; numerical feedback does not break ties. Reference implementations are visible in this teaching repository, so student mode alone does not prove independent work. Preserve all eleven levels and use the measured errors to prepare demonstrations, not to introduce unannounced performance points. See [class budgets and remaining errors](CHALLENGE_EFFICIENCY.md).
