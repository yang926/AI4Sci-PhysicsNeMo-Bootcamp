# Instructor guide

Teach the introduction, Labs 1–4 and all eleven levels of Challenges 1–4 in the [course guide](README.md). Use the [schedule](course-plan.md) for event timing. Slides should follow the actual equations and program sequence in this repository.

## Preparation

1. Create the PhysicsNeMo 2.2.2 environment using the [deployment guide](../environment/SETUP.md), then run the environment-check notebook.
2. Review the hardware, versions and scope in the [validation record](VALIDATION.md). Rehearse completed implementations in a separate instructor workspace on the event hardware; do not distribute answers in the student checkout.
3. Ask participants to complete the marked exercise code: equations, wave speed and conditions in Wave; equations, conditions and chip geometry in Fluid; equations, coefficients, conditions and analytic solutions in Climate; dataset/model functions and PINO's PDE in Operators. See the [task mapping](CHALLENGE_CONTRACTS.md). The student distribution provides no completed answer mode.
4. For training exercises, inspect constraint-specific losses in `loss.csv`, before/after evaluation in `metrics.json`, and plots in `preview.png`. For Lab 4, prepare the model/data cache before class and inspect the forecast, ERA5 reference, persistence baseline and lead-time errors. Lab 4 has no optimizer or training budget to adjust.
5. Agree on teaching and support responsibilities between Mingyu Yang and the co-instructor, then rehearse the 09:30–17:30 agenda with participant editing and questions included.

Recommended background is basic Python functions and arrays, first and second derivatives, and initial/boundary conditions. This is a guided intermediate workshop, not an introduction to Python or a derivation of every PDE from first principles. Use the [final teaching review and pacing notes](FINAL_TEACHING_REVIEW.md) to prepare the dense morning block and the harder Fluid and Neural Operators sessions.

## Student implementation and reruns

All Challenge notebooks run the saved exercise functions. There is no mode toggle or completed answer implementation in the student workflow. Preserve the **EDIT HERE** blocks and required signatures while learners implement their solutions. Challenges 1–3 use 5,000 Adam updates per level; the three Operators retain 3,000. These budgets are not accuracy thresholds.

Each execution uses a fresh result directory. Rerun training and then its result cell after editing; failed attempts must not display older results as new. Commands use Python `-u` to expose training progress while running. **Check saved code** checks syntax and completeness only; correctness is evaluated by the judge.

## Explain the implementation

For a PINN, `FullyConnected` predicts the solution, `PDE` expresses the equations, and `PhysicsInformer` evaluates residuals. The explicit PyTorch loop combines those residuals with initial-condition, boundary-condition and data losses. FNO and AFNO learn an operator from paired data; PINO adds a physics residual. Lab 4 uses AFNO weights already trained by NVIDIA: students perform inference and verification, not another optimization exercise.

Use the [PhysicsNeMo workflow walkthrough](PHYSICSNEMO_WORKFLOW.md) to explain the boundary between library APIs and course code. `create_model`, `create_informer` and `residuals` are helpers in `ETC/runtime/pinn.py`, not PhysicsNeMo API names. Show their definitions once before asking learners to use them. Do not present these three helpers as the execution path for every Lab or for the Neural Operators.

### Walk through one training step

Keep Wave Level 1 and `ETC/runtime/pinn.py` open side by side. This is a reading walkthrough of the existing program, not an additional training run or a change to its recipe.

1. Find `create_model(3, 1, config, args.device)` in `main()`. Follow it to `FullyConnected` and explain why the input is `(x, y, t)` and the output is one displacement field.
2. Follow `student_equations` through `WaveEquation2D` into `create_informer`. A `PDE` stores the SymPy residual expressions; it does not train a model or solve the equation by itself.
3. Open `residuals`. It enables coordinate gradients, predicts the fields, computes time derivatives with PyTorch, and calls `PhysicsInformer.forward`. Spatial derivatives use `autodiff`. The returned `wave` tensor is an equation residual, not an error against the analytical solution.
4. Return to `loss_terms`. Show the separate PDE, initial-displacement, initial-velocity and boundary losses. Point sampling and condition losses here are course code, not an automatic consequence of declaring a `PDE`.
5. Find `zero_grad`, `backward` and `step` in `main()`. Distinguish derivatives with respect to coordinates from gradients with respect to trainable parameters. Both participate in the graph, but answer different questions.

Ask learners to point to the line that changes for a new equation and the lines that remain reusable. In Challenges 2–3, revisit the changed fields, geometry or coefficients without repeating the entire walkthrough. In Challenge 4, contrast this coordinate-to-field path with a whole forcing field passed to FNO/AFNO, and show how PINO adds a spectral residual to its data loss after restoring physical-scale fields.

The existing exercises remain unchanged. Use these questions to check understanding, not as new unannounced scoring requirements. A few correct return statements can pass implementation checks; they do not by themselves demonstrate that the learner understands the full workflow or that the trained solution is accurate.

Use the [learning checkpoints](course-plan.md#learning-checkpoints-and-transitions) to connect lessons. Before the first Challenge, demonstrate an edit, save it, and run the changed program. Before Neural Operators, explain that the input changes from coordinates to an entire forcing field, and that training uses many input/output field pairs. Older `Domain`/`Solver` slides need a separate explanation because these lessons use explicit PyTorch loops.

Keep the notebook and `.py` file open side by side, with JupyterLab's table of contents available. Show how each equation becomes a loss term, then read the before/after table and plot. Expand **Full metrics and run settings** only when discussing details. Editing a Markdown example does not modify the program.

With 110 individual learners, agree who handles setup and first-error triage while the other instructor presents. Before each hands-on block, show the exact file to edit, saved exercise functions and expected output. Use **Check saved code** in the final panel before a long training run; it checks syntax and completeness for selected Levels without training or a judge connection, not answer correctness. Cover all Levels in order before optional parameter experiments, and use training time to discuss the next equation or interpret an earlier result.

## Mathematical and data interpretation

| Topic | Teaching point |
|---|---|
| Wave Level 1 | Initial velocity is nonzero, so the exact solution includes a sine term in time. |
| Wave Levels 2–3 | Level 2 changes speed and initial velocity. Level 3 uses the original Gaussian pair without tapering; its tails are not exactly Robin-compatible at the initial boundary. Students implement the Robin residual, not a supplied replacement. |
| Parameterized PINN | Lab 1 evaluates lengths 1, 1.25, 1.5, 1.75 and 2, with aggregate and `per_length` errors. The preview and legacy `validation_rmse` still show only length 1.5. Five checks do not prove accuracy at every length. |
| Composite bar | Verify continuity of both temperature and physical heat flux. |
| Fluid | Obstacles are fixed. Students construct the chip cutouts. Level 3 returns to one block and viscosity 0.01. Its rest initial state has zero flow rate, while the abrupt inlet and all cross-section targets require unit flow for positive time; discuss this startup incompatibility, not only the inlet corner. |
| Climate | These are educational temperature PDEs. Level 2 restores the original `gamma0=0` baseline. Derive its analytic solutions before a separate nonzero-coupling experiment. Local baseline comparisons use the learner's `student_solution` and are disabled for nonzero coupling; an extension needs a separate derivation. |
| Neural Operators | Compare FNO, AFNO and PINO on the same periodic reaction–diffusion problem; generate fresh data rather than using the historical Poisson HDF5 files. |
| Weather forecasting | Run pretrained FourCastNet from the initial ERA5 state, compare its 48-hour forecast with later ERA5 reanalysis and persistence, and inspect local as well as area-weighted errors. This is inference, not PINN training or general forecast-skill certification. |

### Lab 4: AI Weather Forecasting with FourCastNet

Open the [weather notebook](../../01_labs/04_weather_forecasting/Lab_4_Weather_Forecasting.ipynb). State the objective before running it: generate a 48-hour weather forecast using pretrained FourCastNet, compare it with ERA5 reanalysis, and examine how forecast errors change with lead time.

The input is the global atmosphere at 00 UTC on 1 September 2022: 26 surface and pressure-level variables on a 720-by-1440 grid. NVIDIA's 26-channel FourCastNet1 AFNO checkpoint predicts all 26 fields six hours later. Repeat eight times, feeding each prediction back into the model. Future ERA5 does not enter this loop. The saved checkpoint mean and standard deviation normalize inputs and restore outputs; the output is the complete next state, not a tendency to add to the previous state.

There is no training, fine-tuning or official submission in this Lab. The exercises are to select a weather variable, follow the forecast evolution, compare lead times, and explain errors. ERA5 is a reanalysis combining observations and a numerical model, not direct observations alone. Relative humidity is derived with the documented NVIDIA input conversion; the checkpoint, normalization and forecast outputs are not tuned to this case.

Show ERA5, forecast and signed-error maps side by side with fixed color ranges. Use the East Asia animation to follow the circulation; then compare 6-, 24- and 48-hour errors for surface wind, mean sea-level pressure and 2 m temperature. Ask whether a similar-looking cyclone also has the correct location and intensity. Area-weighted RMSE does not prove that a cyclone core is accurate. The persistence baseline holds the initial weather unchanged; beating it is more informative than showing an attractive animation alone.

The prespecified 24- and 48-hour checks passed for the global and East Asia domains in this one historical case. That does not establish general forecast reliability or operational suitability. Read the [weather validation record](LAB4_WEATHER_VALIDATION.md) for physical-unit metrics and limitations; do not call RMSE a percentage accuracy.

Prepare the cache before students arrive. Model files total about 301 MB and compressed ERA5 source chunks about 622 MB; expanded arrays and saved outputs need additional disk space. The cache lives outside the checkout at `~/.cache/ai4sci/weather`. On a fresh Brev L4, cold download and preparation took 35 seconds; the cached notebook took about 85 seconds for all code cells, including plots and animation (90 seconds including kernel startup and notebook saving). Model computation alone took 3.21 seconds. These are single-instance measurements, not guaranteed timings or evidence of 110-person capacity. Use the remaining lesson time to compare fields, lead times and errors, not to wait for an artificial five-minute run.

Suggested transition: "The previous Labs used equation and condition losses to train a network. Here we use a model already trained on historical weather, then check its forecast against later reanalysis. Fast inference is useful only if we also examine where the forecast is wrong."

The instructor explicitly approved this change in learning objective. The previous [Navier–Stokes PINN notebook](../reference_labs/04_navier_stokes/Lab_4_Navier_Stokes.ipynb), its source data, ParaView workflow and six-hour experiment remain archived for reference, not part of the active Lab sequence. Historical results in [LAB4_EFFICIENCY.md](LAB4_EFFICIENCY.md) and [LAB4_SIX_HOUR.md](LAB4_SIX_HOUR.md) describe those older PINN recipes, not FourCastNet. Neither the analytical Taylor–Green regression nor the old PINN runs validate this weather model.

The [migration notes](MIGRATION.md) record both the earlier unintended adaptations and the restored Wave initial displacement, Fluid startup and Climate baseline. Teach the conditions shown in the current notebooks and distinguish analytical checks from real-world validation.

## Evaluation and event operations

Students submit from the Challenge notebook and inspect their results there.
The projector is a read-only standings display, not an upload site. Configure
each workspace's [private judge connection](../environment/JUDGE_CONNECTION.md)
before the session. The event Launchable opts into automatic enrollment and a
private tunnel; a generic installation does not. The notebook does not collect
Brev passwords. Learners register their public nickname in the final submission
panel, then submit saved code there. Verify enrollment, nickname registration,
one submission and the projector display before class; repository documentation
does not establish that the event service is currently reachable.

Follow the [evaluation guide](ASSESSMENT.md). Training and fixed held-out residual evaluation consume the learner's setup. Small local residuals do not establish that the stated problem was implemented correctly. The v3 judge parses all required functions and checks individual components. An unchanged PDE cannot complete Wave 3 without Robin or Fluid 2 without its three chips. Local metrics remain editable diagnostics, not trusted submissions.

Wave 1 retains its independent analytical comparison. Climate compares against the learner's own derived expression at five times; this is not a correctness check. For Climate, read the per-field errors as well as the combined value. Operator validation/test data is reconstructed independently of the student dataset function. A missing analytical reference is not zero error, and raw losses from different problems must not be added into a ranking.

A short successful run is not a convergence result. Full-course completion requires rehearsal. The current eight-hour plan contains 330 minutes of teaching/Q&A, 90 minutes of lunch and 60 minutes of breaks.

Use a new output directory for each experiment. Require `--device cuda` for GPU checks. Verify successful container execution and GitHub publication independently.

The expected audience is 110 individual participants, each with a planned personal Brev L4 environment. Give each learner a separate writable checkout so edits and results do not conflict. The [embedded judge](../judge/README.md) provides local development and preflight checks; the separate event judge serves submissions and standings. Verify the deployed service, eight-GPU worker configuration and agreed event rules before class. The pilot awards implementation points, not points derived from error scales. Challenge 4's scoring dataset is intentionally smaller than the full lesson dataset; its numerical feedback is not a full-lesson benchmark. Single-instance measurements do not establish 110-person capacity.

The agreed pilot ranks original-task completion, not optimizer or model tuning. Fully correct submissions share a rank; numerical feedback does not break ties. Completed Challenge implementations are not supplied in the student workflow; local execution alone still does not prove independent work. Preserve all eleven levels and use the measured errors to prepare demonstrations, not to introduce unannounced performance points. See [class budgets and remaining errors](CHALLENGE_EFFICIENCY.md).
