# Final teaching review · 29 September 2026

Reviewed GitHub `main` at `8be4ae8`: Setup, Introduction, four Labs, eleven Challenge levels, their source programs, submission contracts and instructor guides. The corrections from this review concern instructions and navigation, not equations, data, architectures, optimizers, budgets or points. No new GPU training or live event-service test was performed in this review.

Current-revision note: this is a historical review of the earlier revision. The subsequent student-only Challenge update removes answer mode and notebook answer snippets, sets PINN defaults to 5,000 updates and keeps Operators at 3,000. Local preflight now checks syntax/completeness only, and local PINN residuals use learner implementations. The historical timings and verification totals below do not validate that later change.

## Audience and difficulty

This is a guided intermediate workshop for participants comfortable with basic Python functions, dictionaries and arrays, first and second derivatives, and initial/boundary conditions. Prior PhysicsNeMo use is not required. Completing short code fragments is easier than understanding the full numerical method; neither a successful run nor 100 implementation points certifies independent mastery.

The difficulty ratings below are teaching judgments, not measured learner outcomes.

| Lesson | Difficulty | Explain before learners run it |
|---|---|---|
| Lab 1: PINNs | Moderate, with a harder inverse example | Forward: find the solution. Parameterized: add the domain length as input. Inverse: infer a source from observations. Separate coordinate derivatives from parameter gradients. |
| Lab 2: Projectile | Low–moderate | A second-order ODE needs both initial position and velocity. The training interval is 0–5 seconds; extrapolation is a separate test, not an accuracy promise. |
| Lab 3: Heat | Moderate | Temperature and physical heat flux must agree across the interface. The two materials need not share a gradient or split the temperature rise equally. |
| Lab 4: Weather | Low coding burden, moderate interpretation | Load pretrained AFNO weights, forecast eight six-hour steps, then compare with held-back ERA5 and persistence. There is no optimizer or training exercise. |
| Challenge 1: Wave | Moderate–high | Start with displacement and velocity as distinct conditions. Level 2 changes speed; Level 3 adds a normal derivative and a Robin boundary. |
| Challenge 2: Fluid | High | Three fields, coupled equations, obstacle geometry, wall/outlet conditions and flow-rate constraints. Level 3 also changes time dependence and the startup conditions. |
| Challenge 3: Climate | Moderate–high | Keep parameter names, field indices and exchange signs consistent. The graded `gamma0=0` baseline is uncoupled; nonzero coupling is a separate experiment. |
| Challenge 4: Operators | High conceptual jump, short code edits | The input is now an entire field, not a coordinate. Explain independent data splits, normalization, architecture changes and the physical-unit PDE term in PINO. |

The main pressure points are the number of concepts in the morning and the Fluid and Operators explanations. Reading every paragraph aloud, introducing Python from scratch, installing desktop tools during the session, or requiring repeated tuning runs is not supported by the available time.

## Morning cue sheet

This is a suggested allocation inside the existing 10:30–11:30 slot, not a changed event agenda or a measured classroom result. All four Labs and their existing modes remain included.

| Time | Activity | Main checkpoint |
|---|---|---|
| 10:30–10:51 | Lab 1: forward, parameterized, inverse | Identify what is known and what is learned in each mode. Run and inspect all three. Explain L-BFGS as the chosen optimizer; a derivation of its update is not required to run the Lab. |
| 10:51–11:01 | Lab 2 | Compare the analytical trajectory, in-domain error and extrapolation. Locate the separate condition losses in TensorBoard and the exported trajectory in ParaView. |
| 11:01–11:16 | Lab 3: fixed and parameterized | Inspect temperature and flux jumps, the small interface-temperature differences, and inference at a new conductivity using reloaded weights. Open the matching material files in ParaView. |
| 11:16–11:26 | Lab 4 | Generate and evaluate the weather forecast. Compare wind, pressure or temperature maps at 6, 24 and 48 hours, then inspect the persistence comparison. |
| 11:26–11:30 | Questions and transition | Distinguish complete Lab programs from Challenge functions learners must implement. Show the Python editor and the marked exercise blocks. |

Complete Setup and the weather cache before this block. During the Introduction, explain one prediction → residual → loss → update path and demonstrate saving a Python file. Have TensorBoard and ParaView ready; the Python setup check does not install or validate a participant's desktop ParaView. Use the existing experiment prompts after the baseline work, not as additional mandatory tasks ahead of the next Lab. Do not drop levels or change recipes to catch up.

Recorded Lab 1–3 command times total about 5.2 minutes; the cached weather notebook adds about 1.5 minutes. These measurements have different scopes: the older Lab 2/3 timings exclude later notebook visualization and reload additions. Roughly seven minutes of recorded computation is not a seven-minute Lab session. Reading, editing, tool navigation and questions dominate the rest.

Sources: [Lab 1 L4 validation](LAB1_LBFGS_VALIDATION.md), [Lab 2/3 FP32 validation](LABS_FP32_VALIDATION.md), [weather L4 validation](LAB4_WEATHER_VALIDATION.md).

## Afternoon delivery

For each Level: state the changed physics, identify every required function, let participants edit and save, use **Check saved code**, then train and inspect the result. Open the final controls directly after setup if needed; this does not require completing the intervening training cells. Select only completed Levels for the local check. The current check establishes syntax/completeness only, needs neither training nor a judge connection, and sends no submission; the judge evaluates correctness.

| Challenge slot | Recorded single-L4 execution | Teaching priority |
|---|---|---|
| Wave: 50 minutes | About 10.1 minutes for three training runs | Spend the first walkthrough on the shared PINN code path and submission controls. Reuse that explanation for Levels 2 and 3. |
| Fluid: 60 minutes | About 13.2 minutes for three training runs | Draw the geometry and identify conditions before coding. Use training time to discuss the next level or inspect an earlier result. |
| Climate: 50 minutes | About 5.4 minutes for two training runs | Check the coefficients and coupled-equation signs, then separate the uncoupled baseline from the optional coupling experiment. |
| Operators: 50 minutes | About 6 minutes for three commands | Generate data once, explain field shapes and split/normalization ownership, then compare architectures and physical residuals. |

Wave/Fluid/Climate numbers are training times, not whole-notebook times. Operators numbers include command startup/evaluation but exclude dataset generation and installation. They come from the [Challenge](CHALLENGE_EFFICIENCY.md) and [Operator](OPERATOR_EFFICIENCY.md) measurement records, not a new run or a 110-person rehearsal. These budgets leave space for one baseline per Level; they do not guarantee completion by every learner or enough time for several tuning attempts.

Before the first real submission, demonstrate **Nickname → Register nickname → select completed Levels → Submit code**. A Challenge score averages all of its Levels, including zeros for omissions. Wave Level 1 at 100 points alone gives 33.33/100 for Challenge 1. Completing Level 2 later means resubmitting Levels 1 and 2 together; separate attempts are not combined. All-correct submissions may tie because numerical feedback does not change completion points.

## Results that need an honest explanation

These are existing limitations, not newly discovered execution failures. Do not describe them as proof that the reference implementations produce highly accurate simulations in every case.

| Example | Evidence and interpretation |
|---|---|
| Wave | Level 1 has pooled five-time relative L2 of 5.59%, but 18.4% at the final time. Level 3's original Gaussian tails are not exactly compatible with its initial Robin boundary; its recorded initial-displacement and Robin RMSEs remain about 0.102 and 0.061. A correctly implemented condition can remain imperfectly fitted. |
| Fluid | This is the largest numerical-quality caveat. Level 1's reported OpenFOAM comparison has u/v/p RMSE about 0.335/0.164/0.920 in the course's field units. Level 3's wall and flux errors remain about 0.224/0.208. Do not turn these into percentages or call the lesson a validated CFD replacement. Levels 2/3 have no independent CFD solution in this course. |
| Climate | The baseline checks educational PDEs, not real climate forecasts. Relative errors can be large late in time when the reference decays; show absolute as well as relative errors. Baseline measurements do not validate nonzero coupling. |
| Operators | AFNO's held-out field relative L2 is about 2.20%, but its physical PDE RMSE is about 7.24, versus about 0.0797 for FNO and 0.0259 for PINO in the recorded run. A good-looking field can have inaccurate derivatives. This is a useful comparison, not evidence that one architecture always wins. |
| Weather Lab | The 24/48-hour checks passed for one historical ERA5 case on L4. The maps and persistence comparison support that example, not general forecast reliability or cyclone-core accuracy. |

Suggested explanation: “The implementation check asks whether we wrote the specified problem correctly. The numerical diagnostics ask how well this finite training run solved it. Those are different questions.” Ask learners to identify the largest remaining condition error rather than promising that full completion points imply an accurate flow field.

## Scope and pre-class checks

Keep equations, data, model settings, optimizer recipes, all eleven Levels and grading rules unchanged for this final instruction pass. Confirm the external slides reflect the approved weather Lab replacement; those files were not reviewed or edited here. Separately check the event Launchable's enrollment, one student nickname/submission and the projector display before class. This repository review neither changes the event services nor confirms current GPU capacity or live connectivity.

## Verification of this revision

The CPU regression suite passed: 1,548 tests and 51 subtests, with 32 skips and 16 warnings. Static validation passed for 142 Python files, 13 notebooks and 405 local links, with no errors or warnings. These are source/regression checks, not new convergence or fleet-capacity measurements.

Compared with the audit base, all nine edited notebooks retain identical executable cells and metadata. Lesson Python sources, configurations, training recipes and grading code are unchanged. The only runtime-module edit corrects the README text placed inside Lab 3's ParaView ZIP; its export logic is unchanged. All new student and instructor guidance is in English.
