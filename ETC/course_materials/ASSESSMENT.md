# Evaluating your Challenge results

The afternoon Challenges are individual exercises. Notebook plots and metrics are practice feedback, not submitted scores. A separate [local scoring pilot](../judge/README.md) supports all four Challenges, with provisional points and individual standings. It is not an approved or deployed event judge.

## Practice versus submission

All four Challenges start with `USE_REFERENCE = False`, regardless of a server's `AI4SCI_REFERENCE` setting. An unfinished function should stop with a message. Setting `True` explicitly runs the provided instructor answer and labels its results as a demonstration; those results are not evidence of a completed student exercise.

The **Submit your code** section sends these saved exercise functions directly from the notebook:

| Challenge | Submitted functions |
|---|---|
| Wave | `student_equations`, `student_speed`, `student_conditions` |
| Fluid | `student_equations`, `student_conditions`, `student_geometry` |
| Climate | `student_equations`, `student_parameters`, `student_conditions`, `student_solution` |
| Neural Operators | Dataset/model factories and PINO's PDE |

Before the first submission, enter a **Nickname** and click **Register nickname** in the same panel. Save the `.py` files, select Levels and click **Submit code**. Running a cell does not submit. Queue status, points and component feedback appear in that panel; there is no website upload. The workspace needs a [private judge connection](../environment/JUDGE_CONNECTION.md).

The selected approach is original-task completion, not an optimizer-tuning competition. Each Level has 100 pilot points divided across its required implementation checks. Initial/boundary conditions, speed, geometry, coefficients and analytic expressions count where requested; correct PDEs alone cannot earn full credit. Fixed-budget numerical errors are separate feedback, with **zero numerical-quality points**. Fully correct submissions tie; submission time is not a hidden tiebreaker. The pilot remains labelled not official until event acceptance.

The server owns training/evaluation and does not trust local `metrics.json`. It retains one best complete submission per Challenge; Levels from different attempts are not merged. Omitted Levels count as zero. The four-Challenge total is 400 pilot points. Format v3 rejects older PDE-only submissions with an update message. Existing score databases are preserved, not mixed with the changed rubric; start a new v3 state.

Challenge 4 retains the course model configurations and 64x64 grid, but its judge pilot uses a fixed 64/16/16 dataset rather than the full lesson dataset. It checks implementation, test prediction errors and independent FFT PDE error. PINO's PhysicsInformer residual is also checked against FFT. This smaller evaluation is not a claim of full-data convergence or event throughput.

## Before comparing two runs

Use the same problem, coefficients, geometry, data split and evaluation settings. Save your Python file, select `USE_REFERENCE = False`, and rerun training before opening the results. Reference mode runs the instructor implementation; it does not test your exercise code.

Training loss tells the optimizer what to reduce. Evaluation checks the resulting solution. In Challenges 1–3, training uses your equations and problem setup; held-out evaluation uses the stated reference equations, conditions and geometry. An incorrect condition can produce a small training loss while solving the wrong problem. Climate's learner-derived analytic expression is checked independently and never becomes the evaluator's ground truth.

For Neural Operators, validation and test loaders are constructed independently from the course data. Student dataset outputs are checked against the required split and normalization. These checks help catch implementation errors; they are not a secure remote judging service.

## What each level checks

| Challenge / Level | Check your implementation | Read the result |
|---|---|---|
| Wave 1 | Constant-speed wave equation, nonzero initial velocity | Analytical error over five times; initial displacement, initial velocity and boundary errors |
| Wave 2 | Spatially varying speed in the specified non-divergence equation | Reference-equation residual, zero initial velocity and boundary errors; no analytical error is claimed |
| Wave 3 | Wave equation on the disk; implement the Robin residual and original Gaussian initial pulses | Reference-equation and Robin errors; Gaussian tails are not exactly compatible with Robin at the initial boundary |
| Fluid 1 | Steady incompressible momentum and continuity | Unweighted PDE errors, inlet/no-slip/outlet conditions, section flux and the supplied OpenFOAM comparison |
| Fluid 2 | Construct all three chip cutouts, as well as the steady PDE and conditions | PDE, boundary and flux errors; the one-block OpenFOAM data is not a reference for this level |
| Fluid 3 | Time derivatives, single-chip geometry, rest initial state and original abrupt inlet | PDE and initial errors; the nonzero inlet meets rest data discontinuously at the initial inlet corner |
| Climate 1 | ADR residual, physical coefficients, initial/boundary targets and baseline analytic solution | Default diffusion reference over five times; nonzero extra terms require a different reference |
| Climate 2 | Both residuals and their opposite exchange signs, coefficients, conditions and baseline analytic solutions | Original `gamma0=0` baseline; a separately labelled `gamma0=0.5` experiment activates coupling |
| Operators 1 / FNO | Correct data split, normalization and FNO construction | Canonical test relative L2, RMSE and independent spectral PDE error |
| Operators 2 / AFNO | The same data with the specified AFNO architecture | The same test metrics as FNO; compare prediction quality, not model names |
| Operators 3 / PINO | FNO plus the reaction–diffusion residual in physical units | The same test metrics, with the physics residual checked independently |

## Reading the saved files

The result cell displays a before/after table and the saved plot. **Full metrics and run settings** contains the complete record, including field-specific errors and evaluation settings. A missing entry is unavailable, not zero. After switching student/reference mode, rerun training; the notebook will not label an old reference result as student work.

- `metrics.json`: evaluation errors and run settings. `assessment.kind` is `local_practice_feedback`; `official_score` is null and `ranking_ready` is false.
- `loss.csv`: training minibatch losses and, for PINNs, the fixed before/after evaluation rows. In student mode these use different equation implementations by design.
- `preview.png`: a view of the prediction, not proof of accuracy over the whole domain. Wave and Climate previews show one time slice; `reference_over_time` checks five times when an analytical reference exists.
- `model.pt` and `predictions.npz`: this run's model and predictions. Keep them with the settings; an old file is not the result of a failed new run.

For `reference_over_time`, read aggregate `relative_l2`, `per_field`, and `per_time` together. A small absolute temperature error late in a decaying solution does not establish accuracy at earlier times. An empty reference result means no analytical comparison is available, not a perfect solution.

PINO includes a physics term in its training objective; FNO and AFNO do not. Do not rank their training-loss values. Do not add raw Wave, Fluid and Climate RMSE values: they measure different quantities on different scales.

## Improve one thing at a time

Check the equation and conditions before increasing training length. Keep a baseline run, change one permitted setting, and compare the same evaluation measures. Inspect which condition remains inaccurate rather than choosing the smallest total loss alone. A short successful run checks execution, not convergence.

## Instructor: event acceptance

The workshop uses individual task completion. Review component weights and ties before publishing official event points. Do not advertise faster training or smaller residuals as a way to increase the current score: model/optimizer changes are not submitted. An optimization competition would require a separately agreed contract. Numerical rehearsal is still required to show meaningful plots in class, even though numerical error no longer contributes points.

An official evaluator must own its equations, data and configuration and recompute metrics from accepted submissions. Local files remain editable, so local JSON and its mode flags cannot prove a valid competition entry. Validate independent reference solutions or residual-based acceptance criteria for levels without analytical truth. Keep practice feedback, submission acceptance and final points distinct.

[Start Here](../../Start_Here.ipynb) · [Instructor guide](INSTRUCTOR.md) · [Schedule](course-plan.md)
