# Evaluating your Challenge results

The afternoon Challenges are individual exercises. Notebook plots and metrics are practice feedback, not submitted scores. A separate [local scoring pilot](../judge/README.md) supports all four Challenges, with provisional points and individual standings. It is not an approved or deployed event judge.

## Practice versus submission

All four Challenges start with `USE_REFERENCE = False`, regardless of a server's `AI4SCI_REFERENCE` setting. An unfinished function should stop with a message. Setting `True` explicitly runs the provided instructor answer and labels its results as a demonstration; those results are not evidence of a completed student exercise.

The **Submit your code** section sends selected exercise implementations directly from the notebook: `student_equations` for Challenges 1-3; dataset/model factories and PINO's PDE for Challenge 4. Before the first submission, enter a **Nickname** and click **Register nickname** in the same panel. The name is reused for all four Challenges and the public scoreboard. Save the `.py` files, select Levels and click **Submit code**. Running the cell does not submit. Queue status, points and evaluation details appear in the same panel; no file download or website upload is needed. The workspace needs a [private judge connection](../environment/JUDGE_CONNECTION.md). The server owns training and evaluation; it does not accept local `metrics.json` as a score. Its pilot board uses one best submission per Challenge, not a mixture of Levels from different attempts. Unsubmitted Levels count as zero within an attempt. The four-Challenge total is 400 provisional points. Final event weights and normalization still need calibration and approval.

Challenge 4 retains the course model configurations and 64x64 grid, but its judge pilot uses a fixed 64/16/16 dataset rather than the full lesson dataset. It checks implementation, test prediction errors and independent FFT PDE error. PINO's PhysicsInformer residual is also checked against FFT. This smaller evaluation is not a claim of full-data convergence or event throughput.

## Before comparing two runs

Use the same problem, coefficients, geometry, data split and evaluation settings. Save your Python file, select `USE_REFERENCE = False`, and rerun training before opening the results. Reference mode runs the instructor implementation; it does not test your exercise code.

Training loss tells the optimizer what to reduce. Evaluation checks the resulting solution. In Challenges 1–3, evaluation uses the provided reference equations even when training uses your equations. An incorrect equation can produce a small training loss but a large evaluation error.

For Neural Operators, validation and test loaders are constructed independently from the course data. Student dataset outputs are checked against the required split and normalization. These checks help catch implementation errors; they are not a secure remote judging service.

## What each level checks

| Challenge / Level | Check your implementation | Read the result |
|---|---|---|
| Wave 1 | Constant-speed wave equation, nonzero initial velocity | Analytical error over five times; initial displacement, initial velocity and boundary errors |
| Wave 2 | Spatially varying speed in the specified non-divergence equation | Reference-equation residual, zero initial velocity and boundary errors; no analytical error is claimed |
| Wave 3 | Wave equation on the disk; Robin boundary with outward normal | Reference-equation and Robin errors; the two initial bumps taper to a compatible boundary |
| Fluid 1 | Steady incompressible momentum and continuity | Unweighted PDE errors, inlet/no-slip/outlet conditions, section flux and the supplied OpenFOAM comparison |
| Fluid 2 | The same steady equations on the three-block geometry | PDE, boundary and flux errors; the one-block OpenFOAM data is not a reference for this level |
| Fluid 3 | Time derivatives with a smooth start from rest | PDE and initial errors; inlet velocity and section flux must use the same time ramp |
| Climate 1 | Advection, diffusion, source and relaxation terms | Default diffusion reference over five times; nonzero extra terms require a different reference |
| Climate 2 | Opposite exchange signs in the two temperature equations | Coupled analytical errors for each field and across five times; coupling is active by default |
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

## Instructor: preparing official scores later

The intended competition is individual, not team-based. Before publishing scores, freeze the problem definitions, allowed changes, learning budget, Level weights, error normalization and tie rules. Calibrate thresholds with measured baseline results. The previously suggested 100 points per Challenge is a proposal, not an approved rubric.

An official evaluator must own its equations, data and configuration and recompute metrics from accepted submissions. Local files remain editable, so local JSON and its mode flags cannot prove a valid competition entry. Validate independent reference solutions or residual-based acceptance criteria for levels without analytical truth. Keep practice feedback, submission acceptance and final points distinct.

[Start Here](../../Start_Here.ipynb) · [Instructor guide](INSTRUCTOR.md) · [Schedule](course-plan.md)
