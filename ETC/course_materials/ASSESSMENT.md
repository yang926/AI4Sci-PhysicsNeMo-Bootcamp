# Evaluating your Challenge results

The afternoon Challenges are individual exercises. Notebook plots and metrics are practice feedback, not submitted scores. A separate [scoring service](../judge/README.md) supports all four Challenges, with provisional points and individual standings. The configured event Launchable enrolls its student workspace through a private tunnel; the service must be running to accept submissions. Deployment and successful rehearsal do not by themselves make the provisional rules approved official event rules.

## Practice versus submission

All four Challenges run only the saved exercise functions. Unfinished code stops with a message. **Check saved code** checks syntax and completeness only; it is not a mathematical answer check. The separate judge evaluates correctness.

The **Submit your code** section sends these saved exercise functions directly from the notebook:

| Challenge | Submitted functions |
|---|---|
| Wave | `student_equations`, `student_speed`, `student_conditions` |
| Fluid | `student_equations`, `student_conditions`, `student_geometry` |
| Climate | `student_equations`, `student_parameters`, `student_conditions`, `student_solution` |
| Neural Operators | Dataset/model factories and PINO's PDE |

Before the first submission, enter a **Nickname** and click **Register nickname** in the same panel. Save the `.py` files, select Levels and click **Submit code**. Running a cell does not submit. Queue status, points and component feedback appear in that panel; there is no website upload. The workspace needs a [private judge connection](../environment/JUDGE_CONNECTION.md).

All Levels start selected. Ctrl/Cmd-click to remove an unfinished Level. Select only finished Levels: a selected unfinished function blocks the whole attempt. Submit all completed Levels together. Omitted Levels score zero, and Levels from different attempts are not combined.

Use the supplied parameter symbols in each PDE, including `rho` even when its default is 1 and Climate coefficients whose default is zero. Keep each residual's displayed sign and exact dictionary keys. The judge compares the stated symbolic residual, not merely its zero set. Wrong but well-formed components lose their own points; malformed functions or wrong keys may invalidate the Level. Climate's local `analytic_exercise` metadata marks the expression as learner-supplied and not independently checked; the judge checks the expression. Restore the stated physical coefficients before submitting.

The selected approach is original-task completion, not an optimizer-tuning competition. Each Level has 100 pilot points divided across its required implementation checks. Initial/boundary conditions, speed, geometry, coefficients and analytic expressions count where requested; correct PDEs alone cannot earn full credit. Fixed-budget numerical errors are separate feedback, with **zero numerical-quality points**. Fully correct submissions tie; submission time is not a hidden tiebreaker. The pilot remains labelled not official until event acceptance.

The server owns training/evaluation and does not trust local `metrics.json`. It retains one best complete submission per Challenge; Levels from different attempts are not merged. Omitted Levels count as zero. The four-Challenge total is 400 pilot points. Format v3 rejects older PDE-only submissions with an update message. Existing score databases are preserved, not mixed with the changed rubric; start a new v3 state.

Challenge 4 retains the course model configurations and 64x64 grid, but its judge pilot uses a fixed 64/16/16 dataset rather than the full lesson dataset. It checks implementation, test prediction errors and independent FFT PDE error. PINO's PhysicsInformer residual is also checked against FFT. This smaller evaluation is not a claim of full-data convergence or event throughput.

## Before comparing two runs

Use the same problem, coefficients, geometry, data split and evaluation settings. Save your Python file and rerun training before opening results. Challenges 1–3 default to 5,000 updates per Level; the three Operators retain 3,000.

Training loss tells the optimizer what to reduce. In Challenges 1–3, fixed held-out residual checks use the same learner-written equations, conditions, coefficients and geometry as training. They measure self-consistency on other samples, not independent correctness. A wrong implementation can have a small residual. Wave 1 adds its independent analytical comparison and Fluid 1 adds the supplied OpenFOAM comparison. Climate plots and legacy `reference_over_time` keys compare against the learner's own `student_solution`; that expression is not trusted ground truth for scoring.

For Neural Operators, the trainer computes statistics from the training split only; `build_datasets` wraps the supplied tensors unchanged. Validation and test loaders are constructed independently from the course data. Student dataset outputs are checked against the supplied tensors and split order. Pass every `model_config` key in the model factory, even if a value equals a library default. These local checks help catch implementation errors; the separate service remains responsible for accepted submissions.

## What each level checks

| Challenge / Level | Check your implementation | Read the result |
|---|---|---|
| Wave 1 | Constant-speed wave equation, nonzero initial velocity | Analytical error over five times; initial displacement, initial velocity and boundary errors |
| Wave 2 | Spatially varying speed in the specified non-divergence equation | Learner-equation residual, zero initial velocity and boundary errors; no analytical error is claimed |
| Wave 3 | Wave equation on the disk; implement the Robin residual and original Gaussian initial pulses | Learner-equation and Robin errors; Gaussian tails are not exactly compatible with Robin at the initial boundary |
| Fluid 1 | Steady incompressible momentum and continuity | Unweighted PDE errors, inlet/no-slip/outlet conditions, section flux and the supplied OpenFOAM comparison |
| Fluid 2 | Construct all three chip cutouts, as well as the steady PDE and conditions | PDE, boundary and flux errors; the one-block OpenFOAM data is not a reference for this level |
| Fluid 3 | Time derivatives, single-chip geometry, rest initial state and original abrupt inlet | PDE, initial and flux errors; `Q=1` through every section for `t>0` is incompatible with rest-state `Q=0`, not only at the inlet corner |
| Climate 1 | ADR residual, physical coefficients, initial/boundary targets and baseline analytic solution | Comparison with your own derived solution at five times; the judge checks its correctness |
| Climate 2 | Both residuals and their opposite exchange signs, coefficients, conditions and baseline analytic solutions | Original `gamma0=0` baseline with learner-derived comparisons; a nonzero-coupling experiment disables the baseline comparison |
| Operators 1 / FNO | Wrap the supplied splits unchanged and construct FNO with every model setting; trainer owns normalization | Canonical test relative L2, RMSE and independent spectral PDE error |
| Operators 2 / AFNO | The same data with the specified AFNO architecture | The same test metrics as FNO; compare prediction quality, not model names |
| Operators 3 / PINO | FNO plus the reaction–diffusion residual in physical units | The same test metrics, with the physics residual checked independently |

## Reading the saved files

The result cell displays a before/after table and the saved plot. **Full metrics and run settings** contains the complete record, including field-specific errors and evaluation settings. A missing entry is unavailable, not zero. After changing code or settings, rerun training; an old artifact is not a successful new attempt.

- `metrics.json`: evaluation errors and run settings. `assessment.kind` is `local_practice_feedback`; `official_score` is null and `ranking_ready` is false.
- Operators additionally record `training_seconds`: optimizer-loop wall time including batch transfers and logging, with CUDA synchronized at both ends. Setup, held-out evaluation and artifact export are excluded; older runs may not contain this field.
- `loss.csv`: training minibatch losses and, for PINNs, the fixed before/after evaluation rows. Both use your equation implementation; only the sample points differ.
- `preview.png`: a view of the prediction, not proof of accuracy over the whole domain. Wave and Climate previews show one time slice. Wave 1 has an independent analytical comparison; Climate `reference_over_time` uses your own derived expression and is not a correctness check.
- `model.pt` and `predictions.npz`: this run's model and predictions. Keep them with the settings; an old file is not the result of a failed new run.

For `reference_over_time`, read aggregate `relative_l2`, `per_field`, and the displayed `per_time` table together. In the default decaying Climate runs, the aggregate denominator is dominated by reference energy at `t=0`; late-time errors still contribute to its numerator. Compare each time's RMSE with the reference temperature amplitude: a small absolute error can still be large relative to a nearly decayed field. An empty reference result means no analytical comparison is available, not a perfect solution.

Climate 2 defaults to `gamma0=0`, so derive the two uncoupled solutions yourself. This run cannot test exchange signs; the judge checks them symbolically. A separate `gamma0=0.5` experiment activates coupling and disables the baseline solution comparison. Restore the original nine coefficients before submission.

PINO includes a physics term in its training objective; FNO and AFNO do not. Do not rank their training-loss values. Do not add raw Wave, Fluid and Climate RMSE values: they measure different quantities on different scales.

## Improve one thing at a time

Check the equation and conditions before increasing training length. Keep a baseline run, change one permitted setting, and compare the same evaluation measures. Inspect which condition remains inaccurate rather than choosing the smallest total loss alone. A short successful run checks execution, not convergence.

## Instructor: event acceptance

The workshop uses individual task completion. Review component weights and ties before publishing official event points. Do not advertise faster training or smaller residuals as a way to increase the current score: model/optimizer changes are not submitted. An optimization competition would require a separately agreed contract. Numerical rehearsal is still required to show meaningful plots in class, even though numerical error no longer contributes points.

An official evaluator must own its equations, data and configuration and recompute metrics from accepted submissions. Local files remain editable, so local JSON cannot prove a valid competition entry. Validate independent reference solutions or residual-based acceptance criteria for levels without analytical truth. Keep practice feedback, submission acceptance and final points distinct.

[Start Here](../../Start_Here.ipynb) · [Instructor guide](INSTRUCTOR.md) · [Schedule](course-plan.md)
