# Challenge 3: L4 numerical validation

Measured on 1 October 2026 with NVIDIA L4, PyTorch 2.10.0+cu128 and PhysicsNeMo 2.2.2. These measurements cover completed baseline exercise functions, not arbitrary learner implementations or official scores.

## What was wrong

The learner's saved baseline equations, conditions, coefficients and derived solutions matched the assigned problem. The previous numerical recipe used 5,000 constant-learning-rate Adam updates with small, newly sampled minibatches. It did not fit that correctly implemented problem sufficiently: the saved Level 1 midpoint RMSE was `0.00741125`, with PDE, initial and boundary errors remaining visible. A completed implementation is not a guarantee of PINN convergence.

## What changed

- Physical coordinates are scaled to `[-1,1]` inside autograd. The spatial and time derivatives remain derivatives with respect to the original physical coordinates.
- The unchanged three-layer, width-64 network remains FP32. Matrix precision was `highest`; TF32 matrix multiplication was disabled.
- The default still makes 5,000 optimizer calls: 1,500 Adam calls with cosine learning-rate decay from `0.001` to `0.00001`, then 3,500 L-BFGS calls.
- L-BFGS uses fixed scrambled Sobol interior points and initial/boundary grids: 4,096 / 2,116 / 2,116 points. A line search therefore reevaluates the same objective.
- Level 1 permits up to two inner quasi-Newton iterations per L-BFGS call; Level 2 permits up to three. This is more computation than 5,000 Adam updates. Actual closure evaluations and training time are saved in `metrics.json`.
- The L-BFGS closure uniformly scales the total loss by 1,000. This preserves relative term weights and the mathematical minimizer while making its absolute stopping tests useful at small losses. Reported losses remain unscaled.

The PDEs, coefficients, physical domain `[0,pi]^2 x [0,2*pi]`, initial/boundary targets, student function signatures and submission contract were not changed. The analytical solution is used only for comparison, never for training. This change does not update or restart the live judge.

## Acceptance checks

The following limits were set before accepting a candidate and were not relaxed after a failed run:

- Initial RMSE at most `0.005` for every field.
- At `t=pi`, RMSE at most `0.0002` and maximum absolute error at most `0.0005` for every field.
- RMSE at most `0.005` at every evaluated time.
- Fresh PDE and boundary RMSE at most `0.003`.
- Aggregate relative L2 at most `0.01`.

Saved models are reloaded and evaluated on a separate 64-by-64 spatial grid at `t=0, 0.25, 0.5, 1, pi/2, pi, 3*pi/2, 2*pi`. Residual/condition checks use fresh points: 4,096 interior, 1,024 initial and 1,024 boundary samples. Baseline expressions were inspected separately; the ordinary lesson comparison still does not certify the correctness of a learner's own expression.

## Measured results

Level 1 final recipe, seed 42: all acceptance checks passed. Training-loop time was 275.71 seconds; 12,591 L-BFGS closure evaluations were recorded. Midpoint RMSE was `0.00006581`, and midpoint maximum absolute error was `0.00025836`. Fresh PDE / initial / boundary RMSE was `0.00036813 / 0.00007437 / 0.00014044`. Aggregate relative L2 across the eight evaluation times was `0.00045235`.

Level 2 final recipe, seed 42: all acceptance checks passed for both fields. Training-loop time was 527.53 seconds; 16,004 L-BFGS closure evaluations were recorded. Midpoint RMSE was `0.00009661` for `Ta` and `0.00011377` for `To`; maximum absolute errors were `0.00036514 / 0.00038138`. Fresh PDE RMSE was `0.00057029 / 0.00054343`, and fresh boundary RMSE was `0.00025129 / 0.00022856`. Aggregate relative L2 was `0.00064337`.

| Final baseline run | Optimizer calls | L-BFGS inner limit | L4 training-loop time | Midpoint RMSE | Midpoint maximum error |
| --- | ---: | ---: | ---: | ---: | ---: |
| Level 1, seed 42 | 5,000 | 2 | 4 min 36 s | T: 0.00006581 | T: 0.00025836 |
| Level 2, seed 42 | 5,000 | 3 | 8 min 48 s | Ta / To: 0.00009661 / 0.00011377 | Ta / To: 0.00036514 / 0.00038138 |

The [machine-readable measurements](CHALLENGE3_L4_MEASUREMENTS.json) retain every evaluation time, raw and relative errors, precision settings, measured optimizer work and acceptance result. Each final recipe has one clean measured run at seed 42; no claim of all-seed convergence is made.

Earlier candidates were not accepted merely because their average error improved. The one-inner-iteration Level 2 candidate failed both midpoint gates. The two-inner-iteration Level 2 candidate passed midpoint RMSE but had maximum errors `0.00050282 / 0.00062430`, exceeding the unchanged `0.0005` limit. Its plots were not treated as evidence that the final recipe had passed.

An earlier Level 1 recipe with one inner iteration, history 50 and the same 5,000-call total passed the gates for seeds 42 and 7. These are supporting experiments, not repeat measurements of the final two-inner-iteration recipe.

## Limits and interpretation

This is an approximate PINN, not an exact symbolic solver. The reference amplitude becomes extremely small at late times, so ordinary relative L2 can still be large despite small absolute error. For example, final-recipe Level 1 has RMSE `0.00008982` at `t=2*pi`, while the reference maximum is about `0.00000349`; its relative L2 is therefore about 52.3. This does not mean the late prediction is relatively accurate. Read raw RMSE, reference amplitude and the fixed-initial-scale error together. No late-time values are clipped or hidden.

Training-loop timings exclude model import, independent evaluation and plotting. They are one-device measurements, not guarantees of runtime on every instance. Changes to physical coefficients or initial/boundary conditions need a new convergence check.

## Regression scope

The focused Climate, PINN, notebook, teaching-contract, student-distribution and visualization tests passed: 210 tests. The static course validator still reports the same two issues on the untouched baseline and this branch: a Hangul-containing string in `ETC/runtime/judge_client.py`, and an existing broken heading link in `ETC/environment/JUDGE_CONNECTION.md`. Two stale budget assertions in `test_teaching_presets.py` also fail identically on the baseline; this fix does not claim the entire repository test suite is green.
