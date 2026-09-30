# Lab 4 training-efficiency comparison

Historical raw-input, 60-hour comparison. These settings and measurements remain
available through `--recipe efficient` and `--recipe upstream`. The teaching
notebook now explicitly selects a different, derived-input six-hour
demonstration; see [its changes, measurements and limitations](LAB4_SIX_HOUR.md).
The L4 timing below applies only to this older raw-input recipe.

## Scope

The experiment keeps the supplied `data_lat.npy`, the periodic square, all three
initial fields, the original nondimensionalization, the 60-hour time interval,
the incompressible equations and FP32 training. Future weather observations are
not available. Initial-field fit and equation residuals are separate diagnostics,
not evidence of weather-forecast skill.

The comparison varies the representation and training budget, not the physical
problem. All candidates keep six 256-unit SiLU hidden layers, weight
normalization, Adam at 0.001 with a factor of 0.95 per 3,000 updates, 2,048 samples
per constraint, and the original observation-sum / interior-area loss scaling.

An optional representation supplies integer periodic frequencies 1, 2, 4 and 8
and expresses network outputs relative to the initial training-field mean and
standard deviation. The model converts its outputs back to the original
normalized u/v/p **before** differentiation and loss evaluation. This does not
change the PDE or substitute an analytical answer. The original single-frequency
representation remains available for comparison.

## Measured choice

Local hardware: NVIDIA GeForce RTX 3080. These are training-loop timings,
including checkpoint diagnostics but excluding dependency installation.
The class preset is **3,000 Adam updates with multiscale periodic features**.
In the runs below it recovered the initial spatial patterns more accurately in
about one third of the time of 10,000 original-recipe updates. This is a bounded
class-time / initial-fit choice, not an optimum across all possible settings.
The 50,000-update upstream run has not been measured here.

| Representation | Adam updates | Seconds | Initial u RMSE / std | Initial v RMSE / std | Initial p RMSE / std |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original, seed 42 | 1,000 | 62.4 | 0.971 | 0.982 | 0.748 |
| Original, seed 42 | 3,000 | 184.9 | 0.852 | 0.888 | 0.686 |
| Original, seed 42 | 10,000 | 639.5 | 0.379 | 0.421 | 0.214 |
| Multiscale, seed 42 | 1,000 | 69.1 | 0.387 | 0.434 | 0.170 |
| Multiscale, seed 42 | 3,000 | 204.6 | 0.257 | 0.294 | 0.121 |
| Multiscale, seed 7 | 3,000 | 205.7 | 0.268 | 0.309 | 0.123 |

The table's initial fit uses a fixed 128-by-128 Cartesian subset of the source
array. For the original recipe that subset can overlap training observations;
it must not be described as an unseen validation set. The candidate comparison
reserves a separate 32-by-32 Cartesian initial grid from training, including
from calculation of its output-normalization statistics. On that strictly
reserved grid, u/v normalized RMSE is 0.248/0.282 for seed 42 and 0.266/0.303
for seed 7. Normalization in this last comparison uses the reserved field's
component standard deviations.

## Important physical limitation

Improved initial fit did **not** improve every physical diagnostic. An independent
64-by-64 cell-center PDE check at six times gave:

| Run | PDE RMSE at t=0 | PDE RMSE at t=0.13 | PDE RMSE at t=1 | RMS pooled over six times |
| --- | ---: | ---: | ---: | ---: |
| Original, 10,000, seed 42 | 2.793 | 0.296 | 0.060 | 1.148 |
| Multiscale, 3,000, seed 42 | 3.358 | 0.276 | 0.030 | 1.376 |
| Multiscale, 3,000, seed 7 | 3.295 | 0.255 | 0.027 | 1.349 |

The input field has substantial discrete divergence in the lesson's planar
periodic coordinates. A diagnostic Fourier Helmholtz decomposition assigns about
33.4% of its fluctuating velocity energy to the divergent component. This is a
discrete diagnostic, not a proof excluding every possible continuous interpolant.
It shows why closely fitting the sampled wind and enforcing incompressibility
can compete. The data were **not** projected, smoothed, or replaced for training.

All eleven saved frames were finite, but velocity RMS fell from 0.110 to 0.018
in the original run and from 0.113–0.116 to 0.016–0.019 in the multiscale runs.
Such strong weakening and small late-time residuals do not certify a correct
flow prediction. There is no future reference with which to validate it. The
notebook now asks learners to examine initial fit and per-time physics errors
separately and makes this limitation explicit.

The production class preset consequently uses the denser 64-by-64 PDE grid.
The separate analytical regression remains a test fixture, never the student
lesson or evidence that these original-data runs converged.

## Reproduction and provenance

Base repository revision: `728cf5076437553cc0191512c8fecbbb50e9e5c9`.
The comparison used that original-data equation/loss implementation plus the
multiscale representation documented above. No hidden future labels, reference
solution, or pretrained model were used.

Original array SHA-256:
`45d2226d51f054d64a9793e31b117bd769a412cae6aa0cadaf109dc15934df01`.

Local benchmark records (not bundled checkpoints):

- `/tmp/lab4-eff-original-seed42/metrics-1000.json`, `metrics-3000.json`,
  `metrics-10000.json`, and `dense-10000.json`.
- `/tmp/lab4-eff-multiscale-seed42/metrics-1000.json`, `metrics-3000.json`,
  and `dense-3000.json`.
- `/tmp/lab4-eff-multiscale-seed7/metrics-3000.json` and `dense-3000.json`.
- Each dense diagnostic also has a fixed-color-scale PNG comparing the actual
  input, predicted initial state, and predicted 60-hour state.

Experimental driver SHA-256:
`18f2e613e724d42680b86d940739361a153f91fe91950ea6295dcb2af8c364fa`
(`/tmp/lab4_efficiency_bench.py`). Independent dense diagnostic SHA-256:
`61a6f3ed1f5a56fe5a31caec4e4aa216cc3fd4285f9fb6536e3de233397bbe28`
(`/tmp/lab4_dense_diagnostics.py`). Timings include the driver's intermediate
evaluation/checkpoint work; they are not end-to-end Launchable startup times.
The driver used the same original loss and learning-rate schedule, but does not
include every production finite-gradient guard, so production time is measured
separately rather than assumed identical.

Run the published class preset from the repository root:

```bash
python ETC/reference_labs/04_navier_stokes/source_code/navier_stokes.py \
  --device cuda --steps 3000 --seed 42 --output-dir outputs/lab4-class-seed42
```

Compare the original representation at the tested 10,000-update budget:

```bash
python ETC/reference_labs/04_navier_stokes/source_code/navier_stokes.py \
  --recipe upstream --device cuda --steps 10000 --seed 42 \
  --output-dir outputs/lab4-upstream-10000-seed42
```

Omitting `--steps` with `--recipe upstream` selects the separate original
50,000-update configuration. Existing output directories are never overwritten.
Production metrics record the recipe, feature frequencies, training-only output
normalization, reserved sample count, elapsed training seconds and per-time errors.

Validation of the production entry point used a two-update, small-network CPU
execution on the unchanged original array; its outputs are at
`/tmp/lab4-efficient-production-smoke`. That checks loading, training, metrics,
checkpoint and eleven-frame export, not convergence. The focused tests cover
original-recipe preservation, periodic values/derivatives, correct chain-rule
scaling, disjoint initial validation and notebook/config agreement.

## Actual NVIDIA L4 confirmation

The published production entry point was run on an existing NVIDIA L4 instance
with the complete 3,000-update, six-by-256 class preset, seed 42, FP32 and the
unchanged original data. **End-to-end command time was 202.69 seconds** (about
3 minutes 23 seconds), including startup, evaluation and artifact export. This
is measured L4 time, not an extrapolation from the RTX 3080 loop timings above.

- Reserved initial u/v/p normalized RMSE: 0.247509 / 0.282137 / 0.118537.
- 64-by-64 PDE RMSE pooled over six times: 1.376000; at t=0: 3.358085;
  at t=1: 0.030214.
- Maximum periodic value / first-gradient mismatch: 5.81e-7 / 6.87e-5.
- Remote artifacts: `/tmp/ai4sci-curriculum-KBqyXp/results/lab4-efficient3k`.

Measured source SHA-256:
`1051144423dada65a828fe220064dbbea510a4773b82b1985bc40042ffff5e41`.
The final source SHA-256 is
`cdd0054d897b2680342ff9822750dc24c6a0a8377dc1ae2d2882f342ed7d1791`.
Reversing only two later edits reproduces the measured source hash exactly:
the default for custom configurations that omit `steps` now follows the selected
recipe, and the unused synthetic/tanh branch uses the computed input width.
Neither edit changes the measured class run: its configuration explicitly sets
3,000 steps and it uses the SiLU branch. The numerical results match the local
seed-42 candidate. They do not remove the physical limitations above.

No globally optimal learning rate, iteration count, or convergence guarantee is
claimed by this bounded comparison.

## Flow-weakening follow-up, 2026-09-28

The published 3,000-update preset was reproduced on the local RTX 3080, retaining
the original data, PDE, FP32 network and scales. Final kinetic energy was only
2.62% of the model's own initial prediction. This is not a display-fading effect
and must not be presented as validated physical dissipation.

Two training alternatives were tested without replacing the initial data:

| Experiment / seed | Updates | Initial u/v NRMSE | Pooled PDE RMSE | Final / predicted initial energy |
| --- | ---: | --- | ---: | ---: |
| Published preset / 42 | 3,000 | 0.2475 / 0.2821 | 1.3760 | 2.62% |
| Increasing time horizon / 42 | 6,000 | 0.2650 / 0.3591 | 0.7359 | 6.70% |
| Integral conservation penalties / 42 | 3,000 | 0.2838 / 0.3155 | 1.2784 | 76.47% |
| Integral conservation penalties / 7 | 3,000 | 0.3579 / 0.4388 | 1.1657 | 94.55% |

The time-horizon experiment explicitly sampled initial/early PDE times, expanded
the interval, and finished with full time coverage. The conservation experiment
added spatially averaged kinetic-energy and mean-momentum balance penalties
derived from the same periodic incompressible equations. It used no future labels.
Neither alternative is promoted to the student recipe: the initial fit worsened,
and energy retention did not establish correct dynamics. For example, conservation
seed 42 still changed mean u from -0.0160 to +0.0120 and produced a simplified
shear-band pattern. There is no future reference that validates this evolution.

The notebook now displays RMS speed, total/fluctuating energy retention, and mean
velocity alongside the unchanged fixed-scale playback. These diagnostics also
work on existing saved predictions. Newly executed runs record them under
`flow_diagnostics` in `metrics.json`. Ratios use the predicted t=0 frame, not the
observed input. Zero-baseline ratios are undefined. The 50% visual review flag
is a heuristic, not an accuracy threshold; retaining energy is not a pass.

The original input, PDE and default training recipe remain unchanged. Further
work on incompressibility-compatible initial velocity/pressure would be an
explicit initial-condition change, not a silent optimizer fix. This bounded
comparison does not prove that all unchanged-data training methods fail. The
50,000-update upstream run has not been measured in this follow-up.

The user stopped that longer run (last printed update 9,500; no completed result)
and requested a fresh **3,000-update upstream** measurement instead. The production
entry point completed in 198.06 training-loop seconds on the local RTX 3080.
Initial u/v/p normalized RMSE on the common 32-by-32 source subset was
0.84362 / 0.88286 / 0.68499; those points may overlap upstream training. Independently
recomputed PDE RMSE on the common 64-by-64 grid at six times was 0.30934.
Final kinetic energy was 116.49% of the model's own predicted t=0 energy.
The fixed-scale image already misses much of the input's structure at t=0:
the smaller PDE residual and lack of later fading therefore do not establish
an improvement. This run does not replace the class preset or validate the
future flow. No 50,000-update completion or L4 timing is claimed for this test.

Local reproducibility record (private validation archive, not distributed):
`lab4-repair-20260928/REPORT.md`,
`compare_training.py`, and the per-run metrics, predictions and checkpoints in
that directory. The analytical checks of the optional penalties verify only
their algebra; no synthetic flow was substituted into this Lab.
