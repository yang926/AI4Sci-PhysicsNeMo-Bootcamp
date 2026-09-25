# Challenge 4: training budget and independent accuracy checks

The first measurements below used PhysicsNeMo 2.2.2, PyTorch 2.10.0+cu128 and one
NVIDIA GeForce RTX 3080 on 2026-09-25. They are **not L4 timings**; a separate
actual L4 check is recorded at the end. No smaller image, dataset or AFNO
embedding was used to make the results faster.

## Classroom choice

Use **3,000 Adam updates for each of FNO, AFNO and PINO**. This is a common
comparison budget with measured field predictions, not a claim that 3,000 is
the optimal stopping point. It corresponds to 12 training-data passes at batch
size 32. The three training loops took a combined 4.8–5.7 minutes in these runs.
Data generation, setup and dependency installation are additional work.

The old 200-update budget leaves AFNO at about 19–20% validation field error.
At 3,000 updates its independent test field error is about 2.1–2.3%. FNO and
PINO are more accurate on this particular smooth periodic dataset. All three
remain genuine trained models, not loaded answers.

**Do not describe AFNO's field fit as a solved PDE.** Its field error is small,
but its differentiated residual remains large. FNO and AFNO minimize the data
loss; PINO also minimizes the physical residual. Differentiation can amplify
small, high-frequency field errors. This comparison does not establish the
precise spatial source of AFNO's residual or rank these architectures on other
datasets.

## Fixed problem and settings

- Equation: `u - Laplacian(u) = f` on the periodic unit square.
- Grid: 64 × 64, excluding the repeated endpoint; Fourier data modes 0–6.
- Independent splits: 8,000 training, 1,000 validation, 1,000 test samples.
  Dataset seed 42; split seeds 2684470948 / 4091952314 / 233227757.
- FP32 training, batch size 32, constant Adam learning rate 0.001, two CPU
  threads. Fourier parameters use the corresponding complex64 representation.
- FNO and PINO: 32 latent channels, 4 Fourier layers, 12 modes, no padding,
  no coordinate features; one decoder layer of width 32.
- AFNO: 8 × 8 patches, embedding width 256, depth 4, 8 blocks, MLP ratio 4.
- PINO physical-loss weight: 1.0. No physical loss was added to FNO or AFNO.
- Normalization uses training data only. The forcing standard deviation is
  `3.4642210006713867`; the solution standard deviation is
  `0.019919253885746002`. The equation is evaluated after undoing normalization.

The dataset is the corrected reaction-diffusion dataset described in the
Challenge 4 notebook, not the historical `Poisson_Fourier` files. These runs
evaluate the current documented problem; they are not numerical reproductions
of those incompatible historical files.

## Validation used for the budget comparison

Field error is `||u_pred - u_true||₂ / ||u_true||₂`, pooled over all validation
samples and pixels. PDE error below is
`RMSE(u_pred - Laplacian(u_pred) - f) / training_f_std`.
It is a dimensionless diagnostic, **not** an official score and not a ratio to
the test split's own forcing RMS.

| Model / seed | Updates | Field relative L2 | PDE RMSE / training f std |
| --- | ---: | ---: | ---: |
| FNO / 42 | 200 | 0.010210 | 0.121396 |
| FNO / 42 | 1,000 | 0.003435 | 0.047827 |
| FNO / 42 | 3,000 | 0.003196 | 0.021721 |
| FNO / 7 | 200 | 0.008105 | 0.099517 |
| FNO / 7 | 1,000 | 0.002333 | 0.037256 |
| FNO / 7 | 3,000 | 0.002251 | 0.019908 |
| AFNO / 42 | 200 | 0.192174 | 12.253268 |
| AFNO / 42 | 1,000 | 0.048705 | 3.820238 |
| AFNO / 42 | 3,000 | 0.021260 | 2.149867 |
| AFNO / 7 | 200 | 0.199512 | 12.436759 |
| AFNO / 7 | 1,000 | 0.043043 | 3.595945 |
| AFNO / 7 | 3,000 | 0.023180 | 2.273686 |
| PINO / 42 | 200 | 0.011626 | 0.013378 |
| PINO / 42 | 1,000 | 0.003506 | 0.003774 |
| PINO / 42 | 3,000 | 0.001756 | 0.005992 |
| PINO / 7 | 200 | 0.006499 | 0.009243 |
| PINO / 7 | 1,000 | 0.002077 | 0.003257 |
| PINO / 7 | 3,000 | 0.002450 | 0.004367 |

PINO's residual is better at 1,000 than at 3,000 in both seeds; its seed-7 field
error also increases slightly. Constant-rate minibatch Adam does not improve
every held-out metric monotonically. Keeping one 3,000-update budget makes the
class comparison simple, but does not make the final weights the best available
checkpoint. No best-test-checkpoint selection was performed.

## Final independent test at 3,000 updates

| Model | Seed | Training seconds | Test relative L2 | Test PDE RMSE | Test PDE RMSE / training f std |
| --- | ---: | ---: | ---: | ---: | ---: |
| FNO | 42 | 51.6 | 0.003219 | 0.075733 | 0.021861 |
| FNO | 7 | 49.9 | 0.002222 | 0.069031 | 0.019927 |
| AFNO | 42 | 129.3 | 0.021370 | 7.445517 | 2.149261 |
| AFNO | 7 | 89.5 | 0.023372 | 7.887540 | 2.276858 |
| PINO | 42 | 163.2 | 0.001751 | 0.020815 | 0.006008 |
| PINO | 7 | 150.7 | 0.002473 | 0.015124 | 0.004366 |

For PINO, the PhysicsInformer and independent FFT test residuals agree:
0.02081469254 versus 0.02081469201 for seed 42, and 0.01512385445 versus
0.01512386384 for seed 7. Accuracy is evaluated on all 1,000 test examples, not
just the picture shown by the notebook.

The training timer synchronizes CUDA and excludes the intermediate validation
and checkpoint writes. Other CPU tests ran during parts of the experiment; clock
and host-load variation means these two seeds are not controlled timing repeats.
Seed-7 command wall times including startup, intermediate checks, final evaluation
and exports were 60.4 / 100.8 / 164.7 seconds. Dataset generation and installation
are excluded. The seed-42 helper did not yet record command wall time.

## Reproduce

From the repository root, choose new output paths:

```bash
python 02_challenges/04_neural_operators/generate_data.py \
  --output-dir /tmp/operator-benchmark-data \
  --train-samples 8000 --val-samples 1000 --test-samples 1000 \
  --grid-size 64 --max-mode 6 --seed 42

python ETC/course_materials/benchmark_operators.py \
  --level 1 --device cuda --seed 42 --checkpoints 200 1000 3000 \
  --data-dir /tmp/operator-benchmark-data \
  --output-dir /tmp/operator-benchmark-fno-seed42
```

Repeat with level 2 (AFNO), level 3 (PINO), and seed 7, each with a fresh output
directory. The command-line maximum checkpoint overrides the YAML update count;
the measured YAML snapshots still contain their previous `steps: 10000` value.
`model.pt` separately records the actual 3,000 updates. All other model and
training settings above are unchanged.

The organizer helper saves `benchmark.json` plus actual intermediate weights.
It only uses validation data for intermediate comparisons; the unchanged lesson
runner reports its initial/final independent test results. The regression test
`ETC/tests/test_operator_benchmark.py` verifies all three model types against an
uninstrumented four-update CPU run: identical final weights, losses, metrics and
RNG state. That regression checks measurement integrity, not convergence.

Measured sources, SHA-256:

- `operator_training.py`:
  `0b44822178d168dd2beaf74e86f404b0c6d43d2184998cf0a1e91351be06c354`
- `generate_data.py`:
  `607efa5265a9de878e53da5ba491f2cdaa202ff0e1a7e45d54b42f665cdd75e7`
- `benchmark_operators.py` (complete seed-7 reports):
  `3f3b898b7344f1dc0deead8c5b8756bf803b4566f60ea8942f65caa34f1f397e`

Local evidence is under `/tmp/ai4sci-operators-full-20260925/`: `fno-seed42`,
`afno-seed42`, `pino-seed42`, `level1-seed7`, `level2-seed7`, `level3-seed7`,
and their sibling `-checkpoints` directories. Each final run has loss history,
metrics, all test predictions and model weights. These temporary evidence files
and pretrained weights are not distributed as student answers.

## Final preset on an actual NVIDIA L4

All three production runners completed with the published 3,000-update YAML
presets, seed 42, full 64-by-64 grid and 8,000/1,000/1,000 samples. Data were
regenerated independently on the existing Brev L4 using the same generator
and seeds, not copied from a reduced smoke test.

| Model | Training seconds | Command seconds | Test relative L2 | Test PDE RMSE / training f std |
| --- | ---: | ---: | ---: | ---: |
| FNO | 48.2 | 64.7 | 0.001953 | 0.023018 |
| AFNO | 91.3 | 108.4 | 0.021952 | 2.090254 |
| PINO | 168.3 | 188.6 | 0.002970 | 0.007489 |

The three training loops total 5.1 minutes; the instrumented commands total
6.0 minutes including intermediate validation/checkpoint writes and final
artifacts. Dataset generation and installation are excluded. This is one L4
instance, not a capacity test for 110 concurrent users.

Absolute test PDE RMSE is 0.079739 / 7.241103 / 0.025944 for FNO / AFNO / PINO.
PINO's independent FFT and PhysicsInformer checks agree to about 6.3e-9. AFNO's
large differentiated residual remains despite a 2.2% field fit. Seed-42 results
are not bitwise identical across GPU architectures; the figures above describe
these runs, not guaranteed errors. No test-based checkpoint selection was used.

Evidence: `/tmp/ai4sci-curriculum-KBqyXp/results/operator-final-level1-3k`,
`operator-final-level2-3k` and `operator-final-level3-3k` on the L4 instance.
Each `benchmark.json` includes the actual GPU, configuration, dataset manifest,
source hash, validation checkpoints, final independent test and command timing.
