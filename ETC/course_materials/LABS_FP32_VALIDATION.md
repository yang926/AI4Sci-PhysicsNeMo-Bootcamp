# Labs 1–4: FP32 accuracy checks

Date: 2026-09-25. PhysicsNeMo 2.2.2, PyTorch 2.10.0+cu128, Python 3.12.
Training models and coordinates use FP32, not FP64 or automatic mixed precision.
The earlier FP64 Lab 1 experiment is retained in `LAB1_VALIDATION.md`; it does
not describe the current student configuration.

## Settings and problems corrected

| Lesson | Finding | Current recipe |
|---|---|---|
| Lab 1, three modes | FP64 was not established as necessary for accurate inverse recovery. | FP32; 1,000 Adam + 2,000 L-BFGS calls. The previous accuracy limits are unchanged. |
| Lab 2 | At 5,000 fixed-rate Adam updates, all three seeds failed at least one initial-condition check. The notebook only requested 200 updates. | Same model, equations and losses; 5,000 Adam updates with cosine learning rate from 0.001 toward 0.000001. |
| Lab 3, two modes | Small errors across the whole temperature range hid large relative errors in the cooler material and flux mismatch. The 5,000-update parameterized runs failed for all three seeds. | FP32, material-dependent output/flux scaling, normalized resistance input, exact supplied outer temperatures, learned interface conditions; 200 Adam + 100 L-BFGS calls. |
| Lab 4 | A near-zero flow could produce small PDE residuals but incorrect velocity. The old 200-update notebook and a 2,000-update comparison failed. | Same network and equations; initial-data weight 10, 1,000 Adam + 2,000 fixed-batch L-BFGS calls. |

One L-BFGS call can evaluate several trial points. Lab 3 allows up to 20
internal iterations per call; Labs 1 and 4 allow one. `loss.csv` records actual
closure evaluations. The call counts above are not equivalent compute budgets.
No analytical solution is used to choose the saved checkpoint. Lab 1 inverse
training uses only the supplied 100 solution observations, not true source
values. Lab 4 synthetic training uses reference data at t=0 only.

Each lesson calls its own optimizer. `setup(args, defaults=...)` only accepts
explicit default values; it does not branch on Lab number or alter global
precision. Precedence is common defaults, lesson defaults, YAML, then explicit
CLI `--steps`. Tests verify one call's defaults cannot leak into another.

## CPU: three initializations per mode

Intel Core i7-11700F; two PyTorch threads per training process. Seeds 42, 43, 7.
All 21 complete FP32 runs passed. CPU runs overlapped other work, so their wall
times are not used as hardware performance comparisons.

| Mode / reported error | Seed 42 | Seed 43 | Seed 7 |
|---|---:|---:|---:|
| Lab 1 forward, u RMSE | 5.97e-6 | 5.47e-6 | 2.27e-6 |
| Lab 1 parameterized, worst-length u RMSE | 0.000372 | 0.000134 | 0.000407 |
| Lab 1 inverse, u RMSE | 4.00e-6 | 3.94e-6 | 7.92e-6 |
| Lab 1 inverse, f RMSE | 0.01106 | 0.01007 | 0.00901 |
| Lab 1 inverse, maximum f error | 0.08586 | 0.08563 | 0.02271 |
| Lab 2, position RMSE (m) | 0.00177 | 0.01248 | 0.01124 |
| Lab 3 fixed, maximum temperature error (K) | 0.000748 | 0.000351 | 0.0000381 |
| Lab 3 parameterized, worst temperature error (K) | 0.00428 | 0.00383 | 0.00300 |
| Lab 4, worst-time velocity RMSE | 0.000709 | 0.000634 | 0.000722 |
| Lab 4, worst-time gauge-aligned pressure RMSE | 0.00110 | 0.000699 | 0.00122 |

Lab 2's original 5,000-update position RMSEs were 0.12565, 0.06753 and 0.15170 m.
Its extrapolation RMSEs after the fix remain about 1.90, 1.62 and 1.62 m over
(5,8] seconds. Training-interval accuracy is not an extrapolation guarantee.
Lab 4's original 200-update worst-time velocity RMSEs were approximately 0.495,
despite small PDE residuals. Accuracy gates prevent that result being a pass.

## L4: complete default-budget runs

NVIDIA L4 on an already-running Brev instance; seed 42, FP32. Every mode ran
sequentially in a separate temporary source copy. No new GPU was provisioned
and the live Jupyter server/course checkout was not replaced. All seven modes
passed the same absolute criteria used on CPU.

| Mode | CLI wall time | Representative final error |
|---|---:|---:|
| Lab 1 forward | 123.2 s | u RMSE 1.22e-5 |
| Lab 1 parameterized | 110.5 s | worst-length u RMSE 0.000614 |
| Lab 1 inverse | 177.8 s | f RMSE 0.00694; maximum f error 0.04007 |
| Lab 2 | 70.8 s | position RMSE 0.00402 m |
| Lab 3 fixed | 23.7 s | maximum temperature error 0.000717 K |
| Lab 3 parameterized | 35.6 s | worst temperature error 0.00967 K |
| Lab 4 | 160.8 s | worst-time velocity RMSE 0.000657 |

These times include process startup, held-out evaluation, plotting and artifact
writing, not just training kernels. They are one-machine observations, not a
110-user capacity estimate or a CPU/GPU speedup claim. L4 was checked with one
seed per mode; CPU was checked with three. CPU/GPU reductions and random
samples need not produce identical weights or predictions.

The L4 evidence was copied locally to
`/tmp/ai4sci-l4-fp32-results.VGB4VQ/{final-l4-seed42,final-l4-lab3-seed42}`.
Each group's report records source hashes, exact commands, time, GPU name,
actual checkpoint dtypes and accuracy diagnostics. The temporary test runs
are not a GitHub publication or a live Launchable update.

## What the checks cover

- Lab 1: 401 points including endpoints; five lengths 1, 1.25, 1.5, 1.75, 2
  for the parameter family. Each requires u RMSE <=0.001, PDE RMSE <=0.01 and
  boundary error <=0.001. Inverse additionally requires u RMSE <=0.0001,
  f RMSE <=0.05 and maximum f error <=0.1.
- Lab 2: 401 times in [0,5] seconds; position RMSE <=0.1 m, maximum error
  <=0.5 m, acceleration residual RMSE <=0.05 m/s², initial position error
  <=0.1 m and initial velocity error <=0.05 m/s.
- Lab 3: 201 points per material; fixed D1=10 or 41 values D1=5:0.5:25.
  Relative temperature RMSE <=5% left and <=0.5% right, maximum temperature
  error <=0.5 K, boundary and interface-temperature errors <=0.1 K, relative
  interface-flux mismatch <=2%, and each material's normalized PDE RMSE <=0.01.
- Lab 4: held-out 24x24 grid at t=0,0.13,0.37,0.61,0.83,1. Each requires
  velocity and spatial-mean-aligned pressure RMSE <=0.02, PDE RMSE <=0.05,
  initial-field RMSE <=0.02, periodic value error <=2e-5 and spatial-gradient
  error <=1e-4. Raw pressure error and the removed offset remain reported.

These are finite-grid teaching acceptance criteria, not global error bounds or
competition scores. Lab 4 accuracy is for the synthetic Taylor–Green fixture,
not a validated forecast from the original weather array. All criteria are
checked from measured errors rather than trusting a stored `passed` flag.

## Reproduce

The final default test suite passed 805 tests and 22 subtests; 28 opt-in or
optional tests were skipped in that invocation. The 21 opt-in Lab training
cases were executed separately as reported above, not counted as untested.
Fifteen pre-existing dependency/test warnings remain. All four notebooks also
executed successfully as temporary two-call copies and produced their expected
plots/artifacts; those short results correctly do not pass lesson accuracy.

The artifact validator verifies actual FP32 tensors/arrays, full step histories
for these Lab recipes, optimizer phase counts and L-BFGS closure totals. It
recomputes accuracy gates and checks parameter/time coverage; a stored pass
flag or small aggregate PDE loss cannot substitute for these checks.

From the repository root in the course environment:

```bash
AI4SCI_RUN_CONVERGENCE=1 python -m pytest \
  ETC/tests/test_lab1_convergence.py ETC/tests/test_lab2_accuracy.py \
  ETC/tests/test_lab3_accuracy.py ETC/tests/test_lab4_accuracy.py -q
```

Set `AI4SCI_CONVERGENCE_DEVICE=cuda` for the GPU regression suite. Short
notebook execution checks are separate:

```bash
python ETC/course_materials/run_notebooks.py \
  --case lab1 --case lab2 --case lab3 --case lab4 \
  --device cpu --steps 2 --output-dir /tmp/labs-notebook-check-new
```

The central runner additionally validates artifacts and held-out improvement:

```bash
python ETC/course_materials/run_validation.py --suite convergence \
  --case pinn_forward --case pinn_parameterized --case pinn_inverse \
  --convergence-steps 3000 --device cuda --output-dir /tmp/lab1-check-new
```

Run `projectile` with 5,000 calls, `diffusion`/`diffusion_parameterized` with
300, and `navier_stokes` with 3,000. The runner's default 500-call budget is not
silently raised. Choose new output directories; previous results are preserved.

Local CPU evidence directories:

- `/tmp/ai4sci-lab1-fp32-cpu-20260925`
- `/tmp/lab2-final-seed42`, `/tmp/lab2-final-seed43`, `/tmp/lab2-final-seed7`
- `/tmp/lab3-fp32-audit/verified-final`
- `/tmp/lab4-audit-final-cpu-seed42`, `/tmp/lab4-audit-final-cpu-seed43`, `/tmp/lab4-audit-final-cpu-seed7`
- `/tmp/ai4sci-all-labs-fp32-notebook-smoke-20260925`

Every full run saves a model, predictions, measured errors, training history
and an actual prediction figure. Checkpoint reloads reproduce saved outputs.
Existing user notebook outputs were preserved; they are not evidence for the
revised code. Rerun the notebook to obtain its current results.
