# Lab 1 earlier FP64 experiment (superseded)

This is the historical FP64 experiment, not the student configuration. The
current source and notebooks use FP32. See `LABS_FP32_VALIDATION.md` for the
subsequent CPU/L4 checks and current lesson settings. The values below must not
be presented as FP32 results.

Measured on 2026-09-25 with PhysicsNeMo 2.2.2, PyTorch 2.10.0+cu128 and Python
3.12. Each run used 3,000 optimizer calls: 1,000 Adam calls followed by 2,000
L-BFGS calls. L-BFGS trial evaluations are counted separately in `loss.csv`.

## What changed

The upstream introductory notebook contains equations and reference images,
not a complete runnable training configuration. The executable Lab 1 is a local
implementation of those same problems. Its previous fixed-rate Adam recipe
did not reliably recover the inverse source, even after 20,000 steps.

The revised implementation uses float64, normalized coordinates, and an Adam
warm-up followed by L-BFGS on fixed equation points. Forward and parameterized
models retain 3 hidden tanh layers of width 64 and soft boundary losses, now
weighted by 10. The inverse model uses two independently learned 2-by-32 tanh
MLPs with `[2x-1, sin(k*pi*x), cos(k*pi*x)]`, `k=1,...,4`, as inputs.

For the inverse problem, `u=x*(1-x)*v`, with `v=0.1*N_u`, enforces the supplied
zero boundaries exactly. The observation loss is
`1000*mean((v-u_observed/[x*(1-x)])**2)` at the same 100 observed interior points
throughout a run. This gives observations near the boundaries more weight.
The two networks still train jointly using `PhysicsInformer` for `u_xx-f`.
No true source values or analytical derivatives enter training. Additional
analytical solution/source values are used only after/before training for
evaluation, never for checkpoint selection.

These feature and scaling choices were selected for this noise-free teaching
problem. The feature range includes its source frequency. They are not original
upstream settings, evidence of unseen-frequency generalization, or a recipe for
noisy observations. Near-boundary observation scaling can amplify noise.

## Measured results

CPU: Intel Core i7-11700F, two PyTorch threads per run. Evaluation uses 401
points per length including endpoints. The parameterized model is checked at
all five lengths `1, 1.25, 1.5, 1.75, 2`.

| Mode / error | Seed 42 | Seed 43 | Seed 7 |
|---|---:|---:|---:|
| Forward u RMSE | 1.44e-6 | 6.68e-6 | 2.38e-6 |
| Parameterized pooled u RMSE | 5.89e-5 | 6.74e-5 | 2.17e-4 |
| Inverse u RMSE | 3.66e-6 | 3.38e-6 | 2.03e-6 |
| Inverse f RMSE | 0.00409 | 0.00518 | 0.00354 |
| Inverse f maximum error | 0.00840 | 0.02843 | 0.02039 |

All nine runs passed the lesson limits, artifact validation and checkpoint
reload checks. The worst parameterized u RMSE over all tested lengths and
seeds was 0.0004193. The nine runs took about 193 seconds in three parallel
groups; individual runs took 46-69 seconds under that shared load.

GPU: NVIDIA GeForce RTX 3080, inverse mode, the same 3,000-call recipe:

| Seed | u RMSE | f RMSE | f maximum error | PDE RMSE |
|---|---:|---:|---:|---:|
| 42 | 3.91e-6 | 0.00724 | 0.05439 | 0.000326 |
| 43 | 4.91e-6 | 0.00743 | 0.03188 | 0.000584 |

Both GPU runs passed. L4/Brev execution has not been revalidated by this local
run. CPU/GPU random coordinates and floating-point reductions differ; these
are accuracy checks, not bitwise cross-device reproducibility claims.

The original reported failure used different settings, hardware and an older
evaluation grid. Its source RMSE near 0.68 is context, not a controlled ablation
isolating one cause. Changing the learning rate alone was not established as a
fix. The reported accuracy belongs to the complete revised recipe.

## Acceptance and reproduction

Each forward/parameterized length must have u RMSE <= 0.001, PDE RMSE <= 0.01,
and endpoint error <= 0.001. Inverse limits additionally require u RMSE <=
0.0001, f RMSE <= 0.05 and maximum f error <= 0.1. These are explicit teaching
acceptance limits, not competition scores or rigorous global error bounds.

The notebook displays `PASS` or `NOT MET` separately from successful execution.
A two-step Jupyter smoke run completed all three modes and correctly reported
`NOT MET`; it did not overwrite the user's saved notebook results.

The commands below run whichever source version is checked out. On current
`main` they validate FP32, not the historical FP64 measurements above. See
`LABS_FP32_VALIDATION.md` for current results and settings.

From the repository root with the course environment active:

```bash
AI4SCI_RUN_CONVERGENCE=1 python -m pytest ETC/tests/test_lab1_convergence.py -q

python ETC/course_materials/run_validation.py --suite convergence \
  --case pinn_forward --case pinn_parameterized --case pinn_inverse \
  --convergence-steps 3000 --device cpu --output-dir /tmp/lab1-verification-new
```

Choose a fresh output directory. The common validator's default 500-call budget
is not silently extended for Lab 1; specify 3,000 for this recipe. To run the
opt-in regression suite on a GPU, also set `AI4SCI_CONVERGENCE_DEVICE=cuda`.

Local evidence for this verification is under:

- `/tmp/ai4sci-lab1-forward-convergence.OfqGUk`
- `/tmp/ai4sci-lab1-parameterized-convergence.eWdFW7`
- `/tmp/ai4sci-lab1-inverse-convergence.27RtyB`
- `/tmp/ai4sci-lab1-convergence-20260925-inverse-cuda-42`
- `/tmp/ai4sci-lab1-convergence-20260925-inverse-cuda-43`
- `/tmp/ai4sci-lab1-notebook-smoke-20260925`

These temporary directories are not distributed course data. Each complete run
contains `metrics.json`, `loss.csv`, `model.pt`, `predictions.npz`, and an actual
prediction plot. The new script requires a fresh run; old notebook outputs are
not evidence for the revised implementation.
