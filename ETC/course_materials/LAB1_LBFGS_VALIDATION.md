# Lab 1: 1,000 L-BFGS calls in FP32

25 September 2026. PhysicsNeMo 2.2.2, PyTorch 2.10.0+cu128, Python 3.12.

Lab 1 now uses L-BFGS from initialization, with no Adam stage. The three
problems, networks, loss weights, observation data, and accuracy limits are
unchanged. The default is 1,000 optimizer calls, not 1,000 loss evaluations.
Strong-Wolfe line search can evaluate the loss several times within a call.
Some calls can return without changing the parameters after convergence.

## Actual training settings

- `lr=1.0`, `max_iter=1`, `max_eval=20`, `history_size=100`.
- `tolerance_grad=1e-10`, `tolerance_change=1e-14`; FP32 parameters and inputs.
- Forward/inverse: a fixed grid of 256 points in [0,1].
- Parameterized: 32 positions at each of 17 lengths, 544 fixed points total.
- YAML `learning_rate` controls the L-BFGS rate; `batch_size` is the fixed
  number of points per length in this lab. The defaults above reproduce the
  measured grids. CLI `--steps` overrides the default 1,000 calls.
- `metrics.json` records actual learning rate, point count, optimizer calls,
  and closure evaluations. `loss.csv` records loss before each optimizer call.
- The final model is saved. Evaluation errors do not select the checkpoint;
  analytical source values are not used to train the inverse model.

## Measured accuracy

All nine CPU runs (three modes, seeds 42/43/7) passed the existing limits.
All three RTX 3080 runs (seed 42) also passed. These are checks of the supplied
noise-free examples, not a guarantee for every initialization or changed PDE.
The unchanged recipe also passed all three modes on NVIDIA L4 with seed 42
on 27 September 2026; timings and errors are recorded below. Earlier L4 results
in [LABS_FP32_VALIDATION.md](LABS_FP32_VALIDATION.md) used the older hybrid recipe
and are not timings for this 1,000-call recipe.

| Mode and error | CPU seed 42 | CPU seed 43 | CPU seed 7 | RTX 3080 seed 42 |
|---|---:|---:|---:|---:|
| Forward, solution RMSE | 1.180e-5 | 2.274e-7 | 2.880e-7 | 3.484e-6 |
| Parameterized, worst-length solution RMSE | 3.878e-4 | 3.022e-4 | 7.039e-4 | 3.017e-4 |
| Inverse, solution RMSE | 2.088e-6 | 4.478e-6 | 6.680e-6 | 5.431e-6 |
| Inverse, source RMSE | 0.004170 | 0.005856 | 0.007932 | 0.005771 |
| Inverse, maximum source error | 0.027831 | 0.011636 | 0.020636 | 0.021693 |

Each solution/PDE check uses 401 points including endpoints. The parameterized
case checks all five lengths 1, 1.25, 1.5, 1.75, 2 separately. Across these
runs, the largest PDE RMSE was 0.002623 (<0.01), and the largest boundary
error was 0.0003612 (<0.001). Forward/parameterized solution RMSE must be
<=0.001; inverse solution RMSE <=0.0001, source RMSE <=0.05, and maximum
source error <=0.1. None of these limits were relaxed.

The checkpoints were reloaded and predictions compared to the saved arrays;
held-out physical residuals were also independently recomputed. Reproduction
checks use the original execution device. The old test compared CUDA results
against CPU inference at a 1e-7 absolute tolerance and failed for differences
up to 4.77e-7. It now tests on the same device without relaxing any tolerance.

## NVIDIA L4 measurement: 27 September 2026

One fresh Brev student instance, `g2-standard-4:nvidia-l4:1`, using the normal
Launchable environment: Python 3.12.11, PhysicsNeMo 2.2.2, PyTorch 2.10.0+cu128,
CUDA runtime 12.8 and NVIDIA driver 595.91.07. The measured checkout was clean
at `8f6ef9c91dda3060023fcadbc6d8f53497709363`.

Each mode ran once in a separate Python process with seed 42, FP32,
`--device cuda --steps 1000`, `OMP_NUM_THREADS=2`, `MKL_NUM_THREADS=2` and
`MPLBACKEND=Agg`. No model, sampling, loss or optimizer setting was changed.
The measured `pinn_basics.py` SHA-256 matches the current training source:
`3fd273b559f5c60c6c9b1874a8a812cda92d47a5e3d4e72e465b6b399928e4ee`.

| Mode | End-to-end seconds | Closure evaluations | Solution RMSE | PDE RMSE | Maximum boundary error |
|---|---:|---:|---:|---:|---:|
| Forward | 53.734 | 4,846 | 1.19137e-5 | 8.10108e-4 | 1.07288e-6 |
| Parameterized | 51.476 | 3,806 | 3.74179e-4 | 2.10571e-3 | 2.58923e-4 |
| Inverse | 75.428 | 4,381 | 2.39430e-6 | 4.27221e-4 | 0 |

All three runs used exactly 1,000 L-BFGS calls and zero Adam calls. The
parameterized row reports the worst value of each metric across the five
evaluation lengths, not the error at the single preview length. Those lengths
are included in the 17-length training grid; this is not an unseen-length
generalization test. Inverse source RMSE was `0.00339907` and maximum source
error was `0.0196496`. All existing accuracy limits passed without relaxation.

Time is measured around the complete CLI subprocess: imports, CUDA
initialization, before/after evaluation, training and artifact export. It
excludes environment installation and the subsequent independent reload
checks. The three commands totalled 180.638 seconds. These are single-run
measurements, not an average, a worst-case bound or a 110-user startup test.

Artifact validation passed for all three runs. Checkpoint floating-point
tensors were finite FP32; reloaded predictions matched the saved arrays on
the same GPU at the existing tolerances. Recomputed solution/PDE/boundary
and inverse-source errors also passed. All three student CLI stderr logs
were empty. The separate verification process emitted one initial cuBLAS
context warning during re-evaluation, without a failed numerical check.
The measurement instance was deleted after its artifacts were copied out.

## Reproduce

Run with the installed course Python environment from the repository root:

```bash
AI4SCI_RUN_CONVERGENCE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
python -m pytest ETC/tests/test_lab1_convergence.py -q
```

To test CUDA explicitly (this will use the local GPU):

```bash
AI4SCI_RUN_CONVERGENCE=1 AI4SCI_CONVERGENCE_DEVICE=cuda \
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
python -m pytest ETC/tests/test_lab1_convergence.py::test_measured_lesson_convergence_at_published_budget -k 42 -q
```

For the three CLI timings above, run each command once from the repository
root, using a fresh output directory. `/usr/bin/time` reports the same
end-to-end scope (the measurement used `time.perf_counter` around each process):

```bash
measurement_dir="$(mktemp -d -t ai4sci-lab1-l4.XXXXXX)"
for mode in forward parameterized inverse; do
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 MPLBACKEND=Agg \
  /usr/bin/time -f '%e seconds' python 01_labs/01_pinn/source_code/pinn_basics.py \
    --mode "$mode" --device cuda --steps 1000 --seed 42 \
    --output-dir "$measurement_dir/$mode"
done
```

## Related classroom fixes

Lab 2 explicitly registers its supplied time derivatives with PhysicsInformer.
This removes the redundant reminder without muting unrelated warnings; a
missing derivative tensor still fails. Its equations and Adam settings are
unchanged. Tests exercise both the derivative graph and the real CLI output.

Lab 3 adds a left-material zoom alongside the full temperature plot, using
the same saved predictions. D1=5,10,25 and D2=0.1 are unchanged. Tests check
that erroneous predictions remain visible rather than being clipped by fixed
plot limits. Lab 4 and the Challenges are unchanged.
