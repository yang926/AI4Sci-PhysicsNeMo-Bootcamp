# Challenge training budgets and remaining error

These are bounded class budgets, not globally optimal settings or certificates
of convergence. Task completion, successful execution and numerical accuracy
are different checks. The v3 judge awards completion points; training errors
do not add points or break ties.

## Current presets

| Exercise | Adam updates | Scope |
| --- | ---: | --- |
| Wave 1 | 20,000 | Two seeds improve over 10,000; later-time error remains |
| Wave 2–3 | 10,000 each | Independent condition checks; no verified exact future solution |
| Fluid 1–3 | 10,000 each | Class run; remaining wall/CFD errors prevent an accuracy claim |
| Climate 1–2 | 10,000 each | Compare every field and time against the analytic baseline |
| FNO / AFNO / PINO | 3,000 each | [Full-size, equal-budget comparison](OPERATOR_EFFICIENCY.md) |

Wave 1 uses interior/initial/boundary batches 512/256/256. Other PINN Challenges
retain 128/64/64; Fluid adds four flux lines with 64 points each. These are current
explicit-loop settings, **not the much larger legacy Hydra batches**. PINN
models remain three 64-unit tanh hidden layers, constant learning rate 0.001,
FP32. Original mathematical conditions are restored; framework assembly and
numerical presets remain adaptations documented in [migration notes](MIGRATION.md).

The former 200-update notebook override is removed. `AI4SCI_STEPS=2` deliberately
checks execution only. Notebook and Level YAML defaults agree. Original
Fluid/Climate 40,000-update budgets remain selectable with `--steps 40000`, but
the current 10,000-update measurements do not certify those longer runs.

## Wave 1 comparisons

Field error is pooled relative L2 against the independently checked analytical
solution at five times, not training loss or one favorable snapshot.

| Device / seed | Experiment | Updates | Field relative L2 | Time |
| --- | --- | ---: | ---: | ---: |
| L4 / 42 | Adam, 128/64/64 batches | 10,000 | 0.15970 | 151.2 s command |
| L4 / 42 | Cosine LR to 1%, same batches | 10,000 | 0.45553 | 144.7 s training |
| L4 / 42 | Adam, 512/256/256 batches | 10,000 | 0.16034 | 143.9 s training |
| L4 / 42 | Adam, 512/256/256 batches | 20,000 | 0.05590 | 286.8 s training |
| RTX 3080 / 7 | Adam, 512/256/256 batches | 10,000 | 0.14201 | 160.5 s training |
| RTX 3080 / 7 | Adam, 512/256/256 batches | 20,000 | 0.05829 | 323.5 s training |

At 20,000, held-out PDE RMSE is 0.00919 / 0.00909 for seeds 42 / 7, but final-time
relative L2 remains 18.4% / 16.0%. Near a small reference amplitude, relative
error can be large despite small absolute error: seed 7 at `t=pi/2` has RMSE
0.00776 and relative L2 37.6%. Show the time slices; pooled 5.6–5.8% is not a
bound at every time.

A separate fixed-point FP32 L-BFGS experiment used 3,000 outer calls, 6,142
closure evaluations and batches 2048/1024/1024 on CPU: relative L2 0.07009 and
PDE RMSE 0.01392. The smaller 1,000-call, 512/256/256 trial had relative L2 0.4189.
These are not controlled GPU timing comparisons or proof that L-BFGS is worse
generally. They did not justify replacing all Challenge optimizers. The
unsuccessful cosine schedule is not adopted either.

## Advanced Wave and Fluid

Seed 42, NVIDIA L4, FP32. Fixed held-out samples are checked at 200, 1,000, 3,000
and 10,000 updates. Training times exclude intermediate checks, setup and final
export; they are not end-to-end Launchable startup times.

| Level at 10,000 | Training seconds | Held-out RMSE |
| --- | ---: | --- |
| Wave 2 | 143.6 | PDE 0.02734; initial u 0.01323; initial ut 0.00988; boundary 0.02807 |
| Wave 3 | 172.6 | PDE 0.02045; initial u 0.10204; initial ut 0.01322; Robin 0.06141 |
| Fluid 1 | 229.8 | Weighted continuity 0.09477; wall 0.13797; flux 0.02506 |
| Fluid 2 | 241.1 | Weighted continuity 0.05963; wall 0.20117; flux 0.05672 |
| Fluid 3 | 318.7 | Weighted continuity 0.08535; wall 0.22364; flux 0.20819; initial rest 0.07753 |

Wave 3 plateaus: initial-u RMSE changes only 0.10992→0.10204 from 3,000→10,000;
Robin error changes 0.05985→0.06141. More iterations alone did not solve it.
The original Gaussian tails are not exactly initial-Robin-compatible. No
analytical future truth is supplied for Wave 2/3; residuals alone cannot certify
solution accuracy.

Fluid 1 compares all 56,942 bundled OpenFOAM points, not just its preview grid.
At 10,000 updates, u/v/p RMSE is 0.33543 / 0.16370 / 0.91964. Unweighted
continuity/x-momentum/y-momentum RMSE is 0.31737 / 0.44219 / 0.16718. Do not
present the smaller weighted training terms as these physical residuals.
No corresponding CFD truth is supplied for Levels 2/3. Full CFD plots reveal
the remaining error.

An additional Fluid 1 fixed-point L-BFGS trial on RTX 3080 used 3,000 calls,
6,109 closure evaluations, batches 512/256/256 and 16 flux lines of 256 points.
It took 257.9 training seconds; full CFD u/v/p RMSE improved to
0.22246 / 0.08239 / 0.26466 and wall RMSE to 0.06990. However, weighted PDE RMSE
at independent samples was 0.15388 / 0.38888 / 0.43654 versus
0.01507 / 0.01207 / 0.00867 on fixed training points. A separate 4,096-point
unweighted check was 0.37909 / 0.93574 / 1.05572. This is substantial
fixed-collocation overfitting, not an overall validated improvement, so it is
not a new student preset. Evidence: `/tmp/ai4sci-fluid1-lbfgs-scale4-seed42-20260925`.

Wave 3 and Fluid still need numerical work before claiming high-accuracy
simulation. Do not hide that by changing initial conditions, removing walls,
replacing real data or displaying a pretrained answer as a learner's result.

## Climate baseline

Seed 42, L4, original coefficients including `gamma0=0` in Level 2. The
five-time aggregate relative L2 improves as follows:

| Level | 200 updates | 1,000 | 3,000 | 10,000 | Training seconds at 10,000 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Climate 1 | 0.45206 | 0.15132 | 0.07017 | 0.04611 | 116.8 |
| Climate 2 | 0.45942 | 0.14520 | 0.04760 | 0.03155 | 206.0 |

At 10,000, Level 1 PDE RMSE is 0.00705. Level 2 atmosphere/ocean PDE RMSE is
0.00444 / 0.00564; separate field relative L2 is 0.03144 / 0.03166. Neither
aggregate is a uniform-in-time guarantee. The analytical temperature decays
toward zero, so late-time relative error is enormous even for an absolute RMSE
around 0.007–0.008. Read the absolute errors and the reference amplitude too;
do not silently drop late-time rows or present this as weather-forecast accuracy.
No nonzero-coupling performance claim follows from this uncoupled baseline.

## Reproduce

Use an isolated checkout and a fresh output path:

```bash
python ETC/course_materials/benchmark_challenges.py \
  02_challenges/01_wave/wave_l1.py --device cuda --seed 42 \
  --checkpoints 200 1000 10000 20000 \
  --output-dir /tmp/wave1-budget-comparison
```

The helper uses complete instructor functions and saves `benchmark.json`,
settings and artifacts. Local student training uses student functions, while
evaluation remains independent. PhysicsNeMo 2.2.2 / PyTorch 2.10.0+cu128 were
used. RTX timings are not estimates for L4 or 110 simultaneous instances.

L4 evidence: isolated existing-instance directory
`/tmp/ai4sci-curriculum-KBqyXp/results/`, including `wave1-original10k`,
`wave1-cosine10k`, `wave1-samples512-20k`, and per-Level `*-baseline10k` folders.
The second Wave seed is at `/tmp/ai4sci-wave1-class-seed7-20260925` locally.
Wave 1 source SHA-256:
`d5daebd893d34f9afaeb043bbfa46750bc0cde527f8d7b5a64f0ecb7b2195d9f`;
measured YAML SHA-256:
`1629afc80e2fcaf17858baf63a9d8e5274c20ae38d3698869758734ba10d593d`.
Earlier reports retain old YAML step defaults; the command-line checkpoint
maximum overrides those. Generated weights and learner notebook outputs are
not published as student answers.
