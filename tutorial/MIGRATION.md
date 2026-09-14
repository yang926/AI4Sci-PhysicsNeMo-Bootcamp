# Tutorial migration to PhysicsNeMo 2.2.2

This change preserves the introduction and all four labs: PINN fundamentals including forward/parameterized/inverse problems; projectile motion; fixed and parameterized two-material diffusion; periodic Navier–Stokes with the original initial-data preparation. It replaces the retired symbolic training orchestration with `physicsnemo.models.mlp.FullyConnected`, `physicsnemo.sym.eq.pde.PDE`, `PhysicsInformer`, and explicit PyTorch optimization loops.

The original source copyright notices and data/images are retained. Historical code remains in Git history. Current notebook code examples, execution cells, configuration, visualization and file links describe the current scripts together, rather than presenting retired APIs as executable instructions.

## Changes to teaching implementation

- `PDE.dim` is explicit. The retired predefined Navier–Stokes module is replaced by the same 2-D constant-density equations written in a local PDE class. PhysicsInformer automatically computes spatial derivatives x/y/z; the examples explicitly provide time derivatives through PyTorch autograd where needed.
- CLI `--device`, `--steps`, `--seed`, `--output-dir`, `--config` makes CPU execution checks and GPU teaching runs use the same scripts. The notebooks default to CPU and 200 steps, configurable with `AI4SCI_DEVICE` and `AI4SCI_STEPS`. Those short runs are execution checks, not certification of convergence.
- Plain YAML stores steps, batch size, learning rate and network size. MLPs use smooth tanh activation, three layers of width 64 by default. All loss scaling is visible in code; these teaching settings require measured convergence checks for the intended hardware and lesson duration.
- `.npz`, `model.pt`, `metrics.json`, `loss.csv`, and `preview.png` replace old framework-specific output directory formats. The ParaView sections explicitly export current predictions to CSV.
- Existing output directories are rejected before training; notebook execution cells create a fresh UUID-suffixed run directory.
- `initial_loss` and `final_loss` report the fixed evaluation objective before training and after the final optimizer update. Per-minibatch losses are separately labeled in metrics and retained in `loss.csv`.
- All training modes record independent fixed-point PDE residual metrics before and after training. Analytical solution RMSE is added where a reference exists. The real-data Navier–Stokes path has no future weather target and does not report invented forecast accuracy.
- Lab 1 now runs the three examples that were previously explanation-only: forward u_xx=1; a length parameter l in [1,2]; an inverse source f(x)=x+sin(4πx) inferred from 100 observations of the original analytical solution. Two MLPs represent u and f in the inverse case. The source target is only used for evaluation.

## Explicit consistency corrections

- Lab 1 summation bounds now use 1…N for N samples. The parameterized boundary-loss expression has the missing square and closing parenthesis restored. Length-weighted residual sampling matches the stated domain integral; the inverse example no longer accidentally includes the unrelated length parameter in u(x).
- Projectile text uses -9.81 m/s², matching the original program. The zero residual `y_tt + 9.81` is algebraically identical to the original target `y_tt = -9.81`. Validation separately reports trained interval 0–5 seconds and extrapolation 5–8 seconds.
- Composite-bar interfaces monitor physical heat-flux continuity D1*T1_x − D2*T2_x, not just bare derivative equality. Parameter D1 is explicitly a spatially constant symbolic coefficient rather than an ambiguous string-defined spatial function. Values D1 in [5,25], D2=0.1, end temperatures 0 and 100, and the piecewise analytical solution are unchanged.
- The Navier–Stokes array path resolves relative to the script rather than the working directory. The original data normalization is preserved, including the unvalidated pressure factor; see [data provenance](navier_stokes/DATA_PROVENANCE.md). Its six-hour output wording is made consistent by writing 11 inclusive frames over a 60-hour scale. The synthetic smoke fixture is a separate, explicitly labeled option.

## Validation boundary

`tests/test_tutorials.py` checks analytical PDE residuals, initial/boundary conditions, composite-bar interface flux, Taylor–Green periodicity and Navier–Stokes residuals, finite forward/backward passes in all modes, and the loader's exact normalization and missing-data failure. CLI and notebook execution results and hardware verification are reported by the repository's overall validation report. Merely passing a short execution test does not prove that every lesson has converged or that a weather model has predictive skill.
