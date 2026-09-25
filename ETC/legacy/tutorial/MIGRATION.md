# Tutorial migration to PhysicsNeMo 2.2.2

This change preserves the introduction and all four labs: PINN fundamentals including forward/parameterized/inverse problems; projectile motion; fixed and parameterized two-material diffusion; periodic Navier–Stokes with the original initial-data preparation. It replaces the retired symbolic training orchestration with `physicsnemo.models.mlp.FullyConnected`, `physicsnemo.sym.eq.pde.PDE`, `PhysicsInformer`, and explicit PyTorch optimization loops.

The original source copyright notices and data/images are retained. Historical code remains in Git history. Current notebook code examples, execution cells, configuration, visualization and file links describe the current scripts together, rather than presenting retired APIs as executable instructions.

## Changes to teaching implementation

- `PDE.dim` is explicit. The retired predefined Navier–Stokes module is replaced by the same 2-D constant-density equations written in a local PDE class. PhysicsInformer automatically computes spatial derivatives x/y/z; the examples explicitly provide time derivatives through PyTorch autograd where needed.
- CLI `--device`, `--steps`, `--seed`, `--output-dir`, `--config` makes CPU execution checks and GPU teaching runs use the same scripts. All four Labs train in FP32. Notebook budgets are Lab 1: 1,000 L-BFGS calls, Lab 2: 5,000 Adam updates, Lab 3: 200 Adam plus 100 L-BFGS calls, and Lab 4: 3,000 Adam updates in the measured class preset. `AI4SCI_STEPS` explicitly overrides the selected budget. L-BFGS can evaluate multiple trial updates per call; the count is recorded separately. Short execution checks are not accuracy certification.
- Plain YAML stores steps, batch size, learning rate and network size. Model and optimizer choices are lesson-specific. Original-data Lab 4 retains the six-layer, width-256 SiLU network; its class preset adds periodic input frequencies and training-data output scaling. All loss scaling is visible in code. See the measured reports before treating a preset as converged.
- `.npz`, `model.pt`, `metrics.json`, `loss.csv`, and `preview.png` retain the run artifacts. Labs 2 and 3 additionally write actual TensorBoard events and native ParaView VTP files; Lab 4 writes the original-data time series. Lab 3 can reload a checkpoint and infer at another conductivity without training.
- Existing output directories are rejected before training; notebook execution cells create a fresh UUID-suffixed run directory.
- `initial_loss` and `final_loss` report the fixed evaluation objective before training and after the final optimizer update. Per-minibatch losses are separately labeled in metrics and retained in `loss.csv`.
- All training modes record independent fixed-point PDE residual metrics before and after training. Analytical solution RMSE is added where a reference exists. The real-data Navier–Stokes path has no future weather target and does not report invented forecast accuracy.
- Lab 1 now runs the three examples that were previously explanation-only: forward u_xx=1; a length parameter l in [1,2]; an inverse source f(x)=x+sin(4πx) inferred from 100 observations of the original analytical solution. Two MLPs represent u and f in the inverse case. The source target is only used for evaluation.

## Explicit consistency corrections

- Lab 1 summation bounds now use 1…N for N samples. The parameterized boundary-loss expression has the missing square and closing parenthesis restored. Length-weighted residual sampling matches the stated domain integral; the inverse example no longer accidentally includes the unrelated length parameter in u(x).
- Projectile text uses -9.81 m/s², matching the original program. The zero residual `y_tt + 9.81` is algebraically identical to the original target `y_tt = -9.81`. Validation separately reports trained interval 0–5 seconds and extrapolation 5–8 seconds.
- Composite-bar interfaces monitor physical heat-flux continuity D1*T1_x − D2*T2_x, not just bare derivative equality. Parameter D1 is explicitly a spatially constant symbolic coefficient rather than an ambiguous string-defined spatial function. Values D1 in [5,25], D2=0.1, end temperatures 0 and 100, and the piecewise analytical solution are unchanged.
- The Navier–Stokes array path resolves relative to the script rather than the working directory. The original data normalization is preserved, including the unvalidated pressure factor; see [data provenance](../../../01_labs/04_navier_stokes/DATA_PROVENANCE.md). Its six-hour output wording is made consistent by writing 11 inclusive frames over a 60-hour scale. The synthetic smoke fixture is a separate, explicitly labeled option.

## Validation boundary

Lab 1's executable implementation was revised after its inverse source recovery
failed a full teaching run. The current notebook uses 1,000 optimizer calls,
single precision (FP32), and fixed-batch L-BFGS without an Adam stage. Forward and
parameterized runs keep soft boundary penalties (weight 10) and normalize their
inputs. The inverse run uses two 32-wide, two-hidden-layer tanh networks with
coordinate features `[2x-1, sin(k*pi*x), cos(k*pi*x)]`, `k=1,...,4`, and enforces
`u=x*(1-x)*v` exactly. It fits the original 100 noise-free observations as
`1000 * mean((v-u_observed/[x*(1-x)])**2)`. This boundary-aware scaling and feature
choice are specific teaching implementation decisions, not original upstream
settings or a general recipe for noisy inverse problems. True source values
remain evaluation-only. The current default supersedes the historical short
execution checks and inverse loss weight described above.

Each Lab 1 run now checks solution, PDE and endpoint errors on 401 points; the
inverse check also requires source RMSE and maximum error, and the parameterized
check applies to every reported length. Failed criteria are shown separately
from successful program execution.

Lab 2 retains its network, equations and loss terms, but decays Adam's learning
rate from 0.001 toward 0.000001 across 5,000 updates. Its checks include the
initial position and velocity, not only the trajectory. Extrapolation beyond
the trained 0–5 second interval remains a separate diagnostic.

Lab 3 rescales each material by its temperature/flux scale and uses normalized
resistance as the parameter input. The supplied outer temperatures are imposed
exactly by an output transform; temperature and physical heat-flux continuity
at the interface remain learned constraints. The two-network model, physical
coefficients and equation are unchanged. It uses 200 Adam calls and 100 L-BFGS
calls (at most 20 internal iterations per L-BFGS call). Accuracy is checked in
each material and at the interface for 41 conductivities across [5,25].

Lab 4 defaults to the original supplied weather array, not Taylor–Green.
The measured class preset uses 3,000 Adam updates; `--recipe upstream`
retains the original 50,000-update representation and schedule for comparison.
The original initial data and PDE are unchanged. Initial-field fit and PDE
residuals are reported separately: there is no future weather target here.
The sampled input contains discrete divergence under the planar periodic
model, so fitting the initial field and reducing continuity error compete.
Taylor–Green is an explicit synthetic verification option only. Its old
1,000 Adam plus 2,000 L-BFGS accuracy results do not validate the weather run.

Each lesson selects its own settings and optimizer explicitly; the common
`setup(args, defaults=...)` helper does not inspect Lab numbers or change the
global floating-point dtype. YAML settings then override those defaults, and
an explicit CLI `--steps` overrides YAML. See
`ETC/course_materials/LAB1_LBFGS_VALIDATION.md`,
`ETC/course_materials/LABS_FP32_VALIDATION.md` (historical recipes), and
`ETC/course_materials/LAB4_EFFICIENCY.md` for their respective measured results.
`LAB1_VALIDATION.md` retains the earlier FP64 experiment as historical evidence.

`ETC/tests/test_tutorials.py` checks analytical PDE residuals, initial/boundary conditions, composite-bar interface flux, Taylor–Green periodicity and Navier–Stokes residuals, finite forward/backward passes in all modes, and the loader's exact normalization and missing-data failure. CLI and notebook execution results and hardware verification are reported by the repository's overall validation report. Merely passing a short execution test does not prove that every lesson has converged or that a weather model has predictive skill.
