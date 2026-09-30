# Lab 4: six-hour flow demonstration

## What students run

The notebook selects `--recipe six_hour`: 3,000 Adam updates in FP32, followed
by playback and ParaView export of 11 frames from 0 to 6 hours. This is a
**simplified, periodic incompressible flow derived from the supplied weather
input**, not a weather forecast. It is not the unchanged upstream experiment.

The original `data_lat.npy` is unchanged. `--recipe efficient` still runs the
raw-input 60-hour comparison; `--recipe upstream` retains the original
representation and its 50,000-update default. The command-line default remains
`efficient`; the teaching notebook explicitly requests `six_hour`.

## Why shortening the interval was not enough

The supplied wind, interpreted on the lesson's planar periodic grid, has a
substantial divergent component, while the equations require divergence-free
velocity. Fitting the raw input and satisfying that equation can compete.
Previous raw-input runs also showed excessive weakening that the very small
prescribed viscosity did not explain. More training alone did not resolve it.

For this demonstration we make the change of initial conditions explicit:

1. Retain the divergence-free part of the original wind using a periodic
   Helmholtz decomposition. This is a velocity-field decomposition, not a
   geographic map projection. It retains **66.91% of the original input's
   kinetic energy**; it is a material change, not a rounding correction.
2. Recalculate compatible initial pressure from the pressure Poisson equation
   using the projected velocity. The original pressure mean only supplies an
   arbitrary pressure gauge. This is not the original meteorological pressure.
3. Sample the same equations on normalized time `[0, 0.1]`. One normalized
   time unit still means 60 hours, so the endpoint is 6 hours. The network uses
   the feature `t / 0.1`; derivatives are taken with respect to the original
   time variable, including the chain rule.
4. Add the spatially integrated kinetic-energy and mean-momentum identities
   of those same equations to the loss. These penalize violations, rather than
   imposing a chosen future energy or copying a future solution.

The notebook displays the original and derived initial fields separately.
Outputs use `data_kind=derived_periodic_incompressible_initial_conditions`.
`initial_fields` contains the derived target and `raw_initial_fields` contains
the original input preview. Transformation diagnostics and the source hash are
saved in `metrics.json`.

## Training settings

- Six hidden layers of width 256, SiLU, weight normalization, periodic spatial
  feature bands 1, 2, 4, 8; no pretrained model.
- Adam, learning rate 0.001 with the existing 0.95-per-3,000-update decay.
- 2,048 initial samples and 2,048 PDE samples per update. Initial loss uses
  the original sample sum; the PDE loss uses the original spatial area.
- Conservation terms have weight 1 and use a 32-by-32 quadrature grid at one
  sampled time per update. The energy balance includes the unchanged viscosity.
- The 32-by-32 initial evaluation grid is excluded from fitting and output
  scaling. PDE evaluation uses a separate 64-by-64 grid at six times.
- No positive-time solution values are supplied during training. No L-BFGS
  refinement is used.

The before/after `objective` reports initial-data and local-PDE errors only;
it does not include the training loop's integrated conservation penalties.
Those components are logged separately in `loss.csv`.

## Measured comparison

Local experiments used an NVIDIA GeForce RTX 3080, seed 42, FP32. They do not
establish the timing on an L4 or robustness across seeds. The selected
experimental training loop took **278.2 seconds** (about 4 minutes 38 seconds).
Installation, initial-field transformation and media export are not included.

The integrated student CLI was then replayed from scratch with the same seed
and complete 3,000-update settings: **263.5 seconds of training** on the same
RTX 3080. It reproduced the experimental initial/PDE errors and 0.955794 energy
retention. Maximum absolute difference across all saved u/v/p predictions was
`7.57e-6` (including a small difference in saved-grid floating-point coordinates).
The derived-result loader, fixed-scale HTML playback and eleven-frame ParaView
ZIP were also exercised on that completed run.

| Six-hour candidate | Initial u/v normalized RMSE | PDE RMSE | E(6 h) / predicted E(0) | Decision |
| --- | ---: | ---: | ---: | --- |
| Derived input, original time feature | 0.261 / 0.294 | 0.300 | 0.923 | Little temporal evolution |
| Derived input, `t / 0.1` feature | 0.259 / 0.277 | 0.278 | 0.853 | Better evolution, excessive energy loss |
| Previous row + 500 L-BFGS calls | 0.301 / 0.292 | 1.032 | Not selected | Rejected: evaluation residual worsened |
| Time conditioning + conservation, Adam 3,000 | 0.256 / 0.275 | 0.268 | 0.956 | Selected for the qualitative demonstration |

Energy retention is relative to the **model's own predicted initial state**,
not the original input or the derived training target. Energy retention alone
does not establish an accurate solution.

### Independent equation-based comparison

A separate CPU pseudo-spectral vorticity solver evolved the same derived
initial-value problem, with periodic boundaries, mean-flow transport,
3/2-dealiased products and RK4 timestepping. Analytic solver regression tests,
timestep refinement and 128/256/512-grid comparisons were performed. Its
outputs were used only for evaluation, never as PINN training targets.

For the selected model at 6 hours, against the 512-grid numerical reference:

| Scope | Relative velocity L2 error | Centered velocity correlation |
| --- | ---: | ---: |
| Full saved 128-by-128 PINN grid | 0.367 | 0.930 |
| Large scales, Fourier modes `abs(kx), abs(ky) < 16` | 0.171 | 0.986 |

The large-scale numerical references differ by 0.383% between grids 256 and
512 at 6 hours. Those modes contain about 78.5% of the 512-grid reference's
energy. Fine-scale/vorticity agreement is weaker: the full shared-grid velocity
difference is about 9.9%, so the 512-grid result is **not a converged truth at
all scales**.

Across the saved times, squared error in the model's large-scale *change from
its own initial prediction* is 50.9% lower than predicting no change. At the
full saved grid this improvement is 16.8%. This checks that the model has not
merely frozen its initial state. It is not operational forecast skill.

The remaining initial-fit and fine-scale errors prevent a claim of an accurate
full-resolution solution. The result supports a **qualitative six-hour
large-scale flow demonstration**, with a visible evolving pattern and much less
spurious weakening. A correlation of 0.986 must not be called "98.6% accuracy."

## Reproduce the classroom run

From the repository root:

```bash
python ETC/reference_labs/04_navier_stokes/source_code/navier_stokes.py \
  --recipe six_hour --device cuda --steps 3000 --seed 42 \
  --output-dir outputs/lab4-six-hour-seed42
```

Choose a new output directory for each run. The notebook uses the same command
and verifies the derived-input label and six-hour time axis before playback or
ParaView export. The original data SHA-256 is:

`45d2226d51f054d64a9793e31b117bd769a412cae6aa0cadaf109dc15934df01`

Local experiment records are under
the private validation archive `lab4-short6h-20260929`:
`candidate-reference-review.json`,
`reference-band-convergence.json`, and
`adam-time-scaled-conservation-3000/metrics.json`. They are validation records,
not dependencies downloaded by the student notebook. The numerical reference
and its large data files are not required for classroom training.

## Suggested explanation

> We start with the supplied wind field, extract its incompressible component
> for this simplified periodic model, and learn its evolution over six hours.
> We compare the initial fit, equation errors and changing flow pattern. This
> demonstrates a physics-informed flow model; it does not validate a weather
> forecast. The original input and the changes made to it are shown separately.
