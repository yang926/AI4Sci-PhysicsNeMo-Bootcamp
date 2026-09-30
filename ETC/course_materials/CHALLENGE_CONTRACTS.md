# Challenge task and submission mapping

Reference: OpenHackathons source commit `9cae27f8303268cdaf7528fe963ce12ba439377f`.
This table records what students implement, where that answer is used, and what
the v3 completion pilot checks. It is not a claim that every old API blank was
copied verbatim.

| Level | Original mathematical task | Current learner function and use |
|---|---|---|
| Wave 1 | Wave residual, constant speed, displacement/velocity initial data, edge condition | `student_equations`, `student_speed`, `student_conditions`; speed enters the PDE, conditions enter training losses |
| Wave 2 | Variable-speed expression, PDE, changed initial velocity, edge condition | Same functions, with a separately checked spatial speed expression |
| Wave 3 | Wave residual, Gaussian initial data, full Robin residual on a disk | `student_conditions` returns `initial_u`, `initial_ut`, `boundary`; the boundary expression includes learner-written normal derivatives |
| Fluid 1 | Steady incompressible flow, inlet/outlet/wall conditions | `student_equations`, `student_conditions`, `student_geometry`; all condition targets and the original single chip reach training |
| Fluid 2 | Three rectangular cutouts and channel subtraction, with steady constraints | `student_geometry` returns three `(xmin, xmax, top)` cutouts; interior, wall, SDF and flux sampling all consume them |
| Fluid 3 | Time-dependent equations and rest initial values, with inlet/outlet/wall constraints | `student_equations` includes time derivatives; conditions include all three initial fields; geometry returns to one chip |
| Climate 1 | ADR residual, six coefficients, initial/boundary values, exact baseline temperature | `student_equations`, `student_parameters`, `student_conditions`, `student_solution` |
| Climate 2 | Both residuals, nine coefficients, both fields' conditions and exact baseline solutions | Same four functions; exchange signs are checked symbolically even though the original baseline has `gamma0=0` |
| Operators 1–3 | Wrap the supplied train/validation/test tensors, construct the specified model, implement PINO physics | `build_datasets` preserves tensors unchanged; the trainer owns normalization; `build_model` passes every `model_config` key; PINO uses `ReactionDiffusionPDE` |

The tensor-loop adaptation supplies framework assembly, including model input
wiring, geometry sampling, zero residual targets and constraint batching.
Students return mathematical expressions and bounded geometry data instead of
constructing legacy `Domain`, `Constraint`, `Node` and `Solver` objects. Those
API construction blanks are not represented as arbitrary executable submissions.

## Local run versus judge

All local Challenge runs use the learner's saved functions. The student distribution has no completed answer mode. Local preflight checks syntax and completeness only; the judge checks correctness. PINN held-out residuals use the learner's equations, conditions and geometry. Climate comparisons use `student_solution` and explicitly record that the expression was not independently checked; the judge never accepts it as trusted scoring truth.

Use supplied parameter symbols in PDEs, including terms with zero default
coefficients; do not replace `c`, `nu`, `rho` or Climate `params[...]` with
numbers. Preserve each residual's displayed sign and exact dictionary keys.
This task checks the stated symbolic residual, not all algebraically equivalent
zero sets: multiplying the whole equation by another constant changes the
submitted residual. A wrong component loses that component's points, not
automatically every point in the Level. Wrong keys can invalidate its format.

For Operators, the trainer computes statistics from the training split only;
`build_datasets` wraps the supplied tensors unchanged. Use every `model_config`
key, even when a value equals a library default. Model/data factories allow local
assignments, a final return, and bounded comprehensions over the supplied short
sequences. `for`/`while` statements and `else` branches are not supported. PINO
assigns `self.equations` once as a dictionary, not by item mutation.

The notebook collects every required function from the saved file. The server
does not import submitted files: it interprets a small bounded language, checks
each setup component, and installs only those interpreted functions into trusted
lesson code. Missing functions are an explicit format error, not a request to
fill in instructor defaults. Old PDE-only submissions must be completed again
under the new contract.

Before sending code, the notebook checks the server's advertised submission
contract version. An older judge cannot silently ignore the added setup
functions: the notebook stops and asks the instructor to update both sides.

All Levels start selected. Ctrl/Cmd-click to remove unfinished Levels. Select
only finished Levels: a selected unfinished function prevents sending the whole
attempt. Submit all completed Levels together. Omitted Levels score zero; Levels
from different attempts are never combined.

Each Level has 100 pilot completion points split equally among its checks. A
wrong but well-formed component earns no credit for that check; other correct
components retain partial credit. An unfinished, missing or malformed function
invalidates that Level. Training feedback runs only after all components pass.
Numerical errors earn no extra points; fully correct submissions tie. This is a
completion workshop, not a model/optimizer tuning competition. No hidden
submission-time tiebreaker is used.

Use a fresh judge state for `bootcamp-task-completion-pilot-v3`. Old databases and
their results are not rewritten or silently mixed with v3 scores.

## Original conditions restored

- Wave 3 uses the original sum of two Gaussians, without the added envelope.
  Its initial Robin compatibility is not exact at the circular boundary.
- Fluid 3 uses the original nonzero parabolic inlet and unit flux with a rest
  initial state, without the added ramp. Incompressibility requires `Q=1` through
  every fluid cross-section for `t>0`, while the rest state has `Q=0`. This startup
  incompatibility is channel-wide, not confined to the inlet corner; it does not
  assert a jump in every velocity component at every point.
- Climate 2 uses the original uncoupled baseline, `gamma0=0`. Nonzero coupling
  remains a separately labelled experiment; the baseline learner-solution comparison is disabled for nonzero coupling.

These limitations are taught explicitly rather than hidden by changing the
problem. Correct implementation and converged numerical results are separate
questions; consult the numerical rehearsal record before choosing class timings.
