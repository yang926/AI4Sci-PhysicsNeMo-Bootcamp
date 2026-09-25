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
| Operators 1–3 | Dataset split/normalization, specified model construction, PINO physics | Existing `build_datasets`, `build_model`, and PINO `ReactionDiffusionPDE` contract retained |

The tensor-loop adaptation supplies framework assembly, including model input
wiring, geometry sampling, zero residual targets and constraint batching.
Students return mathematical expressions and bounded geometry data instead of
constructing legacy `Domain`, `Constraint`, `Node` and `Solver` objects. Those
API construction blanks are not represented as arbitrary executable submissions.

## Local run versus judge

Student mode uses the student's functions for training. Reference mode supplies
the complete instructor setup and cannot be submitted. Held-out diagnostics use
the stated reference equations, conditions and geometry in both modes. Climate's
`student_solution` is checked, but is never accepted as the evaluator's truth.

The notebook collects every required function from the saved file. The server
does not import submitted files: it interprets a small bounded language, checks
each setup component, and installs only those interpreted functions into trusted
lesson code. Missing functions are an explicit format error, not a request to
fill in instructor defaults. Old PDE-only submissions must be completed again
under the new contract.

Before sending code, the notebook checks the server's advertised submission
contract version. An older judge cannot silently ignore the added setup
functions: the notebook stops and asks the instructor to update both sides.

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
  initial state, without the added ramp. The initial inlet corner is discontinuous.
- Climate 2 uses the original uncoupled baseline, `gamma0=0`. Nonzero coupling
  remains a separately labelled experiment, with an independent coupled reference.

These limitations are taught explicitly rather than hidden by changing the
problem. Correct implementation and converged numerical results are separate
questions; consult the numerical rehearsal record before choosing class timings.
