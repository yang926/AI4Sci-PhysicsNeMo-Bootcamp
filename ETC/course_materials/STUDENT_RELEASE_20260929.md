# Student-only Challenge revision

Based on `214301e`, preserving its EDIT HERE guidance and all eleven Levels.

## Changes

- Removed executable instructor answers and the reference-mode switch from
  Challenges 1–4, shared operator helpers, answer-bearing notebook examples,
  and instructor-only test fixtures. These were not moved to another student
  directory. Existing private instructor material is maintained separately.
- Set all eight PINN Levels to 5,000 updates. Operator Levels remain at 3,000.
  Learning rate, model architecture, sampling budgets, data and physical problem
  settings are unchanged. Labs are not changed by this revision.
- Notebook subprocesses use unbuffered Python and PINN progress logs flush,
  so progress does not wait for the process to finish.
- Local saved-code checks report format and unfinished placeholders only.
  They do not award points or verify mathematical correctness. The v3 submitted
  function names and payload remain compatible with the separate event judge.
- Held-out PINN residuals now use the learner's implementation and measure
  self-consistency. Climate comparisons use the learner's derived solution,
  explicitly marked as independently unchecked. Wave 1's independent analytical
  comparison and Fluid 1's supplied OpenFOAM comparison remain available.

## Judge boundary

The deployed judge was not updated or stopped. It must retain its private,
pinned instructor/scoring bundle. Do not deploy the answer-free student checkout
as a replacement for that bundle. Embedded correctness-evaluator entry points
fail explicitly without the instructor implementation; they do not issue zero
scores as a substitute for unavailable checks.

The local notebook nickname and submission controls still send saved code to
the existing judge. No automatic submission was added.

## Verification scope

All eleven Levels passed one-update CPU execution checks using private,
in-memory completed learner functions from the base revision. Each unfilled
template stopped, the removed reference CLI option was rejected, and completed
fixtures exported metrics, model, loss history, predictions and a preview.
Operator smoke data used 4/2/2 samples on the original 64-by-64 grid. Those are
execution tests, not L4 timings or convergence tests of the 5,000-update preset.

The focused 28-file regression suite passed 431 tests (7 skipped; 4 subtests).
The notebook/artifact validation helper suite passed 139 tests. Full-course test
collection succeeded; this does not claim the entire test suite was executed.

Static course checks reported one pre-existing English-only issue in the base
revision: the closed-Challenge message in `ETC/runtime/judge_client.py` includes
a Korean parenthetical. It was preserved rather than overwritten during this
task. Notebook compilation, local links and asset checks had no other failures.

## Limits

The shorter defaults are a requested class-time budget, not a guarantee that
every learner implementation converges. Previously published long-run timing
and accuracy tables are historical and do not certify this preset.

This record describes the 2026-09-29 student-only revision. A subsequent
2026-10-01 cleanup replaced the public branch history with release snapshots.
The base revision above is a historical identifier, not a commit included in
the new public history. Previously downloaded copies and cached commits may
still exist. Current student checkout files do not contain those answer fixtures.
