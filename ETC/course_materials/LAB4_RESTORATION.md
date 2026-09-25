# Lab 4 restoration, 2026-09-25

## What was wrong

The student notebook defaulted to an analytical Taylor–Green test while the
course was meant to teach the original ERA5-derived initial-data example.
The same refactor reduced the network, changed the training loss and budget,
and replaced the original ParaView sequence with one CSV. Passing the analytic
test did not establish that the intended lesson was preserved.

## Source and restored settings

Reference: OpenHackathons commit `9cae27f8303268cdaf7528fe963ce12ba439377f`,
`tutorial/navier_stokes/Weather-forecasting-navier-stokes.ipynb` and its
`source_code/navier_stokes.py` / `conf/config.yaml`.

- `data_lat.npy` is unchanged: Git blob
  `f7b1849d2bb299f4b7d24b7c74fad6a56e0ed331`, shape `(3, 1440, 1440)`.
- Initial u/v/p normalization, periodic domain, density, viscosity and
  sixty-hour time scale are retained. No synthetic fallback is allowed.
- The original-data model uses six hidden layers of 256 units, SiLU and weight
  normalization. The input periodic feature map is implemented explicitly in
  the current PhysicsNeMo tensor API.
- Default training is 50,000 Adam updates, batch 2,048 for each constraint,
  initial learning rate 0.001, and exponential decay of 0.95 per 3,000 updates.
- Initial observation loss sums squared errors over samples and u/v/p.
  Interior loss is domain area times the sum of residual mean squares.
  This restores the source constraint scaling, rather than reusing the test
  fixture's weighted mean loss.

The legacy [PointwiseConstraint implementation](https://github.com/NVIDIA/physicsnemo-sym/blob/main/physicsnemo/sym/domain/constraint/continuous.py)
assigns unit area to observations and area-based weights to interior samples.
[PointwiseLossNorm](https://github.com/NVIDIA/physicsnemo-sym/blob/main/physicsnemo/sym/loss/loss.py)
sums those weighted squared errors. The architecture defaults are recorded in
[FullyConnectedArch](https://docs.nvidia.com/deeplearning/physicsnemo/physicsnemo-sym/_modules/physicsnemo/sym/models/fully_connected.html).

This is a current-API implementation of the original problem and settings,
not a bitwise replay of the old Solver. The direct loop resamples points at
each update instead of using the legacy fixed, shuffled interior dataset.
No pretrained checkpoint is bundled; the inspected upstream archive contains
only the initial data array.

## Student view and ParaView

The notebook shows the original initial field, its supplied demonstration
recording, and a separately labeled animation of the student's prediction.
All predicted frames use the same speed and arrow scales. The original input
at time zero is not labeled as ground truth for later times.

The output contains eleven frames at 0, 6, ..., 60 hours, VTI files on a
connected 2-D grid, a `flow.pvd` collection and a downloadable ZIP. The old
source generated ten inclusive frames, despite describing six-hour spacing.
The corrected eleven-frame output does not change the PDE's time interval.

Interactive output uses a 128 by 128 grid over the whole periodic square.
The old ParaView inferencer used a 1440 by 720 central latitude strip. This is
a display-resolution/extent difference, not a change to the initial data or
training domain. Coordinates and fields stay normalized; pressure is not
relabeled as Pa. The VTI `velocity` vector is `(u, v, 0)`.

## Validation scope

The restored 50,000-update run has not been completed in this verification.
A short successful execution is not proof of convergence or forecast accuracy.
No held-out future weather observations are provided by this lesson. Initial
spatial fit, PDE residuals and temporal changes must be read separately.

The former 64-by-3/3,000-update hybrid recipe was also run on the real array.
Its full-field input fit was poor. Increasing only its width to 256 did not
resolve that result within the same budget. Neither run is supplied as a
converged instructor result.

The student notebook and smoke suite exercise the original data. Analytic
convergence tests explicitly opt into the separate Taylor–Green fixture and
configuration. Their results cannot certify the original-data lesson.

The restored recipe was run for 3,000 updates with seed 42 on a local RTX 3080,
using the original array and FP32. It completed without warnings and produced
eleven finite 128-by-128 u/v/p frames, an FP32 checkpoint, input snapshots and
training diagnostics. This is not an L4 performance or full-budget result.
An independent evaluation of all 2,073,600 initial-data points measured u/v/p
RMSE of 0.10969 / 0.10131 / 0.04594 in normalized units. The velocity errors
are still about 85% / 89% of the input component standard deviations: the
short run does not recover the fine wind structure. It must not be distributed
as a converged model or used to claim forecast quality.

The entire student notebook also executed at a two-update CPU test budget,
including the embedded original recording, the input plot, the eleven-frame
player and the ZIP export. This checks execution and UI output, not training
quality. Saved learner notebook outputs were not used for this verification
and are not included in the published notebook changes.

The exporter was independently read with VTK 9.5.2 for all eleven frames:
each new frame has 16,384 points and 16,129 connected cells. Point ordering,
cell connectivity, coordinates and u/v/p/velocity values matched the source
arrays. ParaView GUI playback on the presentation Mac
still needs rehearsal; its optional display script was syntax/API checked,
not GUI-tested here.
