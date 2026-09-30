# Lab 4: short training and instructor-model inference

## Classroom sequence

1. Inspect the original input and equations. The historical ParaView recording is
   explicitly separate from the current model's output.
2. Part A trains a new student model for 3,000 updates. Compare its initial fit
   and per-time PDE errors. Do not describe the short run as converged.
3. Part B loads a reviewed instructor checkpoint, computes the eleven time
   frames, and displays them. It does not run an optimizer. Setup plus Part B
   can be run without waiting for Part A.
4. Export the desired result to ParaView. Each model has a separate output
   directory and labeled download. Neither result replaces the other.

The problem, original array, normalization, equations, FP32 training and
3,000-update student budget are unchanged. The instructor checkpoint can use
the original representation rather than the student's multiscale features.
Its metadata and review note must disclose that difference.

## Current availability

The [model manifest](../../ETC/reference_labs/04_navier_stokes/source_code/conf/instructor_model.json)
is `pending_review`. No instructor checkpoint has been approved or released.
The 50,000-update run is an investigation, not a promised successful result.
While approval is pending, Part B prints its status and skips inference; Part A
and the original recording remain usable. Do not announce that the instructor
model is ready until the review below is complete.

## Review before release

The reviewer must inspect and retain the following evidence:

- Complete run metadata, source revision, data SHA-256, recipe, actual update
  count, FP32 checkpoint, and finite saved predictions.
- Initial u/v/p error and images on the same spatial points. State whether
  those points were withheld from training. Matching initial data is not a
  future-forecast validation.
- Continuity and both momentum residuals at early and late times on an
  independent spatial grid, not just a final aggregate training loss.
- Fixed-scale flow images, mean momentum, and total/fluctuating kinetic energy
  throughout the retained 60-hour interval. Neither a small late PDE residual
  nor preserved energy alone establishes correct dynamics.
- A written acceptance decision for the classroom demonstration, including
  remaining limitations. No future weather observations are provided, so the
  model must not be called a validated weather forecast.

If these checks do not support the intended demonstration, keep the manifest
pending. Do not auto-approve after 50,000 updates, publish a failed checkpoint,
rescale the animation to hide weakening, or replace the input with a synthetic
field. Changes to the initial data or physical problem require a separate
decision.

## Publish only after approval

Keep large checkpoint binaries out of Git. Publish the reviewed `model.pt` as
an immutable release asset of `yang926/AI4Sci-PhysicsNeMo-Bootcamp`, together
with its review evidence. Never overwrite the asset for an existing release.
Then update the manifest with:

- `status`: `approved` only after the review decision;
- `checkpoint`: a path under `instructor_models/` in the Lab directory;
- `download_url`: the HTTPS URL of that course GitHub release asset;
- `sha256`: the exact checkpoint checksum;
- `source_data_sha256`: the original array checksum;
- `training_steps`: the recorded completed update count;
- `review_note`: reviewer, review date, evidence location and limitations.

Part B checks the manifest and source-data checksum, downloads an approved
asset only when the local cache is absent, checks its size and checksum, and
loads tensor weights with `weights_only=True`. A mismatched file fails closed;
it is not silently replaced. This identifies the reviewed file; a checksum is
not a numerical accuracy test or a substitute for trusted release ownership.

The manifest and code update must reach existing student workspaces through
the normal course update workflow. Editing GitHub does not live-update an
already running notebook kernel. Reopen the updated notebook and rerun setup.
New Launchable checkouts receive the manifest with the course code. No approved
asset is fetched or included while the manifest is pending.
