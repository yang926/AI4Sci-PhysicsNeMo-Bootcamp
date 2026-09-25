# Lab 2 and 3 teaching workflow

The physical problems, FP32 models and previously checked training recipes are
unchanged. These additions restore ways to inspect and reuse their results;
they do not restore the retired Sym `Domain`/`Solver` API.

## Lab 2

- Live TensorBoard events record each actual Adam update's pre-update losses.
- `training_points.npz` saves the actual last initial-condition and interior
  batch without drawing new samples or changing the training random sequence.
- The notebook compares the physical trajectory, held-out position error and
  training points. Extrapolation beyond five seconds stays visibly separate.
- A native ParaView ZIP contains initial/interior input points, connected
  prediction/reference trajectories and a held-out validator dataset. Training
  input points have time as their coordinate; they are not measured trajectories.

## Lab 3

- Both fixed and parameterized training write live TensorBoard events, including
  interface-temperature and flux losses and L-BFGS closure counts. Step means an
  optimizer call, not a closure evaluation. Exact endpoint temperatures remain
  imposed by the model; the interface conditions remain learned.
- New checkpoints include the architecture, FP32 dtype, physical coefficients,
  parameterization mode and supported conductivity range.
- `--mode eval --checkpoint RUN/model.pt --d1 7.5 15 22.5 --output-dir NEW_RUN`
  loads that checkpoint in a fresh process. It never constructs an optimizer or
  saves replacement weights. Out-of-range values and incompatible checkpoints
  are rejected; old checkpoint files are not silently reinterpreted.
- `material_fields.npz` retains both sides of the interface. The notebook shows
  full temperature, a left-material enlargement and physical heat flux
  `q=-D*dT/dx`. It does not average away interface errors.
- ParaView exports each material separately for every requested conductivity.
  Both use the same temperature color scale; temperature and flux references
  are labeled evaluation references, not substituted predictions.

## TensorBoard access

The optional notebook cell starts TensorBoard on loopback with a dynamically
assigned port and links through the existing authenticated Jupyter proxy.
No raw port is published. The installer adds the proxy to the **server**
environment as well as TensorBoard to the separate course-kernel environment.
It installs only missing, pinned dependencies after checking that the managed
server is idle; compatible existing Jupyter dependencies are not replaced.
This first dependency/extension change needs the normal guarded restart. Later
source-only updates retain the no-restart behavior.

References: [Jupyter Server Proxy installation](https://jupyter-server-proxy.readthedocs.io/en/latest/install.html)
and [authenticated port routing](https://jupyter-server-proxy.readthedocs.io/en/latest/arbitrary-ports-hosts.html).

## Verification

Dedicated tests train small CPU models, compare saved/reloaded predictions and
fluxes bit-for-bit at new conductivity values, forbid optimizers during eval,
check physical metadata/dtype/range rejection, inspect VTK topology and fields,
and read the actual TensorBoard event records. A temporary real Jupyter server
test accesses TensorBoard's plugin API through the proxy and rejects a request
without the Jupyter token. These are workflow tests, not new convergence claims.

The full 5,000-step Lab 2 and 300-call Lab 3 numerical validation remains recorded
in `LABS_FP32_VALIDATION.md`. Existing user notebook outputs are not evidence of
the newly restored workflow; run the updated cells to generate the new files.
