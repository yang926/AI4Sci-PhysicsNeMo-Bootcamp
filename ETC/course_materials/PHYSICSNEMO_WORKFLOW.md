# From course helpers to PhysicsNeMo

## Lab 4: pretrained weather inference

The active [weather Lab](../../01_labs/04_weather_forecasting/Lab_4_Weather_Forecasting.ipynb)
uses `physicsnemo.models.afno.AFNO.from_checkpoint` and `torch.inference_mode()`.
It does not construct a `PhysicsInformer`, evaluate a PDE loss, or update weights.
The official 26-channel initial state is normalized with the checkpoint's fixed
statistics, advanced by six hours, converted back to physical units, and fed
back for the next step. Eight steps produce a 48-hour forecast. Future ERA5 is
loaded only by the evaluator, which also scores persistence on the same grid.
The sections below describe the PINN and Challenge training workflows; they do
not describe the active Lab 4 inference program. The former PINN flow code is
preserved under `ETC/reference_labs/04_navier_stokes`.

[Introduction](../../01_Introduction.ipynb) · [Lab 1](../../01_labs/01_pinn/Lab_1_PINN_Fundamentals.ipynb) · [Wave challenge](../../02_challenges/01_wave/Challenge_1_Wave_Dynamics.ipynb) · [Neural operators](../../02_challenges/04_neural_operators/Challenge_4_Neural_Operators.ipynb)

This course uses PhysicsNeMo 2.2.2. A lesson combines a model, an equation, sampled inputs and a PyTorch training loop. The short helpers in `ETC/runtime` connect these pieces; reading through them shows which work belongs to the course and which belongs to the libraries. This guide explains the existing workflow. The lesson's source and configuration remain the authority for its physics, training settings and assessment.

## Which names belong to which layer?

| Name | Where it comes from | What it does |
|---|---|---|
| `symbols`, `Function`, `.diff()` | SymPy | Describe coordinates, unknown fields and derivatives symbolically. They do not evaluate a neural network. |
| `PDE` | `physicsnemo.sym.eq.pde` | Base class for an equation definition. A lesson subclass stores named residual expressions in `self.equations`. |
| `FullyConnected` | `physicsnemo.models.mlp` | PyTorch-compatible network that maps an input tensor to predicted field values. |
| `PhysicsInformer` | `physicsnemo.sym.eq.phy_informer` | Evaluates requested residuals using field tensors and a selected spatial differentiation method. |
| `create_model`, `create_informer`, `residuals` | Course file [ETC/runtime/pinn.py](../runtime/pinn.py) | Construct the challenge model and informer, evaluate fields and prepare the tensors the informer needs. These are not public PhysicsNeMo APIs. |
| `torch.autograd.grad`, `backward`, `torch.optim` | PyTorch | Differentiate tensor computations and update trainable parameters. |

For example, `create_model(3, 1, config, device)` wraps the library call below. This is a source excerpt, using the lesson's `config` and `device`, rather than a standalone script:

```python
from physicsnemo.models.mlp import FullyConnected

model = FullyConnected(
    in_features=3,
    out_features=1,
    layer_size=config["model"]["width"],
    num_layers=config["model"]["layers"],
    activation_fn="tanh",
).to(device)
```

The three inputs are `x`, `y`, `t` in the Wave challenge; the one output is `u`. The network does not know the wave equation from this constructor. The loss will connect its prediction to that equation.

The challenge helper `create_informer(pde, device)` checks the equation mapping and discovers any first or second pure time derivatives it needs to declare as supplied inputs. It then delegates to `informer` in [ETC/runtime/labs.py](../runtime/labs.py). That helper configures `required_outputs=list(pde.equations)`, `equations=pde`, `grad_method="autodiff"` and `device`. For spatial-only equations it constructs `PhysicsInformer` directly. For declared time derivatives it uses the course `_SuppliedDerivativeInformer` subclass, which checks that those tensors are supplied and passes the residual computation to PhysicsInformer. The adapter does not train the model or solve the PDE.

## Follow one Wave Level 1 training step

Open [wave_l1.py](../../02_challenges/01_wave/wave_l1.py) beside [ETC/runtime/pinn.py](../runtime/pinn.py). The names below are the actual functions in those files. The excerpts are for reading; complete and run the exercise through the challenge notebook.

1. **Write the symbolic equation.** `WaveEquation2D(PDE)` creates SymPy symbols and fields, then calls your `student_equations` and `student_speed`. Implement the displayed wave PDE under the required `wave` key, preserving the supplied `c` argument. `.diff()` constructs symbolic derivatives before any sampled coordinates or network values exist. No completed exercise implementation is supplied.

2. **Construct the model and evaluator.** In `main`, `WaveEquation2D()` builds your physics, `create_informer` prepares its numerical residual evaluation, and `create_model(3, 1, config, args.device)` constructs the network. Fixed held-out residual checks use the same learner equations; they do not certify the answer. The separate judge checks the stated mathematical contract.

3. **Sample inputs and predict a field.** `loss_terms` samples interior positions `xy` with shape `[N, 2]` and times with shape `[N, 1]`, then calls `residuals(model, informer, xy, time, FIELD_NAMES)`. The helper makes fresh coordinate and time tensors with gradient tracking enabled. `evaluate_fields` concatenates them to `[N, 3]`, calls the model and names its `[N, 1]` output `"u"`. `FIELD_NAMES = ["u"]` connects this tensor to the symbolic field's name.

4. **Evaluate the equation on that prediction.** `residuals` passes the predicted `u`, spatial `coordinates`, `x`, `y`, `t`, and explicitly computed `u__t` and `u__t__t` tensors to `informer.forward(inputs)`. PhysicsInformer obtains the required spatial derivatives by autodiff and evaluates the named expression. The returned dictionary contains `"wave"`: the pointwise tensor

   $$r_\theta(x,y,t)=u_{\theta,tt}-c^2(u_{\theta,xx}+u_{\theta,yy}).$$

5. **Turn residuals and conditions into a scalar loss.** `loss_terms` computes `pde["wave"].square().mean()` and also computes the initial-displacement, initial-velocity and boundary losses. Level 1 uses both initial targets `sin(x) * sin(y)` and zero displacement on the square's edges. `record_step` checks and sums these four scalar losses, returning `total`. The analytic solution is used for comparison; it is not a training target in this loop.

6. **Update the network.** The relevant sequence in `main` is:

   ```python
   optimizer.zero_grad(set_to_none=True)
   losses = loss_terms(model, informer, config, args.device, exercise)
   total = record_step(step, losses, history)
   total.backward()
   optimizer.step()
   ```

   `total.backward()` computes gradients with respect to the model parameters. Adam then updates those parameters. Subsequent steps repeat the prediction and loss calculation with newly sampled training points.

The full path is `student_equations` → `WaveEquation2D.equations` → `create_informer` → `residuals` / `informer.forward` → `loss_terms` → `total.backward()` → `optimizer.step()`. The network enters that path through `create_model` and `evaluate_fields`.

## Coordinate derivatives and parameter gradients

There are two differentiation tasks. To evaluate a PDE, the program differentiates the prediction with respect to coordinates, such as `u_xx` or `u_tt`. To train the network, it differentiates the scalar loss with respect to the network's weights and biases. The second calculation must remain connected through the first.

In the PINN challenges, PhysicsInformer handles `x` and `y` spatial derivatives with `grad_method="autodiff"`. The course `gradient` helper calls `torch.autograd.grad(..., create_graph=True)` to calculate time derivatives separately. `residuals` computes a first and second time derivative for each predicted field when time is present, and supplies them under names such as `u__t` and `u__t__t`. `create_informer` declares only the time derivatives actually used by the equation. Concatenating time with `x` and `y` for the network input does not make time an automatically handled spatial axis in the informer. This follows the [official PhysicsInformer guidance on temporal derivatives](https://docs.nvidia.com/physicsnemo/latest/user-guide/physics_addition.html#adding-pde-losses).

`create_graph=True` preserves a differentiable graph for those derivative calculations, so `total.backward()` can propagate the residual loss back to the model. Enabling coordinate gradients does not ask Adam to move the sample points: the optimizer was constructed from `model.parameters()`.

An equation residual is also different from solution error. A small value of `u_tt - c²(u_xx + u_yy)` says the predicted field approximately satisfies that equation at the checked points. It does not say the prediction is close to the intended solution everywhere. For example, `u = 0` has zero residual for this homogeneous wave equation but violates Level 1's nonzero initial displacement and velocity. Check the conditions and, when available, errors against a known solution as well. The fixed held-out checks in the challenge are local feedback; they do not replace the notebook's submission and grading workflow.

## Why the training loop is explicit

Writing `zero_grad`, a scalar loss, `backward` and `step` is part of the PhysicsNeMo workflow. The [official physics-guided tutorial](https://docs.nvidia.com/physicsnemo/latest/user-guide/physics_addition.html) shows this composition, and the [2.2.2 migration guide](https://github.com/NVIDIA/physicsnemo/blob/v2.2.2/v2.0-MIGRATION-GUIDE.md) points from older `Solver` / `Domain` workflows to explicit PyTorch examples. PhysicsNeMo provides reusable models and residual tools; the training script chooses the samples, conditions, loss weights, optimizer and evaluation procedure.

The Labs illustrate the same division with different code. Their helpers `mlp(nin, nout, cfg)` and `informer(pde, device, supplied_derivatives=...)` live in [ETC/runtime/labs.py](../runtime/labs.py), and their configuration uses keys such as `layer_size` and `num_layers`. The PINN challenge helper `create_model(inputs, outputs, config, device)` instead reads `config["model"]["width"]` and `config["model"]["layers"]`. These signatures and configuration layouts are not interchangeable. Lab 1 also wraps its networks in `BasicPINN`, while its inverse mode uses the lesson's `FourierMLP`; inspect [pinn_basics.py](../../01_labs/01_pinn/source_code/pinn_basics.py) for the selected mode.

Follow each lesson's existing optimizer recipe. Lab 1's L-BFGS uses `optimizer.step(closure)`, and that closure recomputes the loss and calls `backward` on fixed training points. The Wave excerpt above shows Adam. Both use PyTorch to optimize the model; the different loop shapes reflect the chosen optimizer.

Climate uses [ETC/runtime/climate.py](../runtime/climate.py) for an Adam warm-up followed by L-BFGS. Its default 5,000 optimizer calls consist of 1,500 Adam updates with a cosine learning-rate schedule from `0.001` to `0.00001`, then 3,500 L-BFGS calls. Adam uses fresh samples. L-BFGS reuses 4,096 Sobol interior points and initial/boundary grids including endpoints (2,116 points each), so each closure evaluates the same objective. A closure can run several times within one optimizer call; `STEPS` is not the number of gradient evaluations. A two-call execution check uses Adam only.

The Climate network remains a three-layer, 64-unit FP32 MLP. `ClimateModel.forward` scales the physical inputs to `[-1,1]` before passing them to `FullyConnected`. Because this scaling stays in the autograd graph, spatial and time derivatives retain their physical-coordinate factors. Both optimizers minimize the original PDE, initial-condition and boundary-condition losses; `student_solution` supplies comparison only.

The same helper file's `temperature_errors` checks eight times, including the early transient at `t=0.25`, `0.5` and `1`. Each row reports the learner expression's maximum absolute temperature alongside RMSE. RMSE / initial RMS uses the initial-temperature norm as a fixed scale; ordinary relative L2 still uses the reference norm at each evaluation time. These comparisons help interpret diffusion toward zero without changing the reported absolute errors.

## How Challenge 4 changes the inputs and derivatives

In [Challenge 4](../../02_challenges/04_neural_operators/Challenge_4_Neural_Operators.ipynb), the input is a whole forcing field and the output is a whole solution field, each with shape `[batch, 1, n, n]`. Level 1 builds `physicsnemo.models.fno.FNO`; Level 2 builds `physicsnemo.models.afno.AFNO`; Level 3 uses FNO with an additional physics loss, which makes its training physics-informed neural operator training (PINO). FNO and AFNO name architectures; PINO names the use of physics in the training objective.

Read [fno_physicsnemo_l3.py](../../02_challenges/04_neural_operators/fno_physicsnemo_l3.py) for the symbolic `ReactionDiffusionPDE` and its `build_physics` function. For the periodic equation `u - Δu = f` on `[0, 1)²`, it configures PhysicsInformer with `grad_method="spectral"` and `bounds=[1.0, 1.0]`. Here spatial derivatives come from the periodic grid's Fourier representation. The course `BatchedPhysicsInformer` adapter applies the informer separately to each sample and preserves each sample's autograd graph. This path does not use the coordinate-based PINN `residuals` helper.

The shared loop in [operator_training.py](../../02_challenges/04_neural_operators/operator_training.py) makes the scales explicit:

| Quantity | Scale and use |
|---|---|
| Model input and output | Normalized forcing and predicted solution. Means and standard deviations come from the training split. |
| `data_loss` | Mean squared difference between normalized prediction and normalized target. |
| PDE inputs | Prediction and forcing are converted back to their physical scale before evaluating `u - Δu - f`. |
| `physics_loss` | Mean square of the physical residual divided by the training forcing standard deviation: `(residual / stats["f_std"]).square().mean()`. |
| Total loss | `data_loss + physics_weight * physics_loss`, differentiated by `loss.backward()`. |

Applying the original PDE directly to independently normalized `u` and `f` would change its equation unless the scaling factors and offsets were accounted for. The course restores physical-scale fields first, then scales the residual loss. Spectral differentiation remains differentiable with respect to the FNO prediction, so the physics term can update the same model parameters as the data term.

When reading another lesson, locate its model constructor, `PDE.equations`, named tensors passed to `PhysicsInformer.forward`, scalar loss and optimizer update. Those five places explain the library workflow even when a course helper gives one of them a shorter name.
