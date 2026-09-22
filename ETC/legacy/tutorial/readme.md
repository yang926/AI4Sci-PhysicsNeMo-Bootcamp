# Tutorial learning sequence

[Full course guide](../../course_materials/README.md) · [Start Here](../../../Start_Here.ipynb) · [Environment check](../../../00_Setup.ipynb)

The sequence below follows the introduction and Training Labs in the original `Start_Here.ipynb`. Each notebook also links to the previous and next lesson at its beginning and end.

| Order | Notebook | Activities |
|---|---|---|
| 1 | [PhysicsNeMo introduction](../../../01_Introduction.ipynb) | Read the concepts and components |
| 2 | [PINN fundamentals](../../../01_labs/01_pinn/Lab_1_PINN_Fundamentals.ipynb) | Run forward, parameterized, and inverse PINNs |
| 3 | [Projectile](../../../01_labs/02_projectile/Lab_2_Projectile_Motion.ipynb) | Train with the current API, compare in-domain and extrapolated trajectories, export ParaView CSV |
| 4 | [Diffusion](../../../01_labs/03_heat_conduction/Lab_3_Heat_Conduction.ipynb) | Fixed-conductivity diffusion, parameterized diffusion, and result comparison |
| 5 | [Navier–Stokes](../../../01_labs/04_navier_stokes/Lab_4_Navier_Stokes.ipynb) | Check synthetic execution, select original initial data, inspect time slices |

Continue with [Wave](../../../02_challenges/01_wave/Challenge_1_Wave_Dynamics.ipynb) → [Fluid](../../../02_challenges/02_fluid/Challenge_2_Fluid_Flow.ipynb) → [Climate](../../../02_challenges/03_climate/Challenge_3_Climate_Modeling.ipynb) → [Neural Operators](../../../02_challenges/04_neural_operators/Challenge_4_Neural_Operators.ipynb). Complete the levels of each challenge in order.

## Current environment and results

These tutorials use **nvidia-physicsnemo==2.2.2**, `FullyConnected`, `PDE`, `PhysicsInformer`, and explicit PyTorch training loops. See [MIGRATION.md](MIGRATION.md) for the changes.

Notebooks default to CPU and 200 steps. Override these with `AI4SCI_DEVICE` and `AI4SCI_STEPS`. A short run checks execution; it does not certify convergence. Each execution creates a new run directory containing `model.pt`, `predictions.npz`, `preview.png`, `metrics.json`, and `loss.csv`. Selecting an existing output directory fails instead of overwriting it.

In `metrics.json`, `initial_loss` and `final_loss` compare the objective at the same fixed evaluation coordinates before training and after the final optimizer update. Per-minibatch training losses are recorded separately in `loss.csv`. Where an analytical solution is available, also compare solution errors.

The Navier–Stokes notebook initially uses the explicitly labeled Taylor–Green synthetic fixture. Set `SMOKE_DATA=False` to use the original `data_lat.npy`. These checks do not validate weather-forecast accuracy. Review the [data provenance and assumptions](../../../01_labs/04_navier_stokes/DATA_PROVENANCE.md).

## Why open the Python files?

The notebook contains the problem explanation, code examples, execution cells, and result-inspection steps. The `.py` file is the training program started by an execution cell; `conf/*.yaml` contains training settings.

Tutorials provide complete programs to read and run. Challenges ask you to fill in the designated `FIXME` functions in their `.py` files. Open the file in the JupyterLab editor, make the required edits, save it, then rerun the notebook's execution cell. Editing a Markdown code block does not automatically update the `.py` file.

The challenge option `--reference` selects the instructor's completed implementation. Participants should follow each notebook's instructions, complete and save the FIXME functions, and then run them. Basic tutorials require neither a reference option nor unfinished-code edits.
