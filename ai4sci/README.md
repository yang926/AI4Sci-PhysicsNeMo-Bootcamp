# AI4Sci PhysicsNeMo course guide

Use **PhysicsNeMo 2.2.2**. Start with the [environment check](00_environment_check.ipynb), then follow the complete course below. In JupyterLab, right-click this README and select **Open With → Markdown Preview** to read the guide and open its links.

[Installation](../Deployment_Guide.MD) · [Validation record](VALIDATION.md) · [Start Here](../Start_Here.ipynb)

## Complete course sequence

| Order | Topic | Notebook | Learning outcome |
|---|---|---|---|
| Introduction | Introduction to PhysicsNeMo | [Open notebook](../tutorial/introduction/Getting_Started_PhysicsNeMo.ipynb) | Distinguish physics-informed and data-driven learning. |
| Lab 1 | PINN fundamentals | [Open notebook](../tutorial/introduction/Introductory_Notebook.ipynb) | Run forward, parameterized and inverse PINNs. |
| Lab 2 | Projectile ODEs | [Open notebook](../tutorial/projectile/Getting_Started_Projectile.ipynb) | Connect initial conditions and ODE residuals; compare with the analytical trajectory. |
| Lab 3 | Diffusion PDEs | [Open notebook](../tutorial/diffusion_1d/Diffusion_Problem_Notebook.ipynb) | Solve a two-material bar and verify interface temperature and heat flux. |
| Lab 4 | Navier–Stokes | [Open notebook](../tutorial/navier_stokes/Weather-forecasting-navier-stokes.ipynb) | Prepare data, evaluate incompressible flow equations and inspect predictions. |
| Challenge 1 | Wave dynamics | [Open notebook](../challenge/wave/Advanced_Wave_Dynamics.ipynb) | Level 1–3: constant speed, variable speed, circular Robin boundary. |
| Challenge 2 | Fluid flow | [Open notebook](../challenge/fuild/Fluid_Structure_Interaction.ipynb) | Level 1–3: one obstacle, multiple obstacles, time-dependent flow. |
| Challenge 3 | Climate modeling | [Open notebook](../challenge/climate/Multi-Physics_Climate_Modeling.ipynb) | Level 1–2: atmospheric transport and atmosphere–ocean coupling. |
| Challenge 4 | Neural operators | [Open notebook](../challenge/neural_operator/Advanced_Neural_Operators.ipynb) | Level 1–3: FNO, AFNO and PINO on a shared benchmark. |

All four Labs and all eleven Challenge levels are included. The original README and linked notebooks used different Challenge 2–4 titles; the [migration notes](MIGRATION.md) document that difference. This index follows the actual lesson files.

## Notebooks, Python files and configuration

| File | Purpose | What to do |
|---|---|---|
| `.ipynb` | Problem statement, equations, examples, execution and plots | Read in order and run the execution cells. |
| `.py` | Equations, model, conditions, training and evaluation | Read the completed Labs; fill in `student_*` functions in Challenges. |
| `conf/*.yaml` | Network, batch size, learning rate and step count | Inspect or adjust the settings specified by the lesson. |

A fenced code block in a Markdown cell is explanatory text. Editing it does **not** update the `.py` program. The execution cell runs the actual Python file. If a cell uses `!python`, the `!` asks Jupyter to run a terminal command. Other cells use the current kernel's Python explicitly.

Open the notebook and the Python file side by side in JupyterLab so you can read, edit and run them in one workspace.

## Complete one level

1. Read the PDE, domain, initial and boundary conditions, and the implementation task.
2. Open the linked `.py` file in the event JupyterLab workspace.
3. Complete the `FIXME` / `student_*` function and save with **Ctrl+S** or **Cmd+S**. The introductory Labs provide complete programs.
4. Return to the notebook and run the current level's execution cell with **Shift+Enter**.
5. Run the evaluation and plotting cells. Inspect this run's `metrics.json` and `loss.csv`; compare teams only on the same problem and evaluation settings.
6. Resolve any error before continuing to the next level or the next notebook.

The notebook normally starts in its own directory. Use `%pwd` if a file cannot be found. Recheck the file path, saved edits and error message before reinstalling packages or regenerating data.

## Lab programs

| Lab | Program | Output to inspect |
|---|---|---|
| 1 | [pinn_basics.py](../tutorial/introduction/source_code/pinn_basics.py): `forward`, `parameterized`, `inverse` | Learned solution, parameter dependence and inferred source. |
| 2 | [projectile.py](../tutorial/projectile/source_code/projectile.py), [projectile_eqn.py](../tutorial/projectile/source_code/projectile_eqn.py) | Predicted and analytical trajectories; CSV export for ParaView. |
| 3 | [diffusion_bar.py](../tutorial/diffusion_1d/source_code/diffusion_bar.py), [diffusion_bar_parameterized.py](../tutorial/diffusion_1d/source_code/diffusion_bar_parameterized.py) | Piecewise solution, interface conditions and parameterized predictions. |
| 4 | [navier_stokes.py](../tutorial/navier_stokes/source_code/navier_stokes.py) | Data preparation and time-dependent flow predictions. |

## Challenge programs

| Challenge | Level | Edit this file | Configuration |
|---|---|---|---|
| Wave dynamics | 1 | [wave_l1.py](../challenge/wave/wave_l1.py) | [config_wave.yaml](../challenge/wave/conf/config_wave.yaml) |
| Wave dynamics | 2 | [wave_l2.py](../challenge/wave/wave_l2.py) | [config_wave.yaml](../challenge/wave/conf/config_wave.yaml) |
| Wave dynamics | 3 | [wave_l3.py](../challenge/wave/wave_l3.py) | [config_wave.yaml](../challenge/wave/conf/config_wave.yaml) |
| Fluid flow | 1 | [chip_2d_l1.py](../challenge/fuild/chip_2d_l1.py) | [config_chip_2d.yaml](../challenge/fuild/conf/config_chip_2d.yaml) |
| Fluid flow | 2 | [chip_2d_l2.py](../challenge/fuild/chip_2d_l2.py) | [config_chip_2d.yaml](../challenge/fuild/conf/config_chip_2d.yaml) |
| Fluid flow | 3 | [chip_2d_l3.py](../challenge/fuild/chip_2d_l3.py) | [config_chip_2d.yaml](../challenge/fuild/conf/config_chip_2d.yaml) |
| Climate modeling | 1 | [climate_l1.py](../challenge/climate/climate_l1.py) | [config_atmos.yaml](../challenge/climate/conf/config_atmos.yaml) |
| Climate modeling | 2 | [climate_l2.py](../challenge/climate/climate_l2.py) | [config_coupled.yaml](../challenge/climate/conf/config_coupled.yaml) |
| Neural operators | 1 | [fno_physicsnemo_l1.py](../challenge/neural_operator/fno_physicsnemo_l1.py) | [config_FNO.yaml](../challenge/neural_operator/conf/config_FNO.yaml) |
| Neural operators | 2 | [fno_physicsnemo_l2.py](../challenge/neural_operator/fno_physicsnemo_l2.py) | [config_AFNO.yaml](../challenge/neural_operator/conf/config_AFNO.yaml) |
| Neural operators | 3 | [fno_physicsnemo_l3.py](../challenge/neural_operator/fno_physicsnemo_l3.py) | [config_PINO.yaml](../challenge/neural_operator/conf/config_PINO.yaml) |

For Neural Operators, follow the notebook's data-generation step using [generate_data.py](../challenge/neural_operator/generate_data.py) before training.

## Results and reruns

Every successful run saves `metrics.json` (evaluation), `loss.csv` (training history), `model.pt` (weights), `predictions.npz` (predictions), and `preview.png` (plot). Use a fresh output directory for each run; existing results are not overwritten.

`--reference` runs the completed instructor implementation. The default Challenge mode requires you to complete the student function. The initial 200-step notebook setting is an execution check, not evidence of convergence. Use held-out errors, analytical references and individual constraint residuals to decide whether more training is needed.

## Troubleshooting

| Symptom | Check |
|---|---|
| `Complete student_...` or `NotImplementedError` | Complete the exercise in the actual `.py` file and save it, or use the instructor reference mode. |
| The same error remains after editing | Check that you edited and saved the Python file rather than a Markdown example. |
| `can't open file` | Check `%pwd` and the program path. |
| Import or CUDA error | Run the environment check and select the intended virtual-environment kernel. |
| An old plot remains visible after a failure | Confirm that the current run completed and produced new artifacts. |
| Missing data or figure | Check whether the file is included in the repository or generated by an earlier step. |

[Begin the course](../tutorial/introduction/Getting_Started_PhysicsNeMo.ipynb) · [Schedule](course-plan.md) · [Instructor guide](INSTRUCTOR.md)

## Attribution

Adapted from [OpenHackathons AI-Powered-Physics-Bootcamp](https://github.com/openhackathons-org/AI-Powered-Physics-Bootcamp). Original attribution and licensing are retained. [License](../LICENSE).
