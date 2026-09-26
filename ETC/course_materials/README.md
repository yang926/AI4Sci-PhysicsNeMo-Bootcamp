# Instructor course and code index

Use this index to find lesson programs and troubleshoot a session. Students follow [Start Here](../../Start_Here.ipynb); they do not need to read this guide before class.

Use PhysicsNeMo 2.2.2 and run the lessons in JupyterLab. Keep the notebook and its linked Python file open side by side during exercises.

[Installation](../environment/SETUP.md) · [Evaluation guide](ASSESSMENT.md) · [Validation record](VALIDATION.md) · [Start Here](../../Start_Here.ipynb)

## Your first training problem

Run the [environment check](../../00_Setup.ipynb), read the Introduction, then begin Lab 1.

The first training problem is [Lab 1.1: forward PINN](../../01_labs/01_pinn/Lab_1_PINN_Fundamentals.ipynb#first-problem): learn $u(x)$ on $[0,1]$ with $u''(x)=1$ and $u(0)=u(1)=0$.

Under [Forward PINN execution](../../01_labs/01_pinn/Lab_1_PINN_Fundamentals.ipynb#forward-pinn-execution), run setup, training and plotting in that order. Compare the `analytical` and `PINN` curves before moving to Lab 1.2 (parameterized) and Lab 1.3 (inverse). The Lab programs are complete; no functions need to be filled in.

## Complete course sequence

| Order | Topic | Notebook | Learning outcome |
|---|---|---|---|
| Introduction | Introduction to PhysicsNeMo | [Open notebook](../../01_Introduction.ipynb) | Read about physics-informed and data-driven learning; no code cells to run. |
| Lab 1 | PINN fundamentals | [Open notebook](../../01_labs/01_pinn/Lab_1_PINN_Fundamentals.ipynb) | First executable lesson: 1.1 forward, 1.2 parameterized, 1.3 inverse PINNs. |
| Lab 2 | Projectile ODEs | [Open notebook](../../01_labs/02_projectile/Lab_2_Projectile_Motion.ipynb) | Connect initial conditions and ODE residuals; compare with the analytical trajectory. |
| Lab 3 | Steady heat conduction | [Open notebook](../../01_labs/03_heat_conduction/Lab_3_Heat_Conduction.ipynb) | Solve a two-material bar and verify interface temperature and heat flux. |
| Lab 4 | Incompressible Navier–Stokes | [Open notebook](../../01_labs/04_navier_stokes/Lab_4_Navier_Stokes.ipynb) | Start from the original ERA5-derived array, inspect predicted flow over 60 hours, and play the sequence in ParaView. |
| Challenge 1 | Wave dynamics | [Open notebook](../../02_challenges/01_wave/Challenge_1_Wave_Dynamics.ipynb) | Level 1–3: constant speed, variable speed, circular Robin boundary. |
| Challenge 2 | Fluid flow | [Open notebook](../../02_challenges/02_fluid/Challenge_2_Fluid_Flow.ipynb) | Level 1–3: one fixed obstacle, multiple fixed obstacles, time-dependent flow. |
| Challenge 3 | Educational climate PDEs | [Open notebook](../../02_challenges/03_climate/Challenge_3_Climate_Modeling.ipynb) | Level 1–2: temperature transport and atmosphere–ocean equations. Level 2 defaults to the original uncoupled case (`gamma0=0`); nonzero exchange is a separate local experiment. |
| Challenge 4 | Neural operators | [Open notebook](../../02_challenges/04_neural_operators/Challenge_4_Neural_Operators.ipynb) | Level 1–3: FNO, AFNO and PINO on the same periodic reaction–diffusion problem. |

The course has four Labs and eleven Challenge levels. Background on earlier titles is in the [migration notes](MIGRATION.md).

## Notebooks, Python files and configuration

| File | Purpose | What to do |
|---|---|---|
| `.ipynb` | Problem statement, equations, examples, execution and plots | Read in order and run the execution cells. |
| `.py` | Equations, model, conditions, training and evaluation | Read the completed Labs; fill in the marked exercise code in Challenges. |
| `conf/*.yaml` | Network, batch size, learning rate and step count | Inspect or adjust the settings specified by the lesson. |

A code example in a Markdown cell does not change the `.py` program. Edit the Python file itself. Execution cells run that file with the current kernel's interpreter.

Challenges 1–3 require all marked functions for each level: `student_equations` and `student_conditions`, plus Wave's `student_speed`, Fluid's `student_geometry`, or Climate's `student_parameters` and `student_solution`. See the [task and submission mapping](CHALLENGE_CONTRACTS.md). Challenge 4 instead uses `build_datasets`, `build_model`, and Level 3's `ReactionDiffusionPDE`; follow the linked file and `FIXME` markers rather than looking for a function literally named `student_*` there.

## Complete one level

1. Read the PDE, domain, initial and boundary conditions, and the implementation task.
2. Open the linked `.py` file in the event JupyterLab workspace.
3. For a Challenge, set `USE_REFERENCE = False` in the setup cell and execute it (or run that assignment in a new cell after setup). Complete all marked functions for the level and save with **Ctrl+S** or **Cmd+S**. The introductory Labs provide complete programs and do not need this switch.
4. Return to the notebook and run the current level's execution cell with **Shift+Enter**. Verify the printed student/reference mode; student edits are not used in reference mode.
5. Run the evaluation and plotting cells. The before/after table summarizes the saved diagnostics; expand **Full metrics and run settings** for the complete JSON. Compare attempts only on the same problem and evaluation settings. These errors are practice feedback, not official ranking points.
6. Resolve any error before continuing to the next level or the next notebook.

The notebook normally starts in its own directory. Use `%pwd` if a file cannot be found. Recheck the file path, saved edits and error message before reinstalling packages or regenerating data.

## Lab programs

| Lab | Program | Output to inspect |
|---|---|---|
| 1 | [pinn_basics.py](../../01_labs/01_pinn/source_code/pinn_basics.py): `forward`, `parameterized`, `inverse` | Learned solution, parameter dependence and inferred source. |
| 2 | [projectile.py](../../01_labs/02_projectile/source_code/projectile.py), [projectile_eqn.py](../../01_labs/02_projectile/source_code/projectile_eqn.py) | Predicted and analytical trajectories; VTP export for ParaView. |
| 3 | [diffusion_bar.py](../../01_labs/03_heat_conduction/source_code/diffusion_bar.py), [diffusion_bar_parameterized.py](../../01_labs/03_heat_conduction/source_code/diffusion_bar_parameterized.py) | Piecewise solution, interface conditions and parameterized predictions. |
| 4 | [navier_stokes.py](../../01_labs/04_navier_stokes/source_code/navier_stokes.py) | Original initial wind, 11-frame flow playback, and a downloadable ParaView time series. |

## Challenge programs

| Challenge | Level | Edit this file | Configuration |
|---|---|---|---|
| Wave dynamics | 1 | [wave_l1.py](../../02_challenges/01_wave/wave_l1.py) | [config_wave_l1.yaml](../../02_challenges/01_wave/conf/config_wave_l1.yaml) |
| Wave dynamics | 2 | [wave_l2.py](../../02_challenges/01_wave/wave_l2.py) | [config_wave.yaml](../../02_challenges/01_wave/conf/config_wave.yaml) |
| Wave dynamics | 3 | [wave_l3.py](../../02_challenges/01_wave/wave_l3.py) | [config_wave.yaml](../../02_challenges/01_wave/conf/config_wave.yaml) |
| Fluid flow | 1 | [chip_2d_l1.py](../../02_challenges/02_fluid/chip_2d_l1.py) | [config_chip_2d.yaml](../../02_challenges/02_fluid/conf/config_chip_2d.yaml) |
| Fluid flow | 2 | [chip_2d_l2.py](../../02_challenges/02_fluid/chip_2d_l2.py) | [config_chip_2d.yaml](../../02_challenges/02_fluid/conf/config_chip_2d.yaml) |
| Fluid flow | 3 | [chip_2d_l3.py](../../02_challenges/02_fluid/chip_2d_l3.py) | [config_chip_2d.yaml](../../02_challenges/02_fluid/conf/config_chip_2d.yaml) |
| Climate modeling | 1 | [climate_l1.py](../../02_challenges/03_climate/climate_l1.py) | [config_atmos.yaml](../../02_challenges/03_climate/conf/config_atmos.yaml) |
| Climate modeling | 2 | [climate_l2.py](../../02_challenges/03_climate/climate_l2.py) | [config_coupled.yaml](../../02_challenges/03_climate/conf/config_coupled.yaml) |
| Neural operators | 1 | [fno_physicsnemo_l1.py](../../02_challenges/04_neural_operators/fno_physicsnemo_l1.py) | [config_FNO.yaml](../../02_challenges/04_neural_operators/conf/config_FNO.yaml) |
| Neural operators | 2 | [fno_physicsnemo_l2.py](../../02_challenges/04_neural_operators/fno_physicsnemo_l2.py) | [config_AFNO.yaml](../../02_challenges/04_neural_operators/conf/config_AFNO.yaml) |
| Neural operators | 3 | [fno_physicsnemo_l3.py](../../02_challenges/04_neural_operators/fno_physicsnemo_l3.py) | [config_PINO.yaml](../../02_challenges/04_neural_operators/conf/config_PINO.yaml) |

For Neural Operators, follow the notebook's data-generation step using [generate_data.py](../../02_challenges/04_neural_operators/generate_data.py) before training.

## Results and reruns

Every successful run saves `metrics.json` (evaluation), `loss.csv` (training history), `model.pt` (weights), `predictions.npz` (predictions), and `preview.png` (plot). Use a fresh output directory for each run; existing results are not overwritten.

The shared [notebook helper](../runtime/notebook.py) presents results and checks their identity; it does not train models or assign points. Equations remain in the lesson programs, and launch commands remain visible in each notebook. Result tables need no widgets; the separate submission panel uses the environment's installed `ipywidgets` package.

The notebook training cells choose a new result directory on every execution, including when only that cell is rerun. Run the following plot/evaluation cell after a successful run; failure must not be mistaken for an older successful result. If you use the command line instead, choose a new `--output-dir` yourself and pass an explicit `--steps` value: configuration defaults can be much longer than notebook checks.

`--reference` runs the completed instructor implementation. Student mode requires every marked function for the Level, not just its PDE. The notebook and YAML defaults now use measured class budgets instead of a 200-step execution check. See the [Challenge measurements](CHALLENGE_EFFICIENCY.md); use held-out errors, analytical references and individual constraint residuals to judge learning. A successful run does not certify convergence.

For an instructor demonstration, select `USE_REFERENCE = True` in the setup cell. All four notebooks default to `False` and ignore `AI4SCI_REFERENCE`; the setup cell and each run display the explicit choice. Switch back to `False` for student work. Do not change or delete your exercise code to switch modes.

## Troubleshooting

| Symptom | Check |
|---|---|
| `Complete student_...` or `NotImplementedError` | Complete the exercise in the actual `.py` file and save it, or use the instructor reference mode. |
| The same error remains after editing | Check that you edited and saved the Python file rather than a Markdown example. |
| A student edit has no effect | Check the printed mode. Set `USE_REFERENCE = False` before rerunning; reference mode deliberately bypasses student functions. |
| Saved mode or settings differ from the current run | Rerun training with the intended controls, then rerun the result cell. Do not relabel old metrics. |
| A manually entered command reports an existing output directory | Keep previous results and choose a new `--output-dir`; notebook execution cells do this automatically. |
| `can't open file` | Check `%pwd` and the program path. |
| Import or CUDA error | Run the environment check and select the intended virtual-environment kernel. |
| An old plot remains visible after a failure | Confirm that the current run completed and produced new artifacts. |
| Missing data or figure | Check whether the file is included in the repository or generated by an earlier step. |
| Markdown opens as source text | Right-click the file and select **Open With → Markdown Preview**. |

[Begin the course](../../01_Introduction.ipynb) · [Schedule](course-plan.md) · [Instructor guide](INSTRUCTOR.md)

## Attribution

Adapted from [OpenHackathons AI-Powered-Physics-Bootcamp](https://github.com/openhackathons-org/AI-Powered-Physics-Bootcamp). Original attribution and licensing are retained. [License](../../LICENSE).
