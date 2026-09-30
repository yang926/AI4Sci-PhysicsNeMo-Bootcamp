# Instructor course and code index

Use this index to find lesson programs and troubleshoot a session. Students follow [Start Here](../../Start_Here.ipynb); they do not need to read this guide before class.

Use PhysicsNeMo 2.2.2 and run the lessons in JupyterLab. Keep the notebook and its linked Python file open side by side during exercises.

[Installation](../environment/SETUP.md) · [Evaluation guide](ASSESSMENT.md) · [Validation record](VALIDATION.md) · [Start Here](../../Start_Here.ipynb)

## Understand the PhysicsNeMo workflow

Start with the [Introduction](../../01_Introduction.ipynb), then use the [helper-to-API walkthrough](PHYSICSNEMO_WORKFLOW.md) while reading a lesson's Python file. In Challenges 1–3, `create_model`, `create_informer` and `residuals` are course helpers: they connect PhysicsNeMo models and PDE residual evaluation to an explicit PyTorch training loop. The Labs and Neural Operators use their own wrappers, identified in their notebooks.

Before filling a PINN exercise, identify the model's inputs and outputs, the requested residuals, the condition losses, and the optimizer update. A correct equation is one part of that workflow. Running a helper is not the same as explaining what it does. Lab 4 is a separate inference workflow: load pretrained AFNO weights, advance the weather state, and compare with reanalysis without updating weights.

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
| Lab 4 | AI Weather Forecasting with FourCastNet | [Open notebook](../../01_labs/04_weather_forecasting/Lab_4_Weather_Forecasting.ipynb) | Generate a 48-hour forecast with pretrained AFNO; compare with ERA5 reanalysis and persistence as lead time increases. |
| Challenge 1 | Wave dynamics | [Open notebook](../../02_challenges/01_wave/Challenge_1_Wave_Dynamics.ipynb) | Level 1–3: constant speed, variable speed, circular Robin boundary. |
| Challenge 2 | Fluid flow | [Open notebook](../../02_challenges/02_fluid/Challenge_2_Fluid_Flow.ipynb) | Level 1–3: one fixed obstacle, multiple fixed obstacles, time-dependent flow. |
| Challenge 3 | Educational climate PDEs | [Open notebook](../../02_challenges/03_climate/Challenge_3_Climate_Modeling.ipynb) | Level 1–2: temperature transport and atmosphere–ocean equations. Level 2 defaults to the original uncoupled case (`gamma0=0`); nonzero exchange is a separate local experiment. |
| Challenge 4 | Neural operators | [Open notebook](../../02_challenges/04_neural_operators/Challenge_4_Neural_Operators.ipynb) | Level 1–3: FNO, AFNO and PINO on the same periodic reaction–diffusion problem. |

The course has four Labs and eleven Challenge levels. Lab 4 is an explicitly approved replacement of the previous Navier–Stokes teaching slot, not a claim that the original bootcamp included this weather workflow. The previous [PINN Lab](../reference_labs/04_navier_stokes/Lab_4_Navier_Stokes.ipynb) remains available as reference material. Background on earlier titles is in the [migration notes](MIGRATION.md).

## Notebooks, Python files and configuration

| File | Purpose | What to do |
|---|---|---|
| `.ipynb` | Problem statement, equations, examples, execution and plots | Read in order and run the execution cells. |
| `.py` | Equations, model, conditions, training, inference and evaluation | Read the completed Labs; fill in the marked exercise code in Challenges. |
| `conf/*.yaml` | Network, batch size, learning rate and step count | Inspect or adjust the settings specified by the lesson. |

A code example in a Markdown cell does not change the `.py` program. Edit the Python file itself. Execution cells run that file with the current kernel's interpreter.

Challenges 1–3 require all marked functions for each level: `student_equations` and `student_conditions`, plus Wave's `student_speed`, Fluid's `student_geometry`, or Climate's `student_parameters` and `student_solution`. See the [task and submission mapping](CHALLENGE_CONTRACTS.md). Challenge 4 instead uses `build_datasets`, `build_model`, and Level 3's `ReactionDiffusionPDE`; follow the linked file and `FIXME` markers rather than looking for a function literally named `student_*` there.

## Complete one level

1. Read the PDE, domain, initial and boundary conditions, and the implementation task.
2. Open the linked `.py` file in the event JupyterLab workspace.
3. For a Challenge, complete every **EDIT HERE** block in the linked Python file and save with **Ctrl+S** or **Cmd+S**. Challenges run only your implementation. Labs 1–3 remain complete examples.
4. Before training, run the final submission-panel cell, select only completed Levels and click **Check saved code**. This checks syntax and completeness without training or contacting the judge; correctness is evaluated by the judge after submission. Return to the current training cell and run it with **Shift+Enter**.
5. Run the evaluation and plotting cells. The before/after table summarizes the saved diagnostics; expand **Full metrics and run settings** for the complete JSON. Compare attempts only on the same problem and evaluation settings. These errors are practice feedback, not official ranking points.
6. Resolve any error before continuing to the next level or the next notebook.

The notebook normally starts in its own directory. Use `%pwd` if a file cannot be found. Recheck the file path, saved edits and error message before reinstalling packages or regenerating data.

## Lab programs

| Lab | Program | Output to inspect |
|---|---|---|
| 1 | [pinn_basics.py](../../01_labs/01_pinn/source_code/pinn_basics.py): `forward`, `parameterized`, `inverse` | Learned solution, parameter dependence and inferred source. |
| 2 | [projectile.py](../../01_labs/02_projectile/source_code/projectile.py), [projectile_eqn.py](../../01_labs/02_projectile/source_code/projectile_eqn.py) | Predicted and analytical trajectories; VTP export for ParaView. |
| 3 | [diffusion_bar.py](../../01_labs/03_heat_conduction/source_code/diffusion_bar.py), [diffusion_bar_parameterized.py](../../01_labs/03_heat_conduction/source_code/diffusion_bar_parameterized.py) | Piecewise solution, interface conditions and parameterized predictions. |
| 4 | [run_forecast.py](../../01_labs/04_weather_forecasting/source_code/run_forecast.py), [evaluate_weather.py](../../01_labs/04_weather_forecasting/source_code/evaluate_weather.py) | Forecast, ERA5 reanalysis and errors for wind, pressure and temperature; animation and lead-time error curves. |

Lab 4 uses NVIDIA's pretrained 26-channel FourCastNet1 AFNO checkpoint. It starts from ERA5 at 00 UTC on 1 September 2022 and applies eight six-hour forecast steps on a 720-by-1440 global grid. Future ERA5 is used only for evaluation. There is no optimizer, training step count or judge submission in this Lab. One historical case illustrates forecasting and verification; it does not establish general weather skill. See the [weather validation record](LAB4_WEATHER_VALIDATION.md).

The first preparation downloads about 301 MB of model weights and 622 MB of compressed source data. Files are cached under `~/.cache/ai4sci/weather`, outside the Git checkout, and reused after verification. Rehearse preparation before class; a single successful download does not establish 110-person simultaneous download capacity.

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

Successful training runs in Labs 1–3 and the Challenges save `metrics.json` (evaluation), `loss.csv` (training history), `model.pt` (weights), `predictions.npz` (predictions), and `preview.png` (plot). Lab 4 instead saves `forecast.npz`, `runtime.json`, and a separate evaluation directory with metrics, maps and an animation. Use a fresh output directory for each run; existing results are not overwritten.

The shared [notebook helper](../runtime/notebook.py) presents training results and checks their identity; it does not train models or assign points. Equations remain in the lesson programs, and launch commands remain visible in each notebook. Lab 4 has its own forecast/evaluation path. Training result tables need no widgets; the separate submission panel uses the environment's installed `ipywidgets` package.

The notebook training cells choose a new result directory on every execution, including when only that cell is rerun. Run the following plot/evaluation cell after a successful run; failure must not be mistaken for an older successful result. If you use the command line instead, choose a new `--output-dir` yourself and check the selected recipe and step count. Current notebook defaults are full lesson budgets, not smoke checks. Notebook `STEPS` values are passed as `--steps` and override YAML step counts; `AI4SCI_STEPS`, if set, overrides the notebook defaults.

All Challenge runs require every marked function for the Level, not just its PDE. Challenges 1–3 default to 5,000 Adam updates per level; FNO, AFNO and PINO retain 3,000. These are bounded class budgets, not convergence guarantees. Local PINN residuals use your own implementation; the judge checks correctness. Wave 1 retains an independent analytical comparison, and Fluid 1 retains the supplied OpenFOAM comparison. Climate comparisons use your learner-derived `student_solution`. Lab 4 weather verification remains unchanged.

The student distribution contains no completed Challenge answer mode. Notebook commands use Python `-u` so training progress is shown while the process runs.

## Troubleshooting

| Symptom | Check |
|---|---|
| `Complete student_...` or `NotImplementedError` | Complete the exercise in the actual `.py` file and save it. |
| The same error remains after editing | Check that you edited and saved the Python file rather than a Markdown example. |
| A student edit has no effect | Check the Python file path, save the edit, then rerun training and its result cell. |
| Saved settings differ from the current run | Rerun training with the intended controls, then rerun the result cell. Do not relabel old metrics. |
| A manually entered command reports an existing output directory | Keep previous results and choose a new `--output-dir`; notebook execution cells do this automatically. |
| `can't open file` | Check `%pwd` and the program path. |
| Import or CUDA error | Run the environment check and select the intended virtual-environment kernel. |
| An old plot remains visible after a failure | Confirm that the current run completed and produced new artifacts. |
| Missing data or figure | Check whether the file is included in the repository or generated by an earlier step. |
| Markdown opens as source text | Right-click the file and select **Open With → Markdown Preview**. |

[Begin the course](../../01_Introduction.ipynb) · [Schedule](course-plan.md) · [Instructor guide](INSTRUCTOR.md)

## Attribution

Adapted from [OpenHackathons AI-Powered-Physics-Bootcamp](https://github.com/openhackathons-org/AI-Powered-Physics-Bootcamp). Original attribution and licensing are retained. [License](../../LICENSE).
