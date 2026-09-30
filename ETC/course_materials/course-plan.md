# AI4Sci event schedule

Follow [Start Here](../../Start_Here.ipynb) and the [course guide](README.md). The course includes the Introduction, four Labs and eleven levels across four Challenges: Wave, Fluid, Climate and Neural Operators.

## Complete learning sequence

| Order | Topic | Notebook | Learning outcome |
|---|---|---|---|
| Introduction | Introduction to PhysicsNeMo | [Open notebook](../../01_Introduction.ipynb) | Distinguish physics-informed and data-driven learning. |
| Lab 1 | PINN fundamentals | [Open notebook](../../01_labs/01_pinn/Lab_1_PINN_Fundamentals.ipynb) | Run forward, parameterized and inverse PINNs. |
| Lab 2 | Projectile ODEs | [Open notebook](../../01_labs/02_projectile/Lab_2_Projectile_Motion.ipynb) | Connect initial conditions and ODE residuals; compare with the analytical trajectory. |
| Lab 3 | Steady heat conduction | [Open notebook](../../01_labs/03_heat_conduction/Lab_3_Heat_Conduction.ipynb) | Solve a two-material bar and verify interface temperature and heat flux. |
| Lab 4 | AI Weather Forecasting with FourCastNet | [Open notebook](../../01_labs/04_weather_forecasting/Lab_4_Weather_Forecasting.ipynb) | Generate a 48-hour forecast with pretrained AFNO; compare with ERA5 reanalysis and persistence as lead time increases. |
| Challenge 1 | Wave dynamics | [Open notebook](../../02_challenges/01_wave/Challenge_1_Wave_Dynamics.ipynb) | Level 1–3: constant speed, variable speed, circular Robin boundary. |
| Challenge 2 | Fluid flow | [Open notebook](../../02_challenges/02_fluid/Challenge_2_Fluid_Flow.ipynb) | Level 1–3: one fixed obstacle, multiple fixed obstacles, time-dependent flow. |
| Challenge 3 | Educational climate PDEs | [Open notebook](../../02_challenges/03_climate/Challenge_3_Climate_Modeling.ipynb) | Level 1–2: temperature transport and atmosphere–ocean equations. Level 2 defaults to the original uncoupled case (`gamma0=0`); nonzero exchange is a separate local experiment. |
| Challenge 4 | Neural operators | [Open notebook](../../02_challenges/04_neural_operators/Challenge_4_Neural_Operators.ipynb) | Level 1–3: FNO, AFNO and PINO on the same periodic reaction–diffusion problem. |

The instructor has approved replacing the previous Navier–Stokes Lab 4 with pretrained weather inference. Labs 1–3 and all Challenge problems and scoring contracts are unchanged by that replacement. The [previous PINN Lab](../reference_labs/04_navier_stokes/Lab_4_Navier_Stokes.ipynb) is retained for reference. The shared event sheet and external slides have **not** been changed by this repository update.

## Event details and source check

NVIDIA PhysicsNeMo Tutorial, AI4Science Korea 2026. Wednesday, 30 September 2026, Seoul Dragon City, Seoul. Instructors: Mingyu Yang and Hyungon Ryu, NVIDIA. The [conference website](https://ai4scikorea.org/) describes the wider AI for Science event; this repository contains the PhysicsNeMo tutorial.

The schedule and teaching assignments below follow the instructor's supplied agenda, checked on 2026-09-22. Private planning links and internal coordination notes are not distributed with the course. The event slide deck has not been verified in this check.

## 8-hour notebook-aligned teaching plan

This is the instructor-provided 09:30–17:30 agenda, confirmed on 2026-09-22. It supersedes the earlier local 10:00–17:00 allocation. Do not reuse the earlier registration slot, which overlaps the new first session. The student-facing schedule and notebook links are combined in [Start Here](../../Start_Here.ipynb).

| Time | Minutes | Category | Notebook-aligned topic | Instructor |
|---|---:|---|---|---|
| 09:30–10:20 | 50 | Introduction | Introduction to NVIDIA PhysicsNeMo | Mingyu Yang |
| 10:20–10:30 | 10 | Break | Break | |
| 10:30–11:30 | 60 | Lab | Training Labs: PINN Fundamentals, ODEs and PDEs | Mingyu Yang |
| 11:30–13:00 | 90 | Lunch | Lunch Break | |
| 13:00–13:50 | 50 | Challenge | Challenge 1: Advanced Wave Dynamics | Mingyu Yang |
| 13:50–14:00 | 10 | Break | Break | |
| 14:00–15:00 | 60 | Challenge | Challenge 2: Fluid Flow Around Fixed Obstacles | Hyungon Ryu |
| 15:00–15:30 | 30 | Break | Coffee Break | |
| 15:30–16:20 | 50 | Challenge | Challenge 3: Temperature Transport and Atmosphere–Ocean Coupling | Hyungon Ryu |
| 16:20–16:30 | 10 | Break | Break | |
| 16:30–17:20 | 50 | Challenge | Challenge 4: Neural Operators with FNO, AFNO and PINO | Hyungon Ryu |
| 17:20–17:30 | 10 | Wrap-up | Wrap-up and Q&A | |
| **Total** | **480** | **Teaching 330 / Lunch 90 / Break 60** | **8 hours** | |

The published session title and times are retained above. Within the morning Labs, Labs 1–3 train PINNs and Lab 4 runs pretrained weather inference without training. The afternoon Climate session still uses simplified temperature transport and atmosphere–ocean exchange equations, not a weather-forecasting model. All four Labs and all eleven Challenge levels remain in the course. Labs and Challenges have 270 minutes; measure explanation, editing, execution and interpretation time before promising full completion.

Use the Introduction to establish coordinates, predictions, derivatives and losses, then show how to edit and run the first problem. Complete environment setup before class. The 60-minute Labs and 50-minute Neural Operators session need particular attention in rehearsal; no level is silently removed or converted to a demonstration.

## Learning checkpoints and transitions

| Session | Before continuing, the learner should be able to... |
|---|---|
| Introduction | Trace coordinates → predicted solution → derivatives → equation/condition losses → parameter update; identify PhysicsNeMo, SymPy, PyTorch and course helper responsibilities. |
| Lab 1 | Distinguish forward, parameterized and inverse problems; explain the first problem's boundary conditions and analytical curve. |
| Labs 2–3 | Identify initial versus interface conditions, and compare predictions with the applicable analytical solution. |
| Lab 4 | Trace a 26-field global initial state through eight six-hour AFNO forecast steps; compare forecasts with ERA5 reanalysis and persistence, distinguish regional RMSE from local errors, and explain why inference is not training. |
| Challenges 1–3 | Follow `create_model`, `create_informer` and `residuals` into their PhysicsNeMo calls; complete and save all required functions; local checks establish syntax/completeness only, and interpret individual condition errors. |
| Challenge 4 | Explain input-field → solution-field learning, separate the course factories from FNO/AFNO APIs, and identify the spectral PDE loss added by PINO. Compare using the same held-out data. |

Record explanation, editing, execution, result interpretation and questions separately in rehearsal. L4 execution speed alone cannot establish that every learner will finish. Do not silently drop levels or turn hands-on work into demonstrations to fit the time slots.

Use the [helper-to-API walkthrough](PHYSICSNEMO_WORKFLOW.md) within the existing Introduction and first Challenge slots. It explains the current programs without changing the timetable, exercise contracts, training budgets or scoring rules.

The Lab 4 notebook uses the pretrained 26-channel FourCastNet1 AFNO checkpoint, with ERA5 at 00 UTC on 1 September 2022 as its only initial state. Each prediction becomes the next input; later ERA5 is held back for verification. Students inspect surface wind, mean sea-level pressure and 2 m temperature over 48 hours. Do not add training or extend the run just to fill a time slot. Use the time to interpret the maps, baseline and lead-time errors. This is a single-case teaching result, not an operational forecast validation; see the [weather validation record](LAB4_WEATHER_VALIDATION.md).

Prepare the model and data cache before the session: about 301 MB of checkpoint files and 622 MB of compressed source data, outside the checkout at `~/.cache/ai4sci/weather`. On a fresh Brev L4, cold preparation took 35 seconds and the cached notebook took about 85 seconds for all code cells (90 seconds including kernel startup and notebook saving). Model computation alone took 3.21 seconds. These are separate single-instance measurements, not guaranteed timings or evidence of concurrent-download capacity.

## Instructor decisions

| Item | Action |
|---|---|
| Public materials | Align the external agenda and Main material link with the agreed, published course revision; local edits are not a GitHub release. |
| Participants | Plan for 110 individual participants, not teams. Labs are practice; the four afternoon Challenges are intended for individual ranking. |
| Assessment | Follow the [evaluation guide](ASSESSMENT.md). The implemented pilot scores original-task completion, averages all Levels within each Challenge and keeps the best attempt; omitted Levels score zero and separate attempts are not combined. Numerical feedback does not affect points. Confirm the pilot rules for the event before class. |
| Event environment | The plan is one Brev L4 instance per participant and a separate eight-GPU judge. One new L4 instance verified this Lab 4 update; that does not establish 110-instance availability or concurrent startup capacity. Availability and concurrent startup capacity still require confirmation. |
| Instructor roles | Mingyu Yang: Introduction, Labs and Challenge 1. Hyungon Ryu: Challenges 2–4. Agree on support responsibilities during each segment. |
| Rehearsal | Measure the full introduction, four Labs and eleven Challenge levels, including editing and result interpretation. |
| Slides | Confirm the event deck against the current notebook checkpoints: Labs 1–3 use explicit training loops; Lab 4 is pretrained FourCastNet inference and ERA5 verification. Slide files are outside this repository update. |

[Migration notes](MIGRATION.md) explain API and consistency corrections. The supplementary [Wave student notebook](01_Wave_PINN.ipynb) runs the saved Wave Level 1 exercise implementation; complete its marked functions first. It is not a shortened replacement course.
