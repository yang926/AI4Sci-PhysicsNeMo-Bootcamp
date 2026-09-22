# AI4Sci event schedule

Follow [Start Here](../../Start_Here.ipynb) and the [course guide](README.md). The course includes the Introduction, four Labs and eleven levels across four Challenges: Wave, Fluid, Climate and Neural Operators.

## Complete learning sequence

| Order | Topic | Notebook | Learning outcome |
|---|---|---|---|
| Introduction | Introduction to PhysicsNeMo | [Open notebook](../../01_Introduction.ipynb) | Distinguish physics-informed and data-driven learning. |
| Lab 1 | PINN fundamentals | [Open notebook](../../01_labs/01_pinn/Lab_1_PINN_Fundamentals.ipynb) | Run forward, parameterized and inverse PINNs. |
| Lab 2 | Projectile ODEs | [Open notebook](../../01_labs/02_projectile/Lab_2_Projectile_Motion.ipynb) | Connect initial conditions and ODE residuals; compare with the analytical trajectory. |
| Lab 3 | Steady heat conduction | [Open notebook](../../01_labs/03_heat_conduction/Lab_3_Heat_Conduction.ipynb) | Solve a two-material bar and verify interface temperature and heat flux. |
| Lab 4 | Incompressible Navier–Stokes | [Open notebook](../../01_labs/04_navier_stokes/Lab_4_Navier_Stokes.ipynb) | Check synthetic Taylor–Green flow; distinguish the optional original-data path from validated weather forecasting. |
| Challenge 1 | Wave dynamics | [Open notebook](../../02_challenges/01_wave/Challenge_1_Wave_Dynamics.ipynb) | Level 1–3: constant speed, variable speed, circular Robin boundary. |
| Challenge 2 | Fluid flow | [Open notebook](../../02_challenges/02_fluid/Challenge_2_Fluid_Flow.ipynb) | Level 1–3: one fixed obstacle, multiple fixed obstacles, time-dependent flow. |
| Challenge 3 | Educational climate PDEs | [Open notebook](../../02_challenges/03_climate/Challenge_3_Climate_Modeling.ipynb) | Level 1–2: temperature transport and active atmosphere–ocean exchange, with analytical comparisons for the default cases. |
| Challenge 4 | Neural operators | [Open notebook](../../02_challenges/04_neural_operators/Challenge_4_Neural_Operators.ipynb) | Level 1–3: FNO, AFNO and PINO on the same periodic reaction–diffusion problem. |

The instructor has confirmed the current exercise sequence. The shared event sheet and external slides have **not** been changed by this local update.

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

The Climate session uses simplified temperature transport and atmosphere–ocean exchange equations, not a weather-forecasting model. All four Labs and all eleven Challenge levels remain in the course. Labs and Challenges have 270 minutes; measure explanation, editing, execution and interpretation time before promising full completion.

Use the Introduction to establish coordinates, predictions, derivatives and losses, then show how to edit and run the first problem. Complete environment setup before class. The 60-minute Labs and 50-minute Neural Operators session need particular attention in rehearsal; no level is silently removed or converted to a demonstration.

## Learning checkpoints and transitions

| Session | Before continuing, the learner should be able to... |
|---|---|
| Introduction | Trace coordinates → predicted solution → derivatives → equation/condition losses → parameter update. |
| Lab 1 | Distinguish forward, parameterized and inverse problems; explain the first problem's boundary conditions and analytical curve. |
| Labs 2–3 | Identify initial versus interface conditions, and compare predictions with the applicable analytical solution. |
| Lab 4 | Identify the selected dataset and explain why synthetic-flow accuracy is not weather-forecast accuracy. |
| Challenges 1–3 | Select student mode, edit and save the linked function, rerun safely, and interpret individual condition errors. |
| Challenge 4 | Explain input-field → solution-field learning and compare FNO/AFNO/PINO using the same held-out data. |

Record explanation, editing, execution, result interpretation and questions separately in rehearsal. L40 execution speed alone cannot establish that every learner will finish. Do not silently drop levels or turn hands-on work into demonstrations to fit the time slots.

## Instructor decisions

| Item | Action |
|---|---|
| Public materials | Align the external agenda and Main material link with the agreed, published course revision; local edits are not a GitHub release. |
| Participants | Plan for 110 individual participants, not teams. Labs are practice; the four afternoon Challenges are intended for individual ranking. |
| Assessment | Follow the [evaluation guide](ASSESSMENT.md). Local error metrics are practice feedback; point conversion and official submission rules are not yet finalized. |
| Event environment | Participant training resources are undecided. An eight-GPU Brev judge has been requested separately; no instance or scoreboard is created by this material update. |
| Instructor roles | Mingyu Yang: Introduction, Labs and Challenge 1. Hyungon Ryu: Challenges 2–4. Agree on support responsibilities during each segment. |
| Rehearsal | Measure the full introduction, four Labs and eleven Challenge levels, including editing and result interpretation. |
| Slides | Confirm the event deck and map it to the notebook checkpoints and current explicit training loops. Slide files are outside this repository update. |

[Migration notes](MIGRATION.md) explain API and consistency corrections. The supplementary [Wave reference notebook](01_Wave_PINN.ipynb) invokes the same Wave Level 1 implementation; it is not a shortened replacement course.
