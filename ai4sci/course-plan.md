# AI4Sci event schedule

The full upstream Lab and Challenge curriculum is retained. Follow [Start Here](../Start_Here.ipynb) and the [course guide](README.md); align slides with the actual equations and program sequence.

## Complete learning sequence

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

The original README listed Darcy/AFNO, FourCastNet and MHD/PINO for Challenges 2–4. Those titles were copied into the shared plan, while the original start notebook linked to Fluid, Climate and Neural Operators. The repository index now follows its actual files. The shared sheet remains unchanged.

## 7-hour schedule from the shared plan

The timing and session titles below preserve the [shared event plan](https://docs.google.com/spreadsheets/d/1KS1z-Bmop8Jcn-PKLxuaCopveItRkmx2ImscphaLZfo/edit) as checked on 2026-09-14. Previously blank ten-minute gaps are classified as breaks. Registration is 09:30–10:00; the seven-hour event is 10:00–17:00, including lunch and breaks.

| Time | Minutes | Category | Shared-plan label |
|---|---:|---|---|
| 10:00–10:50 | 50 | Introduction | Introduction to NVIDIA PhysicsNeMo |
| 10:50–11:00 | 10 | Break | Blank interval in the shared plan |
| 11:00–12:00 | 60 | Lab | Training Labs: Fron PINN to PDE system problems |
| 12:00–13:00 | 60 | Lunch | Lunch |
| 13:00–13:50 | 50 | Challenge | Challenge 1: Advanced Wave Dynamics |
| 13:50–14:00 | 10 | Break | Blank interval in the shared plan |
| 14:00–14:50 | 50 | Challenge | Challenge 2: Solving the Darcy-Flow problem using AFNO |
| 14:50–15:00 | 10 | Break | Blank interval in the shared plan |
| 15:00–15:50 | 50 | Challenge | Challenge 3: Forecasting weather using FourCastNet |
| 15:50–16:00 | 10 | Break | Blank interval in the shared plan |
| 16:00–16:50 | 50 | Challenge | Challenge 4: Modeling Magnetohydrodynamics with Physics Informed Neural Operators |
| 16:50–17:00 | 10 | Wrap-up | Wrap-up |
| **Total** | **420** | **Teaching 320 / Lunch 60 / Break 40** | **7 hours** |

`Fron` is the original typo and should read `From`; the sheet itself has not been modified.

Labs and Challenges have 260 minutes in this schedule, compared with the upstream estimate of 120 minutes for Labs plus 240 minutes for Challenges. Keep the complete materials and measure instruction, editing, training and evaluation time in rehearsal before promising full completion.

## 6-hour format

Allocation for six hours including lunch and breaks is **to be confirmed**. All Labs and eleven Challenge levels remain available; this document does not silently remove levels or change them to demonstrations.

## Instructor decisions

| Item | Action |
|---|---|
| Session titles | Align Challenges 2–4 with the actual Fluid, Climate and Neural Operators files; do not advertise unimplemented FourCastNet or MHD models. |
| Instructor roles | Agree on teaching segments and support responsibilities between Mingyu Yang and the co-instructor. |
| Rehearsal | Measure the full introduction, four Labs and eleven Challenge levels, including editing and result interpretation. |
| Slides | Update slides to follow the actual notebooks, equations and transitions. Slide files are outside this repository update. |

[Migration notes](MIGRATION.md) explain API and consistency corrections. The supplementary [Wave reference notebook](01_Wave_PINN.ipynb) invokes the same Wave Level 1 implementation; it is not a shortened replacement course.
