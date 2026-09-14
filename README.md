# AI4Sci PhysicsNeMo Bootcamp

A complete, updated edition of the OpenHackathons AI-Powered-Physics-Bootcamp using **PhysicsNeMo 2.2.2** and Python **3.11–3.14**.

Start with the [course guide](ai4sci/README.md) and [environment check](ai4sci/00_environment_check.ipynb). The course includes the introduction, all four training Labs, and all eleven Challenge levels. All teaching materials and instructor references are in English.

[Installation and tests](Deployment_Guide.MD) · [Validation results](ai4sci/VALIDATION.md) · [Start Here](Start_Here.ipynb)

Researchers learn to combine physics and partial differential equations with neural networks, and compare physics-informed and data-driven approaches. The upstream material estimated two hours of Labs and four hours of Challenges, excluding breaks. The AI4Sci event allows six or seven hours **including lunch and breaks**; see the [event schedule](ai4sci/course-plan.md). Full-course rehearsal is required to establish what can be completed in that time.

<p align="center">
  <img width="600" height="400" src="https://github.com/openhackathons-org/End-to-End-AI-for-Science/blob/d403086ce59c49b26be430bbea0056c37bd4d5f6/workspace/python/jupyter_notebook/omniverse/images/tcwv.gif">
</p>

## Bootcamp contents

| Order | Topic | Notebook | Learning outcome |
|---|---|---|---|
| Introduction | Introduction to PhysicsNeMo | [Open notebook](tutorial/introduction/Getting_Started_PhysicsNeMo.ipynb) | Distinguish physics-informed and data-driven learning. |
| Lab 1 | PINN fundamentals | [Open notebook](tutorial/introduction/Introductory_Notebook.ipynb) | Run forward, parameterized and inverse PINNs. |
| Lab 2 | Projectile ODEs | [Open notebook](tutorial/projectile/Getting_Started_Projectile.ipynb) | Connect initial conditions and ODE residuals; compare with the analytical trajectory. |
| Lab 3 | Diffusion PDEs | [Open notebook](tutorial/diffusion_1d/Diffusion_Problem_Notebook.ipynb) | Solve a two-material bar and verify interface temperature and heat flux. |
| Lab 4 | Navier–Stokes | [Open notebook](tutorial/navier_stokes/Weather-forecasting-navier-stokes.ipynb) | Prepare data, evaluate incompressible flow equations and inspect predictions. |
| Challenge 1 | Wave dynamics | [Open notebook](challenge/wave/Advanced_Wave_Dynamics.ipynb) | Level 1–3: constant speed, variable speed, circular Robin boundary. |
| Challenge 2 | Fluid flow | [Open notebook](challenge/fuild/Fluid_Structure_Interaction.ipynb) | Level 1–3: one obstacle, multiple obstacles, time-dependent flow. |
| Challenge 3 | Climate modeling | [Open notebook](challenge/climate/Multi-Physics_Climate_Modeling.ipynb) | Level 1–2: atmospheric transport and atmosphere–ocean coupling. |
| Challenge 4 | Neural operators | [Open notebook](challenge/neural_operator/Advanced_Neural_Operators.ipynb) | Level 1–3: FNO, AFNO and PINO on a shared benchmark. |

FNO, AFNO and PINO are implemented in the Neural Operators Challenge. The original README advertised Darcy/AFNO, FourCastNet and MHD/PINO for Challenges 2–4, while its linked files contained the topics above. The [migration notes](ai4sci/MIGRATION.md) preserve the original titles and explain the mapping. FourCastNet and MHD are not presented as implemented models.

## Prerequisites and tools

Participants should know Python, differential equations and deep-learning fundamentals. The course uses [NVIDIA PhysicsNeMo](https://developer.nvidia.com/physicsnemo), PyTorch and JupyterLab. [ParaView](https://www.paraview.org/) is optional for exported CSV data.

## Teaching and deployment

- [Course guide](ai4sci/README.md): notebook order, program files and completion checks.
- [Schedule](ai4sci/course-plan.md): timing from the shared event plan.
- [Instructor guide](ai4sci/INSTRUCTOR.md): reference mode, evaluation and rehearsal.
- [Deployment guide](Deployment_Guide.MD): isolated environments and reproducible tests.

This AI4Sci repository retains the upstream history at commit `9cae27f8303268cdaf7528fe963ce12ba439377f`. Source updates, measured CPU validation, GPU/container verification and event deployment are recorded separately.

## Attribution

This material originates from the OpenHackathons GitHub repository. Check out additional materials [here](https://github.com/openhackathons-org)

Don't forget to check out additional [Open Hackathons Resources](https://www.openhackathons.org/s/technical-resources) and join our [OpenACC and Hackathons Slack Channel](https://www.openacc.org/community#slack) to share your experience and get more help from the community.

## Licensing

Copyright © 2026 OpenACC-Standard.org. This material is released by OpenACC-Standard.org, in collaboration with NVIDIA Corporation, under the Creative Commons Attribution 4.0 International (CC BY 4.0). These materials may include references to hardware and software developed by other entities; all applicable licensing and copyrights apply.
