# AI4Sci PhysicsNeMo Bootcamp

This course covers physics-informed neural networks, neural operators and AI weather forecasting using PhysicsNeMo 2.2.2 and Python 3.12.

Recommended background: basic Python functions and arrays, first and second derivatives, and initial/boundary conditions. No prior PhysicsNeMo experience is assumed.

Open [Start Here](Start_Here.ipynb) for the course order. Read the lessons on GitHub or run them in JupyterLab. Labs 1–3 provide complete PINN training examples. Lab 4: AI Weather Forecasting with FourCastNet uses a pretrained AFNO model to generate a 48-hour forecast and compare it with ERA5 reanalysis and a persistence baseline; it does not train the model. In the four Challenges, fill in the **EDIT HERE** blocks in the linked Python files, save them, then run the notebook. Challenge runs use only your implementation; no completed answer mode is included. Challenges 1–3 default to 5,000 updates per level and Challenge 4 to 3,000. **Check saved code** checks syntax/completeness only; the judge evaluates correctness.

Learn how the pieces connect: PhysicsNeMo supplies models and PDE residual evaluation, SymPy expresses the equations, and PyTorch updates the model parameters. The Introduction explains this workflow. Use the [code walkthrough](ETC/course_materials/PHYSICSNEMO_WORKFLOW.md) to look inside course helpers such as `create_model`, `create_informer` and `residuals`; these are not public PhysicsNeMo API names.

Need an environment? Follow the [uv installation guide](ETC/environment/SETUP.md). If JupyterLab is already running, use Start Here; no reinstall is needed.

For the student Brev Launchable, use the [GitHub-based setup and update guide](ETC/launchable/README.md). It connects an isolated course kernel to Brev-managed Jupyter and preserves existing student work when updating.

## Release history

The public Git history starts with a clean release snapshot on 2026-10-01.
If you cloned this repository before that date, keep your existing folder and
saved work, then clone the course into a new folder. The existing updater will
refuse to replace the old history. Fresh clones use the normal update workflow.

## Attribution

Adapted from [OpenHackathons AI-Powered-Physics-Bootcamp](https://github.com/openhackathons-org/AI-Powered-Physics-Bootcamp).

Compatibility and problem-setting changes are recorded in the [migration notes](ETC/course_materials/MIGRATION.md). The instructor-approved weather Lab replaces the previous Navier–Stokes teaching slot; that [PINN notebook and its source data](ETC/reference_labs/04_navier_stokes) remain archived for reference. Labs 1–3 and the Challenge problems are unchanged by this replacement.

The weather Lab uses [NVIDIA FourCastNet1](https://huggingface.co/nvidia/fourcastnet1) and Copernicus/ECMWF ERA5 reanalysis from Google ARCO-ERA5. Model and data downloads are cached outside the checkout. See the weather Lab for source, license and input-processing details.

Additional [Open Hackathons Resources](https://www.openhackathons.org/s/technical-resources) and the [OpenACC and Hackathons Slack Channel](https://www.openacc.org/community#slack) are available for further study and community support.

## Licensing

Copyright © 2026 OpenACC-Standard.org. This material is released by OpenACC-Standard.org, in collaboration with NVIDIA Corporation, under the Creative Commons Attribution 4.0 International (CC BY 4.0). These materials may include references to hardware and software developed by other entities; all applicable licensing and copyrights apply.
