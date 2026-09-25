# Navier–Stokes data provenance and interpretation

The student Lab uses the original `data_lat.npy`, preserved byte for byte, as its initial condition. The upstream notebook describes it as a two-dimensional projection and periodic tiling of ERA5 reanalysis fields containing u velocity, v velocity, and pressure. This repository does not contain enough metadata to verify the original acquisition date, variable identifiers, level, units, projection, or tiling procedure independently. Treat those as **not independently verified**, not as validated dataset provenance.

Original notebook and data source: [OpenHackathons source](https://github.com/openhackathons-org/AI-Powered-Physics-Bootcamp/tree/9cae27f8303268cdaf7528fe963ce12ba439377f/tutorial/navier_stokes). The original notebook also linked a [Google Drive archive](https://drive.google.com/file/d/1IXEGbM3NOO6Dig1sxG1stHubwb09-D2N/view). The inspected archive contains only `data_lat.npy`, not a pretrained checkpoint. The modern notebook uses the already tracked array and trains a new model.

The loader preserves the original channel ordering and normalization:

- x and y samples each span -0.720 to 0.719; model periodic interval is [-0.720, 0.720].
- Length scale is 12,742,000 / 1.440 metres; time scale is 60 hours.
- Velocity scale is length scale / time scale; pressure scale is 1.1614 × velocity scale².
- Pressure is multiplied by the original factor **0.10197** before dividing by pressure scale. That factor was named `Pa_to_kgm3` in the original code, but pressure and mass density have different dimensions. The factor is retained to avoid silently altering the dataset's numerical meaning; the prior name is not endorsed as a valid unit conversion. Pressure units and offsets require validation before physical interpretation.

The mathematical model is a planar periodic, constant-density, incompressible 2-D Navier–Stokes PINN. It omits the full thermodynamics, moisture, rotation and spherical geometry of a numerical weather prediction model. No held-out future weather observations are bundled for forecast validation.

The student notebook has no dataset-selection switch. Missing original data stops execution, and its playback refuses saved synthetic results. The command-line `--smoke-data` option is reserved for tests under `ETC`: it selects a separate analytical Taylor–Green fixture with nondimensional viscosity 0.01. Its synthetic-field accuracy tests do not validate this original-data Lab and must not be presented as its results.

The saved real-data inference times now include 11 frames over the 60-hour scale, giving 6-hour increments. The old 10-frame inclusive linspace gave 60/9-hour intervals despite the six-hour wording. This correction changes the output sampling, not the PDE or training time interval.

The notebook first displays the original input wind. Its time player then holds the original field at 0 hours on the left while the prediction advances on the right. Speed colors and velocity-arrow scales stay fixed across all frames; x, y and field values remain normalized. The input is not a reference for later times. The ParaView export contains the full predicted sequence as VTI files plus `flow.pvd` and a self-contained ZIP, so it can be downloaded and played locally without retraining or port forwarding.
