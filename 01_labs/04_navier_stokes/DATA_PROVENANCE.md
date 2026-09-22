# Navier–Stokes data provenance and interpretation

The original `data_lat.npy` file is preserved byte for byte. The upstream notebook describes it as a two-dimensional projection and periodic tiling of ERA5 reanalysis fields containing u velocity, v velocity, and pressure. This repository does not contain enough metadata to verify the original acquisition date, variable identifiers, level, units, projection, or tiling procedure independently. Treat those as **not independently verified**, not as validated dataset provenance.

Original notebook and data source: [OpenHackathons source](https://github.com/openhackathons-org/AI-Powered-Physics-Bootcamp/tree/9cae27f8303268cdaf7528fe963ce12ba439377f/tutorial/navier_stokes). The original notebook also linked a [Google Drive archive](https://drive.google.com/file/d/1IXEGbM3NOO6Dig1sxG1stHubwb09-D2N/view). The modern notebook uses the already tracked array, and does not download or load historical pretrained weights. Their compatibility and provenance have not been verified for PhysicsNeMo 2.2.2.

The loader preserves the original channel ordering and normalization:

- x and y samples each span -0.720 to 0.719; model periodic interval is [-0.720, 0.720].
- Length scale is 12,742,000 / 1.440 metres; time scale is 60 hours.
- Velocity scale is length scale / time scale; pressure scale is 1.1614 × velocity scale².
- Pressure is multiplied by the original factor **0.10197** before dividing by pressure scale. That factor was named `Pa_to_kgm3` in the original code, but pressure and mass density have different dimensions. The factor is retained to avoid silently altering the dataset's numerical meaning; the prior name is not endorsed as a valid unit conversion. Pressure units and offsets require validation before physical interpretation.

The mathematical model is a planar periodic, constant-density, incompressible 2-D Navier–Stokes PINN. It omits the full thermodynamics, moisture, rotation and spherical geometry of a numerical weather prediction model. No held-out future weather observations are bundled for forecast validation.

`--smoke-data` is a separate deterministic analytical Taylor–Green vortex fixture. It uses nondimensional viscosity 0.01 and is generated independently of the original weather array. The flag is explicit; missing original data fails with an error rather than silently changing datasets. Metrics label the fixture as `synthetic_taylor_green` and always set `weather_forecast_validated` to false. Synthetic-field RMSE measures this test problem only.

The saved real-data inference times now include 11 frames over the 60-hour scale, giving 6-hour increments. The old 10-frame inclusive linspace gave 60/9-hour intervals despite the six-hour wording. This correction changes the output sampling, not the PDE or training time interval.
