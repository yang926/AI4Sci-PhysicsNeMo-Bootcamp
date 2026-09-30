# Lab 4: AI weather forecasting with FourCastNet

Status: the integrated notebook passed local RTX 3080 and Brev L4 execution on
29 September 2026. This is pretrained-model inference and verification, not
training a new model or solving the former PINN problem.

## Scope and provenance

The official NVIDIA 26-channel FourCastNet/AFNO checkpoint is pinned to
`nvidia/fourcastnet1` revision `c67a63995f6c8e0e557eb3d791f32f437e9b02d5`.
It runs in FP32, without TF32 or autocast, on its native 720 x 1440 grid.
The official normalization, architecture and pretrained weights are unchanged.
This is the 26-channel released model, not the original paper's 20-channel setup.

Initial state: ERA5 at 2022-09-01 00 UTC. Eight autoregressive six-hour steps
produce a 48-hour forecast. Future ERA5 enters only the evaluator, never the
forecast program. ERA5 is reanalysis, not a collection of direct observations.
The forecast and persistence baseline are scored on the same cells and times,
using cosine-latitude area weighting. All eight leads are retained.

The preparation records source object checksums, grid orientation, variable
order and units. Relative humidity follows NVIDIA Earth2Studio's documented
mixed-phase derivation and 0..100 percent bounds, with clipping counts recorded;
raw specific humidity and model forecasts are not edited. One slightly negative
source humidity value is retained and handled by that documented transformation.
The raw source snapshot and pinned data plan are included with the lesson assets.

## Prespecified forecast checks

At 24 and 48 hours, globally and over East Asia (10–55 N, 100–160 E), require
wind-vector RMSE <= 6 m/s, mean sea-level pressure RMSE <= 5 hPa and 2 m
temperature RMSE <= 3 K, and improvement over persistence for all three.
These are bounded classroom acceptance checks, fixed before the actual run,
not operational forecast-certification thresholds.

The local and Brev L4 runs passed all 26 checks. At 48 hours (rounded values
agree on both GPUs):

| Region | Wind-vector RMSE | Pressure RMSE | Temperature RMSE |
| --- | ---: | ---: | ---: |
| Global forecast | 2.132 m/s | 1.356 hPa | 1.086 K |
| Global persistence | 7.535 m/s | 8.190 hPa | 2.593 K |
| East Asia forecast | 1.794 m/s | 1.251 hPa | 0.977 K |
| East Asia persistence | 5.550 m/s | 4.433 hPa | 1.878 K |

All eight leads beat persistence on the reported quantities in both regions.
Wind-vector error is sqrt(mean(du^2 + dv^2)), not speed error. Temperature
differences in K have the same magnitude as differences in degrees Celsius.
Maps use fixed scales across time; localized cyclone-core errors remain visible.
One historical initialization and eight correlated leads do not establish broad
forecast skill, accurate storm intensity or safety for operational decisions.

## Runtime evidence

The integrated student notebook executed all nine code cells without errors on
both GPUs. The model and scientific configuration were identical.

| Measurement | Brev NVIDIA L4 | Local NVIDIA RTX 3080 |
| --- | ---: | ---: |
| Cold model/data download, validation and assembly | 35.04 s | Not separately measured |
| Cached notebook, first code cell to last completed cell | 84.78 s | 48.47 s |
| Forecast cell, including loading and output saving | 36.9 s | 22.9 s |
| Eight model steps only, CUDA-synchronized | 3.21 s | 1.67 s |
| Evaluation, maps and animation | 38.6 s | 20.7 s |
| Peak PyTorch-allocated GPU memory | 1.003 GiB | 1.003 GiB |

The L4 host was a fresh `g2-standard-4:nvidia-l4:1` student Launchable instance
with 4 vCPUs and 16 GiB host memory. It used Python 3.12.11, PhysicsNeMo 2.2.2
and PyTorch 2.10.0+cu128. The full `nbconvert --execute` command took 89.62 s,
including kernel startup and notebook output writing; its maximum host RSS was
1.91 GiB. GPU allocation excludes driver overhead. Network preparation and
cached notebook timings are separate measurements, not a five-minute promise.
The first-download measurement includes about 301 MB of checkpoint files and
622 MB of compressed ERA5 source chunks, checksum checks and data conversion.

This is one new L4 instance, not a 110-user capacity test. Prepare caches before
the session and allow time for network variability. No artificial delay or
additional training was added to fill the lesson. Spend the remaining class
time comparing ERA5, forecast, persistence and errors.

## Publication checks

- Full regression suite: 1,539 passed, 32 skipped, 51 subtests passed. GPU tests
  were excluded from this CPU regression run; actual weather GPU execution is
  recorded separately above. Third-party deprecation warnings and one existing
  test-only tensor conversion warning remain in the test log.
- Static materials check: 141 Python files, 13 notebooks, 393 local links and
  45 preserved upstream assets; zero errors and warnings.
- Offline preparation tests cover pinned object checksums, corrupted caches,
  interrupted assembly, retries and model/data separation. All 49 pinned
  source objects were downloaded on L4 and verified.
- Independent review found no changes to the scientific code or recipes of
  Labs 1–3 or Challenges 1–4. Lab 3 and Challenge 1 have navigation-only edits.

## Published-version deployment rehearsal

After publishing implementation commit `8b5d8f1`, the normal command
`bash ~/AI4Sci-PhysicsNeMo-Bootcamp/ETC/launchable/update.sh` updated the fresh
student VM from `978fef3` successfully. It reused the course environment and
verified the model/data cache without restarting managed Jupyter or changing
its authentication. The actual Jupyter API confirmed the course root,
Start Here landing page, course-only kernel, TensorBoard proxy and new Lab 4.

The notebook from the GitHub-updated checkout then passed all nine code cells
again, with zero errors and all 26 forecast checks passing: 81.97 s for the code
cells, 85.73 s for the complete command, and 2.97 s for model computation alone.
These repeat timings illustrate ordinary run-to-run variation. The checkout
remained free of tracked learner changes; execution artifacts were kept separate.

## Retained reference and curriculum change

The instructor explicitly approved replacing the former Lab 4 on 29 September.
The original Navier–Stokes files, recipes, source array and ParaView resources
remain in [the reference folder](../reference_labs/04_navier_stokes/Lab_4_Navier_Stokes.ipynb).
Their earlier numerical results are not evidence for this AFNO lesson. Labs 1–3
and all Challenge problems, training recipes and scoring rules are unchanged.

## Sources and terms

- [NVIDIA FourCastNet model card](https://huggingface.co/nvidia/fourcastnet1): Apache-2.0.
- [Earth2Studio FCN wrapper](https://github.com/NVIDIA/earth2studio/blob/main/earth2studio/models/px/fcn.py).
- [Pinned Earth2Studio humidity derivation](https://github.com/NVIDIA/earth2studio/blob/486c5daa98841b0ce93cb78cb70d0dee7a8a56e6/earth2studio/models/dx/derived.py).
- [Google ARCO-ERA5](https://github.com/google-research/arco-era5), derived from
  Copernicus Climate Change Service / ECMWF ERA5; retain attribution and the
  [data terms](https://cds.climate.copernicus.eu/licences/cc-by).
- [Natural Earth coastline](https://github.com/nvkelso/natural-earth-vector/blob/master/geojson/ne_110m_coastline.geojson): public domain.
