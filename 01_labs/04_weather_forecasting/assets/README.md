# Weather lesson sources and attribution

This lab uses a pretrained model for inference. It does not train a new model.
The fixed case starts on 1 September 2022 at 00:00 UTC and forecasts 48 hours.
The later ERA5 fields are independent verification data, not forecast inputs.

## Model and implementation

- **NVIDIA FourCastNet1 / AFNO**: the official 26-channel checkpoint, revision
  `c67a63995f6c8e0e557eb3d791f32f437e9b02d5` of
  [nvidia/fourcastnet1](https://huggingface.co/nvidia/fourcastnet1).
  The model card identifies the model as Apache-2.0. Checkpoint and normalization
  files are downloaded from that revision and checked against full SHA-256
  hashes before loading. Model files are not included in this repository.
- **NVIDIA Earth2Studio**: normalization and the six-hour recurrent forecast
  follow the [official FCN wrapper](https://github.com/NVIDIA/earth2studio/blob/main/earth2studio/models/px/fcn.py).
  The model predicts the complete next normalized state, not a state increment.
- **Relative humidity preprocessing**: the IFS mixed-phase formula is adapted
  from the [Earth2Studio DerivedRH implementation](https://github.com/NVIDIA/earth2studio/blob/486c5daa98841b0ce93cb78cb70d0dee7a8a56e6/earth2studio/models/dx/derived.py),
  pinned at `486c5daa98841b0ce93cb78cb70d0dee7a8a56e6`.
  Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. Apache-2.0;
  see the repository [license](../../../LICENSE). The local adaptation uses
  NumPy, accepts ERA5 temperature (K), specific humidity (kg/kg), and pressure
  (hPa), and stores the derived relative humidity as FP32 percent. It retains
  the official mixed-phase blend and final 0-100 percent RH bounds. It does not
  change the raw specific humidity or clip model forecasts. Diagnostics record
  the count and range of preprocessing values affected by RH bounds.

## ERA5 reanalysis

ERA5 is produced by the Copernicus Climate Change Service at ECMWF and is
accessed through [Google ARCO-ERA5](https://github.com/google-research/arco-era5).
Refer to the [ERA5 data licence](https://cds.climate.copernicus.eu/licences/cc-by)
and [ARCO citation guidance](https://github.com/google-research/arco-era5#how-to-cite-this-work).
The source is reanalysis, not direct observations or a live operational feed.

`era5-source-plan.json` fixes the exact GCS object generations and source checksums.
`era5-source-metadata.json` records the matching source dimensions and units.
These small metadata documents do not contain the weather arrays. Their SHA-256
hashes are checked by `prepare.py`. The source chunks are checked against GCS
MD5 and size, and prepared files against their recorded SHA-256 hashes.
The raw downloaded metadata SHA and the locally formatted metadata SHA are
recorded separately; JSON formatting does not change the underlying dataset.

Changes from the source fields are explicit: select the required channels and
pressure levels, remove the south-pole row to match the official 720-by-1440
grid, and derive relative humidity using the formula above. No interpolation,
forecast amplitude correction, future-state feedback, or spatial alignment is
used. Full source attribution and transformation diagnostics are saved in the
external cache at `data-20220901/provenance.json`.

## Geographic outlines

`coastline.geojson` is the Natural Earth 1:110m coastline, distributed as public
domain data. [Source file](https://github.com/nvkelso/natural-earth-vector/blob/master/geojson/ne_110m_coastline.geojson)
and [Natural Earth terms](https://www.naturalearthdata.com/about/terms-of-use/).
SHA-256: `851f581ff5ffb844deed8ae1a9ce22e3c4bb3d74fa342cadb5d8e39b41ae7c3c`.
Generated maps identify Natural Earth in their attribution.

## Validation scope

`weather-acceptance.json` contains criteria declared before evaluating this
case. Forecast scores use physical units and cosine-latitude area weights.
The initial frame is excluded from forecast error metrics. All eight positive
lead times are reported, with the initial-state persistence forecast as a
baseline. Passing one historical case is not evidence of operational weather
skill or of runtime on a different GPU.
