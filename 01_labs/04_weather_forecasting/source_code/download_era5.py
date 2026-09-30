#!/usr/bin/env python3
"""Bounded anonymous ARCO-ERA5 download for one prespecified FCN26 hindcast.

No training, regridding, future-input mixing, or fabricated weather fields.
RH is explicitly derived from ERA5 q/T/p using NVIDIA's corrected IFS method.
Only NumPy and numcodecs are required besides the Python standard library.
"""
# The relative-humidity formula is adapted from NVIDIA Earth2Studio.
# Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0.
# Unless required by applicable law or agreed to in writing, software is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied. See the License for the specific language
# governing permissions and limitations under the License.
from __future__ import annotations

import argparse
import base64
import datetime as dt
import hashlib
import json
from pathlib import Path
import tempfile
import urllib.parse
import urllib.request

import numcodecs
import numpy as np


BUCKET = "gcp-public-data-arco-era5"
PREFIX = "ar/full_37-1h-0p25deg-chunk-1.zarr-v3/"
BASE_URL = f"https://storage.googleapis.com/{BUCKET}/{PREFIX}"
DEFAULT_TIME = "2022-09-01T00:00:00Z"
HARD_BYTE_LIMIT = 2_000_000_000
FCN_REVISION = "c67a63995f6c8e0e557eb3d791f32f437e9b02d5"
RH_REVISION = "486c5daa98841b0ce93cb78cb70d0dee7a8a56e6"
RH_SOURCE = f"https://github.com/NVIDIA/earth2studio/blob/{RH_REVISION}/earth2studio/models/dx/derived.py"
MODEL_VARIABLES = (
    "u10m", "v10m", "t2m", "sp", "msl", "t850", "u1000", "v1000",
    "z1000", "u850", "v850", "z850", "u500", "v500", "z500", "t500",
    "z50", "r500", "r850", "tcwv", "u100m", "v100m", "u250", "v250",
    "z250", "t250",
)
TRUTH_VARIABLES = ("u10m", "v10m", "msl", "t2m")
SURFACE = {
    "u10m": "10m_u_component_of_wind", "v10m": "10m_v_component_of_wind",
    "t2m": "2m_temperature", "sp": "surface_pressure",
    "msl": "mean_sea_level_pressure", "tcwv": "total_column_water_vapour",
    "u100m": "100m_u_component_of_wind", "v100m": "100m_v_component_of_wind",
}
PRESSURE = {"t": "temperature", "u": "u_component_of_wind",
            "v": "v_component_of_wind", "z": "geopotential", "q": "specific_humidity"}
SOURCE_UNITS = {
    **{name: "m s**-1" for key, name in SURFACE.items() if key.startswith(("u", "v"))},
    "2m_temperature": "K", "surface_pressure": "Pa", "mean_sea_level_pressure": "Pa",
    "total_column_water_vapour": "kg m**-2", "temperature": "K",
    "u_component_of_wind": "m s**-1", "v_component_of_wind": "m s**-1",
    "geopotential": "m**2 s**-2", "specific_humidity": "kg kg**-1",
}


def parse_time(value: str) -> dt.datetime:
    """Parse an exact UTC hour; reject ambiguous offsets and subhour times."""
    result = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is not None and result.utcoffset() != dt.timedelta(0):
        raise ValueError("time must use UTC")
    result = result.replace(tzinfo=None)
    if result.minute or result.second or result.microsecond:
        raise ValueError("time must be an exact hour")
    return result


def hour_index(value: dt.datetime) -> int:
    return int((value - dt.datetime(1900, 1, 1)).total_seconds() // 3600)


def relative_humidity_percent(temperature, specific_humidity, pressure_hpa, return_diagnostics=False):
    """Corrected Earth2Studio DerivedRH, evaluated in float64 then stored FP32.

    T is kelvin, q is kg/kg (not the erroneous g/kg comment in upstream),
    pressure is hPa. IFS mixed-phase saturation uses alpha clipped to 1, not 1.2.
    Mathematical formula adapted from NVIDIA Earth2Studio, Apache-2.0;
    copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. See RH_SOURCE.
    """
    t, q, p = np.broadcast_arrays(np.asarray(temperature, np.float64),
                                np.asarray(specific_humidity, np.float64),
                                np.asarray(pressure_hpa, np.float64) * 100.0)
    if not all(np.isfinite(x).all() for x in (t, q, p)):
        raise ValueError("RH inputs must be finite")
    epsilon = 0.621981
    denominator = epsilon + (1.0 - epsilon) * q
    if np.any(t <= 0) or np.any(p <= 0) or np.any(q >= 1) or np.any(denominator <= 0):
        raise ValueError("RH requires positive temperature/pressure/denominator and q < 1")
    # Keep raw signed q unchanged, as the upstream implementation does. Tiny
    # negative reanalysis values are handled only by its final RH clipping.
    e = p * q / denominator
    es_water = 611.21 * np.exp(17.502 * (t - 273.16) / (t - 32.19))
    es_ice = 611.21 * np.exp(22.587 * (t - 273.16) / (t + 0.7))
    alpha = np.clip((t - 250.16) / (273.16 - 250.16), 0.0, 1.0) ** 2
    es = alpha * es_water + (1.0 - alpha) * es_ice
    raw_rh = 100.0 * e / es
    result = np.clip(raw_rh, 0.0, 100.0)
    if not np.isfinite(result).all():
        raise ValueError("invalid RH result")
    result = result.astype(np.float32)
    if not return_diagnostics:
        return result
    diagnostics = {
        "raw_q_min_kgkg": float(q.min()), "raw_q_max_kgkg": float(q.max()),
        "raw_q_negative_count": int(np.count_nonzero(q < 0)),
        "raw_q_negative_minmax": [float(q[q < 0].min()), float(q[q < 0].max())] if np.any(q < 0) else None,
        "unclipped_rh_min_percent": float(raw_rh.min()), "unclipped_rh_max_percent": float(raw_rh.max()),
        "lower_clipped_count": int(np.count_nonzero(raw_rh < 0)),
        "upper_clipped_count": int(np.count_nonzero(raw_rh > 100)),
        "lower_clipped_raw_rh_minmax": [float(raw_rh[raw_rh < 0].min()), float(raw_rh[raw_rh < 0].max())] if np.any(raw_rh < 0) else None,
        "upper_clipped_raw_rh_minmax": [float(raw_rh[raw_rh > 100].min()), float(raw_rh[raw_rh > 100].max())] if np.any(raw_rh > 100) else None,
        "raw_q_modified": False,
    }
    return result, diagnostics


def decode_chunk(payload: bytes, metadata: dict) -> np.ndarray:
    """Decode only the simple, bounded Zarr2 chunks used by this archive."""
    if metadata.get("zarr_format") != 2 or metadata.get("order") != "C":
        raise ValueError("only C-ordered Zarr2 arrays are supported")
    if metadata.get("filters") not in (None, []):
        raise ValueError("unexpected Zarr filters")
    dtype = np.dtype(metadata["dtype"])
    if dtype.kind not in "fiu" or dtype.itemsize > 8:
        raise ValueError("unexpected nonnumeric Zarr dtype")
    chunks = tuple(metadata["chunks"])
    if not chunks or any(not isinstance(x, int) or x <= 0 for x in chunks):
        raise ValueError("invalid chunk dimensions")
    expected = int(np.prod(chunks, dtype=np.int64)) * dtype.itemsize
    if expected <= 0 or expected > 200_000_000:
        raise ValueError("decoded chunk exceeds bounded memory limit")
    compressor = metadata.get("compressor")
    if compressor is not None and compressor.get("id") != "blosc":
        raise ValueError("unexpected chunk compressor")
    decoded = numcodecs.get_codec(compressor).decode(payload) if compressor else payload
    if len(decoded) != expected:
        raise ValueError(f"decoded byte size mismatch: {len(decoded)} != {expected}")
    result = np.frombuffer(decoded, dtype=dtype).reshape(chunks)
    if not np.isfinite(result).all():
        raise ValueError("nonfinite ERA5 chunk")
    return result


def validate_weather_metadata(meta: dict, variable: str) -> None:
    array = meta[variable + "/.zarray"]
    attrs = meta[variable + "/.zattrs"]
    dims = attrs["_ARRAY_DIMENSIONS"]
    if len(dims) != len(set(dims)):
        raise ValueError("duplicate array dimensions")
    pressure = variable in PRESSURE.values()
    expected_dims = {"time", "latitude", "longitude"} | ({"level"} if pressure else set())
    if set(dims) != expected_dims or len(dims) != len(array["shape"]):
        raise ValueError("unexpected weather dimensions")
    sizes = dict(zip(dims, array["shape"]))
    chunks = dict(zip(dims, array["chunks"]))
    expected_chunks = {"time": 1, "latitude": 721, "longitude": 1440}
    if pressure:
        expected_chunks["level"] = 37
    if chunks != expected_chunks or any(sizes[k] != v for k, v in expected_chunks.items() if k != "time"):
        raise ValueError("unexpected weather chunk shape")
    if np.dtype(array["dtype"]) != np.dtype("<f4"):
        raise ValueError("weather dtype must be float32")
    if attrs.get("units") != SOURCE_UNITS[variable]:
        raise ValueError("unexpected or missing source units")
    if array.get("zarr_format") != 2 or array.get("order") != "C":
        raise ValueError("unsupported weather storage")


def canonical_field(chunk, dimensions, level=None, levels=None, drop_south_pole=True):
    """Select by named dimensions; only remove the final south-pole row."""
    dims = list(dimensions)
    result = np.asarray(chunk)
    if len(dims) != result.ndim or len(dims) != len(set(dims)):
        raise ValueError("array dimension mismatch")
    if "time" not in dims or result.shape[dims.index("time")] != 1:
        raise ValueError("expected one time per weather chunk")
    result = np.take(result, 0, axis=dims.index("time"))
    dims.remove("time")
    if "level" in dims:
        if level is None or levels is None:
            raise ValueError("an exact pressure level is required")
        matches = np.flatnonzero(np.asarray(levels) == level)
        if len(matches) != 1:
            raise ValueError("missing or duplicate pressure level")
        result = np.take(result, int(matches[0]), axis=dims.index("level"))
        dims.remove("level")
    elif level is not None:
        raise ValueError("level requested for surface array")
    if set(dims) != {"latitude", "longitude"}:
        raise ValueError("unexpected field dimensions")
    result = np.transpose(result, [dims.index("latitude"), dims.index("longitude")])
    if drop_south_pole:
        result = result[:-1]
    if result.size == 0 or not np.isfinite(result).all():
        raise ValueError("empty or nonfinite field")
    return np.array(result, dtype=np.float32, copy=True)


def build_requests(meta: dict, time: dt.datetime) -> list[dict]:
    """Only 13 initial objects and 32 future verification surface objects."""
    base_index = hour_index(time)
    initial_sources = list(SURFACE.values()) + list(PRESSURE.values())
    requests = []
    for lead in range(0, 49, 6):
        variables = initial_sources if lead == 0 else [SURFACE[x] for x in TRUTH_VARIABLES]
        for variable in variables:
            validate_weather_metadata(meta, variable)
            attrs, array = meta[variable + "/.zattrs"], meta[variable + "/.zarray"]
            dims = attrs["_ARRAY_DIMENSIONS"]
            if not 0 <= base_index + lead < array["shape"][dims.index("time")]:
                raise ValueError("requested time is outside array shape")
            coordinates = [base_index + lead if dim == "time" else 0 for dim in dims]
            requests.append({"variable": variable, "key": variable + "/" + ".".join(map(str, coordinates)),
                             "lead_hours": lead, "purpose": "initial" if lead == 0 else "truth"})
    return requests


def verify_file(path: Path, object_metadata: dict) -> dict:
    path = Path(path)
    if path.stat().st_size != int(object_metadata["size"]):
        raise ValueError(f"compressed size mismatch: {path}")
    digest, md5 = hashlib.sha256(), hashlib.md5()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
            md5.update(block)
    encoded_md5 = base64.b64encode(md5.digest()).decode("ascii")
    if not object_metadata.get("md5Hash") or encoded_md5 != object_metadata["md5Hash"]:
        raise ValueError(f"GCS MD5 mismatch or missing checksum: {path}")
    return {"sha256": digest.hexdigest(), "size": path.stat().st_size, "md5": encoded_md5}


class PublicStore:
    def __init__(self, cache: Path, byte_limit: int, offline=False):
        if not 0 < byte_limit <= HARD_BYTE_LIMIT:
            raise ValueError("download budget must be positive and at most 2 GB")
        self.cache, self.byte_limit, self.downloaded_bytes = Path(cache), byte_limit, 0
        self.records = []
        self.offline = offline

    def read_small(self, url: str, limit=2_000_000) -> bytes:
        if self.offline:
            raise RuntimeError("network access is disabled in offline mode")
        with urllib.request.urlopen(url, timeout=90) as response:
            payload = response.read(min(limit, self.byte_limit - self.downloaded_bytes) + 1)
        self.downloaded_bytes += len(payload)
        if len(payload) > limit or self.downloaded_bytes > self.byte_limit:
            raise ValueError("metadata/download byte budget exceeded")
        return payload

    def object_metadata(self, key: str) -> dict:
        quoted = urllib.parse.quote(PREFIX + key, safe="")
        url = f"https://storage.googleapis.com/storage/v1/b/{BUCKET}/o/{quoted}?fields=name,size,md5Hash,crc32c,generation"
        document = json.loads(self.read_small(url, limit=16_384))
        if document.get("name") != PREFIX + key or not document.get("md5Hash"):
            raise ValueError("object name/checksum metadata mismatch")
        return document

    def fetch(self, key: str, descriptor: dict) -> Path:
        if key.startswith("/") or ".." in Path(key).parts:
            raise ValueError("unsafe cache key")
        path = self.cache / key
        size = int(descriptor["size"])
        reused = path.exists()
        # Pin the object generation between metadata and payload GET.
        url = BASE_URL + key + "?generation=" + str(descriptor["generation"])
        if not reused:
            if self.offline:
                raise FileNotFoundError(f"missing verified offline cache object: {path}")
            if self.downloaded_bytes + size > self.byte_limit:
                raise ValueError("payload would exceed total download budget")
            path.parent.mkdir(parents=True, exist_ok=True)
            partial = None
            try:
                with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", suffix=".partial", delete=False) as stream:
                    partial = Path(stream.name)
                    count = 0
                    with urllib.request.urlopen(url, timeout=120) as response:
                        while block := response.read(min(4 * 1024 * 1024, size - count + 1)):
                            count += len(block)
                            self.downloaded_bytes += len(block)
                            if count > size or self.downloaded_bytes > self.byte_limit:
                                raise ValueError("download exceeded declared object size or budget")
                            stream.write(block)
                verify_file(partial, descriptor)
                partial.rename(path)
            finally:
                if partial is not None:
                    partial.unlink(missing_ok=True)
        verified = verify_file(path, descriptor)
        self.records.append({"key": key, "url": url, "cache_path": str(path.resolve()),
                             "reused_cache": reused, "gcs": descriptor, **verified})
        return path


def units_for(variable: str) -> str:
    if variable.startswith("r"):
        return "%"
    name = SURFACE.get(variable) or PRESSURE[variable[0]]
    return SOURCE_UNITS[name]


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--time", default=DEFAULT_TIME)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "data-20220901")
    parser.add_argument("--max-download-bytes", type=int, default=HARD_BYTE_LIMIT)
    parser.add_argument("--plan-only", action="store_true", help="metadata requests only; no chunk payloads")
    parser.add_argument("--offline", action="store_true", help="reassemble only from existing metadata and verified chunk cache; no network")
    parser.add_argument("--source-plan", type=Path, help="Pinned request descriptors, including GCS generations and checksums")
    parser.add_argument("--source-metadata", type=Path, help="Matching pinned Zarr metadata; required with --source-plan")
    args = parser.parse_args(argv)
    if bool(args.source_plan) != bool(args.source_metadata):
        parser.error("--source-plan and --source-metadata must be supplied together")
    init_time = parse_time(args.time)
    output = args.output_dir.resolve()
    if not args.plan_only and any((output / name).exists() for name in ("initial.npz", "truth.npz")):
        raise FileExistsError("preserving existing output NPZs; choose a fresh output directory")
    output.mkdir(parents=True, exist_ok=True)
    store = PublicStore(output / "chunk-cache", args.max_download_bytes, offline=args.offline)
    pinned = args.source_plan is not None
    previous_plan = (json.loads(args.source_plan.read_text()) if pinned else
                     json.loads((output / "download-plan.json").read_text()) if args.offline else None)
    metadata_bytes = (args.source_metadata.read_bytes() if pinned else
                      (output / "source-zmetadata.json").read_bytes() if args.offline else
                      store.read_small(BASE_URL + ".zmetadata"))
    metadata_document = json.loads(metadata_bytes)
    meta = metadata_document["metadata"]
    if metadata_document.get("zarr_consolidated_format") != 1:
        raise ValueError("unexpected consolidated format")
    start = parse_time(meta[".zattrs"]["valid_time_start"])
    stop = parse_time(meta[".zattrs"]["valid_time_stop"]) + dt.timedelta(days=1)
    if not start <= init_time or not init_time + dt.timedelta(hours=48) < stop:
        raise ValueError("requested times are outside finalized ERA5 coverage")
    if meta["time/.zattrs"].get("units") != "hours since 1900-01-01 00:00:00":
        raise ValueError("unexpected time epoch")
    requests = build_requests(meta, init_time)
    coordinate_requests = [{"key": name + "/0", "variable": name} for name in ("latitude", "longitude", "level")]
    time_chunk_length = meta["time/.zarray"]["chunks"][0]
    time_chunk_indices = sorted({hour_index(init_time + dt.timedelta(hours=lead)) // time_chunk_length for lead in range(0, 49, 6)})
    coordinate_requests += [{"key": f"time/{index}", "variable": "time"} for index in time_chunk_indices]
    all_requests = coordinate_requests + requests
    if args.offline or pinned:
        previous_requests = {request["key"]: request for request in previous_plan["requests"]}
        if set(previous_requests) != {request["key"] for request in all_requests}:
            raise ValueError("pinned/offline plan does not match the requested case")
        for request in all_requests:
            request["object_metadata"] = previous_requests[request["key"]]["object_metadata"]
    else:
        for request in all_requests:
            request["object_metadata"] = store.object_metadata(request["key"])
    planned_bytes = sum(int(request["object_metadata"]["size"]) for request in all_requests)
    if planned_bytes + store.downloaded_bytes > store.byte_limit:
        raise ValueError("complete request exceeds the configured byte budget")
    plan = {"initial_time_utc": init_time.isoformat() + "Z", "metadata_url": BASE_URL + ".zmetadata",
            "metadata_sha256": previous_plan["metadata_sha256"] if args.offline or pinned else hashlib.sha256(metadata_bytes).hexdigest(),
            "local_metadata_sha256": hashlib.sha256(metadata_bytes).hexdigest(),
            "planned_compressed_chunk_bytes": planned_bytes, "weather_object_count": len(requests),
            "requests": all_requests, "max_download_bytes": store.byte_limit}
    write_json(output / "download-plan.json", plan)
    write_json(output / "source-zmetadata.json", metadata_document)
    print(json.dumps({"planned_compressed_MiB": planned_bytes / 2**20,
                      "weather_chunks": len(requests), "coordinate_chunks": len(coordinate_requests)}), flush=True)
    if args.plan_only:
        return 0

    coordinates = {}
    for request in coordinate_requests:
        path = store.fetch(request["key"], request["object_metadata"])
        data = decode_chunk(path.read_bytes(), meta[request["variable"] + "/.zarray"])
        coordinates[request["key"]] = data
    latitude, longitude, levels = (coordinates[f"{name}/0"] for name in ("latitude", "longitude", "level"))
    if not np.array_equal(latitude, np.linspace(90, -90, 721)) or not np.array_equal(longitude, np.arange(1440) / 4):
        raise ValueError("grid does not exactly match FCN; no remapping is permitted")
    expected_levels = np.array([1, 2, 3, 5, 7, 10, 20, 30, 50, 70, 100, 125, 150, 175, 200, 225, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 775, 800, 825, 850, 875, 900, 925, 950, 975, 1000])
    if not np.array_equal(levels, expected_levels):
        raise ValueError("unexpected pressure-level coordinates")
    for lead in range(0, 49, 6):
        index = hour_index(init_time + dt.timedelta(hours=lead))
        if coordinates[f"time/{index // time_chunk_length}"][index % time_chunk_length] != index:
            raise ValueError("time coordinate does not match requested UTC hour")

    initial_fields = {}
    truth = np.empty((9, 4, 720, 1440), dtype=np.float32)
    rh_ingredients = {}
    for i, request in enumerate(requests):
        variable, lead = request["variable"], request["lead_hours"]
        path = store.fetch(request["key"], request["object_metadata"])
        chunk = decode_chunk(path.read_bytes(), meta[variable + "/.zarray"])
        dims = meta[variable + "/.zattrs"]["_ARRAY_DIMENSIONS"]
        if lead == 0:
            for channel in MODEL_VARIABLES:
                if channel.startswith("r"):
                    continue
                source = SURFACE.get(channel) or PRESSURE[channel[0]]
                if variable == source:
                    level = None if channel in SURFACE else int(channel[1:])
                    initial_fields[channel] = canonical_field(chunk, dims, level, levels)
            if variable in ("temperature", "specific_humidity"):
                for level in (500, 850):
                    rh_ingredients[(variable, level)] = canonical_field(chunk, dims, level, levels)
        else:
            channel = next(channel for channel in TRUTH_VARIABLES if SURFACE[channel] == variable)
            truth[lead // 6, TRUTH_VARIABLES.index(channel)] = canonical_field(chunk, dims)
        print(f"verified {i + 1}/{len(requests)} {variable} +{lead}h; network {store.downloaded_bytes / 2**20:.1f} MiB", flush=True)
        del chunk
    rh_diagnostics = {}
    for level in (500, 850):
        initial_fields[f"r{level}"], rh_diagnostics[f"r{level}"] = relative_humidity_percent(
            rh_ingredients[("temperature", level)], rh_ingredients[("specific_humidity", level)],
            level, return_diagnostics=True)
    state = np.stack([initial_fields[channel] for channel in MODEL_VARIABLES])
    for i, channel in enumerate(TRUTH_VARIABLES):
        truth[0, i] = state[MODEL_VARIABLES.index(channel)]
    if state.shape != (26, 720, 1440) or not np.isfinite(state).all() or not np.isfinite(truth).all():
        raise ValueError("invalid completed output fields")
    times = np.array([np.datetime64(init_time + dt.timedelta(hours=lead), "h") for lead in range(0, 49, 6)])
    provenance = {
        "data_kind": "ERA5_reanalysis_with_explicit_IFS_derived_RH",
        "source_store": f"gs://{BUCKET}/{PREFIX}", "metadata_url": BASE_URL + ".zmetadata",
        "metadata_sha256": plan["metadata_sha256"], "metadata_group_attrs": meta[".zattrs"],
        "initial_time_utc": init_time.isoformat() + "Z", "fcn_revision": FCN_REVISION,
        "initial_model_variables": list(MODEL_VARIABLES), "truth_variables": list(TRUTH_VARIABLES),
        "weather_object_count": len(requests), "planned_compressed_chunk_bytes": planned_bytes,
        "actual_network_bytes_including_metadata": store.downloaded_bytes,
        "offline_assembly": args.offline,
        "pinned_source_plan": pinned,
        "cached_compressed_chunk_bytes": sum(record["size"] for record in store.records),
        "grid_operation": "exact original 0.25-degree grid; remove only latitude=-90 row; no interpolation",
        "rh_derivation": {"source": RH_SOURCE, "commit": RH_REVISION, "inputs": "ERA5 T(K), q(kg/kg), p(hPa)",
                          "method": "IFS mixed water/ice saturation; alpha=clip((T-250.16)/23,0,1)^2; RH clipped 0..100 percent",
                          "not_direct_archived_RH": True, "float64_calculation_then_float32_storage": True,
                          "diagnostics": rh_diagnostics, "forecast_output_clipping": "none; preprocessing only"},
        "scope": "One prespecified retrospective case; reanalysis is not direct observation or operational forecast verification.",
        "attribution": "ERA5 from the Copernicus Climate Change Service at ECMWF, distributed via Google ARCO-ERA5; modified by explicit RH derivation and south-pole crop.",
        "license_sources": ["https://cds.climate.copernicus.eu/licences/cc-by", "https://github.com/google-research/arco-era5#how-to-cite-this-work"],
        "objects": store.records,
    }
    metadata_string = np.array(json.dumps({key: value for key, value in provenance.items() if key != "objects"}, allow_nan=False))
    np.savez(output / "initial.npz", state=state, variables=np.array(MODEL_VARIABLES),
             time=times[0], latitude=latitude[:-1], longitude=longitude,
             units=np.array([units_for(v) for v in MODEL_VARIABLES]), provenance_json=metadata_string)
    np.savez(output / "truth.npz", reference=truth, variables=np.array(TRUTH_VARIABLES),
             times=times, lead_hours=np.arange(0, 49, 6), latitude=latitude[:-1], longitude=longitude,
             units=np.array([units_for(v) for v in TRUTH_VARIABLES]), provenance_json=metadata_string)
    provenance["outputs"] = {}
    for name in ("initial.npz", "truth.npz"):
        path = output / name
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        provenance["outputs"][name] = {"sha256": digest, "bytes": path.stat().st_size}
    provenance["channel_ranges"] = {channel: {"min": float(state[i].min()), "max": float(state[i].max()), "mean": float(state[i].mean())} for i, channel in enumerate(MODEL_VARIABLES)}
    write_json(output / "provenance.json", provenance)
    print(json.dumps({"initial": str(output / "initial.npz"), "truth": str(output / "truth.npz"),
                      "actual_network_MiB": store.downloaded_bytes / 2**20}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
