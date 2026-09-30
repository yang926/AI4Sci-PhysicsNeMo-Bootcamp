"""Offline cache, retry and integrity tests for weather lesson preparation."""
import copy
import hashlib
import io
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import pytest

WEATHER_SOURCE = Path(__file__).resolve().parents[2] / "01_labs/04_weather_forecasting/source_code"

def load_weather_module(stem):
    spec = importlib.util.spec_from_file_location("weather_" + stem + "_under_test", WEATHER_SOURCE / (stem + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


download_model = load_weather_module("download_model")
download_era5 = load_weather_module("download_era5")
with patch.dict(sys.modules, {"download_model": download_model}):
    weather_prepare = load_weather_module("prepare")


def test_default_cache_is_external_and_configurable(monkeypatch, tmp_path):
    monkeypatch.delenv("AI4SCI_WEATHER_CACHE", raising=False)
    assert weather_prepare.default_cache() == Path.home() / ".cache/ai4sci/weather"
    monkeypatch.setenv("AI4SCI_WEATHER_CACHE", str(tmp_path))
    assert weather_prepare.default_cache() == tmp_path
    with pytest.raises(ValueError, match="outside"):
        weather_prepare.external_cache(WEATHER_SOURCE / "cache")


def test_pinned_source_asset_checksums_and_case():
    plan = weather_prepare.pinned_plan()
    assert plan["initial_time_utc"] == "2022-09-01T00:00:00Z"
    assert plan["weather_object_count"] == 45


def cached_data(tmp_path):
    plan = weather_prepare.pinned_plan()
    source = {"initial_time_utc": weather_prepare.INITIAL_TIME,
              "fcn_revision": weather_prepare.REVISION,
              "source_store": "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3/",
              "objects": [{"key": x["key"], "gcs": x["object_metadata"]} for x in plan["requests"]],
              "outputs": {}}
    for name in ("initial.npz", "truth.npz"):
        payload = ("small fake file " + name).encode()
        (tmp_path / name).write_bytes(payload)
        source["outputs"][name] = {"sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}
    (tmp_path / "provenance.json").write_text(json.dumps(source))
    return plan, source


def test_prepared_file_hashes_verified(tmp_path):
    plan, source = cached_data(tmp_path)
    assert weather_prepare.verify_data(tmp_path, plan)["outputs"] == source["outputs"]
    path = tmp_path / "truth.npz"
    path.write_bytes(b"x" * path.stat().st_size)
    with pytest.raises(ValueError, match="corrupted"):
        weather_prepare.verify_data(tmp_path, plan)


@pytest.mark.parametrize("defect", ["date", "model", "source", "generation", "md5", "missing", "duplicate"])
def test_cache_source_contract_cannot_drift(tmp_path, defect):
    plan, source = cached_data(tmp_path)
    if defect == "date":
        source["initial_time_utc"] = "2022-09-02T00:00:00Z"
    elif defect == "model":
        source["fcn_revision"] = "other"
    elif defect == "source":
        source["source_store"] = "other"
    elif defect == "generation":
        source["objects"][0]["gcs"]["generation"] = "other"
    elif defect == "md5":
        source["objects"][0]["gcs"]["md5Hash"] = "other"
    elif defect == "missing":
        source["objects"].pop()
    else:
        source["objects"][-1] = copy.deepcopy(source["objects"][0])
    (tmp_path / "provenance.json").write_text(json.dumps(source))
    # Re-read the trusted plan: the fixture shares descriptor dicts in memory.
    with pytest.raises(ValueError):
        weather_prepare.verify_data(tmp_path, weather_prepare.pinned_plan())


def test_retry_reuses_completed_files_and_runs_recovery(monkeypatch, tmp_path):
    calls, recovered = [], []
    def run(command, **kwargs):
        calls.append(command)
        if len(calls) < 3:
            raise subprocess.CalledProcessError(1, command)
    monkeypatch.setattr(weather_prepare.subprocess, "run", run)
    monkeypatch.setattr(weather_prepare.time, "sleep", lambda _: None)
    weather_prepare.run_stage("test", ["example"], before_attempt=lambda: recovered.append(True))
    assert len(calls) == len(recovered) == 3


def test_interrupted_assembly_preserved_without_touching_chunks(tmp_path):
    (tmp_path / "initial.npz").write_bytes(b"interrupted")
    (tmp_path / "chunk-cache").mkdir()
    (tmp_path / "chunk-cache" / "verified").write_bytes(b"keep")
    weather_prepare.preserve_incomplete_assembly(tmp_path)
    assert not (tmp_path / "initial.npz").exists()
    assert next(tmp_path.glob("interrupted-assembly-*/initial.npz")).read_bytes() == b"interrupted"
    assert (tmp_path / "chunk-cache" / "verified").read_bytes() == b"keep"


def test_verify_only_never_downloads(monkeypatch, tmp_path):
    monkeypatch.setattr(weather_prepare, "run_stage", lambda *a, **k: pytest.fail("network stage"))
    with pytest.raises(ValueError, match="artifact"):
        weather_prepare.prepare(tmp_path, verify_only=True)


def test_model_download_uses_external_target_and_reuses_verified_files(monkeypatch, tmp_path):
    payload = b"verified official fixture"
    monkeypatch.setattr(download_model, "FILES", {"model.bin": (len(payload), hashlib.sha256(payload).hexdigest())})
    calls = []
    def fetch(url, **kwargs):
        calls.append(url)
        return io.BytesIO(payload)
    monkeypatch.setattr(download_model.urllib.request, "urlopen", fetch)
    download_model.main(["--output-dir", str(tmp_path)])
    download_model.main(["--output-dir", str(tmp_path)])
    assert len(calls) == 1
    assert (tmp_path / "model.bin").read_bytes() == payload
    assert not list(tmp_path.glob("*.partial"))


def test_failed_model_download_cleans_own_partial_and_preserves_other_files(monkeypatch, tmp_path):
    monkeypatch.setattr(download_model, "FILES", {"model.bin": (5, "incorrect")})
    monkeypatch.setattr(download_model.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(b"short"))
    other = tmp_path / "previous.partial"
    other.write_bytes(b"preserve")
    with pytest.raises(ValueError, match="verification"):
        download_model.main(["--output-dir", str(tmp_path)])
    assert list(tmp_path.glob("*.partial")) == [other]
    assert not (tmp_path / "model.bin").exists()


def test_data_pinned_metadata_arguments_must_be_paired():
    with pytest.raises(SystemExit):
        download_era5.main(["--source-plan", "somewhere.json"])


@pytest.mark.skipif(not os.environ.get("AI4SCI_WEATHER_VALIDATION_CACHE"), reason="Optional full cached ERA5 reconstruction")
def test_actual_pinned_source_offline_reassembly(tmp_path):
    """Rebuild from existing source chunks, compare physical arrays, no network."""
    import numpy as np
    original = Path(os.environ["AI4SCI_WEATHER_VALIDATION_CACHE"])
    (tmp_path / "model").symlink_to(original / "model", target_is_directory=True)
    staging = tmp_path / ".data-20220901-building"
    staging.mkdir()
    (staging / "chunk-cache").symlink_to(original / "data-20220901/chunk-cache", target_is_directory=True)
    weather_prepare.prepare(tmp_path, offline=True)
    for name, keys in (("initial.npz", ("state", "time", "variables", "latitude", "longitude")),
                       ("truth.npz", ("reference", "times", "variables", "latitude", "longitude"))):
        with np.load(original / "data-20220901" / name, allow_pickle=False) as reference, \
             np.load(tmp_path / "data-20220901" / name, allow_pickle=False) as current:
            for key in keys:
                np.testing.assert_array_equal(current[key], reference[key])
