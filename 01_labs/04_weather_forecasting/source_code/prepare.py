"""Prepare and verify the fixed weather lesson in a cache outside the checkout.

The first preparation downloads about 301 MB of official model files and
622 MB of compressed ERA5 source chunks. Subsequent runs verify and reuse them.
No CUDA, credentials, training, or future-weather input to the model is needed.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from uuid import uuid4

from download_model import FILES, REVISION, digest


SOURCE = Path(__file__).resolve().parent
ASSETS = SOURCE.parent / "assets"
COURSE = SOURCE.parents[2]
CASE = "data-20220901"
INITIAL_TIME = "2022-09-01T00:00:00Z"
PINNED_ASSETS = {
    "era5-source-plan.json": "667a15be025a6203500dc25477c241f05aaa232d78fc50a90e825bad51c6d7fb",
    "era5-source-metadata.json": "6656a7880b371b6ba1bbf9098442b202f32ffdfbdb65936ee50e8ee3994e8772",
}


def default_cache() -> Path:
    return Path(os.environ.get("AI4SCI_WEATHER_CACHE", "~/.cache/ai4sci/weather")).expanduser()


def external_cache(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path == COURSE or COURSE in path.parents:
        raise ValueError("The weather cache must be outside the course checkout so updates preserve it.")
    return path


def pinned_plan() -> dict:
    for name, expected in PINNED_ASSETS.items():
        if digest(ASSETS / name) != expected:
            raise ValueError(f"Pinned lesson metadata failed verification: {name}")
    plan = json.loads((ASSETS / "era5-source-plan.json").read_text())
    if plan["initial_time_utc"] != INITIAL_TIME:
        raise ValueError("Pinned lesson metadata contains an unexpected date")
    return plan


def verify_model(folder: Path) -> None:
    for name, (size, sha) in FILES.items():
        path = folder / name
        if not path.is_file() or path.stat().st_size != size or digest(path) != sha:
            raise ValueError(f"Missing or invalid pinned model artifact: {path}")


def verify_data(folder: Path, plan: dict) -> dict:
    provenance = json.loads((folder / "provenance.json").read_text())
    if provenance.get("initial_time_utc") != INITIAL_TIME or provenance.get("fcn_revision") != REVISION:
        raise ValueError("Cached weather data belong to a different date or model contract")
    if provenance.get("source_store") != "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3/":
        raise ValueError("Cached weather data have an unexpected source")
    expected_objects = {request["key"]: request["object_metadata"] for request in plan["requests"]}
    records = provenance["objects"]
    if len(records) != len(expected_objects) or {entry["key"] for entry in records} != set(expected_objects):
        raise ValueError("Cached weather data do not contain the complete pinned source manifest")
    for record in records:
        descriptor = expected_objects[record["key"]]
        for key in ("generation", "size", "md5Hash", "name"):
            if record["gcs"].get(key) != descriptor.get(key):
                raise ValueError(f"Cached source generation/checksum mismatch: {record['key']}")
    for name in ("initial.npz", "truth.npz"):
        path = folder / name
        expected = provenance["outputs"][name]
        if not path.is_file() or path.stat().st_size != expected["bytes"] or digest(path) != expected["sha256"]:
            raise ValueError(f"Missing or corrupted prepared weather file: {path}")
    return provenance


@contextmanager
def preparation_lock(cache: Path, wait_seconds: float = 600):
    """Prevent two notebook/installer processes from preparing the same cache."""
    cache.mkdir(parents=True, exist_ok=True)
    with (cache / ".prepare.lock").open("a+") as lock:
        started = time.monotonic()
        waiting = False
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if not waiting:
                    print("Another process is preparing this weather cache; waiting for verification.", flush=True)
                    waiting = True
                if time.monotonic() - started >= wait_seconds:
                    raise TimeoutError("Weather preparation is still running. Retry after it finishes.")
                time.sleep(1)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def run_stage(label: str, command: list[str], attempts: int = 3, before_attempt=None) -> None:
    for attempt in range(1, attempts + 1):
        if before_attempt is not None:
            before_attempt()
        print(f"{label}: attempt {attempt}/{attempts}", flush=True)
        try:
            subprocess.run(command, check=True)
            return
        except subprocess.CalledProcessError:
            if attempt == attempts:
                raise
            print("Retrying; completed verified files and source chunks will be reused.", flush=True)
            time.sleep(2 ** attempt)


def preserve_incomplete_assembly(folder: Path) -> None:
    """Keep interrupted generated NPZ files; leave reusable chunks untouched."""
    existing = [folder / name for name in ("initial.npz", "truth.npz", "provenance.json")
                if (folder / name).exists()]
    if existing:
        saved = folder / ("interrupted-assembly-" + uuid4().hex[:12])
        saved.mkdir()
        for path in existing:
            path.rename(saved / path.name)
        print(f"Preserved interrupted generated files in {saved}; rebuilding from verified chunks.", flush=True)


def prepare(cache: Path, *, offline=False, verify_only=False) -> dict:
    cache = external_cache(cache)
    plan = pinned_plan()
    if verify_only and not cache.is_dir():
        raise FileNotFoundError(f"Weather cache is not prepared: {cache}")
    with preparation_lock(cache):
        model_dir = cache / "model"
        if offline or verify_only:
            verify_model(model_dir)
        else:
            run_stage("Official model", [sys.executable, str(SOURCE / "download_model.py"),
                      "--output-dir", str(model_dir)])
            verify_model(model_dir)

        data_dir = cache / CASE
        if data_dir.exists():
            # Never silently replace a complete but corrupted cache.
            verify_data(data_dir, plan)
            print("Verified cached ERA5 initial state and independent future reference.", flush=True)
        elif verify_only:
            raise FileNotFoundError(f"Weather data are not prepared: {data_dir}")
        else:
            staging = cache / ("." + CASE + "-building")
            staging.mkdir(exist_ok=True)
            # A previous run may have completed assembly but been interrupted
            # before the atomic directory rename. Verify before reusing it.
            complete_staging = False
            if (staging / "provenance.json").exists():
                try:
                    verify_data(staging, plan)
                    complete_staging = True
                except (ValueError, KeyError, OSError):
                    pass
            if not complete_staging:
                preserve_incomplete_assembly(staging)
                command = [sys.executable, str(SOURCE / "download_era5.py"), "--output-dir", str(staging),
                           "--time", INITIAL_TIME, "--source-plan", str(ASSETS / "era5-source-plan.json"),
                           "--source-metadata", str(ASSETS / "era5-source-metadata.json")]
                if offline:
                    command.append("--offline")
                # Interrupted assembled files are preserved between retries;
                # verified source chunks are reused, never downloaded twice.
                run_stage("ERA5 preparation", command, attempts=1 if offline else 3,
                          before_attempt=lambda: preserve_incomplete_assembly(staging))
                verify_data(staging, plan)
            staging.rename(data_dir)
            print(f"Prepared and verified ERA5 data: {data_dir}", flush=True)
        return {"cache_dir": str(cache), "model_dir": str(model_dir),
                "initial": str(data_dir / "initial.npz"), "truth": str(data_dir / "truth.npz"),
                "model_revision": REVISION, "initial_time_utc": INITIAL_TIME,
                "future_reference_is_not_model_input": True}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=default_cache())
    parser.add_argument("--verify-only", action="store_true", help="Verify cached files without any downloads")
    parser.add_argument("--offline", action="store_true", help="Use only existing verified model/source chunks")
    args = parser.parse_args(argv)
    try:
        result = prepare(args.cache_dir, offline=args.offline, verify_only=args.verify_only)
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"Weather preparation stopped: {exc}\nExisting files were preserved. Retry after correcting the reported cause.\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
