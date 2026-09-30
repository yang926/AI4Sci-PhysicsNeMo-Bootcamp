"""Inference from an explicitly reviewed Lab 4 original-data checkpoint.

Approval is an instructor decision recorded in source_code/conf/instructor_model.json.
Completing a training run never changes that decision. This module creates no
optimizer, training history, synthetic targets, or future-reference labels.
"""

import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
import time
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

import numpy as np
import torch

from ETC.runtime.artifacts import staged_output
from ETC.runtime.flow_diagnostics import summarize_flow


_ARCHITECTURE_SHAPE = (256, 6)
_BATCH_SIZE = 2048
_MAX_CHECKPOINT_BYTES = 100 * 1024 * 1024
_RELEASE_PREFIX = "/yang926/AI4Sci-PhysicsNeMo-Bootcamp/releases/download/"
_RECIPES = {"upstream_adam_fp32_v1", "multiscale_adam_fp32_v1"}


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _check_hash(path, expected, label):
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    if _sha256(path) != expected:
        raise ValueError(f"{label} SHA-256 does not match the approved manifest")


def _check_checkpoint(path, expected):
    if path.is_file() and path.stat().st_size > _MAX_CHECKPOINT_BYTES:
        raise ValueError("Instructor checkpoint exceeds the size limit")
    _check_hash(path, expected, "Instructor checkpoint")


def _allowed_download_url(url, *, redirect=False):
    if not isinstance(url, str):
        raise ValueError("download_url must be an official course GitHub release HTTPS URL")
    parsed = urlsplit(url)
    if (parsed.scheme != "https" or parsed.username or parsed.password
            or parsed.port not in (None, 443) or parsed.fragment):
        raise ValueError("Checkpoint downloads require trusted HTTPS URLs")
    course_release = (parsed.hostname == "github.com" and not parsed.query
                      and parsed.path.startswith(_RELEASE_PREFIX)
                      and len(parsed.path[len(_RELEASE_PREFIX):].split("/")) == 2
                      and all(parsed.path[len(_RELEASE_PREFIX):].split("/")))
    asset_redirect = redirect and parsed.hostname in {
        "release-assets.githubusercontent.com", "objects.githubusercontent.com"}
    if not (course_release or asset_redirect):
        raise ValueError("download_url must be an official course GitHub release HTTPS URL")


class _ReleaseRedirects(HTTPRedirectHandler):
    def redirect_request(self, request, response, code, message, headers, newurl):
        _allowed_download_url(newurl, redirect=True)
        return super().redirect_request(request, response, code, message, headers, newurl)


def _manifest(lab_dir):
    lab = Path(lab_dir).resolve()
    path = lab / "source_code/conf/instructor_model.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if (not isinstance(manifest, dict) or not isinstance(manifest.get("status"), str)
            or manifest["status"] not in {"pending_review", "approved"}):
        raise ValueError("Instructor manifest status must be pending_review or approved")
    if manifest["status"] == "pending_review":
        return lab, manifest, None
    if not isinstance(manifest.get("review_note"), str) or not manifest["review_note"].strip():
        raise ValueError("An approved instructor checkpoint needs a nonempty review_note")
    if type(manifest.get("training_steps")) is not int or manifest["training_steps"] <= 0:
        raise ValueError("training_steps must be a positive integer")
    for key in ("sha256", "source_data_sha256"):
        if not isinstance(manifest.get(key), str) or not re.fullmatch("[0-9a-f]{64}", manifest[key]):
            raise ValueError(f"{key} must be a lowercase SHA-256 digest")
    relative = manifest.get("checkpoint")
    if not isinstance(relative, str) or not relative.strip():
        raise ValueError("An approved manifest needs a relative checkpoint path")
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("checkpoint must stay within the Lab directory")
    checkpoint = (lab / relative).resolve()
    if not checkpoint.is_relative_to(lab) or checkpoint == lab:
        raise ValueError("checkpoint must stay within the Lab directory")
    if "download_url" in manifest:
        _allowed_download_url(manifest["download_url"])
        cache = lab / "instructor_models"
        if (relative.parent != Path("instructor_models") or cache.is_symlink()
                or checkpoint.parent != cache):
            raise ValueError("Downloaded checkpoints must use instructor_models/<filename> inside the Lab")
    return lab, manifest, checkpoint


def instructor_model_status(lab_dir):
    """Return status/message for a notebook, without downloading or loading weights.

    approved means the manifest authorizes inference; it is not a claim of
    forecast accuracy. Missing local weights with a valid download_url remain
    approved and will be fetched when run_instructor_inference is called.
    """
    try:
        lab, manifest, checkpoint = _manifest(lab_dir)
        if manifest["status"] == "pending_review":
            return {"status": "pending_review", "message":
                    "Instructor model is pending review. No approved checkpoint is available; "
                    "instructor inference is skipped."}
        _check_hash(lab / "data_lat.npy", manifest["source_data_sha256"], "Original input data")
        if checkpoint.exists() or checkpoint.is_symlink():
            _check_checkpoint(checkpoint, manifest["sha256"])
            message = "Reviewed instructor checkpoint is available for inference."
        elif "download_url" in manifest:
            message = "Reviewed instructor checkpoint will download on the first inference run."
        else:
            raise FileNotFoundError(f"Approved instructor checkpoint is missing: {checkpoint}")
        return {"status": "approved", "message":
                f"{message} Training steps: {manifest['training_steps']}. "
                f"Instructor review: {manifest['review_note']}"}
    except FileNotFoundError as error:
        return {"status": "unavailable", "message": str(error)}
    except (ValueError, OSError) as error:
        return {"status": "invalid", "message": str(error)}


def _download_checkpoint(checkpoint, manifest):
    """Fetch a bounded release asset, then publish verified bytes exclusively."""
    _allowed_download_url(manifest["download_url"])
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    opener = build_opener(_ReleaseRedirects())
    request = Request(manifest["download_url"], headers={"User-Agent": "AI4Sci-Lab4-Inference"})
    temporary = None
    try:
        with opener.open(request, timeout=30) as response:
            _allowed_download_url(response.geturl(), redirect=True)
            length = response.headers.get("Content-Length")
            if length is not None and (int(length) < 0 or int(length) > _MAX_CHECKPOINT_BYTES):
                raise ValueError("Instructor checkpoint download exceeds the size limit")
            digest, total = hashlib.sha256(), 0
            with tempfile.NamedTemporaryFile(prefix=".checkpoint-", dir=checkpoint.parent,
                                             delete=False) as stream:
                temporary = Path(stream.name)
                while block := response.read(1024 * 1024):
                    total += len(block)
                    if total > _MAX_CHECKPOINT_BYTES:
                        raise ValueError("Instructor checkpoint download exceeds the size limit")
                    digest.update(block)
                    stream.write(block)
            if digest.hexdigest() != manifest["sha256"]:
                raise ValueError("Downloaded instructor checkpoint SHA-256 does not match the approved manifest")
        try:
            # Hard-link publication is atomic and cannot replace an existing file.
            os.link(temporary, checkpoint)
        except FileExistsError:
            _check_checkpoint(checkpoint, manifest["sha256"])
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _checkpoint_config(saved, manifest):
    if not isinstance(saved, dict) or not isinstance(saved.get("config"), dict):
        raise ValueError("Instructor checkpoint must contain model_state_dict and config")
    cfg = saved["config"]
    base = {"steps", "batch_size", "learning_rate", "layer_size", "num_layers",
            "lab4_dtype", "lab4_recipe", "lab4_architecture"}
    scaling = {"lab4_feature_bands", "lab4_field_center", "lab4_field_scale"}
    if (not isinstance(cfg.get("lab4_recipe"), str) or cfg["lab4_recipe"] not in _RECIPES
            or cfg.get("lab4_dtype") != "float32"
            or cfg.get("lab4_architecture") != "upstream_silu_weight_norm"):
        raise ValueError("Instructor checkpoint must use a supported original-data FP32 recipe")
    efficient = cfg["lab4_recipe"] == "multiscale_adam_fp32_v1"
    if set(cfg) != (base | scaling if efficient else base):
        raise ValueError("Instructor checkpoint config has missing or unsupported settings")
    if any(type(cfg[key]) is not int or cfg[key] <= 0
           for key in ("steps", "batch_size", "layer_size", "num_layers")):
        raise ValueError("Instructor checkpoint integer settings must be positive integers")
    if ((cfg["layer_size"], cfg["num_layers"]) != _ARCHITECTURE_SHAPE
            or cfg["batch_size"] != 2048 or type(cfg["learning_rate"]) not in (int, float)
            or cfg["learning_rate"] != .001):
        raise ValueError("Instructor checkpoint does not match the original-data architecture/config")
    if (type(saved.get("steps")) is not int or saved["steps"] != manifest["training_steps"]
            or cfg["steps"] != manifest["training_steps"]):
        raise ValueError("Checkpoint training_steps do not match the approved manifest")
    if efficient:
        if cfg["lab4_feature_bands"] != [1, 2, 4, 8]:
            raise ValueError("Instructor checkpoint must use the reviewed periodic feature bands")
        for key in ("lab4_field_center", "lab4_field_scale"):
            values = cfg[key]
            if (not isinstance(values, list) or len(values) != 3
                    or any(type(v) not in (int, float) or not math.isfinite(v) for v in values)
                    or key == "lab4_field_scale" and any(v <= 0 for v in values)):
                raise ValueError("Instructor checkpoint field scaling is invalid")
    state = saved.get("model_state_dict")
    if not isinstance(state, dict) or not state:
        raise ValueError("Instructor checkpoint has no model_state_dict")
    if any(not isinstance(key, str) or not torch.is_tensor(value)
           or value.dtype != torch.float32 or not torch.isfinite(value).all()
           for key, value in state.items()):
        raise ValueError("Instructor checkpoint state must contain finite FP32 tensors")
    if efficient:
        for key, source in (("bands", "lab4_feature_bands"), ("center", "lab4_field_center"),
                            ("scale", "lab4_field_scale")):
            if key not in state or not torch.equal(state[key], torch.tensor(cfg[source], dtype=torch.float32)):
                raise ValueError("Instructor checkpoint scaling buffers disagree with its config")
    return cfg, state


def run_instructor_inference(lab_dir, output_dir, device="auto"):
    """Save 11 original-data frames from approved weights; never train a model.

    The endpoint-excluded 128x128 spatial grid and normalized inputs match the
    lesson. Time endpoints 0 and 60 hours are both included. Outputs contain
    predictions.npz and metrics.json, with no learner loss.csv or copied model.
    """
    output = Path(output_dir).absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Output directory already exists: {output}. Choose a new output directory.")
    lab, manifest, checkpoint = _manifest(lab_dir)
    if manifest["status"] != "approved":
        raise RuntimeError("Instructor model is pending review; no approved checkpoint is available")
    if device not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be auto, cpu or cuda")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; use device='cpu'")
    selected = torch.device("cuda" if device == "auto" and torch.cuda.is_available()
                            else "cpu" if device == "auto" else device)
    _check_hash(lab / "data_lat.npy", manifest["source_data_sha256"], "Original input data")
    if not checkpoint.exists() and not checkpoint.is_symlink() and "download_url" in manifest:
        _download_checkpoint(checkpoint, manifest)
    _check_checkpoint(checkpoint, manifest["sha256"])
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    cfg, state = _checkpoint_config(saved, manifest)
    flow = importlib.import_module("ETC.reference_labs.04_navier_stokes.source_code.navier_stokes")
    # Construction initializes random weights before loading the saved state;
    # preserve the notebook's CPU RNG for an independent student training run.
    with torch.random.fork_rng(devices=[]):
        model = flow.PeriodicFlow(cfg).to(dtype=torch.float32)
    model.load_state_dict(state, strict=True)
    model.to(device=selected, dtype=torch.float32).eval()
    initial = flow.read_wf_data(data_path=lab / "data_lat.npy")
    initial_xy, initial_fields = flow.sample_initial_field(initial, 128)
    axis = torch.linspace(flow.LOWER, flow.LOWER + flow.LENGTH, 129, dtype=torch.float32)[:-1]
    xx, yy = torch.meshgrid(axis, axis, indexing="xy")
    xy = torch.stack((xx.ravel(), yy.ravel()), dim=1)
    times = torch.linspace(0, 1, 11, dtype=torch.float32)
    started = time.perf_counter()
    frames = []
    with torch.inference_mode(), torch.autocast(device_type=selected.type, enabled=False):
        for instant in times:
            chunks = []
            for coordinates in xy.split(_BATCH_SIZE):
                coordinates = coordinates.to(selected)
                prediction = model(coordinates, torch.full_like(coordinates[:, :1], float(instant)))
                if prediction.shape != (len(coordinates), 3) or not torch.isfinite(prediction).all():
                    raise ValueError("Instructor predictions must be finite u, v, p fields")
                chunks.append(prediction.cpu())
            frames.append(torch.cat(chunks))
    prediction = torch.stack(frames).numpy()
    elapsed = time.perf_counter() - started
    arrays = {"xy": xy.numpy(), "times": times.numpy(), "prediction": prediction,
              "times_hours": np.arange(0, 61, 6, dtype=np.float32),
              "initial_xy": initial_xy.numpy(), "initial_fields": initial_fields.numpy()}
    metrics = {
        "problem": "periodic_2d_navier_stokes", "run_kind": "instructor_inference",
        "data_kind": "upstream_data_lat_legacy_normalization", "device": str(selected),
        "dtype": "float32", "weather_forecast_validated": False,
        "status": "inference_complete_from_reviewed_checkpoint",
        "checkpoint": manifest["checkpoint"], "checkpoint_sha256": manifest["sha256"],
        "checkpoint_training_steps": manifest["training_steps"], "review_note": manifest["review_note"],
        "checkpoint_recipe": cfg["lab4_recipe"], "checkpoint_config": cfg,
        "source_data_sha256": manifest["source_data_sha256"], "inference_seconds": elapsed,
        "training_performed": False, "time_unit": "hours", "time_scale_hours": 60,
        "initial_data_source": "data_lat.npy (upstream normalization retained)",
        "initial_data_shape": [len(initial[0]), 3], "legacy_pressure_factor": flow.LEGACY_PRESSURE_FACTOR,
        "flow_diagnostics": summarize_flow(prediction, arrays["times"], arrays["xy"]),
    }
    metrics["flow_diagnostics"]["interpretation"] = (
        "Predicted-frame statistics, not forecast accuracy. Retention is relative to predicted t=0, "
        "not the observed input. Instructor review does not provide future reference observations.")
    serialized = json.dumps(metrics, indent=2, allow_nan=False) + "\n"
    with staged_output(output) as destination:
        np.savez_compressed(destination / "predictions.npz", **arrays)
        (destination / "metrics.json").write_text(serialized, encoding="utf-8")
    return output.resolve()
