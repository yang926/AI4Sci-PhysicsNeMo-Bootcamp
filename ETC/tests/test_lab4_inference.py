"""CPU-only checks of the approval gate and original-data inference artifacts."""

import importlib
import io
import json
from pathlib import Path
import sys
from urllib.request import Request

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from ETC.runtime import lab4_inference as inference
from ETC.runtime.flow_visualization import load_original_flow

flow = importlib.import_module("ETC.reference_labs.04_navier_stokes.source_code.navier_stokes")
RELEASE_URL = "https://github.com/yang926/AI4Sci-PhysicsNeMo-Bootcamp/releases/download/test/model.pt"


def forbidden(*args, **kwargs):
    raise AssertionError("Inference must not train, use synthetic data, or access unrequested services")


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    previous = torch.get_num_threads()
    torch.set_num_threads(min(2, previous))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(inference, "build_opener", forbidden)
    for name in ("optimize_original", "optimize_lab", "training_points", "taylor_green", "evaluate"):
        monkeypatch.setattr(flow, name, forbidden)
    monkeypatch.setattr(torch.optim, "Adam", forbidden)
    monkeypatch.setattr(torch.optim, "LBFGS", forbidden)
    yield
    torch.set_num_threads(previous)


def write_manifest(lab, manifest):
    target = lab / "source_code/conf/instructor_model.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest), encoding="utf-8")


@pytest.fixture
def approved(tmp_path, monkeypatch):
    # Keep the actual PeriodicFlow, loader, state_dict checks, and data reader;
    # only shrink the architecture for these CPU tests, without training it.
    monkeypatch.setattr(inference, "_ARCHITECTURE_SHAPE", (8, 1))
    lab = tmp_path / "lab"
    lab.mkdir()
    values = np.arange(3 * 5 * 7, dtype=np.float32).reshape(3, 5, 7)
    np.save(lab / "data_lat.npy", values)
    cfg = {"steps": 3000, "batch_size": 2048, "learning_rate": .001,
           "layer_size": 8, "num_layers": 1, "lab4_dtype": "float32",
           "lab4_recipe": "upstream_adam_fp32_v1", "lab4_architecture": "upstream_silu_weight_norm"}
    checkpoint = lab / "instructor_models/model.pt"
    checkpoint.parent.mkdir()
    saved = {"config": cfg, "model_state_dict": flow.PeriodicFlow(cfg).state_dict(), "steps": 3000}
    torch.save(saved, checkpoint)
    manifest = {"status": "approved", "checkpoint": "instructor_models/model.pt",
                "sha256": inference._sha256(checkpoint), "source_data_sha256": inference._sha256(lab / "data_lat.npy"),
                "training_steps": 3000, "review_note": "Test-only approval of a small original-data fixture."}
    write_manifest(lab, manifest)
    return lab, manifest, saved


def update_checkpoint(approved):
    lab, manifest, saved = approved
    checkpoint = lab / manifest["checkpoint"]
    torch.save(saved, checkpoint)
    manifest["sha256"] = inference._sha256(checkpoint)
    write_manifest(lab, manifest)


def test_repository_manifest_is_pending_without_a_ready_artifact():
    lab = ROOT / "ETC/reference_labs/04_navier_stokes"
    manifest = json.loads((lab / "source_code/conf/instructor_model.json").read_text())
    assert manifest["status"] == "pending_review"
    assert not {"checkpoint", "sha256", "download_url"} & manifest.keys()
    assert inference.instructor_model_status(lab)["status"] == "pending_review"


def test_pending_review_never_reads_data_loads_weights_or_downloads(tmp_path, monkeypatch):
    write_manifest(tmp_path, {"status": "pending_review", "download_url": RELEASE_URL})
    monkeypatch.setattr(torch, "load", forbidden)
    monkeypatch.setattr(inference, "_sha256", forbidden)
    monkeypatch.setattr(flow, "read_wf_data", forbidden)
    assert inference.instructor_model_status(tmp_path)["status"] == "pending_review"
    output = tmp_path / "output"
    with pytest.raises(RuntimeError, match="pending review"):
        inference.run_instructor_inference(tmp_path, output, device="cpu")
    assert not output.exists()


@pytest.mark.parametrize("recipe", ["upstream_adam_fp32_v1", "multiscale_adam_fp32_v1"])
def test_approved_inference_preserves_normalization_frames_and_provenance(approved, monkeypatch, recipe):
    lab, manifest, saved = approved
    if recipe.startswith("multiscale"):
        saved["config"].update(lab4_recipe=recipe, lab4_feature_bands=[1, 2, 4, 8],
                               lab4_field_center=[.1, -.2, 3.], lab4_field_scale=[1., 2., 3.])
        saved["model_state_dict"] = flow.PeriodicFlow(saved["config"]).state_dict()
        update_checkpoint(approved)
    original_bytes = (lab / "data_lat.npy").read_bytes()
    original_load = torch.load
    loads, batches = [], []

    def load(*args, **kwargs):
        loads.append(kwargs)
        return original_load(*args, **kwargs)

    forward = flow.PeriodicFlow.forward

    def checked_forward(model, xy, time):
        assert not model.training and not torch.is_grad_enabled()
        assert xy.device.type == "cpu" and xy.dtype == time.dtype == torch.float32
        assert not torch.is_autocast_enabled("cpu")
        assert all(p.dtype == torch.float32 for p in model.parameters())
        batches.append(len(xy))
        return forward(model, xy, time)

    monkeypatch.setattr(torch, "load", load)
    assert inference.instructor_model_status(lab)["status"] == "approved"
    assert loads == [], "Status should not deserialize a checkpoint"
    monkeypatch.setattr(flow.PeriodicFlow, "forward", checked_forward)
    output = lab / "inference"
    # An enclosing notebook autocast context must not reduce inference precision.
    with torch.autocast("cpu", dtype=torch.bfloat16):
        result = inference.run_instructor_inference(lab, output, device="auto")
    assert result == output.resolve()
    assert loads == [{"map_location": "cpu", "weights_only": True}]
    assert max(batches) <= 2048 and len(batches) == 11 * 8
    assert {p.name for p in output.iterdir()} == {"metrics.json", "predictions.npz"}
    data = load_original_flow(output)
    assert data["prediction"].shape == (11, 128 * 128, 3)
    assert data["prediction"].dtype == np.float32
    np.testing.assert_array_equal(data["times_hours"], np.arange(0, 61, 6))
    np.testing.assert_allclose(data["times"], np.linspace(0, 1, 11), atol=1e-7)
    assert len(np.unique(data["xy"][:, 0])) == 128
    assert data["xy"][:, 0].max() < flow.LOWER + flow.LENGTH
    expected_xy, expected_fields = flow.read_wf_data(data_path=lab / "data_lat.npy")
    np.testing.assert_array_equal(data["initial_xy"], expected_xy)
    np.testing.assert_array_equal(data["initial_fields"], expected_fields)
    assert (lab / "data_lat.npy").read_bytes() == original_bytes
    metrics = json.loads((output / "metrics.json").read_text())
    assert metrics["run_kind"] == "instructor_inference"
    assert metrics["checkpoint_sha256"] == manifest["sha256"]
    assert metrics["source_data_sha256"] == manifest["source_data_sha256"]
    assert metrics["checkpoint_training_steps"] == 3000
    assert metrics["checkpoint_recipe"] == recipe
    assert metrics["checkpoint_config"] == saved["config"]
    assert metrics["inference_seconds"] >= 0
    assert metrics["training_performed"] is False
    assert metrics["weather_forecast_validated"] is False
    assert metrics["flow_diagnostics"]["baseline"] == "predicted_t0"
    assert len(metrics["flow_diagnostics"]["per_time"]) == 11
    assert not {"loss", "steps", "initial_loss", "final_loss", "heldout_before", "heldout_after"} & metrics.keys()


@pytest.mark.parametrize("field,value", [
    ("status", "ready"), ("status", []), ("review_note", ""), ("training_steps", True),
    ("sha256", "bad"), ("source_data_sha256", "0" * 64),
    ("checkpoint", "../outside.pt"), ("checkpoint", "/tmp/outside.pt"),
    ("download_url", "https://example.com/model.pt"),
])
def test_invalid_manifest_blocks_before_checkpoint_loading(approved, monkeypatch, field, value):
    lab, manifest, _ = approved
    manifest[field] = value
    write_manifest(lab, manifest)
    monkeypatch.setattr(torch, "load", forbidden)
    assert inference.instructor_model_status(lab)["status"] == "invalid"
    with pytest.raises(ValueError):
        inference.run_instructor_inference(lab, lab / "output", device="cpu")
    assert not (lab / "output").exists()


def test_missing_checkpoint_and_tampered_checkpoint_are_not_approved(approved, monkeypatch):
    lab, manifest, _ = approved
    checkpoint = lab / manifest["checkpoint"]
    checkpoint.unlink()
    assert inference.instructor_model_status(lab)["status"] == "unavailable"
    with pytest.raises(FileNotFoundError):
        inference.run_instructor_inference(lab, lab / "output", device="cpu")
    checkpoint.write_bytes(b"not the approved file")
    monkeypatch.setattr(torch, "load", forbidden)
    assert inference.instructor_model_status(lab)["status"] == "invalid"
    with pytest.raises(ValueError, match="SHA-256"):
        inference.run_instructor_inference(lab, lab / "output", device="cpu")


def test_checkpoint_symlink_cannot_escape_lab(approved, tmp_path):
    lab, manifest, _ = approved
    outside = tmp_path / "outside.pt"
    checkpoint = lab / manifest["checkpoint"]
    checkpoint.rename(outside)
    checkpoint.symlink_to(outside)
    assert inference.instructor_model_status(lab)["status"] == "invalid"


@pytest.mark.parametrize("change", ["synthetic", "width", "half", "steps", "unknown", "missing", "extra", "nan"])
def test_invalid_checkpoint_fails_without_publishing(approved, change):
    lab, _, saved = approved
    if change == "synthetic":
        saved["config"]["lab4_recipe"] = "adam_lbfgs_fp32_v1"
    elif change == "width":
        saved["config"]["layer_size"] = 9
    elif change == "half":
        saved["model_state_dict"] = {key: value.half() for key, value in saved["model_state_dict"].items()}
    elif change == "steps":
        saved["steps"] = 50000
    elif change == "unknown":
        saved["config"]["synthetic"] = True
    elif change == "missing":
        saved["model_state_dict"].pop(next(iter(saved["model_state_dict"])))
    elif change == "extra":
        saved["model_state_dict"]["unexpected"] = torch.ones(1)
    elif change == "nan":
        next(iter(saved["model_state_dict"].values())).flatten()[0] = float("nan")
    update_checkpoint(approved)
    with pytest.raises((ValueError, RuntimeError)):
        inference.run_instructor_inference(lab, lab / "output", device="cpu")
    assert not (lab / "output").exists()


def test_existing_output_is_preserved_without_loading(approved, monkeypatch):
    lab, _, _ = approved
    output = lab / "output"
    output.mkdir()
    (output / "keep.txt").write_text("previous run")
    monkeypatch.setattr(torch, "load", forbidden)
    with pytest.raises(FileExistsError):
        inference.run_instructor_inference(lab, output, device="cpu")
    assert (output / "keep.txt").read_text() == "previous run"


def test_cpu_round_trip_preserves_rng_and_matches_saved_model(approved):
    lab, manifest, saved = approved
    model = flow.PeriodicFlow(saved["config"]).eval()
    model.load_state_dict(saved["model_state_dict"], strict=True)
    axis = torch.linspace(flow.LOWER, flow.LOWER + flow.LENGTH, 129)[:-1]
    xx, yy = torch.meshgrid(axis, axis, indexing="xy")
    indices = torch.tensor([0, 127, 128, 8123, 16383])
    xy = torch.stack((xx.ravel(), yy.ravel()), dim=1)[indices]
    times = torch.tensor([0., .5, 1.])
    with torch.inference_mode():
        expected = torch.stack([model(xy, torch.full_like(xy[:, :1], float(t))) for t in times]).numpy()
    torch.manual_seed(73)
    rng_before = torch.get_rng_state().clone()
    output = inference.run_instructor_inference(lab, lab / "output", device="cpu")
    assert torch.equal(torch.get_rng_state(), rng_before)
    actual = load_original_flow(output)["prediction"][[0, 5, 10]][:, indices.numpy()]
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-7)
    status = inference.instructor_model_status(lab)
    assert "3000" in status["message"] and manifest["review_note"] in status["message"]


def test_oversize_local_checkpoint_is_rejected_before_loading(approved, monkeypatch):
    lab, _, _ = approved
    monkeypatch.setattr(inference, "_MAX_CHECKPOINT_BYTES", 1)
    monkeypatch.setattr(torch, "load", forbidden)
    assert inference.instructor_model_status(lab)["status"] == "invalid"
    with pytest.raises(ValueError, match="size limit"):
        inference.run_instructor_inference(lab, lab / "output", device="cpu")


class Download(io.BytesIO):
    def __init__(self, payload, headers=None, url=None):
        super().__init__(payload)
        self.headers = {} if headers is None else headers
        self.url = url or "https://release-assets.githubusercontent.com/github-production-release-asset/test"

    def geturl(self):
        return self.url


def mock_download(monkeypatch, payload, **kwargs):
    calls = []

    class Opener:
        def open(self, request, timeout):
            calls.append(request.full_url)
            assert timeout == 30
            return Download(payload, **kwargs)

    monkeypatch.setattr(inference, "build_opener", lambda handler: Opener())
    return calls


def downloadable(approved):
    lab, manifest, _ = approved
    checkpoint = lab / manifest["checkpoint"]
    payload = checkpoint.read_bytes()
    checkpoint.unlink()
    manifest["download_url"] = RELEASE_URL
    write_manifest(lab, manifest)
    return checkpoint, payload


def test_approved_download_is_verified_cached_and_used_for_inference(approved, monkeypatch):
    lab, manifest, _ = approved
    checkpoint, payload = downloadable(approved)
    calls = mock_download(monkeypatch, payload)
    status = inference.instructor_model_status(lab)
    assert status["status"] == "approved" and "download" in status["message"]
    assert calls == [] and not checkpoint.exists()
    inference.run_instructor_inference(lab, lab / "output", device="cpu")
    assert calls == [RELEASE_URL]
    assert checkpoint.read_bytes() == payload
    assert inference._sha256(checkpoint) == manifest["sha256"]
    assert list(checkpoint.parent.iterdir()) == [checkpoint]
    assert inference.instructor_model_status(lab)["status"] == "approved"


@pytest.mark.parametrize("failure", ["hash", "length", "stream", "redirect"])
def test_failed_download_publishes_no_checkpoint_or_predictions(approved, monkeypatch, failure):
    lab, _, _ = approved
    checkpoint, payload = downloadable(approved)
    kwargs = {}
    if failure == "hash":
        payload = b"wrong weights"
    elif failure == "length":
        kwargs["headers"] = {"Content-Length": str(inference._MAX_CHECKPOINT_BYTES + 1)}
    elif failure == "stream":
        monkeypatch.setattr(inference, "_MAX_CHECKPOINT_BYTES", 10)
    elif failure == "redirect":
        kwargs["url"] = "http://127.0.0.1/model.pt"
    mock_download(monkeypatch, payload, **kwargs)
    with pytest.raises(ValueError):
        inference.run_instructor_inference(lab, lab / "output", device="cpu")
    assert not checkpoint.exists() and not (lab / "output").exists()
    assert list(checkpoint.parent.iterdir()) == []


@pytest.mark.parametrize("url", [
    "http://github.com/yang926/AI4Sci-PhysicsNeMo-Bootcamp/releases/download/test/model.pt",
    "https://github.com/another/project/releases/download/test/model.pt",
    "https://github.com@localhost/model.pt", "https://127.0.0.1/model.pt",
    "https://github.com:444/yang926/AI4Sci-PhysicsNeMo-Bootcamp/releases/download/test/model.pt",
])
def test_untrusted_download_urls_are_rejected(url):
    with pytest.raises(ValueError):
        inference._allowed_download_url(url)


def test_redirect_handler_rejects_untrusted_host_before_request():
    with pytest.raises(ValueError):
        inference._ReleaseRedirects().redirect_request(
            Request(RELEASE_URL), None, 302, "Found", {}, "https://127.0.0.1/model.pt")


def test_scaling_buffers_must_match_reviewed_config(approved):
    lab, _, saved = approved
    cfg = saved["config"]
    cfg.update(lab4_recipe="multiscale_adam_fp32_v1", lab4_feature_bands=[1, 2, 4, 8],
               lab4_field_center=[.1, -.2, 3.], lab4_field_scale=[1., 2., 3.])
    saved["model_state_dict"] = flow.PeriodicFlow(cfg).state_dict()
    saved["model_state_dict"]["scale"][0] = 999
    update_checkpoint(approved)
    with pytest.raises(ValueError, match="buffers disagree"):
        inference.run_instructor_inference(lab, lab / "output", device="cpu")


def test_failed_inference_cleans_staging_and_does_not_publish(approved, monkeypatch):
    lab, _, _ = approved

    def save_failure(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(np, "savez_compressed", save_failure)
    with pytest.raises(OSError, match="disk full"):
        inference.run_instructor_inference(lab, lab / "output", device="cpu")
    assert not (lab / "output").exists()
    assert not list(lab.glob(".output-*"))
