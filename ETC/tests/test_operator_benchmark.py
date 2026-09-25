"""Validation timing must not change the original optimizer or sample stream."""
import json
from pathlib import Path
import sys

import pytest
import torch
import yaml

from ETC.course_materials import benchmark_operators

OPERATOR_DIR = Path(__file__).resolve().parents[2] / "02_challenges/04_neural_operators"
sys.path.insert(0, str(OPERATOR_DIR))
from generate_data import generate_splits
from operator_training import reference_fno, run


@pytest.fixture(autouse=True)
def restore_thread_count():
    previous = torch.get_num_threads()
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("level", [1, 2, 3])
def test_checkpoints_preserve_training_exactly(tmp_path, monkeypatch, level):
    data = tmp_path / "data"
    generate_splits(data, train_samples=6, val_samples=3, test_samples=3,
                    grid_size=16, max_mode=3, seed=42)
    model = ({"latent_channels": 8, "num_fno_layers": 2, "num_fno_modes": 4,
              "padding": 0, "coord_features": False, "decoder_layer_size": 8}
             if level != 2 else {"patch_size": [4, 4], "embed_dim": 16,
                                "depth": 2, "num_blocks": 4})
    config = {"data": {"grid_size": 16}, "model": model,
              "training": {"steps": 4, "batch_size": 2, "learning_rate": .001,
                           "cpu_threads": 2}}
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    plain, measured = tmp_path / "plain", tmp_path / "measured"
    run(level, reference_fno, argv=["--reference", "--device", "cpu", "--seed", "7",
        "--steps", "4", "--data-dir", str(data), "--config", str(config_path),
        "--output-dir", str(plain)])
    plain_rng = torch.get_rng_state()
    monkeypatch.setattr(sys, "argv", ["benchmark_operators.py", "--level", str(level),
        "--device", "cpu", "--seed", "7", "--checkpoints", "1", "3", "4",
        "--data-dir", str(data), "--config", str(config_path),
        "--output-dir", str(measured)])
    benchmark_operators.main()
    assert torch.equal(torch.get_rng_state(), plain_rng)
    assert (plain / "loss.csv").read_bytes() == (measured / "loss.csv").read_bytes()
    for name in ("metrics.json",):
        assert json.loads((plain / name).read_text()) == json.loads((measured / name).read_text())
    plain_state = torch.load(plain / "model.pt", weights_only=True)["model_state_dict"]
    measured_state = torch.load(measured / "model.pt", weights_only=True)["model_state_dict"]
    assert plain_state.keys() == measured_state.keys()
    assert all(torch.equal(plain_state[name], measured_state[name]) for name in plain_state)
    report = json.loads((measured / "benchmark.json").read_text())
    assert [row["step"] for row in report["checkpoints"]] == [1, 3, 4]
    assert report["checkpoints"][-1]["validation"] == json.loads(
        (measured / "metrics.json").read_text())["validation"]
    assert report["config"] == config
    assert report["split_samples"] == {"train": 6, "val": 3, "test": 3}
    assert len(report["operator_training_sha256"]) == 64
    assert report["test_pde_rmse_over_training_f_std"] >= 0
