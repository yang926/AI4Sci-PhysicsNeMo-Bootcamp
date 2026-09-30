"""Check the notebook's actual device selection without allocating a GPU."""

import ast
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("requested,available,required,expected", [
    ("cpu", True, False, "cpu"),
    ("cpu", False, False, "cpu"),
    ("auto", True, False, "cuda"),
    ("auto", False, False, "cpu"),
    ("cuda", True, True, "cuda"),
    ("cpu", True, True, "cuda"),  # Explicit REQUIRE_CUDA override.
])
def test_preflight_honors_requested_device(requested, available, required, expected):
    notebook = json.loads((ROOT / "00_Setup.ipynb").read_text())
    source = "".join(next(c for c in notebook["cells"] if c["id"] == "preflight-3")["source"])
    assignment = next(node for node in ast.parse(source).body
                      if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "device" for t in node.targets))
    expression = compile(ast.Expression(assignment.value), "preflight-device", "eval")
    actual = eval(expression, {"torch": SimpleNamespace(device=lambda value: value),
                               "REQUESTED_DEVICE": requested, "cuda_available": available,
                               "REQUIRE_CUDA": required})
    assert actual == expected


LAB_NOTEBOOKS = sorted(ROOT.glob("01_labs/*/Lab_*.ipynb"))


@pytest.mark.parametrize("notebook_path", LAB_NOTEBOOKS, ids=lambda path: path.stem)
@pytest.mark.parametrize("override", [None, "cpu", "cuda", "auto"])
def test_lab_notebooks_default_to_auto_and_preserve_explicit_override(notebook_path, override, monkeypatch):
    assert len(LAB_NOTEBOOKS) == 4
    if override is None:
        monkeypatch.delenv("AI4SCI_DEVICE", raising=False)
    else:
        monkeypatch.setenv("AI4SCI_DEVICE", override)
    notebook = json.loads(notebook_path.read_text())
    source = "\n".join("".join(cell["source"]) for cell in notebook["cells"]
                       if cell["cell_type"] == "code")
    assignments = [node for node in ast.parse(source).body
                   if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id == "DEVICE"
                           for target in node.targets)]
    assert len(assignments) == 1
    expression = compile(ast.Expression(assignments[0].value), str(notebook_path), "eval")
    assert eval(expression, {"os": os}) == (override or "auto")


@pytest.mark.parametrize("available,expected", [(True, "cuda"), (False, "cpu")])
def test_lab_auto_device_resolves_without_allocating_gpu(available, expected, monkeypatch, tmp_path):
    from ETC.runtime import labs

    monkeypatch.setattr(labs.torch.cuda, "is_available", lambda: available)
    monkeypatch.setattr(labs.torch, "manual_seed", lambda _: None)
    monkeypatch.setattr(labs.torch, "set_num_threads", lambda _: None)
    args = labs.parser("device check").parse_args(["--output-dir", str(tmp_path / "unused")])
    assert args.device == "auto"
    _, device = labs.setup(args)
    assert device.type == expected
    assert not args.output_dir.exists()
