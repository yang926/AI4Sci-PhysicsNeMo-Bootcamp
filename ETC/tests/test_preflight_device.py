"""Check the notebook's actual device selection without allocating a GPU."""

import ast
import json
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
