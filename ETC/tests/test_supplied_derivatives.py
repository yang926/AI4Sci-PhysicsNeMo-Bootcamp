"""Time-derivative registration removes reminders, not missing-input errors."""
import logging
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch
from sympy import Function, Symbol
from physicsnemo.sym.eq.pde import PDE
from physicsnemo.sym.eq.phy_informer import PhysicsInformer

from ETC.runtime.labs import derivative, informer


LOGGER = "physicsnemo.sym.eq.phy_informer"
SUPPLIED = ("x__t__t", "y__t__t")


class Projectile(PDE):
    def __init__(self):
        self.dim = 1
        t = Symbol("t")
        x, y = Function("x")(t), Function("y")(t)
        self.equations = {"ode_x": x.diff(t, 2), "ode_y": y.diff(t, 2) + 9.81}


def test_declared_time_derivatives_are_inputs_without_redundant_warning(caplog):
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        physics = informer(Projectile(), "cpu", supplied_derivatives=SUPPLIED)
        assert set(SUPPLIED) <= set(physics.required_inputs)
        t = torch.linspace(0, 5, 17)[:, None].requires_grad_()
        x, y = 20 * t, 35 * t - 4.905 * t.square()
        actual = physics.forward({"coordinates": t,
                                  "x__t__t": derivative(derivative(x, t), t),
                                  "y__t__t": derivative(derivative(y, t), t)})
    assert max(value.detach().abs().max().item() for value in actual.values()) < 1e-6
    assert not [record for record in caplog.records if record.name == LOGGER]


@pytest.mark.parametrize("missing", SUPPLIED)
def test_declared_but_missing_tensor_still_fails(missing):
    physics = informer(Projectile(), "cpu", supplied_derivatives=SUPPLIED)
    inputs = {name: torch.zeros(2, 1) for name in (*SUPPLIED, "coordinates")}
    inputs.pop(missing)
    with pytest.raises(KeyError, match=missing):
        physics.forward(inputs)


def test_undeclared_derivative_and_unrelated_warning_are_not_hidden(caplog):
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        physics = informer(Projectile(), "cpu", supplied_derivatives=("x__t__t",))
        logging.getLogger(LOGGER).warning("unrelated diagnostic remains visible")
    messages = [record.getMessage() for record in caplog.records if record.name == LOGGER]
    assert sum("must be supplied manually" in message for message in messages) == 2
    assert "unrelated diagnostic remains visible" in messages
    with pytest.raises(KeyError, match="y__t__t"):
        physics.forward({"coordinates": torch.zeros(2, 1), "x__t__t": torch.zeros(2, 1)})


def test_default_helper_keeps_original_physicsinformer_behavior(caplog):
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        physics = informer(Projectile(), "cpu")
    assert type(physics) is PhysicsInformer
    assert sum("must be supplied manually" in record.getMessage()
               for record in caplog.records if record.name == LOGGER) == 4


@pytest.mark.parametrize("name", ["x__t__t__t", "x__x", "coordinates", "x__t__"])
def test_invalid_declarations_are_rejected(name):
    with pytest.raises(ValueError, match="non-spatial derivative"):
        informer(Projectile(), "cpu", supplied_derivatives=(name,))


def test_typo_in_derivative_declaration_is_rejected():
    with pytest.raises(ValueError, match="absent from the equations"):
        informer(Projectile(), "cpu", supplied_derivatives=(*SUPPLIED, "z__t__t"))


def test_supplied_autograd_derivatives_preserve_parameter_gradients():
    physics = informer(Projectile(), "cpu", supplied_derivatives=SUPPLIED)
    t = torch.linspace(0, 1, 9)[:, None].requires_grad_()
    acceleration = torch.nn.Parameter(torch.tensor(1.5))
    x, y = acceleration * t.square(), -4.905 * t.square()
    actual = physics.forward({"coordinates": t,
                              "x__t__t": derivative(derivative(x, t), t),
                              "y__t__t": derivative(derivative(y, t), t)})
    sum(value.square().mean() for value in actual.values()).backward()
    assert acceleration.grad.item() == pytest.approx(12.0)


def test_lab2_training_process_does_not_print_the_reminder(tmp_path):
    root = Path(__file__).resolve().parents[2]
    output = tmp_path / "projectile"
    result = subprocess.run(
        [sys.executable, str(root / "01_labs/02_projectile/source_code/projectile.py"),
         "--device", "cpu", "--steps", "2", "--output-dir", str(output)],
        cwd=root, check=True, capture_output=True, text=True, timeout=60,
        env={**os.environ, "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"})
    assert "must be supplied manually" not in result.stderr + result.stdout
    assert (output / "metrics.json").is_file()
