"""The conductivity comparison must show small changes without changing data."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = ROOT / "01_labs/03_heat_conduction/Lab_3_Heat_Conduction.ipynb"


@pytest.mark.parametrize("left_error", [0.0, 0.01, 4.0])
def test_parameterized_plot_has_honest_full_and_zoom_views(tmp_path, monkeypatch, left_error):
    notebook = json.loads(NOTEBOOK.read_text())
    cell = next(cell for cell in notebook["cells"] if cell.get("id") == "37eca0266f18")
    x = np.linspace(0, 2, 201)
    d1 = np.array([5.0, 10.0, 25.0])
    interface = 100 * 0.1 / (d1 + 0.1)
    reference = np.stack([np.where(x <= 1, x * tb, (x - 1) * 100 + (2 - x) * tb)
                          for tb in interface])[:, :, None]
    prediction = reference.copy()
    prediction[:, x <= 1, 0] += left_error * x[x <= 1]
    np.savez(tmp_path / "predictions.npz", x=x[:, None], D1=d1,
             reference=reference, prediction=prediction)
    monkeypatch.setattr(plt, "show", lambda: None)
    context = {"np": np, "plt": plt, "RUN_DIRS": {}, "RUN_COMPLETED": {},
               "completed_output": lambda *args: tmp_path}
    try:
        exec(compile("".join(cell["source"]), str(NOTEBOOK), "exec"), context)
        axes = context["axes"]
        assert len(axes) == 2
        assert axes[0].get_xlim() == (0, 2)
        assert axes[1].get_xlim() == (0, 1)
        assert "zoom" in axes[1].get_title().lower()
        for ax, region in zip(axes, (np.ones(x.shape, dtype=bool), x <= 1)):
            for i in range(len(d1)):
                exact_line, model_line = ax.lines[2 * i:2 * i + 2]
                np.testing.assert_array_equal(exact_line.get_xdata(), x[region])
                np.testing.assert_array_equal(exact_line.get_ydata(), reference[i, region, 0])
                np.testing.assert_array_equal(model_line.get_ydata(), prediction[i, region, 0])
                assert exact_line.get_color() == model_line.get_color()
                assert exact_line.get_linestyle() == "--"
                assert model_line.get_linestyle() == "-"
                assert model_line.get_marker() == "o"
            low, high = ax.get_ylim()
            assert low < prediction[:, region, 0].min()
            assert high > prediction[:, region, 0].max()
        if left_error == 0:
            assert axes[1].get_ylim()[1] < 2.2
        context["fig"].canvas.draw()
    finally:
        plt.close("all")
