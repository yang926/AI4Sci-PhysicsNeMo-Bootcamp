# Teaching diagrams

These 17 local SVGs are embedded in the eight active Lab and Challenge notebooks.
They show domain/data geometry and the actual program connections before running.
The figures are schematics, not learned results or evidence of convergence.

Edit the SVG source directly. No image generator, external CDN, browser extension,
GPU job, or notebook execution is needed to view them. Keep the diagrams aligned
with the source functions below; do not turn an implementation trace into a completed
Challenge answer.

| Assets | Source of geometry and data flow |
|---|---|
| `lab1-domain.svg`, `lab1-flow.svg` | [pinn_basics.py](../../../01_labs/01_pinn/source_code/pinn_basics.py), [labs.py](../../runtime/labs.py): `BasicPINN`, `loss_terms`, `optimize_lab` |
| `lab2-domain.svg`, `lab2-flow.svg` | [projectile.py](../../../01_labs/02_projectile/source_code/projectile.py): `ProjectileModel`, `loss_terms`, `optimize_projectile` |
| `lab3-domain.svg`, `lab3-flow.svg` | [diffusion_bar.py](../../../01_labs/03_heat_conduction/source_code/diffusion_bar.py), [labs.py](../../runtime/labs.py): two-material sampling, interface residuals, saved-model inference |
| `lab4-domain.svg`, `lab4-workflow.svg` | [run_forecast.py](../../../01_labs/04_weather_forecasting/source_code/run_forecast.py), [evaluate_weather.py](../../../01_labs/04_weather_forecasting/source_code/evaluate_weather.py): global AFNO inference and independent verification |
| `challenge1-domain.svg`, `challenge1-circle.svg`, `challenge1-flow.svg` | [wave_l1.py](../../../02_challenges/01_wave/wave_l1.py), [wave_l2.py](../../../02_challenges/01_wave/wave_l2.py), [wave_l3.py](../../../02_challenges/01_wave/wave_l3.py), [pinn.py](../../runtime/pinn.py): square/disk sampling, separate PDE/network objects, four losses |
| `challenge2-domain.svg`, `challenge2-flow.svg` | [chip_2d_l1.py](../../../02_challenges/02_fluid/chip_2d_l1.py), [chip_2d_l2.py](../../../02_challenges/02_fluid/chip_2d_l2.py), [chip_2d_l3.py](../../../02_challenges/02_fluid/chip_2d_l3.py): masked channel, geometry/condition/PDE paths |
| `challenge3-domain.svg`, `challenge3-flow.svg` | [climate scripts](../../../02_challenges/03_climate): square/time sampling, one or two fields, separate comparison path |
| `challenge4-domain.svg`, `challenge4-flow.svg` | [operator scripts](../../../02_challenges/04_neural_operators): periodic grid, factory contracts, data loss and PINO spectral residual branch |

All labels are English, with local notebook image links and SVG title/description
alternatives. Review the rendered pictures after editing, including labels and arrow
routing. `ETC/tests/test_teaching_visuals.py` checks embeds, vector safety and
unchanged executable notebook cells relative to the illustration baseline.

