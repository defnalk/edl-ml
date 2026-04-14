# edl-ml

[![CI](https://github.com/defnalk/edl-ml/actions/workflows/ci.yml/badge.svg)](https://github.com/defnalk/edl-ml/actions/workflows/ci.yml)
[![Docs](https://github.com/defnalk/edl-ml/actions/workflows/docs.yml/badge.svg)](https://defnalk.github.io/edl-ml)
[![codecov](https://codecov.io/gh/defnalk/edl-ml/branch/main/graph/badge.svg)](https://codecov.io/gh/defnalk/edl-ml)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Checked with mypy](https://www.mypy-lang.org/static/mypy_badge.svg)](https://mypy-lang.org/)

**ML-augmented electric double layer simulator for ion adsorption at
charged electrode interfaces.**

`edl-ml` couples a rigorous Gouy-Chapman-Stern physics core (nonlinear
Poisson-Boltzmann solver, Grahame equation, self-consistent series
capacitance) with a PyTorch MLP surrogate that reproduces `C_dl(E)` at
~10⁴× speedup — fast enough to live inside a Bayesian optimisation loop,
a digital-twin dashboard, or a parameter-identifiability study.

## Scientific motivation

The differential capacitance of an electric double layer controls ion
adsorption, electrosorption energetics, pseudocapacitive storage, and the
kinetics of electrocatalytic reactions. The Gouy-Chapman-Stern model
predicts `C_dl(E)` from first principles but requires an inner-loop
bisection over the self-consistent potential split and is awkward to
differentiate through. An ML surrogate trained on GCS data provides a
smooth, analytically differentiable, vectorised alternative suitable for
large-scale screening and gradient-based process design.

## Architecture

```text
┌─────────────────────────┐      ┌────────────────────────────┐
│  physics                │      │  ml                        │
│                         │      │                            │
│  • Poisson-Boltzmann    │──────▶  • Latin hypercube dataset │
│    (first-integral ODE) │  GCS │  • StandardScaler          │
│  • Grahame equation     │──────▶  • MLP (torch)             │
│  • GCS self-consistent  │ data │  • AdamW + cosine LR       │
│    potential split      │      │  • Optuna TPE HPO          │
└─────────────────────────┘      │  • MLflow logging          │
                                 │  • Kernel SHAP + permutation│
                                 └────────────────────────────┘
                                                │
                                                ▼
                                 ┌──────────────────────────────┐
                                 │  viz / cli / docs            │
                                 │  • parity plots              │
                                 │  • residual distributions    │
                                 │  • loss curves               │
                                 │  • SHAP beeswarm             │
                                 │  • edl CLI                   │
                                 │  • mkdocs-material site      │
                                 └──────────────────────────────┘
```

## Quickstart

```bash
git clone https://github.com/defnalk/edl-ml
cd edl-ml
make install-dev

# end-to-end pipeline
edl generate --n-samples 1000 --output data/processed/dataset.parquet
edl train    --data data/processed/dataset.parquet \
             --checkpoint data/models/model.pt \
             --mlflow-experiment edl-ml
edl evaluate --checkpoint data/models/model.pt \
             --data data/processed/dataset.parquet \
             --figures-dir reports/figures
```

### Single-point simulation (no ML)

```bash
edl simulate --concentration 0.1 --valence 1 \
             --e-min -0.4 --e-max 0.4 --n-points 81
```

### Python API

```python
from edl_ml.physics import GCSParameters, gouy_chapman_stern
import numpy as np

params = GCSParameters(
    concentration_mol_l=0.1,
    valence=1,
    stern_thickness_m=3e-10,
    stern_permittivity=6.0,
)
E = np.linspace(-0.4, 0.4, 201)
sigma, psi_d, C_dl = gouy_chapman_stern(params, E)
```

## Results

Reference pipeline trained on 1 000 LHS samples × 81 potentials = 81 000
rows, split 70/15/15 at the **sweep** level (no feature leakage):

| Metric       | Units      | Test value |
|--------------|------------|------------|
| RMSE         | µF/cm²     | `< 0.15`   |
| MAE          | µF/cm²     | `< 0.10`   |
| R²           | —          | `> 0.999`  |
| Inference    | points/sec | `> 10⁶`    |

Figures are generated by `edl evaluate` and saved to `reports/figures/`.

## Validation

The physics core is validated against closed-form results:

- Debye length matches `1 / sqrt(2 N_A e² z² c / ε_r ε₀ k_B T)` exactly
  across concentration and valence.
- Numerical PB profile matches the analytical Gouy-Chapman expression
  `ψ(x) = (4 k_B T / z e) arctanh(tanh(z e ψ_d / 4 k_B T) · exp(-κ x))`
  to `< 10⁻⁸ V` over a 40-Debye-length domain.
- Grahame surface charge reproduced to 1 ppm relative tolerance across
  randomised diffuse-layer potentials (hypothesis-driven property test).
- Self-consistency `|E − (ψ_H + ψ_d)| < 10⁻⁹ V` at every electrode
  potential in the sampling box.

## Documentation

Full API and guides: **[defnalk.github.io/edl-ml](https://defnalk.github.io/edl-ml)**.

## Citation

```bibtex
@software{edl_ml,
  author  = {Ertugrul, Defne Nihal},
  title   = {edl-ml: ML-augmented electric double layer simulator},
  year    = {2026},
  url     = {https://github.com/defnalk/edl-ml},
  version = {0.1.0}
}
```

## License

MIT. See [LICENSE](LICENSE).
