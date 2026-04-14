# edl-ml

**ML-augmented electric double layer simulator for ion adsorption.**

`edl-ml` couples a rigorous Gouy-Chapman-Stern physics core (nonlinear
Poisson-Boltzmann solver, Grahame equation, series capacitance) with a PyTorch
MLP surrogate that reproduces the differential capacitance curve `C_dl(E)` at
~10⁴× speedup. It is designed for applications where the physics must be
evaluated thousands of times inside a Bayesian optimisation loop, a sensitivity
study, or a digital-twin dashboard.

## Highlights

- :material-function-variant: Nonlinear Poisson-Boltzmann solver validated
  against the analytical Gouy-Chapman closed form to ~10⁻¹² V.
- :material-chart-bell-curve: Gouy-Chapman-Stern self-consistent solver
  satisfying the series identity `E = ψ_H + ψ_d` to machine precision.
- :material-brain: Torch MLP surrogate with Optuna TPE hyperparameter search
  and MLflow experiment tracking.
- :material-magnify-scan: Kernel SHAP and permutation feature importance for
  interpretability.
- :material-progress-check: `hypothesis`-powered property tests that verify
  physical invariants (charge/potential symmetry, Grahame consistency,
  series-rule bound).
- :material-rocket-launch-outline: One-command dataset generation, training,
  tuning and evaluation via the `edl` CLI.

## Quickstart

```bash
pip install -e ".[all]" --extra-index-url https://download.pytorch.org/whl/cpu

# 1. simulate one operating point
edl simulate --concentration 0.1 --valence 1

# 2. generate a training dataset (1000 LHS samples)
edl generate --n-samples 1000 --output data/processed/dataset.parquet

# 3. train the surrogate with MLflow tracking
edl train --data data/processed/dataset.parquet \
          --checkpoint data/models/model.pt \
          --epochs 200 --mlflow-experiment edl-ml

# 4. hyperparameter search
edl tune --data data/processed/dataset.parquet --n-trials 40

# 5. parity + residual figures on the test split
edl evaluate --checkpoint data/models/model.pt \
             --data data/processed/dataset.parquet \
             --figures-dir reports/figures
```

## Next

- [Scientific background](guides/physics.md) — derivation and validation.
- [ML pipeline](guides/ml.md) — architecture, scaling, training loop.
- [API reference](api/physics.md) — docstrings for every public function.
