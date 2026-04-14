# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] — 2026-04-15

### Added
- Nonlinear Poisson-Boltzmann solver for symmetric electrolytes using the
  Gouy-Chapman first-integral formulation, validated against the analytical
  closed form to `< 10⁻⁸ V`.
- Gouy-Chapman-Stern self-consistent bisection solver returning surface
  charge, diffuse potential, and total differential capacitance.
- Latin-hypercube dataset generator with multiprocess sweep execution and
  parquet serialisation.
- PyTorch MLP surrogate with SiLU/GELU/ReLU activations, BatchNorm and
  dropout options, AdamW + cosine annealing, early stopping.
- Optuna TPE hyperparameter search over architecture and optimiser.
- Optional MLflow logging of hyperparameters and metrics.
- KernelSHAP and permutation feature-importance analysis.
- Matplotlib diagnostics: parity, residual histogram, loss curves, SHAP
  beeswarm, ion-density and potential profiles.
- `edl` CLI: `simulate`, `generate`, `train`, `tune`, `evaluate`, `predict`.
- Hypothesis-powered property tests asserting Grahame consistency,
  sign-flip symmetry, Debye-length scaling, and series-capacitor bound.
- `mkdocs-material` documentation site with mkdocstrings API rendering.
- GitHub Actions CI (lint + mypy + pytest on 3.10 & 3.11 + Codecov) and
  Docs workflow auto-publishing to GitHub Pages.
- Pre-commit hooks: ruff, ruff-format, mypy, file hygiene checks.

[Unreleased]: https://github.com/defnalk/edl-ml/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/defnalk/edl-ml/releases/tag/v0.1.0
