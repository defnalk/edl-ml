"""Smoke tests for the plotting helpers."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
from matplotlib.figure import Figure

from edl_ml.physics import PBParameters, solve_poisson_boltzmann
from edl_ml.viz import (
    plot_capacitance_curve,
    plot_error_distribution,
    plot_ion_profiles,
    plot_loss_curves,
    plot_parity,
    plot_shap_summary,
)


def test_plot_ion_profiles_returns_figure() -> None:
    res = solve_poisson_boltzmann(PBParameters(0.01, psi_diffuse_v=0.05))
    fig = plot_ion_profiles(res, title="test")
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 3


def test_plot_capacitance_curve_with_prediction() -> None:
    e = np.linspace(-0.3, 0.3, 25)
    c = 20 + 5 * np.cos(10 * e)
    fig = plot_capacitance_curve(e, c, predicted=c + np.random.default_rng(0).normal(0, 0.5, 25))
    assert isinstance(fig, Figure)


def test_plot_parity_returns_figure() -> None:
    rng = np.random.default_rng(0)
    y = rng.uniform(5, 30, 200)
    fig = plot_parity(y, y + rng.normal(0, 0.3, 200))
    assert isinstance(fig, Figure)


def test_plot_error_distribution_has_zero_axvline() -> None:
    rng = np.random.default_rng(1)
    y = rng.uniform(5, 30, 200)
    fig = plot_error_distribution(y, y + rng.normal(0, 0.4, 200))
    assert isinstance(fig, Figure)
    assert any(line.get_xdata()[0] == 0.0 for line in fig.axes[0].get_lines())


def test_plot_loss_curves_log_scale() -> None:
    fig = plot_loss_curves([1.0, 0.5, 0.25], [1.1, 0.6, 0.3])
    assert fig.axes[0].get_yscale() == "log"


def test_plot_shap_summary_shapes() -> None:
    rng = np.random.default_rng(0)
    sv = rng.normal(size=(100, 4))
    feats = rng.normal(size=(100, 4))
    fig = plot_shap_summary(sv, feats, ["a", "b", "c", "d"])
    assert isinstance(fig, Figure)
