"""Matplotlib-based diagnostic plots for physics and ML outputs.

All functions return a :class:`matplotlib.figure.Figure`. They do not call
``plt.show``, so they can be embedded in notebooks, tests, or saved to disk
from the CLI.
"""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

from edl_ml.physics.pb import PBResult


def plot_ion_profiles(result: PBResult, *, title: str | None = None) -> Figure:
    """Plot potential, field, and ion-density profiles from a PB solve.

    Parameters
    ----------
    result
        Output of :func:`edl_ml.physics.pb.solve_poisson_boltzmann`.
    title
        Optional figure title.

    Returns
    -------
    Figure
        A figure with three stacked axes sharing the x-axis.
    """
    x_nm = result.x_m * 1e9
    fig, axes = plt.subplots(3, 1, figsize=(6.4, 7.2), sharex=True)
    axes[0].plot(x_nm, result.psi_v * 1e3, color="tab:blue")
    axes[0].set_ylabel("Potential (mV)")
    axes[0].grid(alpha=0.3)
    axes[1].plot(x_nm, result.field_v_m * 1e-6, color="tab:orange")
    axes[1].set_ylabel("Field (MV/m)")
    axes[1].grid(alpha=0.3)
    axes[2].semilogy(x_nm, result.cation_density_m3, label="cation", color="tab:red")
    axes[2].semilogy(x_nm, result.anion_density_m3, label="anion", color="tab:green")
    axes[2].set_xlabel("x (nm)")
    axes[2].set_ylabel("Number density (1/m³)")
    axes[2].legend(frameon=False)
    axes[2].grid(alpha=0.3, which="both")
    if title is not None:
        fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_capacitance_curve(
    potentials_v: NDArray[np.float64],
    capacitance_uf_cm2: NDArray[np.float64],
    *,
    predicted: NDArray[np.float64] | None = None,
    title: str | None = None,
) -> Figure:
    """Plot a capacitance–potential curve, optionally with a prediction overlay.

    Parameters
    ----------
    potentials_v
        Electrode potentials, V.
    capacitance_uf_cm2
        Reference capacitance values, µF/cm².
    predicted
        Optional surrogate predictions on the same grid.
    """
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(potentials_v, capacitance_uf_cm2, color="k", label="GCS physics")
    if predicted is not None:
        ax.plot(
            potentials_v,
            predicted,
            color="tab:red",
            linestyle="--",
            label="MLP surrogate",
        )
    ax.set_xlabel("Electrode potential (V)")
    ax.set_ylabel(r"$C_\mathrm{dl}$ (µF/cm²)")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_parity(
    y_true: NDArray[np.float64],
    y_pred: NDArray[np.float64],
    *,
    title: str | None = None,
    unit: str = "µF/cm²",
) -> Figure:
    """Parity (true-vs-predicted) scatter with diagonal reference."""
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    ax.scatter(y_true, y_pred, s=6, alpha=0.4, color="tab:blue")
    lo = float(min(np.min(y_true), np.min(y_pred)))
    hi = float(max(np.max(y_true), np.max(y_pred)))
    ax.plot([lo, hi], [lo, hi], color="k", linestyle="--", linewidth=1)
    ax.set_xlabel(f"True ({unit})")
    ax.set_ylabel(f"Predicted ({unit})")
    ax.grid(alpha=0.3)
    ax.set_aspect("equal", adjustable="box")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_error_distribution(
    y_true: NDArray[np.float64],
    y_pred: NDArray[np.float64],
    *,
    bins: int = 50,
    title: str | None = None,
) -> Figure:
    """Histogram of prediction residuals with summary statistics."""
    err = np.asarray(y_pred) - np.asarray(y_true)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.hist(err, bins=bins, color="tab:blue", alpha=0.8, edgecolor="black")
    ax.axvline(0.0, color="k", linestyle="--", linewidth=1)
    ax.set_xlabel("Prediction residual (µF/cm²)")
    ax.set_ylabel("Count")
    ax.grid(alpha=0.3)
    text = f"mean={err.mean():.3f}\nstd ={err.std():.3f}\nMAE ={np.mean(np.abs(err)):.3f}"
    ax.text(
        0.02,
        0.97,
        text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        family="monospace",
        bbox={"boxstyle": "round", "fc": "white", "alpha": 0.85},
    )
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_loss_curves(
    train_losses: Sequence[float],
    val_losses: Sequence[float],
    *,
    title: str | None = None,
) -> Figure:
    """Log-scale plot of training and validation loss curves."""
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(range(1, len(train_losses) + 1), train_losses, label="train")
    ax.plot(range(1, len(val_losses) + 1), val_losses, label="val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE (scaled)")
    ax.set_yscale("log")
    ax.grid(alpha=0.3, which="both")
    ax.legend(frameon=False)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_shap_summary(
    shap_values: NDArray[np.float64],
    features: NDArray[np.float64],
    feature_names: Sequence[str],
    *,
    top_k: int | None = None,
) -> Figure:
    """Beeswarm-style SHAP summary plot.

    Built with matplotlib alone rather than ``shap.summary_plot`` so it is
    trivially embeddable in reports and tests. Each row of ``shap_values`` is
    one sample; columns are features in the same order as ``feature_names``.
    """
    sv = np.asarray(shap_values)
    feats = np.asarray(features)
    if sv.shape != feats.shape:
        raise ValueError("shap_values and features must have matching shapes")
    if sv.shape[1] != len(feature_names):
        raise ValueError("feature_names length mismatch")

    order = np.argsort(np.mean(np.abs(sv), axis=0))[::-1]
    if top_k is not None:
        order = order[:top_k]
    order = order[::-1]

    fig, ax = plt.subplots(figsize=(6.4, 0.35 * len(order) + 1.5))
    for i, feat_idx in enumerate(order):
        x = sv[:, feat_idx]
        y = np.full_like(x, i, dtype=float)
        y += np.random.default_rng(feat_idx).uniform(-0.15, 0.15, size=len(x))
        col = feats[:, feat_idx]
        col_norm = (col - col.min()) / max(col.max() - col.min(), 1e-12)
        ax.scatter(x, y, c=col_norm, s=8, cmap="coolwarm", alpha=0.7)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([feature_names[i] for i in order])
    ax.axvline(0.0, color="k", linestyle="--", linewidth=1)
    ax.set_xlabel("SHAP value (impact on prediction, µF/cm²)")
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    return fig
