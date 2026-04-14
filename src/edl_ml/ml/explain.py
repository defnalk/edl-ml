"""SHAP and permutation-based explanations for the capacitance surrogate."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray

from edl_ml.ml.dataset import INPUT_COLUMNS, StandardScalerTensor
from edl_ml.ml.model import CapacitanceMLP


@dataclass(frozen=True, slots=True)
class ShapResult:
    """Output of a SHAP explanation run.

    Attributes
    ----------
    values
        SHAP values in *unscaled* capacitance units, shape
        ``(n_samples, n_features)``.
    features
        Corresponding raw feature values, shape ``(n_samples, n_features)``.
    feature_names
        Names of the features in column order.
    base_value
        The expected model output (mean over the background set) in the same
        units as ``values``.
    """

    values: NDArray[np.float64]
    features: NDArray[np.float64]
    feature_names: tuple[str, ...]
    base_value: float


def _predict_unscaled(
    model: CapacitanceMLP,
    x: NDArray[np.float64],
    x_scaler: StandardScalerTensor,
    y_scaler: StandardScalerTensor,
) -> NDArray[np.float64]:
    """Run the MLP on raw features and return unscaled predictions."""
    with torch.no_grad():
        xt = torch.as_tensor(x, dtype=torch.float32)
        xt = x_scaler.transform(xt)
        out = model(xt)
        out = y_scaler.inverse_transform(out)
    return np.asarray(out.cpu().numpy().ravel(), dtype=np.float64)


def shap_explain(
    model: CapacitanceMLP,
    background: NDArray[np.float64],
    samples: NDArray[np.float64],
    x_scaler: StandardScalerTensor,
    y_scaler: StandardScalerTensor,
    *,
    nsamples: int = 200,
) -> ShapResult:
    """Compute KernelSHAP values for the surrogate on raw feature space.

    Parameters
    ----------
    model
        Trained :class:`CapacitanceMLP`.
    background
        Background dataset, shape ``(m, n_features)``. KernelSHAP approximates
        marginal expectations against this reference distribution.
    samples
        Samples to explain, shape ``(n, n_features)``.
    x_scaler, y_scaler
        Scalers used when training ``model``.
    nsamples
        Number of coalitions used by KernelSHAP per sample.

    Returns
    -------
    ShapResult
        Results in unscaled output units.
    """
    import shap

    def f(x: NDArray[np.float64]) -> NDArray[np.float64]:
        return _predict_unscaled(model, x, x_scaler, y_scaler)

    explainer = shap.KernelExplainer(f, background)
    values = np.asarray(explainer.shap_values(samples, nsamples=nsamples))
    base_value = float(np.mean(f(background)))
    return ShapResult(
        values=values,
        features=np.asarray(samples, dtype=float),
        feature_names=INPUT_COLUMNS,
        base_value=base_value,
    )


def permutation_feature_importance(
    model: CapacitanceMLP,
    features: NDArray[np.float64],
    targets: NDArray[np.float64],
    x_scaler: StandardScalerTensor,
    y_scaler: StandardScalerTensor,
    *,
    n_repeats: int = 20,
    seed: int = 0,
) -> NDArray[np.float64]:
    """Return mean permutation feature importance (increase in MAE per feature).

    Useful as a lightweight, no-external-dependency alternative to SHAP. Each
    feature column is independently shuffled ``n_repeats`` times and the mean
    increase in mean-absolute error (µF/cm²) is recorded.
    """
    rng = np.random.default_rng(seed)
    base_pred = _predict_unscaled(model, features, x_scaler, y_scaler)
    base_mae = float(np.mean(np.abs(base_pred - targets)))
    importances = np.zeros(features.shape[1])
    for col in range(features.shape[1]):
        deltas = np.zeros(n_repeats)
        for rep in range(n_repeats):
            shuffled = features.copy()
            rng.shuffle(shuffled[:, col])
            pred = _predict_unscaled(model, shuffled, x_scaler, y_scaler)
            deltas[rep] = float(np.mean(np.abs(pred - targets))) - base_mae
        importances[col] = float(np.mean(deltas))
    return importances
