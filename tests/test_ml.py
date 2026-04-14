"""Tests for the ML layer that do not require a full GPU training run."""

from __future__ import annotations

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from edl_ml.ml import (  # noqa: E402
    CapacitanceMLP,
    MLPConfig,
    TrainConfig,
    build_loaders,
    train_model,
)
from edl_ml.ml.dataset import INPUT_COLUMNS, StandardScalerTensor  # noqa: E402
from edl_ml.ml.explain import permutation_feature_importance  # noqa: E402


def test_standard_scaler_roundtrip() -> None:
    """Scaler inverse is an identity in float32 tolerance."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 4))
    sc = StandardScalerTensor.fit(x)
    xt = torch.as_tensor(x, dtype=torch.float32)
    recovered = sc.inverse_transform(sc.transform(xt))
    assert torch.allclose(recovered, xt, atol=1e-5)


def test_scaler_handles_constant_column() -> None:
    """A column of zero variance yields std=1 (no divide-by-zero)."""
    x = np.zeros((5, 3))
    sc = StandardScalerTensor.fit(x)
    assert torch.all(sc.std == 1.0)


def test_mlp_config_validation() -> None:
    with pytest.raises(ValueError):
        MLPConfig(input_dim=0)
    with pytest.raises(ValueError):
        MLPConfig(hidden_dims=(0,))
    with pytest.raises(ValueError):
        MLPConfig(dropout=1.0)
    with pytest.raises(ValueError):
        MLPConfig(activation="tanh")


def test_mlp_forward_shape() -> None:
    """Forward pass has shape ``(batch, 1)``."""
    cfg = MLPConfig(input_dim=7, hidden_dims=(16, 16))
    model = CapacitanceMLP(cfg)
    y = model(torch.randn(4, 7))
    assert y.shape == (4, 1)
    assert model.count_parameters() > 0


def test_end_to_end_tiny_training() -> None:
    """A tiny training run converges to a finite validation loss.

    Not a performance test — only a smoke test that the plumbing works and
    that loss decreases monotonically on average over a handful of epochs.
    """
    from edl_ml.data import SamplingBounds, build_capacitance_dataset, split_by_sample

    df = build_capacitance_dataset(
        SamplingBounds(potential_n_points=15),
        n_samples=32,
        seed=42,
        parallel=False,
    )
    train, val, test = split_by_sample(df, 0.2, 0.2, seed=4)
    loaders = build_loaders(train, val, test, batch_size=64, seed=4)
    report = train_model(
        loaders,
        MLPConfig(input_dim=len(INPUT_COLUMNS), hidden_dims=(32, 32)),
        TrainConfig(learning_rate=1e-3, max_epochs=30, patience=30, seed=0),
    )
    assert math.isfinite(report.best_val_loss)
    assert report.best_val_loss < report.train_losses[0]
    for key in ("mse", "rmse", "mae", "r2", "mape"):
        assert key in report.test_metrics
    assert report.test_metrics["rmse"] >= 0
    assert report.test_metrics["r2"] <= 1.0 + 1e-9


def test_permutation_importance_shape_and_values() -> None:
    """Permutation importance returns one finite score per feature."""
    from edl_ml.data import SamplingBounds, build_capacitance_dataset, split_by_sample

    df = build_capacitance_dataset(
        SamplingBounds(potential_n_points=15),
        n_samples=32,
        seed=41,
        parallel=False,
    )
    train, val, test = split_by_sample(df, 0.2, 0.2, seed=5)
    loaders = build_loaders(train, val, test, batch_size=64, seed=5)
    report = train_model(
        loaders,
        MLPConfig(input_dim=len(INPUT_COLUMNS), hidden_dims=(32, 32)),
        TrainConfig(learning_rate=1e-3, max_epochs=20, patience=20, seed=0),
    )
    feats = test[list(INPUT_COLUMNS)].to_numpy(dtype=np.float32)
    targets = test["capacitance_uf_cm2"].to_numpy(dtype=np.float32)
    importances = permutation_feature_importance(
        report.model,
        feats,
        targets,
        loaders.x_scaler,
        loaders.y_scaler,
        n_repeats=4,
        seed=0,
    )
    assert importances.shape == (len(INPUT_COLUMNS),)
    assert np.all(np.isfinite(importances))
