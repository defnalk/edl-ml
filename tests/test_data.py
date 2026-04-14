"""Tests for the dataset-generation pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from edl_ml.data import (
    FEATURE_COLUMNS,
    SamplingBounds,
    build_capacitance_dataset,
    latin_hypercube_samples,
    save_dataset,
    split_by_sample,
    summarise_dataset,
)


def test_latin_hypercube_shape_and_bounds() -> None:
    """LHS samples respect the declared bounds and size."""
    bounds = SamplingBounds()
    samples = latin_hypercube_samples(bounds, 32, seed=1)
    assert samples.shape == (32, 5)
    assert samples[:, 0].min() >= bounds.log10_concentration_min - 1e-12
    assert samples[:, 0].max() <= bounds.log10_concentration_max + 1e-12
    assert set(np.unique(samples[:, 1]).astype(int)).issubset(set(bounds.valence_choices))
    assert samples[:, 2].min() >= bounds.temperature_min_k - 1e-9


@given(n=st.integers(min_value=2, max_value=20))
@settings(max_examples=15, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_dataset_row_count(n: int) -> None:
    """Dataset size equals ``n_samples * potential_n_points``."""
    bounds = SamplingBounds(potential_n_points=7)
    df = build_capacitance_dataset(bounds, n, seed=3, parallel=False)
    assert len(df) == n * bounds.potential_n_points


def test_dataset_physical_reasonable(small_dataset: pd.DataFrame) -> None:
    """Capacitance values are positive and bounded in µF/cm²."""
    assert (small_dataset["capacitance_uf_cm2"] > 0).all()
    assert (small_dataset["capacitance_uf_cm2"] < 500.0).all()


def test_split_by_sample_no_leakage(small_dataset: pd.DataFrame) -> None:
    """Train/val/test splits share no physical feature vector."""
    train, val, test = split_by_sample(small_dataset, 0.25, 0.25, seed=2)
    train_keys = {tuple(r) for r in train[list(FEATURE_COLUMNS)].to_numpy()}
    val_keys = {tuple(r) for r in val[list(FEATURE_COLUMNS)].to_numpy()}
    test_keys = {tuple(r) for r in test[list(FEATURE_COLUMNS)].to_numpy()}
    assert train_keys.isdisjoint(val_keys)
    assert train_keys.isdisjoint(test_keys)
    assert val_keys.isdisjoint(test_keys)


def test_save_and_load_roundtrip(small_dataset: pd.DataFrame, tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Saving and loading a parquet dataset preserves content."""
    path = tmp_path / "roundtrip.parquet"
    save_dataset(small_dataset, path)
    from edl_ml.data import load_dataset

    loaded = load_dataset(path)
    pd.testing.assert_frame_equal(
        loaded.reset_index(drop=True), small_dataset.reset_index(drop=True)
    )


def test_summarise_dataset_keys(small_dataset: pd.DataFrame) -> None:
    stats = summarise_dataset(small_dataset)
    expected = {"n_rows", "n_unique_samples", "cap_mean", "cap_std", "cap_min", "cap_max"}
    assert set(stats) == expected
    assert stats["n_rows"] > 0
    assert stats["cap_min"] > 0


def test_split_rejects_bad_fractions(small_dataset: pd.DataFrame) -> None:
    with pytest.raises(ValueError):
        split_by_sample(small_dataset, val_fraction=0.0, test_fraction=0.2)
    with pytest.raises(ValueError):
        split_by_sample(small_dataset, val_fraction=0.6, test_fraction=0.6)
