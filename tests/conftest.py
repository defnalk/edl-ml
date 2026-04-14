"""Shared pytest fixtures."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
import pytest

from edl_ml.data import SamplingBounds, build_capacitance_dataset


@pytest.fixture(scope="session")
def small_dataset() -> pd.DataFrame:
    """A small dataset used across data-oriented tests."""
    return build_capacitance_dataset(
        SamplingBounds(potential_n_points=21),
        n_samples=12,
        seed=7,
        parallel=False,
    )


@pytest.fixture(scope="session")
def feature_vectors(small_dataset: pd.DataFrame) -> np.ndarray:
    """Unique feature vectors from the small dataset."""
    cols: Iterable[str] = (
        "log10_concentration_mol_l",
        "valence",
        "temperature_k",
        "stern_thickness_ang",
        "stern_permittivity",
    )
    return small_dataset.drop_duplicates(subset=list(cols))[list(cols)].to_numpy()
