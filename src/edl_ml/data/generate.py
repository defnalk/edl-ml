"""High-throughput dataset generation driven by the Gouy-Chapman-Stern solver."""

from __future__ import annotations

from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from edl_ml.data.features import (
    FEATURE_COLUMNS,
    SamplingBounds,
    latin_hypercube_samples,
)
from edl_ml.physics.gcs import GCSParameters, gouy_chapman_stern


@dataclass(frozen=True, slots=True)
class SweepResult:
    """Outputs of a single Gouy-Chapman-Stern sweep over electrode potential.

    Attributes
    ----------
    features
        Length-5 feature vector, matching :data:`FEATURE_COLUMNS`.
    potentials_v
        Electrode potentials, V.
    capacitance_f_m2
        Total differential capacitance at each potential, F/m².
    surface_charge_c_m2
        Diffuse-layer surface charge, C/m².
    """

    features: NDArray[np.float64]
    potentials_v: NDArray[np.float64]
    capacitance_f_m2: NDArray[np.float64]
    surface_charge_c_m2: NDArray[np.float64]


def run_single_sweep(
    feature_vector: NDArray[np.float64],
    potentials_v: NDArray[np.float64],
) -> SweepResult:
    """Run the GCS solver for one feature vector over an electrode potential grid.

    Parameters
    ----------
    feature_vector
        Five-element array matching :data:`FEATURE_COLUMNS`.
    potentials_v
        Electrode potentials to sweep, V.

    Returns
    -------
    SweepResult
    """
    (
        log10_c,
        valence,
        temperature,
        stern_thickness_ang,
        stern_permittivity,
    ) = feature_vector
    params = GCSParameters(
        concentration_mol_l=float(10.0**log10_c),
        valence=int(round(valence)),
        temperature_k=float(temperature),
        stern_thickness_m=float(stern_thickness_ang) * 1e-10,
        stern_permittivity=float(stern_permittivity),
    )
    sigma, _, cap = gouy_chapman_stern(params, potentials_v)
    return SweepResult(
        features=np.asarray(feature_vector, dtype=np.float64),
        potentials_v=potentials_v,
        capacitance_f_m2=cap,
        surface_charge_c_m2=sigma,
    )


def _sweep_worker(
    feature_vector: NDArray[np.float64],
    potentials_v: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Pickle-friendly worker used by the process pool."""
    r = run_single_sweep(feature_vector, potentials_v)
    return r.features, r.capacitance_f_m2, r.surface_charge_c_m2


def build_capacitance_dataset(
    bounds: SamplingBounds,
    n_samples: int,
    *,
    seed: int | None = 0,
    parallel: bool = True,
    max_workers: int | None = None,
) -> pd.DataFrame:
    """Build a tidy dataset of capacitance values for ML training.

    The returned DataFrame is long-format: every row represents one
    ``(features, electrode_potential)`` pair with the corresponding total
    differential capacitance. This layout is convenient for scikit-learn and
    torch dataset consumers.

    Parameters
    ----------
    bounds
        Sampling bounds object.
    n_samples
        Number of Latin hypercube samples.
    seed
        Random seed.
    parallel
        Whether to run sweeps in a process pool.
    max_workers
        Process pool size. ``None`` uses the default.

    Returns
    -------
    DataFrame with columns:
        ``log10_concentration_mol_l``, ``valence``, ``temperature_k``,
        ``stern_thickness_ang``, ``stern_permittivity``,
        ``electrode_potential_v``, ``capacitance_uf_cm2``,
        ``surface_charge_uc_cm2``.
    """
    samples = latin_hypercube_samples(bounds, n_samples, seed=seed)
    potentials = np.linspace(
        bounds.potential_min_v,
        bounds.potential_max_v,
        bounds.potential_n_points,
    )

    rows: list[dict[str, float]] = []
    if parallel and n_samples > 1:
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(_sweep_worker, sample, potentials) for sample in samples]
            results = [f.result() for f in as_completed(futures)]
    else:
        results = [_sweep_worker(sample, potentials) for sample in samples]

    for features, cap, sigma in results:
        for e, c, s in zip(potentials, cap, sigma, strict=True):
            row: dict[str, float] = dict(zip(FEATURE_COLUMNS, features, strict=True))
            row["electrode_potential_v"] = float(e)
            row["capacitance_uf_cm2"] = float(c) * 100.0  # F/m² → µF/cm²
            row["surface_charge_uc_cm2"] = float(s) * 100.0
            rows.append(row)
    df = pd.DataFrame(rows)
    return df.sort_values(list(FEATURE_COLUMNS) + ["electrode_potential_v"]).reset_index(drop=True)


def save_dataset(df: pd.DataFrame, path: Path | str) -> None:
    """Save a dataset to a parquet file, creating parent directories as needed."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)


def load_dataset(path: Path | str) -> pd.DataFrame:
    """Load a dataset previously produced by :func:`build_capacitance_dataset`."""
    return pd.read_parquet(path)


def summarise_dataset(df: pd.DataFrame) -> dict[str, float]:
    """Return simple summary statistics for logging.

    Returns
    -------
    dict
        Keys: ``n_rows``, ``n_unique_samples``, ``cap_mean``, ``cap_std``,
        ``cap_min``, ``cap_max``.
    """
    unique_samples = df.drop_duplicates(subset=list(FEATURE_COLUMNS))
    return {
        "n_rows": len(df),
        "n_unique_samples": len(unique_samples),
        "cap_mean": float(df["capacitance_uf_cm2"].mean()),
        "cap_std": float(df["capacitance_uf_cm2"].std()),
        "cap_min": float(df["capacitance_uf_cm2"].min()),
        "cap_max": float(df["capacitance_uf_cm2"].max()),
    }


def split_by_sample(
    df: pd.DataFrame,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int | None = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split the dataset so that every sweep is entirely in one split.

    Splitting at the sweep level (rather than the row level) prevents
    information leakage between train and test capacitance curves that share
    the same physical parameters.

    Parameters
    ----------
    df
        Output of :func:`build_capacitance_dataset`.
    val_fraction, test_fraction
        Fractions in (0, 1). Their sum must be strictly below 1.
    seed
        RNG seed.

    Returns
    -------
    tuple
        ``(train_df, val_df, test_df)``.
    """
    if not 0 < val_fraction < 1 or not 0 < test_fraction < 1:
        raise ValueError("fractions must lie in (0, 1)")
    if val_fraction + test_fraction >= 1:
        raise ValueError("val + test fractions must be < 1")

    unique_samples = df.drop_duplicates(subset=list(FEATURE_COLUMNS))[
        list(FEATURE_COLUMNS)
    ].to_numpy()
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(unique_samples))
    n_val = int(round(val_fraction * len(unique_samples)))
    n_test = int(round(test_fraction * len(unique_samples)))

    test_idx = set(map(int, order[:n_test]))
    val_idx = set(map(int, order[n_test : n_test + n_val]))

    train_rows: list[int] = []
    val_rows: list[int] = []
    test_rows: list[int] = []
    key_to_idx = {tuple(row): i for i, row in enumerate(unique_samples)}
    for i, row in enumerate(df[list(FEATURE_COLUMNS)].to_numpy()):
        sample_idx = key_to_idx[tuple(row)]
        if sample_idx in test_idx:
            test_rows.append(i)
        elif sample_idx in val_idx:
            val_rows.append(i)
        else:
            train_rows.append(i)
    return (
        df.iloc[train_rows].reset_index(drop=True),
        df.iloc[val_rows].reset_index(drop=True),
        df.iloc[test_rows].reset_index(drop=True),
    )


__all__ = [
    "SweepResult",
    "build_capacitance_dataset",
    "load_dataset",
    "run_single_sweep",
    "save_dataset",
    "split_by_sample",
    "summarise_dataset",
]


def _ensure_iterable(x: float | Iterable[float]) -> Iterable[float]:  # pragma: no cover
    if hasattr(x, "__iter__"):
        return x  # type: ignore[return-value]
    return [x]  # type: ignore[return-value]
