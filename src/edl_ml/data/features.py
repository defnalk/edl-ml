"""Feature definitions and sampling utilities for the EDL dataset.

The ML surrogate is trained to reproduce the full differential capacitance
curve ``C_dl(E)`` produced by the Gouy-Chapman-Stern solver. Each training
sample is parameterised by five physical variables drawn from a Latin
hypercube inside ``SamplingBounds``. Concentration is sampled on a
log-uniform grid because the diffuse-layer capacitance depends on
:math:`\\sqrt{c}` through the Debye length, giving poor coverage under a
uniform scale.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

try:
    from pyDOE2 import lhs
except ImportError:  # pragma: no cover - fallback for local tests
    lhs = None  # type: ignore[assignment]

FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    "log10_concentration_mol_l",
    "valence",
    "temperature_k",
    "stern_thickness_ang",
    "stern_permittivity",
)
"""Ordered feature names stored in every dataset row."""

TARGET_COLUMN: Final[str] = "capacitance_uf_cm2"
"""Target variable name (differential capacitance)."""


@dataclass(frozen=True, slots=True)
class SamplingBounds:
    """Inclusive sampling bounds for the five input features.

    Defaults bracket physically reasonable aqueous electrochemistry:
    1 mM–1 M symmetric electrolyte, z=1 or 2, 283–343 K, Stern thickness
    2.5–6 Å, Stern permittivity 5–15.
    """

    log10_concentration_min: float = -3.0
    log10_concentration_max: float = 0.0
    valence_choices: tuple[int, ...] = (1, 2)
    temperature_min_k: float = 283.15
    temperature_max_k: float = 343.15
    stern_thickness_min_ang: float = 2.5
    stern_thickness_max_ang: float = 6.0
    stern_permittivity_min: float = 5.0
    stern_permittivity_max: float = 15.0
    potential_min_v: float = -0.4
    potential_max_v: float = 0.4
    potential_n_points: int = 81

    def __post_init__(self) -> None:
        if self.log10_concentration_min >= self.log10_concentration_max:
            raise ValueError("invalid concentration range")
        if not self.valence_choices:
            raise ValueError("valence_choices must be non-empty")
        if self.temperature_min_k >= self.temperature_max_k:
            raise ValueError("invalid temperature range")
        if self.potential_n_points < 5:
            raise ValueError("potential_n_points too small")


def latin_hypercube_samples(
    bounds: SamplingBounds,
    n_samples: int,
    seed: int | None = 0,
) -> NDArray[np.float64]:
    """Generate a Latin hypercube sample of feature vectors.

    Parameters
    ----------
    bounds
        Sampling bounds object.
    n_samples
        Number of samples to draw.
    seed
        Random seed for reproducibility.

    Returns
    -------
    ndarray of shape ``(n_samples, 5)``
        Columns in the order given by :data:`FEATURE_COLUMNS`. Valence is
        rounded to an element of ``bounds.valence_choices`` after uniform
        sampling on ``[0, 1]``.
    """
    rng = np.random.default_rng(seed)
    if n_samples < 1:
        raise ValueError("n_samples must be positive")

    if lhs is not None:
        unit = lhs(5, samples=n_samples, random_state=seed)
    else:
        # Deterministic LHS fallback so tests do not depend on pyDOE2.
        unit = np.zeros((n_samples, 5))
        for dim in range(5):
            perm = rng.permutation(n_samples)
            u = rng.uniform(0.0, 1.0, size=n_samples)
            unit[:, dim] = (perm + u) / n_samples

    samples = np.zeros_like(unit)
    samples[:, 0] = bounds.log10_concentration_min + unit[:, 0] * (
        bounds.log10_concentration_max - bounds.log10_concentration_min
    )
    idx = np.clip(
        np.floor(unit[:, 1] * len(bounds.valence_choices)).astype(int),
        0,
        len(bounds.valence_choices) - 1,
    )
    samples[:, 1] = np.asarray(bounds.valence_choices, dtype=float)[idx]
    samples[:, 2] = bounds.temperature_min_k + unit[:, 2] * (
        bounds.temperature_max_k - bounds.temperature_min_k
    )
    samples[:, 3] = bounds.stern_thickness_min_ang + unit[:, 3] * (
        bounds.stern_thickness_max_ang - bounds.stern_thickness_min_ang
    )
    samples[:, 4] = bounds.stern_permittivity_min + unit[:, 4] * (
        bounds.stern_permittivity_max - bounds.stern_permittivity_min
    )
    return samples
