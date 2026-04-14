"""Tests for the Gouy-Chapman-Stern composite model."""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from edl_ml.physics.gcs import (
    GCSParameters,
    diffuse_capacitance,
    gouy_chapman_stern,
    stern_capacitance,
    total_capacitance,
)


def test_stern_capacitance_units() -> None:
    """Stern capacitance at 3 Å and ε_r=6 is ~17.7 µF/cm²."""
    params = GCSParameters(0.1)
    value_uf_cm2 = stern_capacitance(params) * 100.0
    assert math.isclose(value_uf_cm2, 17.71, rel_tol=1e-3)


def test_diffuse_capacitance_monotone_in_psi() -> None:
    """Diffuse-layer C should be minimum at psi=0 and symmetric about it."""
    params = GCSParameters(0.05)
    grid = np.linspace(-0.2, 0.2, 41)
    cap = np.asarray(diffuse_capacitance(params, grid))
    assert float(np.argmin(cap)) == float(len(grid) // 2)
    assert np.allclose(cap, cap[::-1], atol=1e-12)


def test_total_capacitance_series_rule() -> None:
    """Total capacitance is the series combination of Stern and diffuse."""
    params = GCSParameters(0.1)
    c_h = stern_capacitance(params)
    c_d = float(diffuse_capacitance(params, 0.05))
    c_tot = float(total_capacitance(params, 0.05))
    assert math.isclose(1.0 / c_tot, 1.0 / c_h + 1.0 / c_d, rel_tol=1e-12)


def test_gcs_self_consistency() -> None:
    """Potential split E = psi_H + psi_d holds at every point."""
    params = GCSParameters(0.1)
    c_h = stern_capacitance(params)
    E = np.linspace(-0.3, 0.3, 31)
    sigma, psid, _ = gouy_chapman_stern(params, E)
    psi_h = sigma / c_h
    err = float(np.max(np.abs(E - (psi_h + psid))))
    assert err < 1e-9


def test_gcs_pzc_gives_zero_charge() -> None:
    """At E=0 the diffuse-layer charge and potential are zero."""
    params = GCSParameters(0.05)
    sigma, psid, cap = gouy_chapman_stern(params, np.array([0.0]))
    assert abs(float(sigma[0])) < 1e-12
    assert abs(float(psid[0])) < 1e-12
    assert cap[0] > 0


@given(
    log_c=st.floats(min_value=-3.0, max_value=0.0),
    z=st.integers(min_value=1, max_value=3),
    d_ang=st.floats(min_value=2.0, max_value=8.0),
    eps_h=st.floats(min_value=3.0, max_value=20.0),
    E=st.floats(min_value=-0.25, max_value=0.25),
)
@settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_gcs_cap_below_stern(log_c: float, z: int, d_ang: float, eps_h: float, E: float) -> None:
    """The series combination must always lie below the Stern capacitance."""
    params = GCSParameters(
        concentration_mol_l=10.0**log_c,
        valence=z,
        stern_thickness_m=d_ang * 1e-10,
        stern_permittivity=eps_h,
    )
    c_h = stern_capacitance(params)
    _, _, cap = gouy_chapman_stern(params, np.array([E]))
    assert float(cap[0]) <= c_h + 1e-12


def test_gcs_rejects_bad_params() -> None:
    """Negative stern thickness or permittivity is rejected."""
    with pytest.raises(ValueError):
        GCSParameters(0.1, stern_thickness_m=0.0)
    with pytest.raises(ValueError):
        GCSParameters(0.1, stern_permittivity=-1.0)
