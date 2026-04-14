"""Tests for the Poisson-Boltzmann solver and Debye length."""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from edl_ml.physics.pb import (
    PBParameters,
    debye_length,
    solve_poisson_boltzmann,
)


def test_debye_length_known_value() -> None:
    """The Debye length at 0.1 M, z=1, 298.15 K is ~0.961 nm."""
    value_nm = debye_length(0.1, 1, 298.15) * 1e9
    assert math.isclose(value_nm, 0.961, rel_tol=5e-3)


def test_debye_length_valence_scaling() -> None:
    """Doubling the valence halves the Debye length."""
    k1 = debye_length(0.01, 1)
    k2 = debye_length(0.01, 2)
    assert math.isclose(k1 / k2, 2.0, rel_tol=1e-10)


@given(
    c=st.floats(min_value=1e-4, max_value=1.0, allow_nan=False, allow_infinity=False),
    z=st.integers(min_value=1, max_value=3),
    t=st.floats(min_value=273.0, max_value=350.0),
)
@settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_debye_length_concentration_scaling(c: float, z: int, t: float) -> None:
    """``kd ∝ 1/sqrt(c)``: doubling c should scale kd by ``1/sqrt(2)``."""
    assume_ratio = debye_length(c, z, t) / debye_length(2.0 * c, z, t)
    assert math.isclose(assume_ratio, math.sqrt(2.0), rel_tol=1e-10)


def test_pb_profile_matches_analytical_gouy_chapman() -> None:
    """PB numerical profile matches the analytical Gouy-Chapman closed form."""
    from edl_ml.physics.constants import (
        BOLTZMANN,
        ELEMENTARY_CHARGE,
    )

    psi_d = 0.05
    c = 0.05
    params = PBParameters(
        concentration_mol_l=c,
        psi_diffuse_v=psi_d,
        domain_debye_lengths=40.0,
    )
    res = solve_poisson_boltzmann(params)
    kd = res.debye_length_m
    kbT_over_ze = BOLTZMANN * params.temperature_k / ELEMENTARY_CHARGE
    tanh_term = math.tanh(psi_d / (4.0 * kbT_over_ze)) * np.exp(-res.x_m / kd)
    psi_analytic = 4.0 * kbT_over_ze * np.arctanh(tanh_term)
    err = float(np.max(np.abs(res.psi_v - psi_analytic)))
    assert err < 1e-8  # volts


def test_pb_symmetry_under_sign_flip() -> None:
    """Flipping psi_d flips psi(x) but not |sigma|."""
    a = solve_poisson_boltzmann(PBParameters(0.05, psi_diffuse_v=+0.08))
    b = solve_poisson_boltzmann(PBParameters(0.05, psi_diffuse_v=-0.08))
    assert np.allclose(a.psi_v, -b.psi_v, atol=1e-8)
    assert math.isclose(abs(a.surface_charge_c_m2), abs(b.surface_charge_c_m2), rel_tol=1e-10)
    assert a.surface_charge_c_m2 * b.surface_charge_c_m2 < 0


def test_pb_bulk_boundary_condition() -> None:
    """Potential decays to ~0 at the bulk boundary of the domain."""
    res = solve_poisson_boltzmann(PBParameters(0.1, psi_diffuse_v=0.1))
    assert abs(res.psi_v[-1]) < 1e-6


def test_pb_rejects_bad_parameters() -> None:
    """Negative or zero physical parameters are rejected."""
    with pytest.raises(ValueError):
        PBParameters(concentration_mol_l=-1.0)
    with pytest.raises(ValueError):
        PBParameters(concentration_mol_l=0.1, valence=0)
    with pytest.raises(ValueError):
        PBParameters(concentration_mol_l=0.1, temperature_k=0.0)


@given(
    psi_d=st.floats(
        min_value=-0.25,
        max_value=0.25,
        allow_nan=False,
        allow_infinity=False,
    ),
)
@settings(max_examples=25, deadline=None)
def test_pb_grahame_consistency(psi_d: float) -> None:
    """Surface charge in the solver matches the closed-form Grahame equation."""
    from edl_ml.physics.constants import (
        AVOGADRO,
        BOLTZMANN,
        ELEMENTARY_CHARGE,
        VACUUM_PERMITTIVITY,
        WATER_PERMITTIVITY,
    )

    if abs(psi_d) < 1e-6:
        return
    c = 0.01
    res = solve_poisson_boltzmann(PBParameters(c, psi_diffuse_v=psi_d))
    prefactor = math.sqrt(
        8.0 * WATER_PERMITTIVITY * VACUUM_PERMITTIVITY * c * 1e3 * AVOGADRO * BOLTZMANN * 298.15
    )
    expected = (
        np.sign(psi_d)
        * prefactor
        * math.sinh(ELEMENTARY_CHARGE * abs(psi_d) / (2.0 * BOLTZMANN * 298.15))
    )
    assert math.isclose(res.surface_charge_c_m2, expected, rel_tol=1e-10)
