"""Gouy-Chapman-Stern composite double-layer model.

The Stern layer behaves as a molecular condenser with thickness equal to the
closest approach of a hydrated ion to the electrode surface, and a dielectric
permittivity significantly reduced compared with bulk water due to the
orientation of water molecules near the interface.

The total differential capacitance is obtained by placing the Stern and
diffuse-layer capacitances in series,

.. math::

    \\frac{1}{C_\\text{dl}} = \\frac{1}{C_H} + \\frac{1}{C_d},

with :math:`C_H = \\epsilon_r^H \\epsilon_0 / d_H` the Helmholtz (Stern)
capacitance and :math:`C_d` the diffuse-layer capacitance from the
Gouy-Chapman model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from edl_ml.physics.constants import (
    AVOGADRO,
    BOLTZMANN,
    ELEMENTARY_CHARGE,
    ROOM_TEMPERATURE,
    VACUUM_PERMITTIVITY,
    WATER_PERMITTIVITY,
)


@dataclass(frozen=True, slots=True)
class GCSParameters:
    """Parameters defining a Gouy-Chapman-Stern electrode-electrolyte interface.

    Parameters
    ----------
    concentration_mol_l
        Bulk electrolyte concentration, mol/L.
    valence
        Ionic valence :math:`z` for a symmetric electrolyte.
    temperature_k
        Temperature in Kelvin.
    stern_thickness_m
        Thickness of the Stern (inner Helmholtz) layer, in metres. Typically
        3–6 Å.
    stern_permittivity
        Relative permittivity of the Stern layer, dimensionless. For water the
        oriented interfacial value is ~6.
    bulk_permittivity
        Relative permittivity of the bulk solvent, dimensionless.
    """

    concentration_mol_l: float
    valence: int = 1
    temperature_k: float = ROOM_TEMPERATURE
    stern_thickness_m: float = 3e-10
    stern_permittivity: float = 6.0
    bulk_permittivity: float = WATER_PERMITTIVITY

    def __post_init__(self) -> None:
        if self.concentration_mol_l <= 0:
            raise ValueError("concentration must be positive")
        if self.valence <= 0:
            raise ValueError("valence must be a positive integer")
        if self.temperature_k <= 0:
            raise ValueError("temperature must be positive")
        if self.stern_thickness_m <= 0:
            raise ValueError("stern_thickness must be positive")
        if self.stern_permittivity <= 0:
            raise ValueError("stern_permittivity must be positive")
        if self.bulk_permittivity <= 0:
            raise ValueError("bulk_permittivity must be positive")


def stern_capacitance(params: GCSParameters) -> float:
    """Return the Stern (Helmholtz) capacitance per unit area in F/m².

    Parameters
    ----------
    params
        GCS parameters.

    Returns
    -------
    float
        :math:`C_H = \\epsilon_r^H \\epsilon_0 / d_H`.
    """
    return float(params.stern_permittivity * VACUUM_PERMITTIVITY / params.stern_thickness_m)


def diffuse_capacitance(
    params: GCSParameters,
    psi_diffuse_v: float | NDArray[np.float64],
) -> float | NDArray[np.float64]:
    """Gouy-Chapman diffuse-layer differential capacitance.

    Parameters
    ----------
    params
        GCS parameters. Only the bulk-phase properties are used.
    psi_diffuse_v
        Potential at the Stern / diffuse boundary, V.

    Returns
    -------
    float or ndarray
        :math:`C_d = \\epsilon_r \\epsilon_0 \\kappa
        \\cosh(z e \\psi_d / 2 k_B T)`, in F/m².

    Notes
    -----
    The closed form is obtained by differentiating the Grahame equation with
    respect to the diffuse-layer potential.
    """
    from edl_ml.physics.pb import debye_length

    kd = debye_length(
        params.concentration_mol_l,
        params.valence,
        params.temperature_k,
        params.bulk_permittivity,
    )
    eps = params.bulk_permittivity * VACUUM_PERMITTIVITY
    kappa = 1.0 / kd
    thermal = BOLTZMANN * params.temperature_k
    argument = params.valence * ELEMENTARY_CHARGE * psi_diffuse_v / (2.0 * thermal)
    return eps * kappa * np.cosh(argument)


def total_capacitance(
    params: GCSParameters,
    psi_diffuse_v: float | NDArray[np.float64],
) -> float | NDArray[np.float64]:
    """Series combination of the Stern and diffuse-layer capacitances.

    Parameters
    ----------
    params
        GCS parameters.
    psi_diffuse_v
        Potential at the Stern / diffuse boundary, V.

    Returns
    -------
    float or ndarray
        :math:`C_\\text{dl}` in F/m².
    """
    c_h = stern_capacitance(params)
    c_d = diffuse_capacitance(params, psi_diffuse_v)
    return 1.0 / (1.0 / c_h + 1.0 / c_d)


def gouy_chapman_stern(
    params: GCSParameters,
    electrode_potentials_v: NDArray[np.float64],
    *,
    max_iter: int = 80,
    tol: float = 1e-10,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Solve the self-consistent Gouy-Chapman-Stern problem.

    The electrode potential :math:`E` relative to the point of zero charge is
    split across the Stern and diffuse layers,

    .. math::

        E = \\psi_H + \\psi_d,\\qquad \\psi_H = \\sigma / C_H,

    with :math:`\\sigma` given by the Grahame equation as a function of
    :math:`\\psi_d`. We solve the resulting nonlinear equation in
    :math:`\\psi_d` by bisection for each electrode potential.

    Parameters
    ----------
    params
        GCS parameters.
    electrode_potentials_v
        Grid of electrode potentials in volts, relative to the point of zero
        charge.
    max_iter
        Maximum bisection iterations per potential.
    tol
        Absolute tolerance on the residual ``E - (psi_H + psi_d)`` in volts.

    Returns
    -------
    surface_charge
        Array of diffuse-layer surface charge densities, C/m².
    psi_diffuse
        Array of diffuse-layer potentials, V.
    differential_capacitance
        Array of total differential capacitances at each electrode potential,
        F/m².
    """
    c_h = stern_capacitance(params)
    thermal = BOLTZMANN * params.temperature_k
    n0 = params.concentration_mol_l * 1e3 * AVOGADRO
    eps = params.bulk_permittivity * VACUUM_PERMITTIVITY
    prefactor = np.sqrt(8.0 * eps * n0 * thermal)

    def sigma_of_psid(psi_d: float) -> float:
        """Grahame equation."""
        return float(
            np.sign(psi_d)
            * prefactor
            * np.sinh(params.valence * ELEMENTARY_CHARGE * abs(psi_d) / (2.0 * thermal))
        )

    def residual(psi_d: float, e: float) -> float:
        return e - psi_d - sigma_of_psid(psi_d) / c_h

    sigma = np.zeros_like(electrode_potentials_v)
    psi_d_arr = np.zeros_like(electrode_potentials_v)
    for idx, e in enumerate(electrode_potentials_v):
        lo, hi = -abs(e) - 1.0, abs(e) + 1.0
        f_lo = residual(lo, float(e))
        f_hi = residual(hi, float(e))
        if f_lo * f_hi > 0:
            # Expand bracket; the residual is monotone in psi_d so this
            # terminates quickly.
            for _ in range(20):
                lo *= 2.0
                hi *= 2.0
                f_lo = residual(lo, float(e))
                f_hi = residual(hi, float(e))
                if f_lo * f_hi <= 0:
                    break
        psi_d = 0.5 * (lo + hi)
        for _ in range(max_iter):
            psi_d = 0.5 * (lo + hi)
            f_mid = residual(psi_d, float(e))
            if abs(f_mid) < tol:
                break
            if f_lo * f_mid < 0:
                hi, f_hi = psi_d, f_mid
            else:
                lo, f_lo = psi_d, f_mid
        psi_d_arr[idx] = psi_d
        sigma[idx] = sigma_of_psid(psi_d)
    cap = total_capacitance(params, psi_d_arr)
    return sigma, psi_d_arr, np.asarray(cap, dtype=np.float64)
