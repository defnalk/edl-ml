"""Poisson-Boltzmann solver for the diffuse layer of a symmetric electrolyte.

The full nonlinear one-dimensional Poisson-Boltzmann equation for a :math:`z:z`
symmetric electrolyte reads

.. math::

    \\frac{d^2 \\psi}{dx^2} = \\frac{2 z e n_0}{\\epsilon_r \\epsilon_0}
                             \\sinh\\!\\left(\\frac{z e \\psi}{k_B T}\\right),

with :math:`\\psi` the electrostatic potential, :math:`n_0` the bulk number
density of each ion, :math:`z` the valence, :math:`T` the temperature and
:math:`\\epsilon_r` the relative permittivity of the solvent.

For a one-dimensional symmetric electrolyte in contact with a planar electrode
there is a closed-form first integral

.. math::

    \\frac{d\\psi}{dx} = -\\,\\mathrm{sgn}(\\psi)\\;\\frac{2 k_B T \\kappa}{z e}
                         \\sinh\\!\\left(\\frac{z e \\psi}{2 k_B T}\\right),

which lets us evaluate the diffuse-layer surface charge density analytically
via Gauss's law while still integrating the full nonlinear ODE numerically to
recover the spatial profile.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

from edl_ml.physics.constants import (
    AVOGADRO,
    BOLTZMANN,
    ELEMENTARY_CHARGE,
    ROOM_TEMPERATURE,
    VACUUM_PERMITTIVITY,
    WATER_PERMITTIVITY,
)


@dataclass(frozen=True, slots=True)
class PBParameters:
    """Input parameters to the Poisson-Boltzmann solver.

    Parameters
    ----------
    concentration_mol_l
        Bulk electrolyte concentration in mol/L.
    valence
        Ionic valence :math:`z` for the symmetric electrolyte.
    temperature_k
        Absolute temperature in Kelvin.
    relative_permittivity
        Relative permittivity of the solvent.
    psi_diffuse_v
        Potential at the Stern / diffuse boundary, in volts.
    domain_debye_lengths
        Length of the integration domain measured in Debye lengths.
    n_points
        Number of points used to represent the solution on export.
    """

    concentration_mol_l: float
    valence: int = 1
    temperature_k: float = ROOM_TEMPERATURE
    relative_permittivity: float = WATER_PERMITTIVITY
    psi_diffuse_v: float = 0.05
    domain_debye_lengths: float = 30.0
    n_points: int = 400

    def __post_init__(self) -> None:
        if self.concentration_mol_l <= 0:
            raise ValueError("concentration must be positive")
        if self.valence <= 0:
            raise ValueError("valence must be a positive integer")
        if self.temperature_k <= 0:
            raise ValueError("temperature must be positive")
        if self.relative_permittivity <= 0:
            raise ValueError("relative_permittivity must be positive")
        if self.domain_debye_lengths <= 0:
            raise ValueError("domain_debye_lengths must be positive")
        if self.n_points < 10:
            raise ValueError("n_points must be at least 10")


@dataclass(frozen=True, slots=True)
class PBResult:
    """Output of the Poisson-Boltzmann solver.

    Attributes
    ----------
    x_m
        Position array in metres, measured from the Stern / diffuse boundary.
    psi_v
        Electrostatic potential in volts.
    field_v_m
        Electric field :math:`-d\\psi/dx` in V/m.
    cation_density_m3
        Cation number density as a function of position, in 1/m³.
    anion_density_m3
        Anion number density as a function of position, in 1/m³.
    surface_charge_c_m2
        Diffuse-layer surface charge density, in C/m² (Grahame equation).
    debye_length_m
        Debye screening length, in metres.
    """

    x_m: NDArray[np.float64]
    psi_v: NDArray[np.float64]
    field_v_m: NDArray[np.float64]
    cation_density_m3: NDArray[np.float64]
    anion_density_m3: NDArray[np.float64]
    surface_charge_c_m2: float
    debye_length_m: float


def debye_length(
    concentration_mol_l: float,
    valence: int = 1,
    temperature_k: float = ROOM_TEMPERATURE,
    relative_permittivity: float = WATER_PERMITTIVITY,
) -> float:
    """Return the Debye screening length for a symmetric electrolyte.

    Parameters
    ----------
    concentration_mol_l
        Bulk concentration of each ion in mol/L.
    valence
        Ionic valence.
    temperature_k
        Temperature in Kelvin.
    relative_permittivity
        Relative permittivity of the solvent.

    Returns
    -------
    float
        Debye length in metres.

    Notes
    -----
    The Debye length in a symmetric electrolyte is

    .. math::

        \\kappa^{-1} = \\sqrt{\\frac{\\epsilon_r \\epsilon_0 k_B T}
                                     {2 N_A e^2 z^2 c}}

    where :math:`c` is the bulk concentration in mol/m³.
    """
    c_m3 = concentration_mol_l * 1e3
    denom = 2.0 * AVOGADRO * ELEMENTARY_CHARGE**2 * valence**2 * c_m3
    num = relative_permittivity * VACUUM_PERMITTIVITY * BOLTZMANN * temperature_k
    return float(np.sqrt(num / denom))


def _grahame_surface_charge(
    psi_diffuse_v: float,
    concentration_mol_l: float,
    valence: int,
    temperature_k: float,
    relative_permittivity: float,
) -> float:
    """Closed-form diffuse-layer charge density from the Grahame equation.

    .. math::

        \\sigma_d = \\mathrm{sgn}(\\psi_d)\\sqrt{8 \\epsilon_r \\epsilon_0
                    n_0 k_B T}\\,
                    \\sinh\\!\\left(\\frac{z e \\psi_d}{2 k_B T}\\right).
    """
    eps = relative_permittivity * VACUUM_PERMITTIVITY
    n0 = concentration_mol_l * 1e3 * AVOGADRO
    thermal = BOLTZMANN * temperature_k
    prefactor = np.sqrt(8.0 * eps * n0 * thermal)
    return float(
        np.sign(psi_diffuse_v)
        * prefactor
        * np.sinh(valence * ELEMENTARY_CHARGE * abs(psi_diffuse_v) / (2.0 * thermal))
    )


def _first_integral_rhs(
    x: float,
    y: NDArray[np.float64],
    kappa: float,
    thermal_scale: float,
) -> NDArray[np.float64]:
    """Right-hand side of the Gouy-Chapman first-order ODE for ``psi(x)``.

    The second-order PB equation admits an analytical first integral subject
    to the bulk boundary condition :math:`\\psi(\\infty) = \\psi'(\\infty) = 0`:

    .. math::

        \\frac{d\\psi}{dx} = -\\,\\mathrm{sgn}(\\psi_d)\\,
                             \\frac{2 k_B T \\kappa}{z e}\\,
                             \\sinh\\!\\left(\\frac{z e \\psi}{2 k_B T}\\right).

    Integrating this first-order form is numerically stable because the ODE is
    monotonically contractive toward the bulk state, unlike the second-order
    form whose :math:`\\psi = 0` fixed point is hyperbolic.
    """
    psi = y[0]
    reduced = np.clip(psi / (2.0 * thermal_scale), -40.0, 40.0)
    sign = np.sign(psi) if psi != 0.0 else 0.0
    return np.array(
        [
            -sign * 2.0 * thermal_scale * kappa * np.sinh(abs(reduced)),
        ]
    )


def solve_poisson_boltzmann(params: PBParameters) -> PBResult:
    """Solve the nonlinear Poisson-Boltzmann equation for the diffuse layer.

    The first-integral formulation of the Gouy-Chapman equation is integrated
    forward from ``x = 0`` with ``psi(0) = psi_d``. This formulation is
    equivalent to the full second-order PB equation under the boundary
    condition ``psi → 0`` as ``x → infinity``, and is numerically stable
    because the bulk fixed point is attracting on forward integration. The
    surface charge density is reported via the Grahame equation.

    Parameters
    ----------
    params
        Solver parameters.

    Returns
    -------
    PBResult
        Potential, field and ion density profiles with the derived surface
        charge density and Debye length.
    """
    kd = debye_length(
        params.concentration_mol_l,
        params.valence,
        params.temperature_k,
        params.relative_permittivity,
    )
    thermal_scale = BOLTZMANN * params.temperature_k / (params.valence * ELEMENTARY_CHARGE)
    length = params.domain_debye_lengths * kd
    kappa = 1.0 / kd

    psi_d = params.psi_diffuse_v
    x_eval = np.linspace(0.0, length, params.n_points)
    sol = solve_ivp(
        _first_integral_rhs,
        (0.0, length),
        np.array([psi_d]),
        args=(kappa, thermal_scale),
        method="LSODA",
        rtol=1e-10,
        atol=1e-14,
        t_eval=x_eval,
    )
    if not sol.success:
        raise RuntimeError(f"PB solver failed: {sol.message}")

    psi = sol.y[0]
    # Derivative from the same first-integral relation.
    reduced_half = np.clip(psi / (2.0 * thermal_scale), -40.0, 40.0)
    dpsi = -np.sign(psi) * 2.0 * thermal_scale * kappa * np.sinh(np.abs(reduced_half))
    field = -dpsi

    c_m3 = params.concentration_mol_l * 1e3 * AVOGADRO
    reduced = np.clip(psi / thermal_scale, -80.0, 80.0)
    cation = c_m3 * np.exp(-reduced)
    anion = c_m3 * np.exp(reduced)

    sigma = _grahame_surface_charge(
        psi_d,
        params.concentration_mol_l,
        params.valence,
        params.temperature_k,
        params.relative_permittivity,
    )

    return PBResult(
        x_m=x_eval,
        psi_v=psi,
        field_v_m=field,
        cation_density_m3=cation,
        anion_density_m3=anion,
        surface_charge_c_m2=sigma,
        debye_length_m=kd,
    )
