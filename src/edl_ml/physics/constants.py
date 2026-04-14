"""Physical constants used throughout the electric double layer model.

All constants are in SI units unless otherwise noted. Values follow CODATA 2018.
"""

from __future__ import annotations

from typing import Final

BOLTZMANN: Final[float] = 1.380_649e-23
"""Boltzmann constant, J/K."""

ELEMENTARY_CHARGE: Final[float] = 1.602_176_634e-19
"""Elementary charge, C."""

AVOGADRO: Final[float] = 6.022_140_76e23
"""Avogadro constant, 1/mol."""

VACUUM_PERMITTIVITY: Final[float] = 8.854_187_812_8e-12
"""Vacuum permittivity, F/m."""

WATER_PERMITTIVITY: Final[float] = 78.4
"""Relative permittivity of water at 298.15 K (dimensionless)."""

FARADAY: Final[float] = ELEMENTARY_CHARGE * AVOGADRO
"""Faraday constant, C/mol."""

GAS_CONSTANT: Final[float] = BOLTZMANN * AVOGADRO
"""Ideal gas constant, J/(mol K)."""

ROOM_TEMPERATURE: Final[float] = 298.15
"""Reference temperature, K."""
