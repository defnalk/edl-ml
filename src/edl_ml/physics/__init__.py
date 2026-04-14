"""Physics layer: Gouy-Chapman-Stern and Poisson-Boltzmann solvers."""

from edl_ml.physics.constants import (
    AVOGADRO,
    BOLTZMANN,
    ELEMENTARY_CHARGE,
    VACUUM_PERMITTIVITY,
    WATER_PERMITTIVITY,
)
from edl_ml.physics.gcs import (
    GCSParameters,
    diffuse_capacitance,
    gouy_chapman_stern,
    stern_capacitance,
    total_capacitance,
)
from edl_ml.physics.pb import (
    PBParameters,
    PBResult,
    debye_length,
    solve_poisson_boltzmann,
)

__all__ = [
    "AVOGADRO",
    "BOLTZMANN",
    "ELEMENTARY_CHARGE",
    "VACUUM_PERMITTIVITY",
    "WATER_PERMITTIVITY",
    "GCSParameters",
    "PBParameters",
    "PBResult",
    "debye_length",
    "diffuse_capacitance",
    "gouy_chapman_stern",
    "solve_poisson_boltzmann",
    "stern_capacitance",
    "total_capacitance",
]
