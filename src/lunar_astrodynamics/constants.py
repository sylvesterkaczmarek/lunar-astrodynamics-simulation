"""Physical constants used by the demonstration models."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LunarJ2Model:
    """Parameters for an axisymmetric lunar J2 gravity model.

    ``reference_radius_m`` belongs to the gravity model and is distinct from
    ``collision_radius_m``, which is a simple mean-radius surface used only for
    impact termination in this educational propagator.
    """

    mu_m3_s2: float
    reference_radius_m: float
    j2: float
    collision_radius_m: float
    name: str


# GRGM1200A archive metadata gives GM = 4902.80011526323 km^3/s^2 and a
# reference radius of 1738.0 km. The J2 value below is the rounded low-degree
# GRGM1200A value reported in the literature (203.224e-6).
#
# The 1737.4 km collision radius is the JPL mean lunar radius. It is not used as
# the spherical-harmonic reference radius.
GRGM1200A_J2 = LunarJ2Model(
    mu_m3_s2=4.90280011526323e12,
    reference_radius_m=1_738_000.0,
    j2=203.224e-6,
    collision_radius_m=1_737_400.0,
    name="GRGM1200A low-degree J2 approximation",
)

# Rounded DE440 value published by JPL, kept for comparison and provenance.
MOON_GM_DE440_M3_S2 = 4.902800e12
MOON_MEAN_RADIUS_M = 1_737_400.0
