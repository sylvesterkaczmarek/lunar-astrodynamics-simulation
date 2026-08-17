"""Physical constants and product metadata for lunar gravity models."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LunarJ2Model:
    """Parameters for an axisymmetric lunar J2 gravity model."""

    mu_m3_s2: float
    reference_radius_m: float
    j2: float
    collision_radius_m: float
    name: str


@dataclass(frozen=True)
class GravityProduct:
    """Metadata for an archived spherical-harmonic gravity product."""

    name: str
    product_id: str
    source_url: str
    label_url: str
    expected_size_bytes: int
    max_degree: int
    max_order: int
    reference_radius_m: float
    mu_m3_s2: float
    normalization: str
    body_fixed_frame: str


GRGM1200A_J2 = LunarJ2Model(
    mu_m3_s2=4.90280011526323e12,
    reference_radius_m=1_738_000.0,
    j2=203.224e-6,
    collision_radius_m=1_737_400.0,
    name="GRGM1200A low-degree J2 approximation",
)

# JPL DE440 mass parameters.  The Earth and Sun values are used by the
# ephemeris-driven third-body helpers; callers can supply different values when
# reproducing another ephemeris solution.
SUN_GM_DE440_M3_S2 = 1.32712440041279419e20
EARTH_GM_DE440_M3_S2 = 3.98600435507e14
MOON_GM_DE440_M3_S2 = 4.902800e12

# IAU 2012 exact astronomical unit and IAU 2015 nominal solar radius.
ASTRONOMICAL_UNIT_M = 149_597_870_700.0
SUN_NOMINAL_RADIUS_M = 695_700_000.0
MOON_MEAN_RADIUS_M = 1_737_400.0

# Representative mean direct solar momentum flux at one astronomical unit.
# This is configurable in SolarRadiationPressure; it is not treated as exact.
SOLAR_RADIATION_PRESSURE_1_AU_N_M2 = 4.56e-6

GRGM1200A = GravityProduct(
    name="GRGM1200A",
    product_id="GGGRX_1200A_SHA.TAB",
    source_url=(
        "https://pds-geosciences.wustl.edu/grail/grail-l-lgrs-5-rdr-v1/"
        "grail_1001/shadr/gggrx_1200a_sha.tab"
    ),
    label_url=(
        "https://pds-geosciences.wustl.edu/grail/grail-l-lgrs-5-rdr-v1/"
        "grail_1001/shadr/gggrx_1200a_sha.lbl"
    ),
    expected_size_bytes=88_059_844,
    max_degree=1200,
    max_order=1200,
    reference_radius_m=1_738_000.0,
    mu_m3_s2=4.90280011526323e12,
    normalization="4pi",
    body_fixed_frame="DE430 lunar principal-axes frame",
)
