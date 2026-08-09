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

MOON_GM_DE440_M3_S2 = 4.902800e12
MOON_MEAN_RADIUS_M = 1_737_400.0

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
