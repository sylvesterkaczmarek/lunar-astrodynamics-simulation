"""Classical orbital-element conversions for non-singular elliptic orbits."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]
_TWO_PI = 2.0 * np.pi
_EPS = 1e-12


def _wrap(angle_rad: float) -> float:
    return float(angle_rad % _TWO_PI)


@dataclass(frozen=True)
class ClassicalElements:
    """Classical osculating elements in SI units and radians."""

    semi_major_axis_m: float
    eccentricity: float
    inclination_rad: float
    raan_rad: float
    argument_of_periapsis_rad: float
    true_anomaly_rad: float

    def __post_init__(self) -> None:
        values = np.array(
            [
                self.semi_major_axis_m,
                self.eccentricity,
                self.inclination_rad,
                self.raan_rad,
                self.argument_of_periapsis_rad,
                self.true_anomaly_rad,
            ],
            dtype=float,
        )
        if not np.all(np.isfinite(values)):
            raise ValueError("orbital elements must be finite")
        if self.semi_major_axis_m <= 0.0:
            raise ValueError("semi-major axis must be positive")
        if not 0.0 <= self.eccentricity < 1.0:
            raise ValueError("eccentricity must satisfy 0 <= e < 1")
        if not 0.0 <= self.inclination_rad <= np.pi:
            raise ValueError("inclination must be within [0, pi]")

    @property
    def semilatus_rectum_m(self) -> float:
        return self.semi_major_axis_m * (1.0 - self.eccentricity**2)


def state_from_elements(elements: ClassicalElements, mu_m3_s2: float) -> FloatArray:
    """Convert non-singular elliptic classical elements to Cartesian state."""
    a = float(elements.semi_major_axis_m)
    e = float(elements.eccentricity)
    inc = float(elements.inclination_rad)
    raan = float(elements.raan_rad)
    argp = float(elements.argument_of_periapsis_rad)
    nu = float(elements.true_anomaly_rad)

    if not np.isfinite(mu_m3_s2) or mu_m3_s2 <= 0.0:
        raise ValueError("mu_m3_s2 must be finite and positive")

    p = a * (1.0 - e * e)
    radius = p / (1.0 + e * np.cos(nu))
    r_pf = np.array([radius * np.cos(nu), radius * np.sin(nu), 0.0])
    v_pf = np.sqrt(mu_m3_s2 / p) * np.array([-np.sin(nu), e + np.cos(nu), 0.0])

    cO, sO = np.cos(raan), np.sin(raan)
    ci, si = np.cos(inc), np.sin(inc)
    co, so = np.cos(argp), np.sin(argp)
    rotation = np.array(
        [
            [cO * co - sO * so * ci, -cO * so - sO * co * ci, sO * si],
            [sO * co + cO * so * ci, -sO * so + cO * co * ci, -cO * si],
            [so * si, co * si, ci],
        ]
    )
    return np.concatenate((rotation @ r_pf, rotation @ v_pf)).astype(float)


def elements_from_state(state: ArrayLike, mu_m3_s2: float) -> ClassicalElements:
    """Convert Cartesian state to classical elements."""
    y = np.asarray(state, dtype=float)
    if y.shape != (6,) or not np.all(np.isfinite(y)):
        raise ValueError("state must be a finite six-vector")
    if not np.isfinite(mu_m3_s2) or mu_m3_s2 <= 0.0:
        raise ValueError("mu_m3_s2 must be finite and positive")

    r = y[:3]
    v = y[3:]
    r_norm = float(np.linalg.norm(r))
    v_norm = float(np.linalg.norm(v))
    if r_norm == 0.0:
        raise ValueError("position cannot be the central-body origin")

    h = np.cross(r, v)
    h_norm = float(np.linalg.norm(h))
    if h_norm == 0.0:
        raise ValueError("state has zero angular momentum")

    k = np.array([0.0, 0.0, 1.0])
    node = np.cross(k, h)
    node_norm = float(np.linalg.norm(node))
    e_vec = np.cross(v, h) / mu_m3_s2 - r / r_norm
    e = float(np.linalg.norm(e_vec))

    if node_norm < _EPS * h_norm:
        raise ValueError("RAAN is undefined for an equatorial orbit")
    if e < _EPS:
        raise ValueError("argument of periapsis is undefined for a circular orbit")

    energy = 0.5 * v_norm * v_norm - mu_m3_s2 / r_norm
    if energy >= 0.0:
        raise ValueError("this helper supports bound elliptic states only")
    a = -mu_m3_s2 / (2.0 * energy)

    inc = float(np.arccos(np.clip(h[2] / h_norm, -1.0, 1.0)))
    raan = _wrap(np.arctan2(node[1], node[0]))
    argp_y = np.dot(np.cross(node, e_vec), h) / (node_norm * e * h_norm)
    argp_x = np.dot(node, e_vec) / (node_norm * e)
    argp = _wrap(np.arctan2(argp_y, argp_x))
    nu_y = np.dot(np.cross(e_vec, r), h) / (e * r_norm * h_norm)
    nu_x = np.dot(e_vec, r) / (e * r_norm)
    nu = _wrap(np.arctan2(nu_y, nu_x))
    return ClassicalElements(a, e, inc, raan, argp, nu)


def mean_motion_rad_s(semi_major_axis_m: float, mu_m3_s2: float) -> float:
    if (
        not np.isfinite(semi_major_axis_m)
        or not np.isfinite(mu_m3_s2)
        or semi_major_axis_m <= 0.0
        or mu_m3_s2 <= 0.0
    ):
        raise ValueError("semi-major axis and mu must be finite and positive")
    return float(np.sqrt(mu_m3_s2 / semi_major_axis_m**3))


def orbital_period_s(semi_major_axis_m: float, mu_m3_s2: float) -> float:
    return float(_TWO_PI / mean_motion_rad_s(semi_major_axis_m, mu_m3_s2))


def analytical_j2_secular_rates(
    elements: ClassicalElements,
    mu_m3_s2: float,
    reference_radius_m: float,
    j2: float,
) -> tuple[float, float]:
    if (
        not np.isfinite(mu_m3_s2)
        or not np.isfinite(reference_radius_m)
        or not np.isfinite(j2)
        or mu_m3_s2 <= 0.0
        or reference_radius_m <= 0.0
    ):
        raise ValueError("J2 model parameters must be finite with positive mu and radius")
    p = elements.semilatus_rectum_m
    n = mean_motion_rad_s(elements.semi_major_axis_m, mu_m3_s2)
    scale = j2 * (reference_radius_m / p) ** 2
    cos_i = np.cos(elements.inclination_rad)
    raan_rate = -1.5 * n * scale * cos_i
    argp_rate = 0.75 * n * scale * (5.0 * cos_i * cos_i - 1.0)
    return float(raan_rate), float(argp_rate)
