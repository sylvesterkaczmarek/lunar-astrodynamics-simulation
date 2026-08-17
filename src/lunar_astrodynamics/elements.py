"""Classical, vector, and nonsingular orbital representations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]
_TWO_PI = 2.0 * np.pi
_EPS = 1e-12
_MEE_RETROGRADE_DENOM_EPS = 1e-14


def _wrap(angle_rad: float) -> float:
    return float(angle_rad % _TWO_PI)


def _validate_mu(mu_m3_s2: float) -> float:
    mu = float(mu_m3_s2)
    if not np.isfinite(mu) or mu <= 0.0:
        raise ValueError("mu_m3_s2 must be finite and positive")
    return mu


def _validate_state(state: ArrayLike) -> FloatArray:
    y = np.asarray(state, dtype=float)
    if y.shape != (6,) or not np.all(np.isfinite(y)):
        raise ValueError("state must be a finite six-vector")
    if float(np.linalg.norm(y[:3])) == 0.0:
        raise ValueError("position cannot be the central-body origin")
    return y


@dataclass(frozen=True)
class ClassicalElements:
    """Classical osculating elements for a bound elliptic orbit.

    These remain useful when the orbit is neither equatorial nor circular.
    ``elements_from_state`` intentionally raises instead of inventing RAAN or
    argument of periapsis values when those classical angles are undefined.
    """

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


@dataclass(frozen=True)
class OrbitalVectors:
    """Coordinate-safe osculating quantities derived directly from Cartesian state."""

    eccentricity_vector: FloatArray
    specific_angular_momentum_vector_m2_s: FloatArray
    orbital_plane_normal: FloatArray
    semi_major_axis_m: float
    semilatus_rectum_m: float
    eccentricity: float
    inclination_rad: float
    specific_orbital_energy_m2_s2: float

    def __post_init__(self) -> None:
        for field_name in (
            "eccentricity_vector",
            "specific_angular_momentum_vector_m2_s",
            "orbital_plane_normal",
        ):
            value = np.asarray(getattr(self, field_name), dtype=float).copy()
            if value.shape != (3,) or not np.all(np.isfinite(value)):
                raise ValueError(f"{field_name} must be a finite three-vector")
            value.setflags(write=False)
            object.__setattr__(self, field_name, value)
        scalars = np.array(
            [
                self.semilatus_rectum_m,
                self.eccentricity,
                self.inclination_rad,
                self.specific_orbital_energy_m2_s2,
            ],
            dtype=float,
        )
        if not np.all(np.isfinite(scalars)):
            raise ValueError("orbital vector scalars must be finite")
        if self.semilatus_rectum_m <= 0.0 or self.eccentricity < 0.0:
            raise ValueError("semilatus rectum must be positive and eccentricity non-negative")
        if not 0.0 <= self.inclination_rad <= np.pi:
            raise ValueError("inclination must be within [0, pi]")
        if not (np.isfinite(self.semi_major_axis_m) or np.isinf(self.semi_major_axis_m)):
            raise ValueError("semi_major_axis_m must be finite or infinite for a parabolic state")

    @property
    def periselene_radius_m(self) -> float:
        return float(self.semilatus_rectum_m / (1.0 + self.eccentricity))

    @property
    def aposelene_radius_m(self) -> float:
        if self.eccentricity >= 1.0:
            return float(np.inf)
        return float(self.semilatus_rectum_m / (1.0 - self.eccentricity))


@dataclass(frozen=True)
class ModifiedEquinoctialElements:
    """Walker-style prograde modified equinoctial elements ``(p,f,g,h,k,L)``.

    Conventions are

    ``p = a(1-e^2)``
    ``f = e cos(Omega + omega)``
    ``g = e sin(Omega + omega)``
    ``h = tan(i/2) cos(Omega)``
    ``k = tan(i/2) sin(Omega)``
    ``L = Omega + omega + nu``

    This tangent/prograde formulation is nonsingular for circular, prograde
    equatorial, and polar orbits. It is singular at exactly ``i = pi`` and is
    increasingly ill-conditioned close to that retrograde-equatorial limit.
    """

    semilatus_rectum_m: float
    f: float
    g: float
    h: float
    k: float
    true_longitude_rad: float

    def __post_init__(self) -> None:
        values = np.array(
            [
                self.semilatus_rectum_m,
                self.f,
                self.g,
                self.h,
                self.k,
                self.true_longitude_rad,
            ],
            dtype=float,
        )
        if not np.all(np.isfinite(values)):
            raise ValueError("modified equinoctial elements must be finite")
        if self.semilatus_rectum_m <= 0.0:
            raise ValueError("semilatus_rectum_m must be positive")
        object.__setattr__(self, "true_longitude_rad", _wrap(self.true_longitude_rad))

    @property
    def eccentricity(self) -> float:
        return float(np.hypot(self.f, self.g))

    @property
    def inclination_rad(self) -> float:
        return float(2.0 * np.arctan(np.hypot(self.h, self.k)))

    @property
    def semi_major_axis_m(self) -> float:
        denominator = 1.0 - self.eccentricity**2
        if abs(denominator) <= np.finfo(float).eps:
            return float(np.inf)
        return float(self.semilatus_rectum_m / denominator)

    @property
    def longitude_of_periapsis_rad(self) -> float | None:
        if self.eccentricity < _EPS:
            return None
        return _wrap(np.arctan2(self.g, self.f))

    @property
    def raan_rad(self) -> float | None:
        if np.hypot(self.h, self.k) < _EPS:
            return None
        return _wrap(np.arctan2(self.k, self.h))


def orbital_vectors_from_state(state: ArrayLike, mu_m3_s2: float) -> OrbitalVectors:
    """Return nonsingular vector orbital quantities from a Cartesian state."""
    y = _validate_state(state)
    mu = _validate_mu(mu_m3_s2)
    r = y[:3]
    v = y[3:]
    radius = float(np.linalg.norm(r))
    speed_squared = float(np.dot(v, v))
    angular_momentum = np.cross(r, v)
    angular_momentum_norm = float(np.linalg.norm(angular_momentum))
    if angular_momentum_norm == 0.0:
        raise ValueError("state has zero angular momentum")
    plane_normal = angular_momentum / angular_momentum_norm
    eccentricity_vector = np.cross(v, angular_momentum) / mu - r / radius
    eccentricity = float(np.linalg.norm(eccentricity_vector))
    energy = 0.5 * speed_squared - mu / radius
    energy_scale = mu / radius
    if abs(energy) <= 16.0 * np.finfo(float).eps * energy_scale:
        semi_major_axis = float(np.inf)
    else:
        semi_major_axis = float(-mu / (2.0 * energy))
    semilatus_rectum = float(angular_momentum_norm**2 / mu)
    inclination = float(np.arccos(np.clip(plane_normal[2], -1.0, 1.0)))
    return OrbitalVectors(
        eccentricity_vector=eccentricity_vector,
        specific_angular_momentum_vector_m2_s=angular_momentum,
        orbital_plane_normal=plane_normal,
        semi_major_axis_m=semi_major_axis,
        semilatus_rectum_m=semilatus_rectum,
        eccentricity=eccentricity,
        inclination_rad=inclination,
        specific_orbital_energy_m2_s2=float(energy),
    )


def _mee_basis(h: float, k: float) -> tuple[FloatArray, FloatArray]:
    denominator = 1.0 + h * h + k * k
    f_hat = np.array([1.0 + h * h - k * k, 2.0 * h * k, -2.0 * k], dtype=float) / denominator
    g_hat = np.array([2.0 * h * k, 1.0 - h * h + k * k, 2.0 * h], dtype=float) / denominator
    return f_hat, g_hat


def modified_equinoctial_from_state(
    state: ArrayLike,
    mu_m3_s2: float,
) -> ModifiedEquinoctialElements:
    """Convert Cartesian state to prograde modified equinoctial elements.

    No classical RAAN or argument of periapsis is constructed. The only
    orientation singularity is the prograde formulation's exact ``i = pi``
    limit, detected from the orbital-plane normal.
    """
    y = _validate_state(state)
    mu = _validate_mu(mu_m3_s2)
    vectors = orbital_vectors_from_state(y, mu)
    plane = vectors.orbital_plane_normal
    retrograde_denominator = 1.0 + float(plane[2])
    if retrograde_denominator <= _MEE_RETROGRADE_DENOM_EPS:
        raise ValueError(
            "prograde modified equinoctial elements are singular at the retrograde-equatorial i=pi limit"
        )
    h = -float(plane[1]) / retrograde_denominator
    k = float(plane[0]) / retrograde_denominator
    f_hat, g_hat = _mee_basis(h, k)
    f = float(np.dot(vectors.eccentricity_vector, f_hat))
    g = float(np.dot(vectors.eccentricity_vector, g_hat))
    position_hat = y[:3] / np.linalg.norm(y[:3])
    true_longitude = _wrap(
        np.arctan2(float(np.dot(position_hat, g_hat)), float(np.dot(position_hat, f_hat)))
    )
    return ModifiedEquinoctialElements(
        semilatus_rectum_m=vectors.semilatus_rectum_m,
        f=f,
        g=g,
        h=h,
        k=k,
        true_longitude_rad=true_longitude,
    )


def state_from_modified_equinoctial(
    elements: ModifiedEquinoctialElements,
    mu_m3_s2: float,
) -> FloatArray:
    """Convert prograde modified equinoctial elements to Cartesian state."""
    if not isinstance(elements, ModifiedEquinoctialElements):
        raise TypeError("elements must be ModifiedEquinoctialElements")
    mu = _validate_mu(mu_m3_s2)
    p = float(elements.semilatus_rectum_m)
    f = float(elements.f)
    g = float(elements.g)
    h = float(elements.h)
    k = float(elements.k)
    longitude = float(elements.true_longitude_rad)
    cosine = float(np.cos(longitude))
    sine = float(np.sin(longitude))
    radial_denominator = 1.0 + f * cosine + g * sine
    if radial_denominator <= 0.0 or not np.isfinite(radial_denominator):
        raise ValueError("modified equinoctial elements place the state outside the physical conic branch")
    f_hat, g_hat = _mee_basis(h, k)
    radius = p / radial_denominator
    position = radius * (cosine * f_hat + sine * g_hat)
    velocity = np.sqrt(mu / p) * (-(g + sine) * f_hat + (f + cosine) * g_hat)
    state = np.concatenate((position, velocity)).astype(float)
    if not np.all(np.isfinite(state)):
        raise ValueError("modified equinoctial conversion produced a non-finite state")
    return state


def modified_equinoctial_from_classical(elements: ClassicalElements) -> ModifiedEquinoctialElements:
    """Convert an explicitly supplied classical elliptic orbit to MEE."""
    longitude_of_periapsis = elements.raan_rad + elements.argument_of_periapsis_rad
    return ModifiedEquinoctialElements(
        semilatus_rectum_m=elements.semilatus_rectum_m,
        f=elements.eccentricity * np.cos(longitude_of_periapsis),
        g=elements.eccentricity * np.sin(longitude_of_periapsis),
        h=np.tan(0.5 * elements.inclination_rad) * np.cos(elements.raan_rad),
        k=np.tan(0.5 * elements.inclination_rad) * np.sin(elements.raan_rad),
        true_longitude_rad=longitude_of_periapsis + elements.true_anomaly_rad,
    )


def classical_from_modified_equinoctial(
    elements: ModifiedEquinoctialElements,
) -> ClassicalElements:
    """Convert MEE to classical elements only when all classical angles exist.

    Circular or prograde-equatorial MEE are valid states, but their classical
    argument of periapsis or RAAN is not. This function raises for those cases
    instead of selecting arbitrary zero angles.
    """
    eccentricity = elements.eccentricity
    inclination = elements.inclination_rad
    if np.hypot(elements.h, elements.k) < _EPS:
        raise ValueError("RAAN is undefined for an equatorial orbit")
    if eccentricity < _EPS:
        raise ValueError("argument of periapsis is undefined for a circular orbit")
    if eccentricity >= 1.0:
        raise ValueError("ClassicalElements supports bound elliptic states only")
    raan = _wrap(np.arctan2(elements.k, elements.h))
    longitude_of_periapsis = _wrap(np.arctan2(elements.g, elements.f))
    argument_of_periapsis = _wrap(longitude_of_periapsis - raan)
    true_anomaly = _wrap(elements.true_longitude_rad - longitude_of_periapsis)
    return ClassicalElements(
        semi_major_axis_m=elements.semi_major_axis_m,
        eccentricity=eccentricity,
        inclination_rad=inclination,
        raan_rad=raan,
        argument_of_periapsis_rad=argument_of_periapsis,
        true_anomaly_rad=true_anomaly,
    )


def state_from_elements(elements: ClassicalElements, mu_m3_s2: float) -> FloatArray:
    """Convert bound elliptic classical elements to Cartesian state."""
    a = float(elements.semi_major_axis_m)
    e = float(elements.eccentricity)
    inc = float(elements.inclination_rad)
    raan = float(elements.raan_rad)
    argp = float(elements.argument_of_periapsis_rad)
    nu = float(elements.true_anomaly_rad)
    mu = _validate_mu(mu_m3_s2)

    p = a * (1.0 - e * e)
    radius = p / (1.0 + e * np.cos(nu))
    r_pf = np.array([radius * np.cos(nu), radius * np.sin(nu), 0.0])
    v_pf = np.sqrt(mu / p) * np.array([-np.sin(nu), e + np.cos(nu), 0.0])

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
    """Convert Cartesian state to classical elements when the angles are defined."""
    y = _validate_state(state)
    mu = _validate_mu(mu_m3_s2)
    vectors = orbital_vectors_from_state(y, mu)
    r = y[:3]
    h = vectors.specific_angular_momentum_vector_m2_s
    h_norm = float(np.linalg.norm(h))
    e_vec = vectors.eccentricity_vector
    e = vectors.eccentricity

    reference_axis = np.array([0.0, 0.0, 1.0])
    node = np.cross(reference_axis, h)
    node_norm = float(np.linalg.norm(node))
    if node_norm < _EPS * h_norm:
        raise ValueError("RAAN is undefined for an equatorial orbit")
    if e < _EPS:
        raise ValueError("argument of periapsis is undefined for a circular orbit")
    if not np.isfinite(vectors.semi_major_axis_m) or vectors.semi_major_axis_m <= 0.0 or e >= 1.0:
        raise ValueError("this helper supports bound elliptic states only")

    r_norm = float(np.linalg.norm(r))
    inc = vectors.inclination_rad
    raan = _wrap(np.arctan2(node[1], node[0]))
    argp_y = np.dot(np.cross(node, e_vec), h) / (node_norm * e * h_norm)
    argp_x = np.dot(node, e_vec) / (node_norm * e)
    argp = _wrap(np.arctan2(argp_y, argp_x))
    nu_y = np.dot(np.cross(e_vec, r), h) / (e * r_norm * h_norm)
    nu_x = np.dot(e_vec, r) / (e * r_norm)
    nu = _wrap(np.arctan2(nu_y, nu_x))
    return ClassicalElements(vectors.semi_major_axis_m, e, inc, raan, argp, nu)


def mean_motion_rad_s(semi_major_axis_m: float, mu_m3_s2: float) -> float:
    if not np.isfinite(semi_major_axis_m) or semi_major_axis_m <= 0.0:
        raise ValueError("semi-major axis and mu must be finite and positive")
    mu = _validate_mu(mu_m3_s2)
    return float(np.sqrt(mu / semi_major_axis_m**3))


def orbital_period_s(semi_major_axis_m: float, mu_m3_s2: float) -> float:
    return float(_TWO_PI / mean_motion_rad_s(semi_major_axis_m, mu_m3_s2))


def analytical_j2_secular_rates(
    elements: ClassicalElements,
    mu_m3_s2: float,
    reference_radius_m: float,
    j2: float,
) -> tuple[float, float]:
    if (
        not np.isfinite(reference_radius_m)
        or not np.isfinite(j2)
        or reference_radius_m <= 0.0
    ):
        raise ValueError("J2 model parameters must be finite with positive mu and radius")
    mu = _validate_mu(mu_m3_s2)
    p = elements.semilatus_rectum_m
    n = mean_motion_rad_s(elements.semi_major_axis_m, mu)
    scale = j2 * (reference_radius_m / p) ** 2
    cos_i = np.cos(elements.inclination_rad)
    raan_rate = -1.5 * n * scale * cos_i
    argp_rate = 0.75 * n * scale * (5.0 * cos_i * cos_i - 1.0)
    return float(raan_rate), float(argp_rate)
