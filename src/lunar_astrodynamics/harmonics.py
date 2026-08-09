"""4pi-normalized lunar spherical-harmonic gravity models and SHADR parsing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .frames import RotationProvider, validate_rotation_matrix

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class SphericalHarmonicModel:
    """Fully normalized geodesy-4pi gravity coefficients in a body-fixed frame."""

    mu_m3_s2: float
    reference_radius_m: float
    c: FloatArray
    s: FloatArray
    name: str = "spherical-harmonic gravity model"
    frame: str = "body-fixed"
    normalization: str = "4pi"

    def __post_init__(self) -> None:
        c = np.asarray(self.c, dtype=float)
        s = np.asarray(self.s, dtype=float)
        if c.ndim != 2 or c.shape[0] != c.shape[1] or s.shape != c.shape:
            raise ValueError("c and s must be matching square coefficient arrays")
        if self.mu_m3_s2 <= 0.0 or self.reference_radius_m <= 0.0:
            raise ValueError("mu and reference radius must be positive")
        if self.normalization.lower() != "4pi":
            raise ValueError("this evaluator requires geodesy 4pi-normalized coefficients")
        if not np.all(np.isfinite(c)) or not np.all(np.isfinite(s)):
            raise ValueError("coefficients must be finite")
        if not np.isclose(c[0, 0], 1.0, atol=1e-15, rtol=0.0):
            raise ValueError("c[0,0] must be 1 for the central gravity term")
        if not np.isclose(s[0, 0], 0.0, atol=1e-15, rtol=0.0):
            raise ValueError("s[0,0] must be 0")
        upper = np.triu_indices(c.shape[0], k=1)
        if np.any(c[upper] != 0.0) or np.any(s[upper] != 0.0):
            raise ValueError("coefficients with order m > degree n must be zero")
        object.__setattr__(self, "c", c)
        object.__setattr__(self, "s", s)

    @property
    def max_degree(self) -> int:
        return self.c.shape[0] - 1

    @property
    def max_order(self) -> int:
        nonzero = np.argwhere((self.c != 0.0) | (self.s != 0.0))
        return int(nonzero[:, 1].max()) if nonzero.size else 0

    def truncated(self, max_degree: int, max_order: int | None = None) -> "SphericalHarmonicModel":
        degree = min(int(max_degree), self.max_degree)
        if degree < 0:
            raise ValueError("max_degree must be non-negative")
        order = degree if max_order is None else min(int(max_order), degree)
        if order < 0:
            raise ValueError("max_order must be non-negative")
        c = self.c[: degree + 1, : degree + 1].copy()
        s = self.s[: degree + 1, : degree + 1].copy()
        if order < degree:
            for n in range(degree + 1):
                c[n, order + 1 :] = 0.0
                s[n, order + 1 :] = 0.0
        return SphericalHarmonicModel(
            self.mu_m3_s2,
            self.reference_radius_m,
            c,
            s,
            name=f"{self.name} (n<={degree}, m<={order})",
            frame=self.frame,
        )


def _parse_float(token: str) -> float:
    return float(token.replace("D", "E").replace("d", "e"))


def _header_from_text(text: str) -> tuple[float, float, int, int, int] | None:
    fields = text.split()
    if len(fields) < 6:
        return None
    try:
        radius_km = _parse_float(fields[0])
        mu_km3_s2 = _parse_float(fields[1])
        _parse_float(fields[2])
        degree = int(fields[3])
        order = int(fields[4])
        normalization = int(fields[5])
    except ValueError:
        return None
    if radius_km <= 0.0 or mu_km3_s2 <= 0.0 or degree < 0 or order < 0:
        return None
    return radius_km, mu_km3_s2, degree, order, normalization


def read_shadr(
    path_or_file: str | Path | TextIO,
    *,
    max_degree: int | None = None,
    name: str | None = None,
    frame: str = "body-fixed principal-axes frame",
) -> SphericalHarmonicModel:
    """Read a PDS SHADR ASCII gravity model with normalized coefficients.

    GRAIL SHADR files place one logical header row across the first two
    fixed-length records and begin the coefficient table at record 3. The
    coefficient table omits C00, so this reader inserts C00=1 explicitly.
    """
    close = False
    if hasattr(path_or_file, "read"):
        handle = path_or_file  # type: ignore[assignment]
    else:
        handle = open(path_or_file, "r", encoding="ascii", errors="strict")
        close = True

    try:
        first = handle.readline()
        second = handle.readline()
        header = _header_from_text(first + " " + second)
        if header is None:
            raise ValueError("could not parse SHADR header from the first two records")
        radius_km, mu_km3_s2, file_degree, file_order, normalization = header
        if normalization != 1:
            raise ValueError("SHADR coefficients must be normalized (normalization state 1)")

        degree = file_degree if max_degree is None else min(int(max_degree), file_degree)
        if degree < 0:
            raise ValueError("max_degree must be non-negative")
        c = np.zeros((degree + 1, degree + 1), dtype=float)
        s = np.zeros_like(c)
        c[0, 0] = 1.0

        for raw_line in handle:
            fields = raw_line.split()
            if len(fields) < 6:
                continue
            try:
                n = int(fields[0])
                m = int(fields[1])
                c_nm = _parse_float(fields[2])
                s_nm = _parse_float(fields[3])
                _parse_float(fields[4])
                _parse_float(fields[5])
            except ValueError:
                continue
            if n < 0 or m < 0 or m > n:
                raise ValueError(f"invalid SHADR degree/order pair ({n}, {m})")
            if n <= degree:
                c[n, m] = c_nm
                s[n, m] = s_nm

        return SphericalHarmonicModel(
            mu_m3_s2=mu_km3_s2 * 1e9,
            reference_radius_m=radius_km * 1e3,
            c=c,
            s=s,
            name=name or getattr(path_or_file, "name", "SHADR gravity model"),
            frame=frame,
        )
    finally:
        if close:
            handle.close()


def normalized_legendre_4pi(
    latitude_rad: float, max_degree: int
) -> tuple[FloatArray, FloatArray]:
    """Return geodesy-4pi normalized Pbar_nm and dPbar_nm/d(latitude).

    The forward recursion operates directly on normalized functions, avoiding
    factorial overflow and remaining finite at the poles. No Condon-Shortley
    phase is included, matching geodetic gravity-field convention.
    """
    nmax = int(max_degree)
    if nmax < 0:
        raise ValueError("max_degree must be non-negative")
    if not np.isfinite(latitude_rad):
        raise ValueError("latitude must be finite")

    x = float(np.sin(latitude_rad))
    t = float(np.cos(latitude_rad))
    p = np.zeros((nmax + 1, nmax + 1), dtype=float)
    dp = np.zeros_like(p)
    p[0, 0] = 1.0
    if nmax == 0:
        return p, dp

    sqrt3 = np.sqrt(3.0)
    p[1, 0] = sqrt3 * x
    dp[1, 0] = sqrt3 * t
    p[1, 1] = sqrt3 * t
    dp[1, 1] = -sqrt3 * x

    for n in range(2, nmax + 1):
        diag = np.sqrt((2.0 * n + 1.0) / (2.0 * n))
        prev_diag = p[n - 1, n - 1]
        prev_diag_d = dp[n - 1, n - 1]
        p[n, n] = diag * t * prev_diag
        dp[n, n] = diag * (-x * prev_diag + t * prev_diag_d)

        subdiag = np.sqrt(2.0 * n + 1.0)
        p[n, n - 1] = subdiag * x * prev_diag
        dp[n, n - 1] = subdiag * (t * prev_diag + x * prev_diag_d)

        m = np.arange(n - 1, dtype=float)
        denominator = n * n - m * m
        a = np.sqrt((4.0 * n * n - 1.0) / denominator)
        b = np.sqrt(
            (2.0 * n + 1.0)
            * ((n - 1.0) ** 2 - m * m)
            / ((2.0 * n - 3.0) * denominator)
        )
        p[n, : n - 1] = a * x * p[n - 1, : n - 1] - b * p[n - 2, : n - 1]
        dp[n, : n - 1] = (
            a * (t * p[n - 1, : n - 1] + x * dp[n - 1, : n - 1])
            - b * dp[n - 2, : n - 1]
        )
    return p, dp


def _evaluation_limits(
    model: SphericalHarmonicModel,
    max_degree: int | None,
    max_order: int | None,
) -> tuple[int, int]:
    nmax = model.max_degree if max_degree is None else min(int(max_degree), model.max_degree)
    if nmax < 0:
        raise ValueError("max_degree must be non-negative")
    mmax = nmax if max_order is None else min(int(max_order), nmax)
    if mmax < 0:
        raise ValueError("max_order must be non-negative")
    return nmax, mmax


def _body_fixed_spherical(
    position_m: ArrayLike,
) -> tuple[FloatArray, float, float, float, float]:
    r_vec = np.asarray(position_m, dtype=float)
    if r_vec.shape != (3,) or not np.all(np.isfinite(r_vec)):
        raise ValueError("position must be a finite 3-vector")
    radius = float(np.linalg.norm(r_vec))
    if radius == 0.0:
        raise ValueError("position cannot be the central-body origin")
    transverse = float(np.hypot(r_vec[0], r_vec[1]))
    latitude = float(np.arctan2(r_vec[2], transverse))
    longitude = float(np.arctan2(r_vec[1], r_vec[0])) if transverse > 0.0 else 0.0
    return r_vec, radius, latitude, longitude, transverse


def gravity_potential_body_fixed(
    position_m: ArrayLike,
    model: SphericalHarmonicModel,
    *,
    max_degree: int | None = None,
    max_order: int | None = None,
) -> float:
    """Evaluate positive gravitational potential U in m^2/s^2."""
    _, radius, latitude, longitude, _ = _body_fixed_spherical(position_m)
    nmax, mmax = _evaluation_limits(model, max_degree, max_order)
    p, _ = normalized_legendre_4pi(latitude, nmax)
    m = np.arange(mmax + 1, dtype=float)
    cos_m = np.cos(m * longitude)
    sin_m = np.sin(m * longitude)
    radial_ratio = model.reference_radius_m / radius
    radial_power = 1.0
    total = 0.0
    for n in range(nmax + 1):
        upper = min(n, mmax) + 1
        harmonics = model.c[n, :upper] * cos_m[:upper] + model.s[n, :upper] * sin_m[:upper]
        total += radial_power * float(np.dot(p[n, :upper], harmonics))
        radial_power *= radial_ratio
    return float(model.mu_m3_s2 / radius * total)


def gravity_acceleration_body_fixed(
    position_m: ArrayLike,
    model: SphericalHarmonicModel,
    *,
    max_degree: int | None = None,
    max_order: int | None = None,
) -> FloatArray:
    """Evaluate spherical-harmonic gravitational acceleration in m/s^2."""
    r_vec, radius, latitude, longitude, transverse = _body_fixed_spherical(position_m)
    nmax, mmax = _evaluation_limits(model, max_degree, max_order)

    has_non_zonal = mmax > 0 and (
        np.any(model.c[: nmax + 1, 1 : mmax + 1] != 0.0)
        or np.any(model.s[: nmax + 1, 1 : mmax + 1] != 0.0)
    )
    if transverse <= 1e-14 * radius and has_non_zonal:
        eps = 1e-10
        sign = 1.0 if r_vec[2] >= 0.0 else -1.0
        proxy = np.array([eps * radius, 0.0, sign * radius * np.sqrt(1.0 - eps * eps)])
        return gravity_acceleration_body_fixed(
            proxy, model, max_degree=max_degree, max_order=max_order
        )
    p, dp = normalized_legendre_4pi(latitude, nmax)
    m = np.arange(mmax + 1, dtype=float)
    cos_m = np.cos(m * longitude)
    sin_m = np.sin(m * longitude)

    radial_ratio = model.reference_radius_m / radius
    radial_power = 1.0
    sum_r = 0.0
    sum_lat = 0.0
    sum_lon = 0.0
    for n in range(nmax + 1):
        upper = min(n, mmax) + 1
        c = model.c[n, :upper]
        s = model.s[n, :upper]
        trig = c * cos_m[:upper] + s * sin_m[:upper]
        p_row = p[n, :upper]
        weighted = radial_power
        sum_r += (n + 1.0) * weighted * float(np.dot(p_row, trig))
        sum_lat += weighted * float(np.dot(dp[n, :upper], trig))
        if upper > 1:
            dlon = m[:upper] * (-c * sin_m[:upper] + s * cos_m[:upper])
            sum_lon += weighted * float(np.dot(p_row, dlon))
        radial_power *= radial_ratio

    mu_over_r2 = model.mu_m3_s2 / radius**2
    a_r = -mu_over_r2 * sum_r
    a_lat = mu_over_r2 * sum_lat
    cos_lat = float(np.cos(latitude))
    a_lon = (
        0.0
        if mmax == 0 or abs(cos_lat) < 1e-15
        else mu_over_r2 * sum_lon / cos_lat
    )

    sin_lat = float(np.sin(latitude))
    cos_lon = float(np.cos(longitude))
    sin_lon = float(np.sin(longitude))
    e_r = np.array([cos_lat * cos_lon, cos_lat * sin_lon, sin_lat])
    e_lat = np.array([-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat])
    e_lon = np.array([-sin_lon, cos_lon, 0.0])
    return (a_r * e_r + a_lat * e_lat + a_lon * e_lon).astype(float)


def gravity_acceleration_inertial(
    time_s: float,
    position_inertial_m: ArrayLike,
    model: SphericalHarmonicModel,
    body_fixed_from_inertial: RotationProvider,
    *,
    max_degree: int | None = None,
    max_order: int | None = None,
) -> FloatArray:
    """Evaluate body-fixed harmonics and rotate acceleration back to inertial."""
    r_i = np.asarray(position_inertial_m, dtype=float)
    if r_i.shape != (3,) or not np.all(np.isfinite(r_i)):
        raise ValueError("position must be a finite 3-vector")
    rotation = validate_rotation_matrix(body_fixed_from_inertial(float(time_s)))
    r_b = rotation @ r_i
    a_b = gravity_acceleration_body_fixed(
        r_b, model, max_degree=max_degree, max_order=max_order
    )
    return (rotation.T @ a_b).astype(float)
