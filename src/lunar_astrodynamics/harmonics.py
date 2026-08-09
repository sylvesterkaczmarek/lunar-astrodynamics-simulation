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
        c = np.array(self.c, dtype=float, copy=True)
        s = np.array(self.s, dtype=float, copy=True)
        if c.ndim != 2 or c.shape[0] != c.shape[1] or s.shape != c.shape:
            raise ValueError("c and s must be matching square coefficient arrays")
        if (
            not np.isfinite(self.mu_m3_s2)
            or not np.isfinite(self.reference_radius_m)
            or self.mu_m3_s2 <= 0.0
            or self.reference_radius_m <= 0.0
        ):
            raise ValueError("mu and reference radius must be finite and positive")
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
        c.setflags(write=False)
        s.setflags(write=False)
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
            c[:, order + 1 :] = 0.0
            s[:, order + 1 :] = 0.0
        return SphericalHarmonicModel(
            self.mu_m3_s2,
            self.reference_radius_m,
            c,
            s,
            name=f"{self.name} (n<={degree}, m<={order})",
            frame=self.frame,
        )


def _parse_float(token: str) -> float:
    return float(token.strip().replace("D", "E").replace("d", "e"))


def _without_record_ending(raw_line: str) -> str:
    if raw_line.endswith("\r\n"):
        return raw_line[:-2]
    if raw_line.endswith("\n") or raw_line.endswith("\r"):
        return raw_line[:-1]
    return raw_line


def _field(record: str, start: int, width: int, label: str, record_number: int) -> str:
    value = record[start : start + width].strip()
    if not value:
        raise ValueError(f"empty {label} in SHADR record {record_number}")
    return value


def _header_from_record(record: str) -> tuple[float, float, int, int, int, float, float]:
    """Parse the 137-byte SHADR header data region using PDS column offsets."""
    if len(record) < 137:
        raise ValueError(
            f"SHADR header is too short: {len(record)} characters; expected at least 137"
        )
    try:
        radius_km = _parse_float(_field(record, 0, 23, "reference radius", 1))
        mu_km3_s2 = _parse_float(_field(record, 24, 23, "constant", 1))
        _parse_float(_field(record, 48, 23, "uncertainty in constant", 1))
        degree = int(_field(record, 72, 5, "degree", 1))
        order = int(_field(record, 78, 5, "order", 1))
        normalization = int(_field(record, 84, 5, "normalization state", 1))
        reference_longitude_deg = _parse_float(
            _field(record, 90, 23, "reference longitude", 1)
        )
        reference_latitude_deg = _parse_float(
            _field(record, 114, 23, "reference latitude", 1)
        )
    except ValueError as exc:
        raise ValueError(f"invalid SHADR header: {exc}") from exc

    if (
        not np.isfinite(radius_km)
        or not np.isfinite(mu_km3_s2)
        or radius_km <= 0.0
        or mu_km3_s2 <= 0.0
        or degree < 0
        or order < 0
        or order > degree
    ):
        raise ValueError("invalid SHADR header values")
    return (
        radius_km,
        mu_km3_s2,
        degree,
        order,
        normalization,
        reference_longitude_deg,
        reference_latitude_deg,
    )


def _coefficient_from_record(
    record: str, record_number: int
) -> tuple[int, int, float, float]:
    """Parse the 107-byte SHADR coefficient data region using PDS offsets."""
    if len(record) < 107:
        raise ValueError(
            f"SHADR coefficient record {record_number} is too short: "
            f"{len(record)} characters; expected at least 107"
        )
    try:
        n = int(_field(record, 0, 5, "coefficient degree", record_number))
        m = int(_field(record, 6, 5, "coefficient order", record_number))
        c_nm = _parse_float(_field(record, 12, 23, "C coefficient", record_number))
        s_nm = _parse_float(_field(record, 36, 23, "S coefficient", record_number))
        sigma_c = _parse_float(_field(record, 60, 23, "C uncertainty", record_number))
        sigma_s = _parse_float(_field(record, 84, 23, "S uncertainty", record_number))
    except ValueError as exc:
        raise ValueError(f"invalid SHADR coefficient record {record_number}: {exc}") from exc

    values = np.array([c_nm, s_nm, sigma_c, sigma_s], dtype=float)
    if n < 0 or m < 0 or m > n or not np.all(np.isfinite(values)):
        raise ValueError(f"invalid SHADR coefficient record {record_number}")
    return n, m, c_nm, s_nm


def read_shadr(
    path_or_file: str | Path | TextIO,
    *,
    max_degree: int | None = None,
    name: str | None = None,
    frame: str = "body-fixed principal-axes frame",
) -> SphericalHarmonicModel:
    """Read a PDS SHADR ASCII gravity model with normalized coefficients.

    PDS SHADR files use fixed 122-byte physical records. The header is one
    logical 244-byte line containing 137 bytes of delimited data, 105 padding
    bytes, and CRLF. Coefficient lines contain 107 bytes of comma-delimited
    data, 13 padding bytes, and CRLF. The format does not require coefficient
    rows to be ordered or complete, so this reader scans every row and indexes
    coefficients by their explicit degree and order.

    The evaluator assumes geodesy 4pi normalization. A SHADR normalization
    state of 1 is therefore necessary but callers must also verify that the
    specific product documentation defines that normalized convention.
    """
    close = False
    if hasattr(path_or_file, "read"):
        handle = path_or_file  # type: ignore[assignment]
    else:
        handle = open(
            path_or_file,
            "r",
            encoding="ascii",
            errors="strict",
            newline="",
        )
        close = True

    try:
        raw_header = handle.readline()
        if raw_header == "":
            raise ValueError("empty SHADR file")
        header_record = _without_record_ending(raw_header)
        (
            radius_km,
            mu_km3_s2,
            file_degree,
            file_order,
            normalization,
            reference_longitude_deg,
            reference_latitude_deg,
        ) = _header_from_record(header_record)

        if normalization != 1:
            raise ValueError("SHADR coefficients must be normalized (normalization state 1)")
        if abs(reference_longitude_deg) > 1e-12 or abs(reference_latitude_deg) > 1e-12:
            raise ValueError(
                "non-zero SHADR reference longitude/latitude is not supported by this evaluator"
            )

        degree = file_degree if max_degree is None else min(int(max_degree), file_degree)
        if degree < 0:
            raise ValueError("max_degree must be non-negative")
        c = np.zeros((degree + 1, degree + 1), dtype=float)
        s = np.zeros_like(c)
        c[0, 0] = 1.0
        seen: set[tuple[int, int]] = set()
        coefficient_rows = 0

        for record_number, raw_line in enumerate(handle, start=3):
            record = _without_record_ending(raw_line)
            if not record.strip():
                continue
            n, m, c_nm, s_nm = _coefficient_from_record(record, record_number)
            coefficient_rows += 1
            if n > file_degree or m > file_order:
                raise ValueError(
                    f"SHADR coefficient ({n}, {m}) exceeds header degree/order "
                    f"({file_degree}, {file_order})"
                )
            if n <= degree:
                key = (n, m)
                if key in seen:
                    raise ValueError(f"duplicate SHADR coefficient ({n}, {m})")
                seen.add(key)
                c[n, m] = c_nm
                s[n, m] = s_nm

        if coefficient_rows == 0 and file_degree > 0:
            raise ValueError("SHADR file contains no coefficient records")

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
