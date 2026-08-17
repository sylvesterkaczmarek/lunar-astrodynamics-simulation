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
    sigma_c: FloatArray | None = None
    sigma_s: FloatArray | None = None
    mu_sigma_m3_s2: float | None = None

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

        if (self.sigma_c is None) != (self.sigma_s is None):
            raise ValueError("sigma_c and sigma_s must either both be provided or both be omitted")
        sigma_c: FloatArray | None = None
        sigma_s: FloatArray | None = None
        if self.sigma_c is not None and self.sigma_s is not None:
            sigma_c = np.array(self.sigma_c, dtype=float, copy=True)
            sigma_s = np.array(self.sigma_s, dtype=float, copy=True)
            if sigma_c.shape != c.shape or sigma_s.shape != c.shape:
                raise ValueError("sigma_c and sigma_s must match coefficient array shapes")
            if (
                not np.all(np.isfinite(sigma_c))
                or not np.all(np.isfinite(sigma_s))
                or np.any(sigma_c < 0.0)
                or np.any(sigma_s < 0.0)
            ):
                raise ValueError("coefficient uncertainties must be finite and non-negative")
            if np.any(sigma_c[upper] != 0.0) or np.any(sigma_s[upper] != 0.0):
                raise ValueError("uncertainties with order m > degree n must be zero")
            sigma_c.setflags(write=False)
            sigma_s.setflags(write=False)

        mu_sigma = self.mu_sigma_m3_s2
        if mu_sigma is not None and (not np.isfinite(mu_sigma) or mu_sigma < 0.0):
            raise ValueError("mu_sigma_m3_s2 must be finite and non-negative when provided")

        c.setflags(write=False)
        s.setflags(write=False)
        object.__setattr__(self, "c", c)
        object.__setattr__(self, "s", s)
        object.__setattr__(self, "sigma_c", sigma_c)
        object.__setattr__(self, "sigma_s", sigma_s)

    @property
    def max_degree(self) -> int:
        return self.c.shape[0] - 1

    @property
    def max_order(self) -> int:
        nonzero = np.argwhere((self.c != 0.0) | (self.s != 0.0))
        return int(nonzero[:, 1].max()) if nonzero.size else 0

    @property
    def has_coefficient_uncertainty(self) -> bool:
        return self.sigma_c is not None and self.sigma_s is not None

    def coefficient_uncertainty(self, degree: int, order: int) -> tuple[float, float]:
        """Return the archived C/S uncertainty fields for coefficient (n, m).

        The SHADR SIS calls these values coefficient uncertainties. Whether they
        are calibrated standard deviations is product-specific; GRGM1200A's
        product documentation identifies its archived coefficient uncertainties
        as calibrated uncertainties. This method does not imply independence.
        """
        n = int(degree)
        m = int(order)
        if n < 0 or n > self.max_degree or m < 0 or m > n:
            raise ValueError("coefficient degree/order is outside the model")
        if self.sigma_c is None or self.sigma_s is None:
            raise ValueError("this gravity model does not include coefficient uncertainties")
        return float(self.sigma_c[n, m]), float(self.sigma_s[n, m])

    def truncated(self, max_degree: int, max_order: int | None = None) -> "SphericalHarmonicModel":
        degree = min(int(max_degree), self.max_degree)
        if degree < 0:
            raise ValueError("max_degree must be non-negative")
        order = degree if max_order is None else min(int(max_order), degree)
        if order < 0:
            raise ValueError("max_order must be non-negative")
        c = self.c[: degree + 1, : degree + 1].copy()
        s = self.s[: degree + 1, : degree + 1].copy()
        sigma_c = (
            None
            if self.sigma_c is None
            else self.sigma_c[: degree + 1, : degree + 1].copy()
        )
        sigma_s = (
            None
            if self.sigma_s is None
            else self.sigma_s[: degree + 1, : degree + 1].copy()
        )
        if order < degree:
            c[:, order + 1 :] = 0.0
            s[:, order + 1 :] = 0.0
            if sigma_c is not None and sigma_s is not None:
                sigma_c[:, order + 1 :] = 0.0
                sigma_s[:, order + 1 :] = 0.0
        return SphericalHarmonicModel(
            self.mu_m3_s2,
            self.reference_radius_m,
            c,
            s,
            name=f"{self.name} (n<={degree}, m<={order})",
            frame=self.frame,
            normalization=self.normalization,
            sigma_c=sigma_c,
            sigma_s=sigma_s,
            mu_sigma_m3_s2=self.mu_sigma_m3_s2,
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


def _header_from_record(
    record: str,
) -> tuple[float, float, float, int, int, int, float, float]:
    """Parse the 137-byte SHADR header data region using PDS column offsets."""
    if len(record) < 137:
        raise ValueError(
            f"SHADR header is too short: {len(record)} characters; expected at least 137"
        )
    try:
        radius_km = _parse_float(_field(record, 0, 23, "reference radius", 1))
        mu_km3_s2 = _parse_float(_field(record, 24, 23, "constant", 1))
        mu_sigma_km3_s2 = _parse_float(
            _field(record, 48, 23, "uncertainty in constant", 1)
        )
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
        or not np.isfinite(mu_sigma_km3_s2)
        or radius_km <= 0.0
        or mu_km3_s2 <= 0.0
        or mu_sigma_km3_s2 < 0.0
        or degree < 0
        or order < 0
        or order > degree
    ):
        raise ValueError("invalid SHADR header values")
    return (
        radius_km,
        mu_km3_s2,
        mu_sigma_km3_s2,
        degree,
        order,
        normalization,
        reference_longitude_deg,
        reference_latitude_deg,
    )


def _coefficient_from_record(
    record: str, record_number: int
) -> tuple[int, int, float, float, float, float]:
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
    if (
        n < 0
        or m < 0
        or m > n
        or not np.all(np.isfinite(values))
        or sigma_c < 0.0
        or sigma_s < 0.0
    ):
        raise ValueError(f"invalid SHADR coefficient record {record_number}")
    return n, m, c_nm, s_nm, sigma_c, sigma_s


def read_shadr(
    path_or_file: str | Path | TextIO,
    *,
    max_degree: int | None = None,
    name: str | None = None,
    frame: str = "body-fixed principal-axes frame",
) -> SphericalHarmonicModel:
    """Read a PDS SHADR ASCII gravity model with coefficients and uncertainties.

    PDS SHADR coefficient records contain Cnm, Snm and their associated
    uncertainty fields. These values are retained verbatim. They are not
    interpreted here as an independent covariance model.
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
            mu_sigma_km3_s2,
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
        sigma_c = np.zeros_like(c)
        sigma_s = np.zeros_like(c)
        c[0, 0] = 1.0
        seen: set[tuple[int, int]] = set()
        coefficient_rows = 0

        for record_number, raw_line in enumerate(handle, start=3):
            record = _without_record_ending(raw_line)
            if not record.strip():
                continue
            n, m, c_nm, s_nm, sigma_c_nm, sigma_s_nm = _coefficient_from_record(
                record, record_number
            )
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
                sigma_c[n, m] = sigma_c_nm
                sigma_s[n, m] = sigma_s_nm

        if coefficient_rows == 0 and file_degree > 0:
            raise ValueError("SHADR file contains no coefficient records")

        return SphericalHarmonicModel(
            mu_m3_s2=mu_km3_s2 * 1e9,
            reference_radius_m=radius_km * 1e3,
            c=c,
            s=s,
            name=name or getattr(path_or_file, "name", "SHADR gravity model"),
            frame=frame,
            sigma_c=sigma_c,
            sigma_s=sigma_s,
            mu_sigma_m3_s2=mu_sigma_km3_s2 * 1e9,
        )
    finally:
        if close:
            handle.close()


def _normalized_legendre_4pi_from_direction(
    sin_latitude: float,
    cos_latitude: float,
    max_degree: int,
    *,
    longitude_limit: bool,
) -> tuple[FloatArray, FloatArray, FloatArray | None]:
    """Evaluate normalized ALFs, latitude derivatives, and optional Pbar/cos(phi).

    ``longitude_limit`` requests the auxiliary quantity

        Qbar_nm = Pbar_nm / cos(phi),  m >= 1,

    evaluated from its own recurrence rather than by division. Qbar is the
    finite analytical factor needed by the longitudinal component of the
    spherical gradient. Its recurrence remains defined at the geographic
    poles, where the usual ``(1 / cos(phi)) dU/dlambda`` expression is a
    coordinate singularity. Qbar[:, 0] is intentionally zero because the
    zonal terms have no longitudinal derivative.
    """
    nmax = int(max_degree)
    if nmax < 0:
        raise ValueError("max_degree must be non-negative")
    if not np.isfinite(sin_latitude) or not np.isfinite(cos_latitude):
        raise ValueError("latitude direction cosines must be finite")

    x = float(sin_latitude)
    t = float(cos_latitude)
    p = np.zeros((nmax + 1, nmax + 1), dtype=float)
    dp = np.zeros_like(p)
    q = np.zeros_like(p) if longitude_limit else None
    p[0, 0] = 1.0
    if nmax == 0:
        return p, dp, q

    sqrt3 = np.sqrt(3.0)
    p[1, 0] = sqrt3 * x
    dp[1, 0] = sqrt3 * t
    p[1, 1] = sqrt3 * t
    dp[1, 1] = -sqrt3 * x
    if q is not None:
        q[1, 1] = sqrt3

    for n in range(2, nmax + 1):
        diag = np.sqrt((2.0 * n + 1.0) / (2.0 * n))
        prev_diag = p[n - 1, n - 1]
        prev_diag_d = dp[n - 1, n - 1]
        p[n, n] = diag * t * prev_diag
        dp[n, n] = diag * (-x * prev_diag + t * prev_diag_d)
        if q is not None:
            q[n, n] = diag * prev_diag

        subdiag = np.sqrt(2.0 * n + 1.0)
        p[n, n - 1] = subdiag * x * prev_diag
        dp[n, n - 1] = subdiag * (t * prev_diag + x * prev_diag_d)
        if q is not None:
            q[n, n - 1] = subdiag * x * q[n - 1, n - 1]

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
        if q is not None and n > 2:
            q[n, 1 : n - 1] = (
                a[1:] * x * q[n - 1, 1 : n - 1]
                - b[1:] * q[n - 2, 1 : n - 1]
            )

    return p, dp, q


def normalized_legendre_4pi(
    latitude_rad: float, max_degree: int
) -> tuple[FloatArray, FloatArray]:
    """Return geodesy-4pi normalized Pbar_nm and dPbar_nm/d(latitude).

    The forward recursion operates directly on normalized functions, avoiding
    factorial overflow. No Condon-Shortley phase is included, matching the
    geodetic convention used by the GRAIL SHADR gravity products.
    """
    if not np.isfinite(latitude_rad):
        raise ValueError("latitude must be finite")
    p, dp, _ = _normalized_legendre_4pi_from_direction(
        float(np.sin(latitude_rad)),
        float(np.cos(latitude_rad)),
        max_degree,
        longitude_limit=False,
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


def _body_fixed_geometry(
    position_m: ArrayLike,
) -> tuple[FloatArray, float, float, float, float, float, float]:
    """Return pole-safe body-fixed spherical geometry from a Cartesian position."""
    r_vec = np.asarray(position_m, dtype=float)
    if r_vec.shape != (3,) or not np.all(np.isfinite(r_vec)):
        raise ValueError("position must be a finite 3-vector")
    radius = float(np.linalg.norm(r_vec))
    if radius == 0.0:
        raise ValueError("position cannot be the central-body origin")

    transverse = float(np.hypot(r_vec[0], r_vec[1]))
    sin_latitude = float(r_vec[2] / radius)
    cos_latitude = float(transverse / radius)
    if transverse > 0.0:
        cos_longitude = float(r_vec[0] / transverse)
        sin_longitude = float(r_vec[1] / transverse)
        longitude = float(np.arctan2(r_vec[1], r_vec[0]))
    else:
        # Longitude is undefined exactly on the axis. Choosing lambda=0 fixes
        # the local basis to the body x/y axes; the Cartesian gravity limit is
        # unique because Qbar_nm supplies the exact m=1 pole contribution.
        cos_longitude = 1.0
        sin_longitude = 0.0
        longitude = 0.0

    return (
        r_vec,
        radius,
        sin_latitude,
        cos_latitude,
        cos_longitude,
        sin_longitude,
        longitude,
    )


def gravity_potential_body_fixed(
    position_m: ArrayLike,
    model: SphericalHarmonicModel,
    *,
    max_degree: int | None = None,
    max_order: int | None = None,
) -> float:
    """Evaluate positive gravitational potential U in m^2/s^2."""
    (
        _,
        radius,
        sin_latitude,
        cos_latitude,
        _,
        _,
        longitude,
    ) = _body_fixed_geometry(position_m)
    nmax, mmax = _evaluation_limits(model, max_degree, max_order)
    p, _, _ = _normalized_legendre_4pi_from_direction(
        sin_latitude,
        cos_latitude,
        nmax,
        longitude_limit=False,
    )
    m = np.arange(mmax + 1, dtype=float)
    cos_m = np.cos(m * longitude)
    sin_m = np.sin(m * longitude)
    radial_ratio = model.reference_radius_m / radius
    radial_power = 1.0
    total = 0.0
    for n in range(nmax + 1):
        upper = min(n, mmax) + 1
        harmonics = (
            model.c[n, :upper] * cos_m[:upper]
            + model.s[n, :upper] * sin_m[:upper]
        )
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
    """Evaluate pole-safe spherical-harmonic gravitational acceleration in m/s^2.

    The radial and latitudinal sums use Pbar_nm and dPbar_nm/dphi. The
    longitudinal sum uses Qbar_nm = Pbar_nm/cos(phi), synthesized directly by
    recurrence. This removes the coordinate singularity at the poles without
    displacing the evaluation point or suppressing non-zonal gravity terms.
    """
    (
        _,
        radius,
        sin_latitude,
        cos_latitude,
        cos_longitude,
        sin_longitude,
        longitude,
    ) = _body_fixed_geometry(position_m)
    nmax, mmax = _evaluation_limits(model, max_degree, max_order)
    p, dp, q = _normalized_legendre_4pi_from_direction(
        sin_latitude,
        cos_latitude,
        nmax,
        longitude_limit=True,
    )
    assert q is not None

    m = np.arange(mmax + 1, dtype=float)
    cos_m = np.cos(m * longitude)
    sin_m = np.sin(m * longitude)

    radial_ratio = model.reference_radius_m / radius
    radial_power = 1.0
    sum_r = 0.0
    sum_lat = 0.0
    sum_lon_over_cos = 0.0
    for n in range(nmax + 1):
        upper = min(n, mmax) + 1
        c = model.c[n, :upper]
        s = model.s[n, :upper]
        trig = c * cos_m[:upper] + s * sin_m[:upper]
        weighted = radial_power
        sum_r += (n + 1.0) * weighted * float(np.dot(p[n, :upper], trig))
        sum_lat += weighted * float(np.dot(dp[n, :upper], trig))
        if upper > 1:
            dlon = m[:upper] * (-c * sin_m[:upper] + s * cos_m[:upper])
            sum_lon_over_cos += weighted * float(np.dot(q[n, :upper], dlon))
        radial_power *= radial_ratio

    mu_over_r2 = model.mu_m3_s2 / radius**2
    a_r = -mu_over_r2 * sum_r
    a_lat = mu_over_r2 * sum_lat
    a_lon = mu_over_r2 * sum_lon_over_cos

    e_r = np.array(
        [
            cos_latitude * cos_longitude,
            cos_latitude * sin_longitude,
            sin_latitude,
        ]
    )
    e_lat = np.array(
        [
            -sin_latitude * cos_longitude,
            -sin_latitude * sin_longitude,
            cos_latitude,
        ]
    )
    e_lon = np.array([-sin_longitude, cos_longitude, 0.0])
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
