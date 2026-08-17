"""Run the external/independent lunar astrodynamics validation campaign.

This script intentionally contains an independent SHADR parser, an independent
SHTOOLS gravity path, direct SpiceyPy frame/ephemeris calls, an independent
third-body implementation, and a separate SciPy propagation RHS. The candidate
side imports the production package. Large public data are downloaded by
``download_independent_validation_data.py`` and are not committed.

The campaign distinguishes implementation cross-validation from comparison to
an externally reconstructed spacecraft trajectory. The latter is not forced
through a pass/fail tolerance when the propagated force model deliberately
omits forces estimated by the LRO precision-orbit-determination process.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import platform
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

try:
    import mpmath as mp
    import pyshtools as pysh
    import spiceypy as spice
except ImportError as exc:  # pragma: no cover - external validation dependency
    raise SystemExit(
        "Independent validation requires: pip install -e .[validation]"
    ) from exc

from lunar_astrodynamics import (
    EARTH_GM_DE440_M3_S2,
    MOON_MEAN_RADIUS_M,
    SUN_GM_DE440_M3_S2,
    PropagationSettings,
    gravity_acceleration_body_fixed,
    gravity_acceleration_inertial,
    ground_track_history,
    load_lola_pds_global_gdr,
    propagate_with_acceleration,
    read_shadr,
    spice_ephemeris_from_et,
    spice_rotation_provider,
    third_body_acceleration,
)

FloatArray = NDArray[np.float64]

FRAME = "MOON_PA_DE421"
INERTIAL_FRAME = "J2000"
MOON_ID = 301


@dataclass(frozen=True)
class IndependentGravityModel:
    gm_m3_s2: float
    reference_radius_m: float
    coefficients: FloatArray  # (2, degree+1, degree+1), 4pi normalized
    file_degree: int


@dataclass(frozen=True)
class PropagationComparison:
    horizon_s: float
    sample_count: int
    final_position_difference_m: float
    maximum_position_difference_m: float
    rms_position_difference_m: float
    final_velocity_difference_m_s: float
    maximum_velocity_difference_m_s: float
    rms_velocity_difference_m_s: float


@dataclass(frozen=True)
class LroComparison:
    horizon_s: float
    gravity_degree: int
    sample_count: int
    final_position_difference_m: float
    maximum_position_difference_m: float
    rms_position_difference_m: float
    final_velocity_difference_m_s: float
    maximum_velocity_difference_m_s: float
    radial_rms_m: float
    along_track_rms_m: float
    cross_track_rms_m: float
    numerical_convergence_final_position_m: float | None
    numerical_convergence_final_velocity_m_s: float | None


def _float(token: str) -> float:
    return float(token.strip().replace("D", "E").replace("d", "e"))


def read_shadr_independent(path: Path, max_degree: int) -> IndependentGravityModel:
    """Minimal independent parser using the published fixed-width SHADR layout."""
    with path.open("r", encoding="ascii", newline="") as stream:
        header = stream.readline().rstrip("\r\n")
        if len(header) < 89:
            raise ValueError("independent SHADR parser: invalid header")
        radius_km = _float(header[0:23])
        gm_km3_s2 = _float(header[24:47])
        file_degree = int(header[72:77].strip())
        normalization = int(header[84:89].strip())
        if normalization != 1:
            raise ValueError("independent SHADR parser requires normalized coefficients")
        degree = min(int(max_degree), file_degree)
        cilm = np.zeros((2, degree + 1, degree + 1), dtype=float)
        cilm[0, 0, 0] = 1.0
        for raw in stream:
            line = raw.rstrip("\r\n")
            if not line.strip():
                continue
            if len(line) < 59:
                raise ValueError("independent SHADR parser: short coefficient record")
            n = int(line[0:5].strip())
            m = int(line[6:11].strip())
            if n <= degree:
                cilm[0, n, m] = _float(line[12:35])
                cilm[1, n, m] = _float(line[36:59])
    return IndependentGravityModel(
        gm_m3_s2=gm_km3_s2 * 1e9,
        reference_radius_m=radius_km * 1e3,
        coefficients=cilm,
        file_degree=file_degree,
    )


def spherical_geometry(position_m: FloatArray) -> tuple[float, float, float]:
    radius = float(np.linalg.norm(position_m))
    latitude = math.degrees(math.asin(float(position_m[2] / radius)))
    longitude = math.degrees(math.atan2(float(position_m[1]), float(position_m[0]))) % 360.0
    return radius, latitude, longitude


def shtools_gravity_cartesian(
    position_body_fixed_m: FloatArray,
    model: IndependentGravityModel,
    degree: int,
) -> FloatArray:
    """Evaluate 4pi-normalized gravity with SHTOOLS and convert r/theta/phi to xyz."""
    radius, latitude_deg, longitude_deg = spherical_geometry(position_body_fixed_m)
    components = np.asarray(
        pysh.gravmag.MakeGravGridPoint(
            model.coefficients,
            model.gm_m3_s2,
            model.reference_radius_m,
            radius,
            latitude_deg,
            longitude_deg,
            lmax=int(degree),
        ),
        dtype=float,
    )
    a_r, a_theta, a_phi = components
    latitude = math.radians(latitude_deg)
    longitude = math.radians(longitude_deg)
    cphi, sphi = math.cos(latitude), math.sin(latitude)
    clam, slam = math.cos(longitude), math.sin(longitude)
    e_r = np.array([cphi * clam, cphi * slam, sphi])
    # SHTOOLS theta is colatitude, so positive theta points south.
    e_theta = np.array([sphi * clam, sphi * slam, -cphi])
    e_phi = np.array([-slam, clam, 0.0])
    return (a_r * e_r + a_theta * e_theta + a_phi * e_phi).astype(float)


def gravity_roundoff_tolerance_relative(degree: int) -> float:
    """Conservative a-priori double-precision recurrence/summation envelope.

    The scale is fixed before examining validation residuals. It grows with
    l^(3/2), allowing recurrence error plus cancellation in high-degree sums,
    and has a floor for low-degree cases. It is intentionally far tighter than
    mission-level trajectory tolerances.
    """
    eps = np.finfo(float).eps
    return float(max(5e-13, 200.0 * eps * (degree + 1.0) ** 1.5))


def body_fixed_sample_positions(reference_radius_m: float) -> tuple[FloatArray, ...]:
    definitions = (
        (50_000.0, 0.0, 0.0),
        (100_000.0, 0.0, 90.0),
        (75_000.0, 45.0, 45.0),
        (75_000.0, -45.0, 315.0),
        (30_000.0, 80.0, 10.0),
        (30_000.0, -80.0, 200.0),
        (150_000.0, 20.0, 179.5),
        (250_000.0, -10.0, 270.0),
    )
    result: list[FloatArray] = []
    for altitude, latitude_deg, longitude_deg in definitions:
        radius = reference_radius_m + altitude
        lat = math.radians(latitude_deg)
        lon = math.radians(longitude_deg)
        result.append(
            np.array(
                [
                    radius * math.cos(lat) * math.cos(lon),
                    radius * math.cos(lat) * math.sin(lon),
                    radius * math.sin(lat),
                ],
                dtype=float,
            )
        )
    return tuple(result)


def validate_gravity_model(path: Path, degrees: tuple[int, ...], frame: str) -> dict[str, object]:
    max_degree = max(degrees)
    independent = read_shadr_independent(path, max_degree)
    candidate = read_shadr(path, max_degree=max_degree, name=path.stem, frame=frame)
    points = body_fixed_sample_positions(independent.reference_radius_m)
    rows: list[dict[str, object]] = []
    overall = True
    for degree in degrees:
        tolerance = gravity_roundoff_tolerance_relative(degree)
        max_abs = 0.0
        max_rel = 0.0
        point_rows = []
        for position in points:
            reference = shtools_gravity_cartesian(position, independent, degree)
            actual = gravity_acceleration_body_fixed(
                position, candidate, max_degree=degree, max_order=degree
            )
            difference = actual - reference
            absolute = float(np.linalg.norm(difference))
            relative = absolute / float(np.linalg.norm(reference))
            max_abs = max(max_abs, absolute)
            max_rel = max(max_rel, relative)
            radius, latitude, longitude = spherical_geometry(position)
            point_rows.append(
                {
                    "latitude_deg": latitude,
                    "longitude_deg_east": longitude,
                    "altitude_above_gravity_reference_m": radius - independent.reference_radius_m,
                    "absolute_acceleration_difference_m_s2": absolute,
                    "relative_acceleration_difference": relative,
                }
            )
        accepted = bool(max_rel <= tolerance)
        overall = overall and accepted
        rows.append(
            {
                "degree": degree,
                "max_absolute_acceleration_difference_m_s2": max_abs,
                "max_relative_acceleration_difference": max_rel,
                "a_priori_relative_tolerance": tolerance,
                "accepted": accepted,
                "points": point_rows,
            }
        )
    return {
        "reference_implementation": "SHTOOLS MakeGravGridPoint with independently parsed 4pi coefficients",
        "candidate_implementation": "lunar_astrodynamics.gravity_acceleration_body_fixed",
        "gravity_file": path.name,
        "gravity_file_degree": independent.file_degree,
        "gm_m3_s2": independent.gm_m3_s2,
        "reference_radius_m": independent.reference_radius_m,
        "degrees": rows,
        "accepted": overall,
    }


def load_spice_context(data_dir: Path) -> None:
    spice.kclear()
    for filename in (
        "naif0012.tls",
        "de421.bsp",
        "moon_pa_de421_1900-2050.bpc",
        "moon_080317.tf",
    ):
        spice.furnsh(str(data_dir / filename))


def lro_object_and_coverage(lro_spk: Path) -> tuple[int, tuple[float, float]]:
    ids = [int(value) for value in spice.spkobj(str(lro_spk))]
    if len(ids) != 1:
        raise RuntimeError(f"expected one LRO object in {lro_spk.name}, found {ids}")
    object_id = ids[0]
    coverage = spice.spkcov(str(lro_spk), object_id)
    intervals = [spice.wnfetd(coverage, i) for i in range(spice.wncard(coverage))]
    if not intervals:
        raise RuntimeError("LRO SPK has no coverage")
    interval = max(intervals, key=lambda pair: pair[1] - pair[0])
    return object_id, (float(interval[0]), float(interval[1]))


def choose_validation_epoch(start_et: float, end_et: float, required_horizon_s: float) -> float:
    available = end_et - start_et
    if available <= required_horizon_s + 7200.0:
        raise RuntimeError("LRO SPK coverage is too short for requested validation horizon")
    # Keep a two-hour margin from the beginning and place the arc well inside
    # the reconstruction interval. This rule is deterministic and not selected
    # after inspecting propagation residuals.
    return float(start_et + max(7200.0, 0.35 * (available - required_horizon_s)))


def lro_state_m(object_id: int, et: float) -> FloatArray:
    state_km, _ = spice.spkez(object_id, float(et), INERTIAL_FRAME, "NONE", MOON_ID)
    state = np.asarray(state_km, dtype=float)
    return np.concatenate((state[:3] * 1000.0, state[3:] * 1000.0))


def validate_frames(
    data_dir: Path, object_id: int, epoch_et: float, lro_state: FloatArray
) -> dict[str, object]:
    provider = spice_rotation_provider(INERTIAL_FRAME, FRAME, et_offset_s=0.0)
    epochs = [epoch_et, epoch_et + 3600.0, epoch_et + 6.0 * 3600.0]
    matrix_rows = []
    max_matrix = 0.0
    for et in epochs:
        candidate = provider(et)
        sxform = np.asarray(spice.sxform(INERTIAL_FRAME, FRAME, et), dtype=float)[:3, :3]
        error = float(np.max(np.abs(candidate - sxform)))
        max_matrix = max(max_matrix, error)
        matrix_rows.append({"et_s": et, "max_abs_matrix_difference": error})

    sample_times = np.array([0.0, 1800.0, 3600.0])
    inertial_positions = np.vstack(
        [lro_state_m(object_id, epoch_et + dt)[:3] for dt in sample_times]
    )
    elapsed_provider = spice_rotation_provider(
        INERTIAL_FRAME, FRAME, et_offset_s=epoch_et
    )
    track = ground_track_history(
        sample_times,
        inertial_positions,
        elapsed_provider,
        body_fixed_frame=FRAME,
        reference_radius_m=MOON_MEAN_RADIUS_M,
    )
    coordinate_rows = []
    max_radius = max_lat = max_lon = 0.0
    for i, dt in enumerate(sample_times):
        rotation = np.asarray(spice.pxform(INERTIAL_FRAME, FRAME, epoch_et + dt))
        body = rotation @ inertial_positions[i] / 1000.0
        radius_km, longitude_rad, latitude_rad = spice.reclat(body)
        longitude_deg = math.degrees(longitude_rad) % 360.0
        latitude_deg = math.degrees(latitude_rad)
        radius_error = abs(track.reference_altitude_m[i] - (radius_km * 1000.0 - MOON_MEAN_RADIUS_M))
        lat_error = abs(track.latitude_deg[i] - latitude_deg)
        lon_delta = (track.longitude_deg_east[i] - longitude_deg + 180.0) % 360.0 - 180.0
        lon_error = abs(lon_delta)
        max_radius = max(max_radius, radius_error)
        max_lat = max(max_lat, lat_error)
        max_lon = max(max_lon, lon_error)
        coordinate_rows.append(
            {
                "elapsed_time_s": float(dt),
                "altitude_difference_m": radius_error,
                "latitude_difference_deg": lat_error,
                "longitude_difference_deg": lon_error,
            }
        )

    matrix_tolerance = 5e-13
    coordinate_angle_tolerance_deg = 1e-10
    coordinate_radius_tolerance_m = 1e-6
    accepted = bool(
        max_matrix <= matrix_tolerance
        and max_radius <= coordinate_radius_tolerance_m
        and max_lat <= coordinate_angle_tolerance_deg
        and max_lon <= coordinate_angle_tolerance_deg
    )
    return {
        "reference": "NAIF CSPICE sxform/reclat called directly through SpiceyPy",
        "candidate": "spice_rotation_provider plus ground_track_history",
        "frame": FRAME,
        "matrix_rows": matrix_rows,
        "coordinate_rows": coordinate_rows,
        "max_abs_matrix_difference": max_matrix,
        "max_altitude_difference_m": max_radius,
        "max_latitude_difference_deg": max_lat,
        "max_longitude_difference_deg": max_lon,
        "tolerances": {
            "matrix_max_abs": matrix_tolerance,
            "altitude_m": coordinate_radius_tolerance_m,
            "angle_deg": coordinate_angle_tolerance_deg,
            "basis": "double-precision CSPICE consistency; tolerances are orders above machine epsilon but far below physical mapping accuracy",
        },
        "accepted": accepted,
    }


def mp_third_body(spacecraft_m: FloatArray, body_m: FloatArray, gm: float) -> FloatArray:
    mp.mp.dps = 80
    r = [mp.mpf(str(float(value))) for value in spacecraft_m]
    b = [mp.mpf(str(float(value))) for value in body_m]
    mu = mp.mpf(str(float(gm)))
    delta = [b[i] - r[i] for i in range(3)]
    dn = mp.sqrt(sum(value * value for value in delta))
    bn = mp.sqrt(sum(value * value for value in b))
    result = [mu * (delta[i] / dn**3 - b[i] / bn**3) for i in range(3)]
    return np.array([float(value) for value in result], dtype=float)


def validate_third_bodies(epoch_et: float, spacecraft_m: FloatArray) -> dict[str, object]:
    rows = []
    accepted = True
    for target, gm in (
        ("EARTH", EARTH_GM_DE440_M3_S2),
        ("SUN", SUN_GM_DE440_M3_S2),
    ):
        position_km, _ = spice.spkpos(target, epoch_et, INERTIAL_FRAME, "NONE", "MOON")
        body_m = np.asarray(position_km, dtype=float) * 1000.0
        actual = third_body_acceleration(spacecraft_m, body_m, gm)
        reference = mp_third_body(spacecraft_m, body_m, gm)
        absolute = float(np.linalg.norm(actual - reference))
        relative = absolute / float(np.linalg.norm(reference))
        delta = body_m - spacecraft_m
        term1 = gm * delta / np.linalg.norm(delta) ** 3
        term2 = gm * body_m / np.linalg.norm(body_m) ** 3
        condition = float((np.linalg.norm(term1) + np.linalg.norm(term2)) / np.linalg.norm(reference))
        tolerance = float(max(1e-13, 100.0 * np.finfo(float).eps * condition))
        ok = relative <= tolerance
        accepted = accepted and ok
        rows.append(
            {
                "target": target,
                "absolute_acceleration_difference_m_s2": absolute,
                "relative_acceleration_difference": relative,
                "subtraction_condition_estimate": condition,
                "a_priori_relative_tolerance": tolerance,
                "accepted": ok,
            }
        )
    return {
        "reference": "80-decimal-digit mpmath Newtonian differential acceleration using NAIF DE421 body positions",
        "candidate": "lunar_astrodynamics.third_body_acceleration",
        "rows": rows,
        "accepted": accepted,
    }


def _pds_value(label: str, key: str) -> str:
    match = re.search(rf"^\s*{re.escape(key)}\s*=\s*(.+?)\s*$", label, re.MULTILINE)
    if match is None:
        raise ValueError(f"independent PDS decoder missing {key}")
    value = re.sub(r"\s*<[^>]+>\s*$", "", match.group(1).strip()).strip()
    return value.strip("'\"")


def validate_terrain(data_dir: Path) -> dict[str, object]:
    image = data_dir / "LDEM_4.IMG"
    label_path = data_dir / "LDEM_4.LBL"
    label = label_path.read_text(encoding="ascii")
    lines = int(_pds_value(label, "LINES"))
    samples = int(_pds_value(label, "LINE_SAMPLES"))
    sample_type = _pds_value(label, "SAMPLE_TYPE").upper()
    scale = float(_pds_value(label, "SCALING_FACTOR"))
    reference_radius_m = float(_pds_value(label, "OFFSET"))
    resolution = float(_pds_value(label, "MAP_RESOLUTION"))
    endian = ">" if sample_type.startswith("MSB") else "<"
    raw = np.memmap(image, mode="r", dtype=np.dtype(f"{endian}i2"), shape=(lines, samples))
    candidate = load_lola_pds_global_gdr(image, label_path)

    raw_indices = (
        (0, 0),
        (0, samples // 2),
        (lines // 4, samples // 8),
        (lines // 2 - 1, 0),
        (lines // 2 - 1, samples // 4),
        (lines - 1, samples // 2),
    )
    rows = []
    max_elevation = max_lat = max_lon = 0.0
    for raw_i, j in raw_indices:
        latitude_deg = 90.0 - (raw_i + 0.5) / resolution
        longitude_deg = (j + 0.5) / resolution
        reference_elevation = float(raw[raw_i, j]) * scale
        candidate_i = lines - 1 - raw_i
        actual_elevation = float(candidate.elevation_grid_m[candidate_i, j])
        elevation_error = abs(actual_elevation - reference_elevation)
        latitude_error = abs(float(candidate.latitude_deg[candidate_i]) - latitude_deg)
        longitude_error = abs(float(candidate.longitude_deg_east[j]) - longitude_deg)
        max_elevation = max(max_elevation, elevation_error)
        max_lat = max(max_lat, latitude_error)
        max_lon = max(max_lon, longitude_error)
        rows.append(
            {
                "raw_row": raw_i,
                "column": j,
                "latitude_deg": latitude_deg,
                "longitude_deg_east": longitude_deg,
                "reference_elevation_m": reference_elevation,
                "candidate_elevation_m": actual_elevation,
                "elevation_difference_m": elevation_error,
            }
        )
    accepted = bool(
        max_elevation <= 1e-6
        and max_lat <= 1e-12
        and max_lon <= 1e-12
        and abs(candidate.reference_radius_m - reference_radius_m) <= 1e-9
    )
    return {
        "reference": "independent NumPy raw signed-16-bit decode using official PDS label fields",
        "candidate": "load_lola_pds_global_gdr",
        "product": "LDEM_4",
        "frame_from_label": candidate.frame,
        "reference_radius_m": candidate.reference_radius_m,
        "rows": rows,
        "max_elevation_difference_m": max_elevation,
        "max_latitude_difference_deg": max_lat,
        "max_longitude_difference_deg": max_lon,
        "tolerances": {
            "elevation_m": 1e-6,
            "coordinate_deg": 1e-12,
            "basis": "direct integer-grid decode; these are file-decoding tolerances, not LOLA measurement-accuracy claims",
        },
        "accepted": accepted,
    }


def independent_third_body_numpy(spacecraft_m: FloatArray, body_m: FloatArray, gm: float) -> FloatArray:
    delta = body_m - spacecraft_m
    return gm * (delta / np.linalg.norm(delta) ** 3 - body_m / np.linalg.norm(body_m) ** 3)


def candidate_acceleration_factory(
    gravity_path: Path,
    degree: int,
    epoch_et: float,
) -> tuple[Callable[[float, FloatArray], FloatArray], object]:
    gravity = read_shadr(gravity_path, max_degree=degree, name="GRGM900C", frame=FRAME)
    rotation = spice_rotation_provider(INERTIAL_FRAME, FRAME, et_offset_s=epoch_et)
    ephemeris = spice_ephemeris_from_et(
        epoch_et,
        inertial_frame=INERTIAL_FRAME,
        observer="MOON",
        snapshot_kernels=False,
    )

    def acceleration(time_s: float, position_m: FloatArray) -> FloatArray:
        total = gravity_acceleration_inertial(
            time_s,
            position_m,
            gravity,
            rotation,
            max_degree=degree,
            max_order=degree,
        )
        total += third_body_acceleration(
            position_m, ephemeris.position_m("EARTH", time_s), EARTH_GM_DE440_M3_S2
        )
        total += third_body_acceleration(
            position_m, ephemeris.position_m("SUN", time_s), SUN_GM_DE440_M3_S2
        )
        return total

    return acceleration, gravity


def independent_acceleration_factory(
    independent: IndependentGravityModel,
    degree: int,
    epoch_et: float,
) -> Callable[[float, FloatArray], FloatArray]:
    def acceleration(time_s: float, position_m: FloatArray) -> FloatArray:
        et = epoch_et + float(time_s)
        rotation = np.asarray(spice.pxform(INERTIAL_FRAME, FRAME, et), dtype=float)
        body_position = rotation @ position_m
        gravity_body = shtools_gravity_cartesian(body_position, independent, degree)
        total = rotation.T @ gravity_body
        earth_km, _ = spice.spkpos("EARTH", et, INERTIAL_FRAME, "NONE", "MOON")
        sun_km, _ = spice.spkpos("SUN", et, INERTIAL_FRAME, "NONE", "MOON")
        total += independent_third_body_numpy(
            position_m, np.asarray(earth_km) * 1000.0, EARTH_GM_DE440_M3_S2
        )
        total += independent_third_body_numpy(
            position_m, np.asarray(sun_km) * 1000.0, SUN_GM_DE440_M3_S2
        )
        return np.asarray(total, dtype=float)

    return acceleration


def independent_propagate(
    initial_state: FloatArray,
    times_s: FloatArray,
    acceleration: Callable[[float, FloatArray], FloatArray],
    *,
    rtol: float,
    position_atol_m: float,
    velocity_atol_m_s: float,
    max_step_s: float,
):
    def rhs(time_s: float, state: FloatArray) -> FloatArray:
        derivative = np.empty(6, dtype=float)
        derivative[:3] = state[3:]
        derivative[3:] = acceleration(time_s, state[:3])
        return derivative

    return solve_ivp(
        rhs,
        (0.0, float(times_s[-1])),
        initial_state,
        method="DOP853",
        t_eval=times_s,
        rtol=rtol,
        atol=np.array([position_atol_m] * 3 + [velocity_atol_m_s] * 3),
        max_step=max_step_s,
    )


def compare_state_histories(candidate: FloatArray, reference: FloatArray, times_s: FloatArray) -> PropagationComparison:
    position = np.linalg.norm(candidate[:3].T - reference[:3].T, axis=1)
    velocity = np.linalg.norm(candidate[3:].T - reference[3:].T, axis=1)
    return PropagationComparison(
        horizon_s=float(times_s[-1]),
        sample_count=int(times_s.size),
        final_position_difference_m=float(position[-1]),
        maximum_position_difference_m=float(np.max(position)),
        rms_position_difference_m=float(np.sqrt(np.mean(position**2))),
        final_velocity_difference_m_s=float(velocity[-1]),
        maximum_velocity_difference_m_s=float(np.max(velocity)),
        rms_velocity_difference_m_s=float(np.sqrt(np.mean(velocity**2))),
    )


def validate_independent_propagation(
    gravity_path: Path,
    initial_state: FloatArray,
    epoch_et: float,
    degree: int = 120,
    horizon_s: float = 2.0 * 3600.0,
) -> dict[str, object]:
    times = np.linspace(0.0, horizon_s, 121)
    candidate_acceleration, _ = candidate_acceleration_factory(gravity_path, degree, epoch_et)
    independent_gravity = read_shadr_independent(gravity_path, degree)
    reference_acceleration = independent_acceleration_factory(
        independent_gravity, degree, epoch_et
    )
    settings = PropagationSettings(
        rtol=2e-12,
        position_atol_m=2e-6,
        velocity_atol_m_s=2e-9,
        max_step_s=60.0,
    )
    candidate_solution = propagate_with_acceleration(
        initial_state,
        horizon_s,
        candidate_acceleration,
        collision_radius_m=MOON_MEAN_RADIUS_M,
        sample_times_s=times,
        settings=settings,
    )
    reference_solution = independent_propagate(
        initial_state,
        times,
        reference_acceleration,
        rtol=settings.rtol,
        position_atol_m=settings.position_atol_m,
        velocity_atol_m_s=settings.velocity_atol_m_s,
        max_step_s=settings.max_step_s,
    )
    if not candidate_solution.success or not reference_solution.success:
        raise RuntimeError("same-model independent propagation failed")
    comparison = compare_state_histories(candidate_solution.y, reference_solution.y, times)

    # Derive an a-priori propagation envelope from independent instantaneous
    # acceleration disagreement and the numerical integration settings. The
    # factor 50 allows bounded dynamical amplification over several lunar
    # revolutions without tuning the threshold to the final state residual.
    accel_differences = []
    for index in np.linspace(0, times.size - 1, 17, dtype=int):
        state = reference_solution.y[:, index]
        accel_differences.append(
            np.linalg.norm(
                candidate_acceleration(float(times[index]), state[:3])
                - reference_acceleration(float(times[index]), state[:3])
            )
        )
    max_accel_difference = float(max(accel_differences))
    acceleration_position_bound = 0.5 * max_accel_difference * horizon_s**2
    tolerance_position = float(max(0.02, 50.0 * acceleration_position_bound + 0.02))
    tolerance_velocity = float(max(2e-5, 50.0 * max_accel_difference * horizon_s + 2e-5))
    accepted = bool(
        comparison.maximum_position_difference_m <= tolerance_position
        and comparison.maximum_velocity_difference_m_s <= tolerance_velocity
    )
    return {
        "reference": "independent SHTOOLS gravity + direct SpiceyPy frames/ephemerides + separate SciPy RHS",
        "candidate": "production harmonic/frame/ephemeris/third-body/propagation stack",
        "degree_order": [degree, degree],
        "comparison": asdict(comparison),
        "max_sampled_acceleration_difference_m_s2": max_accel_difference,
        "a_priori_tolerances": {
            "maximum_position_difference_m": tolerance_position,
            "maximum_velocity_difference_m_s": tolerance_velocity,
            "basis": "50x bounded acceleration-error growth plus explicit integration absolute-error floor; fixed formula independent of final residual",
        },
        "accepted": accepted,
    }


def rtn_errors(reference_states: FloatArray, candidate_states: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray]:
    radial = np.empty(reference_states.shape[1])
    along = np.empty_like(radial)
    cross = np.empty_like(radial)
    for i in range(reference_states.shape[1]):
        r = reference_states[:3, i]
        v = reference_states[3:, i]
        er = r / np.linalg.norm(r)
        h = np.cross(r, v)
        ec = h / np.linalg.norm(h)
        ea = np.cross(ec, er)
        delta = candidate_states[:3, i] - r
        radial[i] = np.dot(delta, er)
        along[i] = np.dot(delta, ea)
        cross[i] = np.dot(delta, ec)
    return radial, along, cross


def compare_to_lro_spk(
    gravity_path: Path,
    lro_spk: Path,
    object_id: int,
    epoch_et: float,
    *,
    degree: int,
    horizon_s: float,
    sample_count: int,
    convergence_check: bool,
) -> LroComparison:
    times = np.linspace(0.0, horizon_s, sample_count)
    truth = np.column_stack([lro_state_m(object_id, epoch_et + dt) for dt in times])
    initial_state = truth[:, 0].copy()
    acceleration, _ = candidate_acceleration_factory(gravity_path, degree, epoch_et)
    settings = PropagationSettings(
        rtol=1e-11,
        position_atol_m=1e-5,
        velocity_atol_m_s=1e-8,
        max_step_s=120.0,
    )
    solution = propagate_with_acceleration(
        initial_state,
        horizon_s,
        acceleration,
        collision_radius_m=MOON_MEAN_RADIUS_M,
        sample_times_s=times,
        settings=settings,
    )
    if not solution.success:
        raise RuntimeError(f"LRO comparison propagation failed: {solution.message}")
    position = np.linalg.norm(solution.y[:3] - truth[:3], axis=0)
    velocity = np.linalg.norm(solution.y[3:] - truth[3:], axis=0)
    radial, along, cross = rtn_errors(truth, solution.y)

    convergence_pos = convergence_vel = None
    if convergence_check:
        tight = propagate_with_acceleration(
            initial_state,
            horizon_s,
            acceleration,
            collision_radius_m=MOON_MEAN_RADIUS_M,
            sample_times_s=np.array([0.0, horizon_s]),
            settings=PropagationSettings(
                rtol=2e-12,
                position_atol_m=2e-6,
                velocity_atol_m_s=2e-9,
                max_step_s=60.0,
            ),
        )
        if not tight.success:
            raise RuntimeError("tight LRO convergence propagation failed")
        convergence_pos = float(np.linalg.norm(tight.y[:3, -1] - solution.y[:3, -1]))
        convergence_vel = float(np.linalg.norm(tight.y[3:, -1] - solution.y[3:, -1]))

    return LroComparison(
        horizon_s=horizon_s,
        gravity_degree=degree,
        sample_count=sample_count,
        final_position_difference_m=float(position[-1]),
        maximum_position_difference_m=float(np.max(position)),
        rms_position_difference_m=float(np.sqrt(np.mean(position**2))),
        final_velocity_difference_m_s=float(velocity[-1]),
        maximum_velocity_difference_m_s=float(np.max(velocity)),
        radial_rms_m=float(np.sqrt(np.mean(radial**2))),
        along_track_rms_m=float(np.sqrt(np.mean(along**2))),
        cross_track_rms_m=float(np.sqrt(np.mean(cross**2))),
        numerical_convergence_final_position_m=convergence_pos,
        numerical_convergence_final_velocity_m_s=convergence_vel,
    )


def validate_lro_reconstruction(
    data_dir: Path,
    gravity_path: Path,
    object_id: int,
    coverage: tuple[float, float],
) -> dict[str, object]:
    max_horizon = 24.0 * 3600.0
    epoch = choose_validation_epoch(coverage[0], coverage[1], max_horizon)
    short = compare_to_lro_spk(
        gravity_path,
        data_dir / "LRO_ES_36_GRGM900C_L600.BSP",
        object_id,
        epoch,
        degree=600,
        horizon_s=6.0 * 3600.0,
        sample_count=73,
        convergence_check=True,
    )
    long = compare_to_lro_spk(
        gravity_path,
        data_dir / "LRO_ES_36_GRGM900C_L600.BSP",
        object_id,
        epoch,
        degree=120,
        horizon_s=max_horizon,
        sample_count=145,
        convergence_check=False,
    )
    return {
        "truth_source": "PDS LRO Radio Science GEODYN monthly reconstructed SPK",
        "truth_file": "LRO_ES_36_GRGM900C_L600.BSP",
        "lro_object_id": object_id,
        "spk_coverage_et_s": list(coverage),
        "validation_epoch_et_s": epoch,
        "validation_epoch_utc": spice.et2utc(epoch, "ISOC", 3),
        "candidate_force_model": {
            "lunar_gravity": "GRGM900C spherical harmonics",
            "short_arc_degree_order": [600, 600],
            "long_arc_degree_order": [120, 120],
            "third_bodies": ["Earth point mass", "Sun point mass"],
            "planetary_ephemeris": "DE421",
            "lunar_orientation": FRAME,
            "excluded_relative_to_precision_OD": [
                "solar radiation pressure and detailed attitude/area model",
                "lunar solid tides/time-variable gravity",
                "Jupiter and other planetary third bodies",
                "spacecraft maneuvers and momentum management",
                "tracking-data parameter estimation",
            ],
        },
        "six_hour_degree600": asdict(short),
        "twenty_four_hour_degree120": asdict(long),
        "pass_fail_policy": (
            "No mission-truth pass threshold is imposed because this open-loop model deliberately omits forces and estimated parameters present in GEODYN POD. "
            "The reconstructed SPK is used to quantify physical-model residuals; numerical convergence is reported separately."
        ),
    }


def software_versions() -> dict[str, str]:
    packages = {}
    for name in ("numpy", "scipy", "spiceypy", "pyshtools", "mpmath"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "not installed"
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        **packages,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/validation"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/independent_validation.json"),
    )
    parser.add_argument(
        "--skip-grgm1200b",
        action="store_true",
        help="skip the full degree-1200B SHTOOLS point validation",
    )
    args = parser.parse_args()

    required = (
        "naif0012.tls",
        "de421.bsp",
        "moon_pa_de421_1900-2050.bpc",
        "moon_080317.tf",
        "gggrx_0900c_sha.tab",
        "LRO_ES_36_GRGM900C_L600.BSP",
        "LDEM_4.IMG",
        "LDEM_4.LBL",
        "manifest.json",
    )
    if not args.skip_grgm1200b:
        required += ("gggrx_1200b_sha.tab",)
    missing = [name for name in required if not (args.data_dir / name).exists()]
    if missing:
        raise SystemExit(
            "Missing validation data. Run scripts/download_independent_validation_data.py first. Missing: "
            + ", ".join(missing)
        )

    manifest = json.loads((args.data_dir / "manifest.json").read_text(encoding="utf-8"))
    load_spice_context(args.data_dir)
    lro_spk = args.data_dir / "LRO_ES_36_GRGM900C_L600.BSP"
    spice.furnsh(str(lro_spk))
    object_id, coverage = lro_object_and_coverage(lro_spk)
    epoch = choose_validation_epoch(coverage[0], coverage[1], 24.0 * 3600.0)
    initial_state = lro_state_m(object_id, epoch)

    gravity900 = validate_gravity_model(
        args.data_dir / "gggrx_0900c_sha.tab",
        (10, 60, 120, 300, 600),
        FRAME,
    )
    gravity1200 = None
    if not args.skip_grgm1200b:
        # Acceleration-only cross-validation does not require a DE430 orientation
        # kernel because both evaluators receive identical explicit body-fixed
        # Cartesian points. No DE421/DE430 frame equivalence is assumed.
        gravity1200 = validate_gravity_model(
            args.data_dir / "gggrx_1200b_sha.tab",
            (60, 120, 300, 600, 1200),
            "GRGM1200B published principal-axes coefficient frame",
        )

    frames = validate_frames(args.data_dir, object_id, epoch, initial_state)
    third_bodies = validate_third_bodies(epoch, initial_state[:3])
    terrain = validate_terrain(args.data_dir)
    independent_propagation = validate_independent_propagation(
        args.data_dir / "gggrx_0900c_sha.tab",
        initial_state,
        epoch,
        degree=120,
        horizon_s=2.0 * 3600.0,
    )
    lro = validate_lro_reconstruction(
        args.data_dir,
        args.data_dir / "gggrx_0900c_sha.tab",
        object_id,
        coverage,
    )

    hard_checks = {
        "grgm900c_shtools": bool(gravity900["accepted"]),
        "frame_transformations": bool(frames["accepted"]),
        "third_body_arithmetic": bool(third_bodies["accepted"]),
        "terrain_file_coordinates": bool(terrain["accepted"]),
        "independent_same_model_propagation": bool(independent_propagation["accepted"]),
    }
    if gravity1200 is not None:
        hard_checks["grgm1200b_shtools"] = bool(gravity1200["accepted"])

    result = {
        "campaign": "Independent scientific validation of lunar-astrodynamics-simulation",
        "data_manifest": manifest,
        "software_versions": software_versions(),
        "hard_acceptance_checks": hard_checks,
        "hard_acceptance_passed": all(hard_checks.values()),
        "gravity_grgm900c_vs_shtools": gravity900,
        "gravity_grgm1200b_vs_shtools": gravity1200,
        "frame_validation": frames,
        "third_body_validation": third_bodies,
        "terrain_validation": terrain,
        "same_model_independent_propagation": independent_propagation,
        "lro_reconstructed_trajectory_comparison": lro,
        "validation_scope_note": (
            "Hard acceptance applies only where identical mathematical models are compared independently. The LRO reconstructed-SPK comparison is deliberately reported as a physical model residual because the repository propagation does not reproduce the complete GEODYN estimation/force setup."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["hard_acceptance_passed"]:
        raise SystemExit("one or more independent hard-validation checks failed")


if __name__ == "__main__":
    main()
