"""Quantitative classical and nonsingular analysis for propagated lunar trajectories."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .constants import MOON_MEAN_RADIUS_M
from .elements import elements_from_state, modified_equinoctial_from_state, orbital_vectors_from_state
from .frames import RotationProvider
from .terrain import TerrainShapeModel, terrain_clearance_m

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True)
class ElementHistory:
    """Legacy classical-element history for non-circular, non-equatorial trajectories."""

    time_s: FloatArray
    semi_major_axis_m: FloatArray
    eccentricity: FloatArray
    inclination_rad: FloatArray
    raan_rad_unwrapped: FloatArray
    argument_of_periapsis_rad_unwrapped: FloatArray


@dataclass(frozen=True)
class ScalarEvolutionStatistics:
    """Separate a best-fit secular drift from bounded residual oscillation."""

    initial: float
    final: float
    minimum: float
    maximum: float
    mean: float
    peak_to_peak: float
    linear_rate_per_s: float
    linear_drift_over_span: float
    detrended_rms: float
    detrended_peak_to_peak: float


@dataclass(frozen=True)
class VectorEvolutionStatistics:
    """Per-component drift and bounded oscillation statistics for a vector history."""

    initial: FloatArray
    final: FloatArray
    minimum: FloatArray
    maximum: FloatArray
    linear_rate_per_s: FloatArray
    linear_drift_over_span: FloatArray
    detrended_rms: FloatArray
    detrended_peak_to_peak: FloatArray


@dataclass(frozen=True)
class DirectionEvolutionStatistics:
    """Angular evolution of a direction that may be undefined at some samples."""

    defined_fraction: float
    reference_time_s: float | None
    final_defined_time_s: float | None
    final_change_rad: float | None
    maximum_change_rad: float | None


@dataclass(frozen=True)
class OrbitEvolutionStatistics:
    semi_major_axis_m: ScalarEvolutionStatistics
    eccentricity: ScalarEvolutionStatistics
    inclination_rad: ScalarEvolutionStatistics
    periselene_altitude_m: ScalarEvolutionStatistics
    aposelene_altitude_m: ScalarEvolutionStatistics
    reference_radius_altitude_m: ScalarEvolutionStatistics
    modified_equinoctial_f: ScalarEvolutionStatistics
    modified_equinoctial_g: ScalarEvolutionStatistics
    modified_equinoctial_h: ScalarEvolutionStatistics
    modified_equinoctial_k: ScalarEvolutionStatistics
    eccentricity_vector: VectorEvolutionStatistics
    orbital_plane_normal: VectorEvolutionStatistics
    orbital_plane_direction: DirectionEvolutionStatistics
    apsidal_direction: DirectionEvolutionStatistics
    terrain_clearance_m: ScalarEvolutionStatistics | None


@dataclass(frozen=True)
class OrbitHistory:
    """Nonsingular osculating history intended for stability and frozen-orbit work."""

    time_s: FloatArray
    semi_major_axis_m: FloatArray
    semilatus_rectum_m: FloatArray
    eccentricity: FloatArray
    inclination_rad: FloatArray
    eccentricity_vector: FloatArray
    specific_angular_momentum_vector_m2_s: FloatArray
    orbital_plane_normal: FloatArray
    periselene_radius_m: FloatArray
    aposelene_radius_m: FloatArray
    periselene_altitude_m: FloatArray
    aposelene_altitude_m: FloatArray
    reference_radius_altitude_m: FloatArray
    modified_equinoctial_p_m: FloatArray
    modified_equinoctial_f: FloatArray
    modified_equinoctial_g: FloatArray
    modified_equinoctial_h: FloatArray
    modified_equinoctial_k: FloatArray
    modified_equinoctial_true_longitude_rad_unwrapped: FloatArray
    apsis_defined: BoolArray
    apsidal_direction: FloatArray
    orbital_plane_change_rad: FloatArray
    apsidal_direction_change_rad: FloatArray
    terrain_clearance_m: FloatArray | None
    reference_radius_m: float
    statistics: OrbitEvolutionStatistics

    @property
    def minimum_reference_altitude_m(self) -> float:
        return float(np.min(self.reference_radius_altitude_m))

    @property
    def maximum_reference_altitude_m(self) -> float:
        return float(np.max(self.reference_radius_altitude_m))

    @property
    def minimum_terrain_clearance_m(self) -> float | None:
        if self.terrain_clearance_m is None:
            return None
        return float(np.min(self.terrain_clearance_m))


def _validate_history_inputs(time_s: FloatArray, states: FloatArray) -> tuple[FloatArray, FloatArray]:
    t = np.asarray(time_s, dtype=float)
    y = np.asarray(states, dtype=float)
    if t.ndim != 1 or t.size < 2 or not np.all(np.isfinite(t)):
        raise ValueError("time_s must be a finite one-dimensional array with at least two samples")
    if np.any(np.diff(t) <= 0.0):
        raise ValueError("time_s must be strictly increasing")
    if y.shape != (6, t.size) or not np.all(np.isfinite(y)):
        raise ValueError("states must be finite with shape (6, len(time_s))")
    return t, y


def linear_rate(time_s: FloatArray, values: FloatArray) -> float:
    """Fit and return a linear rate per second for a scalar history."""
    t = np.asarray(time_s, dtype=float)
    value = np.asarray(values, dtype=float)
    if t.shape != value.shape or t.ndim != 1 or t.size < 2:
        raise ValueError("time and value histories must be matching one-dimensional arrays")
    if not np.all(np.isfinite(t)) or not np.all(np.isfinite(value)):
        raise ValueError("time and value histories must be finite")
    centered_t = t - np.mean(t)
    centered_value = value - np.mean(value)
    denominator = float(np.dot(centered_t, centered_t))
    if denominator == 0.0:
        raise ValueError("time history has zero span")
    return float(np.dot(centered_t, centered_value) / denominator)


def scalar_evolution_statistics(time_s: FloatArray, values: FloatArray) -> ScalarEvolutionStatistics:
    """Return secular linear drift and residual bounded-oscillation statistics."""
    t = np.asarray(time_s, dtype=float)
    value = np.asarray(values, dtype=float)
    if t.shape != value.shape or t.ndim != 1 or t.size < 2:
        raise ValueError("time and value histories must be matching one-dimensional arrays")
    if not np.all(np.isfinite(t)) or not np.all(np.isfinite(value)):
        raise ValueError("time and value histories must be finite")
    rate = linear_rate(t, value)
    intercept = float(np.mean(value) - rate * np.mean(t))
    trend = intercept + rate * t
    residual = value - trend
    return ScalarEvolutionStatistics(
        initial=float(value[0]),
        final=float(value[-1]),
        minimum=float(np.min(value)),
        maximum=float(np.max(value)),
        mean=float(np.mean(value)),
        peak_to_peak=float(np.ptp(value)),
        linear_rate_per_s=rate,
        linear_drift_over_span=float(rate * (t[-1] - t[0])),
        detrended_rms=float(np.sqrt(np.mean(residual * residual))),
        detrended_peak_to_peak=float(np.ptp(residual)),
    )


def vector_evolution_statistics(time_s: FloatArray, vectors: FloatArray) -> VectorEvolutionStatistics:
    t = np.asarray(time_s, dtype=float)
    values = np.asarray(vectors, dtype=float)
    if values.shape != (3, t.size) or not np.all(np.isfinite(values)):
        raise ValueError("vectors must be finite with shape (3, len(time_s))")
    stats = [scalar_evolution_statistics(t, values[index]) for index in range(3)]
    return VectorEvolutionStatistics(
        initial=values[:, 0].copy(),
        final=values[:, -1].copy(),
        minimum=np.min(values, axis=1),
        maximum=np.max(values, axis=1),
        linear_rate_per_s=np.array([item.linear_rate_per_s for item in stats]),
        linear_drift_over_span=np.array([item.linear_drift_over_span for item in stats]),
        detrended_rms=np.array([item.detrended_rms for item in stats]),
        detrended_peak_to_peak=np.array([item.detrended_peak_to_peak for item in stats]),
    )


def _direction_changes(vectors: FloatArray, defined: BoolArray) -> tuple[FloatArray, DirectionEvolutionStatistics]:
    changes = np.full(defined.size, np.nan, dtype=float)
    indices = np.flatnonzero(defined)
    if indices.size == 0:
        return changes, DirectionEvolutionStatistics(0.0, None, None, None, None)
    reference_index = int(indices[0])
    reference = vectors[:, reference_index]
    for index in indices:
        cosine = float(np.dot(reference, vectors[:, index]))
        changes[index] = float(np.arccos(np.clip(cosine, -1.0, 1.0)))
    return changes, DirectionEvolutionStatistics(
        defined_fraction=float(indices.size / defined.size),
        reference_time_s=float(reference_index),
        final_defined_time_s=float(indices[-1]),
        final_change_rad=float(changes[indices[-1]]),
        maximum_change_rad=float(np.nanmax(changes)),
    )


def _direction_statistics(
    time_s: FloatArray,
    vectors: FloatArray,
    defined: BoolArray,
) -> tuple[FloatArray, DirectionEvolutionStatistics]:
    changes, raw = _direction_changes(vectors, defined)
    indices = np.flatnonzero(defined)
    if indices.size == 0:
        return changes, raw
    return changes, DirectionEvolutionStatistics(
        defined_fraction=raw.defined_fraction,
        reference_time_s=float(time_s[indices[0]]),
        final_defined_time_s=float(time_s[indices[-1]]),
        final_change_rad=raw.final_change_rad,
        maximum_change_rad=raw.maximum_change_rad,
    )


def element_history(time_s: FloatArray, states: FloatArray, mu_m3_s2: float) -> ElementHistory:
    """Convert a state history to classical elements when their angles exist.

    Circular/equatorial samples raise with guidance to use :func:`orbit_history`
    rather than returning arbitrary RAAN or argument-of-periapsis values.
    """
    t, y = _validate_history_inputs(time_s, states)
    elements = []
    for index in range(t.size):
        try:
            elements.append(elements_from_state(y[:, index], mu_m3_s2))
        except ValueError as exc:
            raise ValueError(
                f"classical element history is undefined at sample {index}: {exc}; use orbit_history for nonsingular analysis"
            ) from exc
    return ElementHistory(
        time_s=t,
        semi_major_axis_m=np.array([e.semi_major_axis_m for e in elements]),
        eccentricity=np.array([e.eccentricity for e in elements]),
        inclination_rad=np.array([e.inclination_rad for e in elements]),
        raan_rad_unwrapped=np.unwrap(np.array([e.raan_rad for e in elements])),
        argument_of_periapsis_rad_unwrapped=np.unwrap(
            np.array([e.argument_of_periapsis_rad for e in elements])
        ),
    )


def orbit_history(
    time_s: FloatArray,
    states: FloatArray,
    mu_m3_s2: float,
    *,
    reference_radius_m: float = MOON_MEAN_RADIUS_M,
    terrain: TerrainShapeModel | None = None,
    terrain_body_fixed_from_inertial: RotationProvider | None = None,
    terrain_frame: str | None = None,
    apsis_eccentricity_threshold: float = 1e-10,
) -> OrbitHistory:
    """Build a nonsingular lunar-orbit history for stability/frozen-orbit analysis.

    Reference-radius altitude is always reported. If terrain inputs are supplied,
    actual radial terrain clearance is additionally evaluated and kept separate.
    The trajectory must remain bound and elliptic because aposelene is part of
    this analysis product.
    """
    t, y = _validate_history_inputs(time_s, states)
    if not np.isfinite(reference_radius_m) or reference_radius_m <= 0.0:
        raise ValueError("reference_radius_m must be finite and positive")
    if not np.isfinite(apsis_eccentricity_threshold) or apsis_eccentricity_threshold < 0.0:
        raise ValueError("apsis_eccentricity_threshold must be finite and non-negative")
    terrain_requested = terrain is not None or terrain_body_fixed_from_inertial is not None or terrain_frame is not None
    if terrain_requested and (terrain is None or terrain_body_fixed_from_inertial is None or terrain_frame is None):
        raise ValueError("terrain, terrain_body_fixed_from_inertial, and terrain_frame must be supplied together")

    vectors = [orbital_vectors_from_state(y[:, index], mu_m3_s2) for index in range(t.size)]
    if any((not np.isfinite(item.semi_major_axis_m)) or item.semi_major_axis_m <= 0.0 or item.eccentricity >= 1.0 for item in vectors):
        raise ValueError("orbit_history currently requires a bound elliptic trajectory")
    equinoctial = [modified_equinoctial_from_state(y[:, index], mu_m3_s2) for index in range(t.size)]

    semi_major_axis = np.array([item.semi_major_axis_m for item in vectors])
    semilatus_rectum = np.array([item.semilatus_rectum_m for item in vectors])
    eccentricity = np.array([item.eccentricity for item in vectors])
    inclination = np.array([item.inclination_rad for item in vectors])
    eccentricity_vector = np.column_stack([item.eccentricity_vector for item in vectors])
    angular_momentum = np.column_stack(
        [item.specific_angular_momentum_vector_m2_s for item in vectors]
    )
    plane_normal = np.column_stack([item.orbital_plane_normal for item in vectors])
    periselene_radius = np.array([item.periselene_radius_m for item in vectors])
    aposelene_radius = np.array([item.aposelene_radius_m for item in vectors])
    periselene_altitude = periselene_radius - reference_radius_m
    aposelene_altitude = aposelene_radius - reference_radius_m
    reference_altitude = np.linalg.norm(y[:3], axis=0) - reference_radius_m

    mee_p = np.array([item.semilatus_rectum_m for item in equinoctial])
    mee_f = np.array([item.f for item in equinoctial])
    mee_g = np.array([item.g for item in equinoctial])
    mee_h = np.array([item.h for item in equinoctial])
    mee_k = np.array([item.k for item in equinoctial])
    mee_longitude = np.unwrap(np.array([item.true_longitude_rad for item in equinoctial]))

    apsis_defined = eccentricity >= apsis_eccentricity_threshold
    apsidal_direction = np.full_like(eccentricity_vector, np.nan)
    if np.any(apsis_defined):
        apsidal_direction[:, apsis_defined] = (
            eccentricity_vector[:, apsis_defined] / eccentricity[apsis_defined]
        )
    plane_defined = np.ones(t.size, dtype=bool)
    plane_changes, plane_direction_stats = _direction_statistics(t, plane_normal, plane_defined)
    apsidal_changes, apsidal_direction_stats = _direction_statistics(
        t, apsidal_direction, apsis_defined
    )

    terrain_clearance = None
    if terrain is not None and terrain_body_fixed_from_inertial is not None and terrain_frame is not None:
        terrain_clearance = np.array(
            [
                terrain_clearance_m(
                    float(t[index]),
                    y[:3, index],
                    terrain,
                    terrain_body_fixed_from_inertial,
                    terrain_frame=terrain_frame,
                )
                for index in range(t.size)
            ],
            dtype=float,
        )

    statistics = OrbitEvolutionStatistics(
        semi_major_axis_m=scalar_evolution_statistics(t, semi_major_axis),
        eccentricity=scalar_evolution_statistics(t, eccentricity),
        inclination_rad=scalar_evolution_statistics(t, inclination),
        periselene_altitude_m=scalar_evolution_statistics(t, periselene_altitude),
        aposelene_altitude_m=scalar_evolution_statistics(t, aposelene_altitude),
        reference_radius_altitude_m=scalar_evolution_statistics(t, reference_altitude),
        modified_equinoctial_f=scalar_evolution_statistics(t, mee_f),
        modified_equinoctial_g=scalar_evolution_statistics(t, mee_g),
        modified_equinoctial_h=scalar_evolution_statistics(t, mee_h),
        modified_equinoctial_k=scalar_evolution_statistics(t, mee_k),
        eccentricity_vector=vector_evolution_statistics(t, eccentricity_vector),
        orbital_plane_normal=vector_evolution_statistics(t, plane_normal),
        orbital_plane_direction=plane_direction_stats,
        apsidal_direction=apsidal_direction_stats,
        terrain_clearance_m=(
            None if terrain_clearance is None else scalar_evolution_statistics(t, terrain_clearance)
        ),
    )

    return OrbitHistory(
        time_s=t,
        semi_major_axis_m=semi_major_axis,
        semilatus_rectum_m=semilatus_rectum,
        eccentricity=eccentricity,
        inclination_rad=inclination,
        eccentricity_vector=eccentricity_vector,
        specific_angular_momentum_vector_m2_s=angular_momentum,
        orbital_plane_normal=plane_normal,
        periselene_radius_m=periselene_radius,
        aposelene_radius_m=aposelene_radius,
        periselene_altitude_m=periselene_altitude,
        aposelene_altitude_m=aposelene_altitude,
        reference_radius_altitude_m=reference_altitude,
        modified_equinoctial_p_m=mee_p,
        modified_equinoctial_f=mee_f,
        modified_equinoctial_g=mee_g,
        modified_equinoctial_h=mee_h,
        modified_equinoctial_k=mee_k,
        modified_equinoctial_true_longitude_rad_unwrapped=mee_longitude,
        apsis_defined=apsis_defined,
        apsidal_direction=apsidal_direction,
        orbital_plane_change_rad=plane_changes,
        apsidal_direction_change_rad=apsidal_changes,
        terrain_clearance_m=terrain_clearance,
        reference_radius_m=float(reference_radius_m),
        statistics=statistics,
    )
