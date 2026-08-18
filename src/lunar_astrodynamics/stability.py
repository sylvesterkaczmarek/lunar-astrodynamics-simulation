"""Automated low-lunar-orbit stability and frozen-orbit search.

The search layer deliberately separates physical stability metrics from ranking.
A candidate that survives propagation is not automatically a frozen orbit.  The
primary metrics follow quantities used in practical low-lunar-orbit design:
osculating periselene/aposelene spread, eccentricity-vector evolution,
eccentricity variation, apsidal/plane evolution, terrain clearance and lifetime.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field, replace
from itertools import product
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
import csv
import json

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .analysis import OrbitHistory, orbit_history
from .constants import GRGM1200A_J2, MOON_MEAN_RADIUS_M, LunarJ2Model
from .dynamics import total_acceleration
from .elements import (
    ClassicalElements,
    modified_equinoctial_from_state,
    state_from_elements,
)
from .forces import ForceComponent
from .frames import RotationProvider, validate_rotation_matrix
from .harmonics import SphericalHarmonicModel, gravity_acceleration_inertial
from .propagation import AccelerationFunction, PropagationSettings, propagate_with_acceleration
from .terrain import TerrainShapeModel, propagate_with_terrain

FloatArray = NDArray[np.float64]
_TWO_PI = 2.0 * np.pi


def _wrap(angle_rad: float) -> float:
    return float(angle_rad % _TWO_PI)


def _float_tuple(values: Sequence[float], *, name: str) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if not result or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain at least one finite value")
    return result


def _jsonable(value: object) -> object:
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "__dataclass_fields__"):
        return _jsonable(asdict(value))
    return str(value)


@dataclass(frozen=True)
class SearchDynamics:
    """One complete acceleration model used by stability search.

    ``analysis_reference_radius_m`` is deliberately distinct from any gravity
    coefficient reference radius.  For GRGM1200A searches it normally remains
    the lunar mean radius (1737.4 km), while gravity synthesis uses the model's
    own 1738 km reference radius internally.
    """

    name: str
    mu_m3_s2: float
    analysis_reference_radius_m: float
    collision_radius_m: float
    acceleration: AccelerationFunction = field(repr=False, compare=False)
    fidelity: str = "custom"
    harmonic_degree: int | None = None
    harmonic_order: int | None = None
    parallel_safe: bool = False
    provenance_data: Mapping[str, object] = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.name or not self.fidelity:
            raise ValueError("search dynamics name and fidelity must be non-empty")
        if (
            not np.isfinite(self.mu_m3_s2)
            or not np.isfinite(self.analysis_reference_radius_m)
            or not np.isfinite(self.collision_radius_m)
            or self.mu_m3_s2 <= 0.0
            or self.analysis_reference_radius_m <= 0.0
            or self.collision_radius_m <= 0.0
        ):
            raise ValueError("search dynamics mu and radii must be finite and positive")
        if self.harmonic_degree is not None and self.harmonic_degree < 0:
            raise ValueError("harmonic_degree must be non-negative")
        if self.harmonic_order is not None and self.harmonic_order < 0:
            raise ValueError("harmonic_order must be non-negative")

    def provenance(self) -> dict[str, object]:
        return {
            "name": self.name,
            "fidelity": self.fidelity,
            "mu_m3_s2": float(self.mu_m3_s2),
            "analysis_reference_radius_m": float(self.analysis_reference_radius_m),
            "collision_radius_m": float(self.collision_radius_m),
            "harmonic_degree": self.harmonic_degree,
            "harmonic_order": self.harmonic_order,
            "parallel_safe": bool(self.parallel_safe),
            **dict(self.provenance_data),
        }


def j2_search_dynamics(
    *,
    model: LunarJ2Model = GRGM1200A_J2,
    include_j2: bool = True,
    body_fixed_from_inertial: RotationProvider | None = None,
    analysis_reference_radius_m: float = MOON_MEAN_RADIUS_M,
) -> SearchDynamics:
    """Build a low-degree central/J2 search model.

    When ``body_fixed_from_inertial`` is supplied, the J2 symmetry axis follows
    that body-fixed frame.  Without it, the historical simple model uses the
    inertial z-axis and is best treated as a screening/demo fidelity.
    """
    j2 = float(model.j2 if include_j2 else 0.0)

    if body_fixed_from_inertial is None:
        def acceleration(_time_s: float, position_m: FloatArray) -> FloatArray:
            return total_acceleration(position_m, model.mu_m3_s2, model.reference_radius_m, j2)

        axis_note = "inertial z-axis"
    else:
        def acceleration(time_s: float, position_m: FloatArray) -> FloatArray:
            rotation = validate_rotation_matrix(body_fixed_from_inertial(float(time_s)))
            body_position = rotation @ np.asarray(position_m, dtype=float)
            body_acceleration = total_acceleration(
                body_position,
                model.mu_m3_s2,
                model.reference_radius_m,
                j2,
            )
            return (rotation.T @ body_acceleration).astype(float)

        axis_note = "caller-supplied rotating body-fixed z-axis"

    return SearchDynamics(
        name=f"{model.name}{'' if include_j2 else ' central only'}",
        mu_m3_s2=model.mu_m3_s2,
        analysis_reference_radius_m=analysis_reference_radius_m,
        collision_radius_m=model.collision_radius_m,
        acceleration=acceleration,
        fidelity="central + J2" if include_j2 else "central gravity",
        harmonic_degree=2 if include_j2 else 0,
        harmonic_order=0,
        parallel_safe=True,
        provenance_data={
            "gravity_reference_radius_m": float(model.reference_radius_m),
            "j2": j2,
            "j2_axis": axis_note,
        },
    )


def harmonic_search_dynamics(
    gravity_model: SphericalHarmonicModel,
    body_fixed_from_inertial: RotationProvider,
    *,
    max_degree: int | None = None,
    max_order: int | None = None,
    additional_forces: Sequence[ForceComponent] = (),
    analysis_reference_radius_m: float = MOON_MEAN_RADIUS_M,
    collision_radius_m: float = MOON_MEAN_RADIUS_M,
    parallel_safe: bool | None = None,
    name: str | None = None,
) -> SearchDynamics:
    """Build high-degree lunar gravity plus optional composed perturbation forces."""
    degree = gravity_model.max_degree if max_degree is None else min(int(max_degree), gravity_model.max_degree)
    if degree < 0:
        raise ValueError("max_degree must be non-negative")
    order = degree if max_order is None else min(int(max_order), degree)
    if order < 0:
        raise ValueError("max_order must be non-negative")
    extras = tuple(additional_forces)

    def acceleration(time_s: float, position_m: FloatArray) -> FloatArray:
        total = gravity_acceleration_inertial(
            time_s,
            position_m,
            gravity_model,
            body_fixed_from_inertial,
            max_degree=degree,
            max_order=order,
        )
        for component in extras:
            contribution = np.asarray(component(float(time_s), np.asarray(position_m, dtype=float)), dtype=float)
            if contribution.shape != (3,) or not np.all(np.isfinite(contribution)):
                raise ValueError(f"force component '{component.name}' returned an invalid acceleration")
            total = total + contribution
        return np.asarray(total, dtype=float)

    extra_provenance: list[dict[str, object]] = []
    for component in extras:
        method = getattr(component, "provenance", None)
        extra_provenance.append(dict(method()) if callable(method) else {"name": component.name})

    safe = (not extras) if parallel_safe is None else bool(parallel_safe)
    return SearchDynamics(
        name=name or f"{gravity_model.name} n<={degree} m<={order}",
        mu_m3_s2=gravity_model.mu_m3_s2,
        analysis_reference_radius_m=analysis_reference_radius_m,
        collision_radius_m=collision_radius_m,
        acceleration=acceleration,
        fidelity="spherical harmonics" + (" + perturbation forces" if extras else ""),
        harmonic_degree=degree,
        harmonic_order=order,
        parallel_safe=safe,
        provenance_data={
            "gravity_model": gravity_model.name,
            "gravity_frame": gravity_model.frame,
            "gravity_reference_radius_m": float(gravity_model.reference_radius_m),
            "gravity_normalization": gravity_model.normalization,
            "additional_forces": extra_provenance,
        },
    )


def harmonic_ensemble_dynamics(
    gravity_models: Sequence[SphericalHarmonicModel],
    body_fixed_from_inertial: RotationProvider,
    *,
    max_degree: int | None = None,
    max_order: int | None = None,
    additional_forces: Sequence[ForceComponent] = (),
    analysis_reference_radius_m: float = MOON_MEAN_RADIUS_M,
    collision_radius_m: float = MOON_MEAN_RADIUS_M,
    parallel_safe: bool | None = None,
) -> tuple[SearchDynamics, ...]:
    """Convert compatible gravity realizations into complete search dynamics."""
    models = tuple(gravity_models)
    if not models:
        raise ValueError("at least one gravity realization is required")
    if len({model.frame for model in models}) != 1:
        raise ValueError("gravity realizations must use the same body-fixed frame")
    return tuple(
        harmonic_search_dynamics(
            model,
            body_fixed_from_inertial,
            max_degree=max_degree,
            max_order=max_order,
            additional_forces=additional_forces,
            analysis_reference_radius_m=analysis_reference_radius_m,
            collision_radius_m=collision_radius_m,
            parallel_safe=parallel_safe,
            name=f"uncertainty realization {index + 1}: {model.name}",
        )
        for index, model in enumerate(models)
    )


@dataclass(frozen=True)
class OrbitSearchPoint:
    semi_major_axis_m: float
    semi_major_axis_altitude_m: float
    eccentricity: float
    inclination_rad: float
    raan_rad: float
    periapsis_parameter_rad: float
    initial_anomaly_rad: float
    periapsis_parameterization: str

    @property
    def argument_of_periapsis_rad(self) -> float:
        if self.periapsis_parameterization == "argument_of_periapsis":
            return _wrap(self.periapsis_parameter_rad)
        return _wrap(self.periapsis_parameter_rad - self.raan_rad)

    @property
    def longitude_of_periapsis_rad(self) -> float:
        if self.periapsis_parameterization == "longitude_of_periapsis":
            return _wrap(self.periapsis_parameter_rad)
        return _wrap(self.raan_rad + self.periapsis_parameter_rad)

    def initial_state(self, mu_m3_s2: float) -> FloatArray:
        return state_from_elements(
            ClassicalElements(
                self.semi_major_axis_m,
                self.eccentricity,
                self.inclination_rad,
                self.raan_rad,
                self.argument_of_periapsis_rad,
                self.initial_anomaly_rad,
            ),
            mu_m3_s2,
        )

    def as_dict(self) -> dict[str, float | str]:
        return {
            "semi_major_axis_m": float(self.semi_major_axis_m),
            "semi_major_axis_altitude_m": float(self.semi_major_axis_altitude_m),
            "eccentricity": float(self.eccentricity),
            "inclination_rad": float(self.inclination_rad),
            "inclination_deg": float(np.rad2deg(self.inclination_rad)),
            "raan_rad": float(_wrap(self.raan_rad)),
            "raan_deg": float(np.rad2deg(_wrap(self.raan_rad))),
            "argument_of_periapsis_rad": float(self.argument_of_periapsis_rad),
            "argument_of_periapsis_deg": float(np.rad2deg(self.argument_of_periapsis_rad)),
            "longitude_of_periapsis_rad": float(self.longitude_of_periapsis_rad),
            "longitude_of_periapsis_deg": float(np.rad2deg(self.longitude_of_periapsis_rad)),
            "initial_anomaly_rad": float(_wrap(self.initial_anomaly_rad)),
            "initial_anomaly_deg": float(np.rad2deg(_wrap(self.initial_anomaly_rad))),
            "periapsis_parameterization": self.periapsis_parameterization,
        }


@dataclass(frozen=True)
class StabilitySearchSpace:
    """Deterministic Cartesian product of physically interpretable orbit parameters.

    Exactly one of ``semi_major_axis_altitudes_m`` or ``semi_major_axes_m`` is
    supplied.  Altitude here means semimajor-axis altitude above the analysis
    reference radius, not instantaneous terrain clearance.
    """

    semi_major_axis_altitudes_m: tuple[float, ...] | None
    semi_major_axes_m: tuple[float, ...] | None
    eccentricities: tuple[float, ...]
    inclinations_rad: tuple[float, ...]
    raan_rad: tuple[float, ...]
    periapsis_rad: tuple[float, ...]
    initial_anomaly_rad: tuple[float, ...] = (0.0,)
    periapsis_parameterization: str = "longitude_of_periapsis"

    def __post_init__(self) -> None:
        if (self.semi_major_axis_altitudes_m is None) == (self.semi_major_axes_m is None):
            raise ValueError("provide exactly one of semi_major_axis_altitudes_m or semi_major_axes_m")
        if self.semi_major_axis_altitudes_m is not None:
            altitude = _float_tuple(self.semi_major_axis_altitudes_m, name="semi_major_axis_altitudes_m")
            if any(value <= 0.0 for value in altitude):
                raise ValueError("semimajor-axis altitudes must be positive")
            object.__setattr__(self, "semi_major_axis_altitudes_m", altitude)
        if self.semi_major_axes_m is not None:
            axes = _float_tuple(self.semi_major_axes_m, name="semi_major_axes_m")
            if any(value <= 0.0 for value in axes):
                raise ValueError("semimajor axes must be positive")
            object.__setattr__(self, "semi_major_axes_m", axes)
        eccentricity = _float_tuple(self.eccentricities, name="eccentricities")
        if any(value < 0.0 or value >= 1.0 for value in eccentricity):
            raise ValueError("search eccentricities must satisfy 0 <= e < 1")
        inclination = _float_tuple(self.inclinations_rad, name="inclinations_rad")
        if any(value < 0.0 or value >= np.pi for value in inclination):
            raise ValueError("prograde-MEE search inclinations must satisfy 0 <= i < pi")
        object.__setattr__(self, "eccentricities", eccentricity)
        object.__setattr__(self, "inclinations_rad", inclination)
        object.__setattr__(self, "raan_rad", _float_tuple(self.raan_rad, name="raan_rad"))
        object.__setattr__(self, "periapsis_rad", _float_tuple(self.periapsis_rad, name="periapsis_rad"))
        object.__setattr__(self, "initial_anomaly_rad", _float_tuple(self.initial_anomaly_rad, name="initial_anomaly_rad"))
        if self.periapsis_parameterization not in {"argument_of_periapsis", "longitude_of_periapsis"}:
            raise ValueError(
                "periapsis_parameterization must be 'argument_of_periapsis' or 'longitude_of_periapsis'"
            )

    @property
    def raw_grid_size(self) -> int:
        radial = self.semi_major_axis_altitudes_m or self.semi_major_axes_m or ()
        return int(
            len(radial)
            * len(self.eccentricities)
            * len(self.inclinations_rad)
            * len(self.raan_rad)
            * len(self.periapsis_rad)
            * len(self.initial_anomaly_rad)
        )

    def points(self, analysis_reference_radius_m: float) -> tuple[OrbitSearchPoint, ...]:
        if not np.isfinite(analysis_reference_radius_m) or analysis_reference_radius_m <= 0.0:
            raise ValueError("analysis_reference_radius_m must be finite and positive")
        if self.semi_major_axis_altitudes_m is not None:
            radial = tuple(
                (analysis_reference_radius_m + altitude, altitude)
                for altitude in self.semi_major_axis_altitudes_m
            )
        else:
            assert self.semi_major_axes_m is not None
            radial = tuple(
                (axis, axis - analysis_reference_radius_m) for axis in self.semi_major_axes_m
            )
        return tuple(
            OrbitSearchPoint(
                semi_major_axis_m=axis,
                semi_major_axis_altitude_m=altitude,
                eccentricity=eccentricity,
                inclination_rad=inclination,
                raan_rad=_wrap(raan),
                periapsis_parameter_rad=_wrap(periapsis),
                initial_anomaly_rad=_wrap(anomaly),
                periapsis_parameterization=self.periapsis_parameterization,
            )
            for (axis, altitude), eccentricity, inclination, raan, periapsis, anomaly in product(
                radial,
                self.eccentricities,
                self.inclinations_rad,
                self.raan_rad,
                self.periapsis_rad,
                self.initial_anomaly_rad,
            )
        )

    def as_dict(self) -> dict[str, object]:
        return _jsonable(asdict(self))  # type: ignore[return-value]


def default_low_lunar_search_space() -> StabilitySearchSpace:
    """Return the canonical surface-safe default search grid.

    This compatibility wrapper delegates to :mod:`search_defaults` so direct
    imports from ``lunar_astrodynamics.stability`` cannot diverge from the
    package-level default.
    """
    from .search_defaults import default_low_lunar_search_space as canonical_default

    return canonical_default()


@dataclass(frozen=True)
class StabilityMetrics:
    duration_s: float
    impact_free_lifetime_s: float
    survived_duration_fraction: float
    impacted: bool
    impact_time_s: float | None
    minimum_reference_altitude_m: float
    maximum_reference_altitude_m: float
    semi_major_axis_peak_to_peak_m: float
    periselene_altitude_min_m: float
    periselene_altitude_max_m: float
    periselene_altitude_peak_to_peak_m: float
    aposelene_altitude_min_m: float
    aposelene_altitude_max_m: float
    aposelene_altitude_peak_to_peak_m: float
    eccentricity_minimum: float
    eccentricity_maximum: float
    eccentricity_peak_to_peak: float
    eccentricity_vector_final_change_norm: float
    eccentricity_vector_linear_drift_norm: float
    eccentricity_vector_detrended_max_radius: float
    apsidal_direction_defined_fraction: float
    apsidal_direction_max_change_rad: float | None
    orbital_plane_final_change_rad: float
    orbital_plane_max_change_rad: float
    minimum_terrain_clearance_m: float | None

    def as_dict(self) -> dict[str, object]:
        return _jsonable(asdict(self))  # type: ignore[return-value]


@dataclass(frozen=True)
class UncertaintyStabilitySummary:
    realization_count: int
    adverse_percentile: float
    impact_fraction: float
    minimum_lifetime_s: float
    minimum_terrain_clearance_m: float | None
    periselene_altitude_peak_to_peak_m: float
    aposelene_altitude_peak_to_peak_m: float
    eccentricity_peak_to_peak: float
    eccentricity_vector_linear_drift_norm: float
    eccentricity_vector_detrended_max_radius: float
    apsidal_direction_max_change_rad: float | None
    orbital_plane_max_change_rad: float

    def as_dict(self) -> dict[str, object]:
        return _jsonable(asdict(self))  # type: ignore[return-value]


@dataclass(frozen=True)
class StabilityConstraints:
    """Optional hard filters, separate from the ranking convenience metric."""

    require_full_duration: bool = True
    minimum_reference_altitude_m: float | None = None
    minimum_terrain_clearance_m: float | None = None
    maximum_periselene_spread_m: float | None = None
    maximum_aposelene_spread_m: float | None = None
    maximum_eccentricity_variation: float | None = None
    maximum_eccentricity_vector_drift: float | None = None
    maximum_orbital_plane_change_rad: float | None = None
    apply_to_uncertainty: bool = True

    def __post_init__(self) -> None:
        for name in (
            "minimum_reference_altitude_m",
            "minimum_terrain_clearance_m",
            "maximum_periselene_spread_m",
            "maximum_aposelene_spread_m",
            "maximum_eccentricity_variation",
            "maximum_eccentricity_vector_drift",
            "maximum_orbital_plane_change_rad",
        ):
            value = getattr(self, name)
            if value is not None and (not np.isfinite(value) or value < 0.0):
                raise ValueError(f"{name} must be finite and non-negative when supplied")


@dataclass(frozen=True)
class StabilityRankingPolicy:
    """Transparent weighted normalisation used only to order otherwise reported metrics.

    The score is a convenience, not the definition of a frozen orbit.  NASA LRO
    experience motivates the larger default weights on periselene spread and
    eccentricity-vector drift.  All scales and weights are caller configurable.
    """

    periselene_spread_scale_m: float = 10_000.0
    aposelene_spread_scale_m: float = 10_000.0
    eccentricity_vector_drift_scale: float = 1.0e-3
    eccentricity_variation_scale: float = 1.0e-3
    apsidal_change_scale_rad: float = float(np.deg2rad(30.0))
    plane_change_scale_rad: float = float(np.deg2rad(1.0))
    periselene_weight: float = 3.0
    aposelene_weight: float = 1.0
    eccentricity_vector_drift_weight: float = 3.0
    eccentricity_variation_weight: float = 1.0
    apsidal_change_weight: float = 1.0
    plane_change_weight: float = 1.0
    lifetime_shortfall_weight: float = 6.0
    minimum_clearance_target_m: float | None = None
    clearance_shortfall_weight: float = 4.0
    use_uncertainty_when_available: bool = True

    def __post_init__(self) -> None:
        positive = (
            self.periselene_spread_scale_m,
            self.aposelene_spread_scale_m,
            self.eccentricity_vector_drift_scale,
            self.eccentricity_variation_scale,
            self.apsidal_change_scale_rad,
            self.plane_change_scale_rad,
        )
        if any(not np.isfinite(value) or value <= 0.0 for value in positive):
            raise ValueError("ranking scales must be finite and positive")
        weights = (
            self.periselene_weight,
            self.aposelene_weight,
            self.eccentricity_vector_drift_weight,
            self.eccentricity_variation_weight,
            self.apsidal_change_weight,
            self.plane_change_weight,
            self.lifetime_shortfall_weight,
            self.clearance_shortfall_weight,
        )
        if any(not np.isfinite(value) or value < 0.0 for value in weights):
            raise ValueError("ranking weights must be finite and non-negative")
        if self.minimum_clearance_target_m is not None and (
            not np.isfinite(self.minimum_clearance_target_m)
            or self.minimum_clearance_target_m <= 0.0
        ):
            raise ValueError("minimum_clearance_target_m must be finite and positive")


@dataclass(frozen=True)
class RankingBreakdown:
    normalised_terms: Mapping[str, float]
    weighted_contributions: Mapping[str, float]
    applicable_weight_sum: float
    penalty: float

    def as_dict(self) -> dict[str, object]:
        return _jsonable(asdict(self))  # type: ignore[return-value]


@dataclass(frozen=True)
class StabilitySearchSettings:
    duration_s: float = 7.0 * 86400.0
    sample_count: int = 257
    propagation: PropagationSettings = PropagationSettings()
    workers: int = 1
    apsis_eccentricity_threshold: float = 1.0e-6
    terrain_clearance_search_samples: int = 513
    uncertainty_adverse_percentile: float = 95.0
    constraints: StabilityConstraints = StabilityConstraints()
    ranking: StabilityRankingPolicy = StabilityRankingPolicy()

    def __post_init__(self) -> None:
        if not np.isfinite(self.duration_s) or self.duration_s <= 0.0:
            raise ValueError("duration_s must be finite and positive")
        if self.sample_count < 3:
            raise ValueError("sample_count must be at least three")
        if self.workers < 1:
            raise ValueError("workers must be at least one")
        if (
            not np.isfinite(self.apsis_eccentricity_threshold)
            or self.apsis_eccentricity_threshold < 0.0
        ):
            raise ValueError("apsis_eccentricity_threshold must be finite and non-negative")
        if self.terrain_clearance_search_samples < 3:
            raise ValueError("terrain_clearance_search_samples must be at least three")
        if (
            not np.isfinite(self.uncertainty_adverse_percentile)
            or self.uncertainty_adverse_percentile <= 50.0
            or self.uncertainty_adverse_percentile > 100.0
        ):
            raise ValueError("uncertainty_adverse_percentile must lie within (50, 100]")


@dataclass(frozen=True)
class StabilityCandidate:
    candidate_id: str
    parameters: OrbitSearchPoint
    initial_state: tuple[float, float, float, float, float, float]
    metrics: StabilityMetrics
    uncertainty_metrics: tuple[StabilityMetrics, ...]
    uncertainty_summary: UncertaintyStabilitySummary | None
    passed_constraints: bool
    constraint_violations: tuple[str, ...]
    ranking: RankingBreakdown
    rank: int | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "rank": self.rank,
            "parameters": self.parameters.as_dict(),
            "initial_state": list(self.initial_state),
            "metrics": self.metrics.as_dict(),
            "uncertainty_metrics": [item.as_dict() for item in self.uncertainty_metrics],
            "uncertainty_summary": (
                None if self.uncertainty_summary is None else self.uncertainty_summary.as_dict()
            ),
            "passed_constraints": self.passed_constraints,
            "constraint_violations": list(self.constraint_violations),
            "ranking": self.ranking.as_dict(),
        }


@dataclass(frozen=True)
class StabilitySearchResult:
    candidates: tuple[StabilityCandidate, ...]
    raw_grid_size: int
    unique_candidate_count: int
    dynamics_provenance: Mapping[str, object]
    uncertainty_dynamics_provenance: tuple[Mapping[str, object], ...]
    search_definition: Mapping[str, object]
    settings: StabilitySearchSettings

    def best_candidates(self, count: int = 10, *, feasible_only: bool = True) -> tuple[StabilityCandidate, ...]:
        if count < 1:
            raise ValueError("count must be at least one")
        pool = tuple(
            candidate
            for candidate in self.candidates
            if candidate.passed_constraints or not feasible_only
        )
        return pool[:count]

    def to_dict(self) -> dict[str, object]:
        return {
            "raw_grid_size": self.raw_grid_size,
            "unique_candidate_count": self.unique_candidate_count,
            "dynamics": _jsonable(self.dynamics_provenance),
            "uncertainty_dynamics": _jsonable(self.uncertainty_dynamics_provenance),
            "search_definition": _jsonable(self.search_definition),
            "settings": _jsonable(asdict(self.settings)),
            "candidates": [candidate.as_dict() for candidate in self.candidates],
        }

    def write_json(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

    def write_csv(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        rows: list[dict[str, object]] = []
        for candidate in self.candidates:
            row: dict[str, object] = {
                "candidate_id": candidate.candidate_id,
                "rank": candidate.rank,
                "passed_constraints": candidate.passed_constraints,
                "constraint_violations": "; ".join(candidate.constraint_violations),
                "ranking_penalty": candidate.ranking.penalty,
            }
            row.update(candidate.parameters.as_dict())
            row.update({f"metric_{key}": value for key, value in candidate.metrics.as_dict().items()})
            for key, value in candidate.ranking.normalised_terms.items():
                row[f"rank_term_{key}"] = value
            if candidate.uncertainty_summary is not None:
                row.update(
                    {
                        f"uncertainty_{key}": value
                        for key, value in candidate.uncertainty_summary.as_dict().items()
                    }
                )
            rows.append(row)
        fieldnames = sorted({key for row in rows for key in row})
        with destination.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


@dataclass(frozen=True)
class RefinementSettings:
    top_candidates: int = 3
    points_per_axis: int = 3
    spacing_fraction: float = 0.5
    refine_axes: tuple[str, ...] = (
        "semi_major_axis_m",
        "eccentricity",
        "inclination_rad",
        "raan_rad",
        "periapsis_parameter_rad",
    )

    def __post_init__(self) -> None:
        if self.top_candidates < 1:
            raise ValueError("top_candidates must be at least one")
        if self.points_per_axis < 1 or self.points_per_axis % 2 == 0:
            raise ValueError("points_per_axis must be a positive odd integer")
        if not np.isfinite(self.spacing_fraction) or self.spacing_fraction <= 0.0:
            raise ValueError("spacing_fraction must be finite and positive")
        allowed = {
            "semi_major_axis_m",
            "eccentricity",
            "inclination_rad",
            "raan_rad",
            "periapsis_parameter_rad",
            "initial_anomaly_rad",
        }
        if any(axis not in allowed for axis in self.refine_axes):
            raise ValueError("refine_axes contains an unsupported parameter")


@dataclass(frozen=True)
class CoarseToFineSearchResult:
    coarse: StabilitySearchResult
    refined: StabilitySearchResult
    seed_candidate_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "seed_candidate_ids": list(self.seed_candidate_ids),
            "coarse": self.coarse.to_dict(),
            "refined": self.refined.to_dict(),
        }

    def write_json(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")


@dataclass(frozen=True)
class StabilityMap:
    x_parameter: str
    y_parameter: str
    metric: str
    x_values: tuple[float, ...]
    y_values: tuple[float, ...]
    values: tuple[tuple[float | None, ...], ...]
    candidate_ids: tuple[tuple[str | None, ...], ...]
    fixed_parameters: Mapping[str, float]

    def to_dict(self) -> dict[str, object]:
        return _jsonable(asdict(self))  # type: ignore[return-value]

    def write_csv(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8", newline="") as stream:
            fieldnames = [self.x_parameter, self.y_parameter, self.metric, "candidate_id"]
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            for row_index, y_value in enumerate(self.y_values):
                for column_index, x_value in enumerate(self.x_values):
                    writer.writerow(
                        {
                            self.x_parameter: x_value,
                            self.y_parameter: y_value,
                            self.metric: self.values[row_index][column_index],
                            "candidate_id": self.candidate_ids[row_index][column_index],
                        }
                    )


def _physical_key(point: OrbitSearchPoint, mu_m3_s2: float) -> tuple[float, ...]:
    state = point.initial_state(mu_m3_s2)
    mee = modified_equinoctial_from_state(state, mu_m3_s2)
    return (
        round(mee.semilatus_rectum_m, 6),
        round(mee.f, 13),
        round(mee.g, 13),
        round(mee.h, 13),
        round(mee.k, 13),
        round(float(np.cos(mee.true_longitude_rad)), 13),
        round(float(np.sin(mee.true_longitude_rad)), 13),
    )


def _unique_points(
    points: Sequence[OrbitSearchPoint], mu_m3_s2: float
) -> tuple[OrbitSearchPoint, ...]:
    seen: set[tuple[float, ...]] = set()
    unique: list[OrbitSearchPoint] = []
    for point in points:
        key = _physical_key(point, mu_m3_s2)
        if key not in seen:
            seen.add(key)
            unique.append(point)
    return tuple(unique)


def _trajectory_with_event_sample(solution: object) -> tuple[FloatArray, FloatArray, bool, float | None]:
    time = np.asarray(solution.t, dtype=float)  # type: ignore[attr-defined]
    states = np.asarray(solution.y, dtype=float)  # type: ignore[attr-defined]
    event_times = solution.t_events[0]  # type: ignore[attr-defined]
    impacted = bool(len(event_times))
    impact_time = float(event_times[0]) if impacted else None
    if impacted:
        event_state = np.asarray(solution.y_events[0][0], dtype=float)  # type: ignore[attr-defined]
        if time.size == 0 or abs(float(time[-1]) - impact_time) > 1.0e-9:
            time = np.concatenate((time, np.array([impact_time])))
            states = np.column_stack((states, event_state))
    if time.size < 2:
        raise ValueError("trajectory analysis requires at least two samples")
    return time, states, impacted, impact_time


def _eccentricity_vector_detrended_max_radius(history: OrbitHistory) -> float:
    time = history.time_s
    vectors = history.eccentricity_vector
    rates = history.statistics.eccentricity_vector.linear_rate_per_s
    intercept = np.mean(vectors, axis=1) - rates * float(np.mean(time))
    trend = intercept[:, None] + rates[:, None] * time[None, :]
    residual = vectors - trend
    return float(np.max(np.linalg.norm(residual, axis=0)))


def _metrics_from_history(
    history: OrbitHistory,
    *,
    duration_s: float,
    lifetime_s: float,
    impacted: bool,
    impact_time_s: float | None,
    minimum_terrain_clearance_m: float | None,
) -> StabilityMetrics:
    e_vector = history.eccentricity_vector
    e_linear_drift = history.statistics.eccentricity_vector.linear_drift_over_span
    apsidal = history.statistics.apsidal_direction
    plane = history.statistics.orbital_plane_direction
    return StabilityMetrics(
        duration_s=float(duration_s),
        impact_free_lifetime_s=float(lifetime_s),
        survived_duration_fraction=float(min(1.0, lifetime_s / duration_s)),
        impacted=bool(impacted),
        impact_time_s=impact_time_s,
        minimum_reference_altitude_m=history.minimum_reference_altitude_m,
        maximum_reference_altitude_m=history.maximum_reference_altitude_m,
        semi_major_axis_peak_to_peak_m=float(np.ptp(history.semi_major_axis_m)),
        periselene_altitude_min_m=float(np.min(history.periselene_altitude_m)),
        periselene_altitude_max_m=float(np.max(history.periselene_altitude_m)),
        periselene_altitude_peak_to_peak_m=float(np.ptp(history.periselene_altitude_m)),
        aposelene_altitude_min_m=float(np.min(history.aposelene_altitude_m)),
        aposelene_altitude_max_m=float(np.max(history.aposelene_altitude_m)),
        aposelene_altitude_peak_to_peak_m=float(np.ptp(history.aposelene_altitude_m)),
        eccentricity_minimum=float(np.min(history.eccentricity)),
        eccentricity_maximum=float(np.max(history.eccentricity)),
        eccentricity_peak_to_peak=float(np.ptp(history.eccentricity)),
        eccentricity_vector_final_change_norm=float(np.linalg.norm(e_vector[:, -1] - e_vector[:, 0])),
        eccentricity_vector_linear_drift_norm=float(np.linalg.norm(e_linear_drift)),
        eccentricity_vector_detrended_max_radius=_eccentricity_vector_detrended_max_radius(history),
        apsidal_direction_defined_fraction=float(apsidal.defined_fraction),
        apsidal_direction_max_change_rad=(
            None if apsidal.maximum_change_rad is None else float(apsidal.maximum_change_rad)
        ),
        orbital_plane_final_change_rad=float(plane.final_change_rad or 0.0),
        orbital_plane_max_change_rad=float(plane.maximum_change_rad or 0.0),
        minimum_terrain_clearance_m=(
            None if minimum_terrain_clearance_m is None else float(minimum_terrain_clearance_m)
        ),
    )


def _propagate_candidate(
    initial_state: FloatArray,
    dynamics: SearchDynamics,
    settings: StabilitySearchSettings,
    *,
    terrain: TerrainShapeModel | None,
    terrain_body_fixed_from_inertial: RotationProvider | None,
    terrain_frame: str | None,
) -> StabilityMetrics:
    sample_times = np.linspace(0.0, float(settings.duration_s), int(settings.sample_count))
    minimum_terrain_clearance: float | None = None
    if terrain is None:
        solution = propagate_with_acceleration(
            initial_state,
            settings.duration_s,
            dynamics.acceleration,
            collision_radius_m=dynamics.collision_radius_m,
            sample_times_s=sample_times,
            settings=settings.propagation,
        )
    else:
        assert terrain_body_fixed_from_inertial is not None
        assert terrain_frame is not None
        terrain_result = propagate_with_terrain(
            initial_state,
            settings.duration_s,
            dynamics.acceleration,
            terrain,
            terrain_body_fixed_from_inertial,
            terrain_frame=terrain_frame,
            sample_times_s=sample_times,
            settings=settings.propagation,
            clearance_search_samples=settings.terrain_clearance_search_samples,
        )
        solution = terrain_result.solution
        minimum_terrain_clearance = float(terrain_result.clearance.minimum_clearance_m)

    time, states, impacted, impact_time = _trajectory_with_event_sample(solution)
    lifetime = impact_time if impacted and impact_time is not None else float(settings.duration_s)
    history = orbit_history(
        time,
        states,
        dynamics.mu_m3_s2,
        reference_radius_m=dynamics.analysis_reference_radius_m,
        apsis_eccentricity_threshold=settings.apsis_eccentricity_threshold,
    )
    return _metrics_from_history(
        history,
        duration_s=settings.duration_s,
        lifetime_s=lifetime,
        impacted=impacted,
        impact_time_s=impact_time,
        minimum_terrain_clearance_m=minimum_terrain_clearance,
    )


def _percentile(values: Sequence[float], percentile: float) -> float:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    return float(np.percentile(array, percentile)) if array.size else float("nan")


def _uncertainty_summary(
    metrics: Sequence[StabilityMetrics], percentile: float
) -> UncertaintyStabilitySummary:
    values = tuple(metrics)
    if not values:
        raise ValueError("at least one uncertainty realization metric is required")
    apsidal = [
        item.apsidal_direction_max_change_rad
        for item in values
        if item.apsidal_direction_max_change_rad is not None
    ]
    clearances = [
        item.minimum_terrain_clearance_m
        for item in values
        if item.minimum_terrain_clearance_m is not None
    ]
    return UncertaintyStabilitySummary(
        realization_count=len(values),
        adverse_percentile=float(percentile),
        impact_fraction=float(np.mean([item.impacted for item in values])),
        minimum_lifetime_s=float(min(item.impact_free_lifetime_s for item in values)),
        minimum_terrain_clearance_m=(
            None if not clearances else float(min(float(item) for item in clearances))
        ),
        periselene_altitude_peak_to_peak_m=_percentile(
            [item.periselene_altitude_peak_to_peak_m for item in values], percentile
        ),
        aposelene_altitude_peak_to_peak_m=_percentile(
            [item.aposelene_altitude_peak_to_peak_m for item in values], percentile
        ),
        eccentricity_peak_to_peak=_percentile(
            [item.eccentricity_peak_to_peak for item in values], percentile
        ),
        eccentricity_vector_linear_drift_norm=_percentile(
            [item.eccentricity_vector_linear_drift_norm for item in values], percentile
        ),
        eccentricity_vector_detrended_max_radius=_percentile(
            [item.eccentricity_vector_detrended_max_radius for item in values], percentile
        ),
        apsidal_direction_max_change_rad=(
            None if not apsidal else _percentile([float(item) for item in apsidal], percentile)
        ),
        orbital_plane_max_change_rad=_percentile(
            [item.orbital_plane_max_change_rad for item in values], percentile
        ),
    )


def _effective_metric_values(
    nominal: StabilityMetrics,
    uncertainty: UncertaintyStabilitySummary | None,
    policy: StabilityRankingPolicy,
) -> dict[str, float | None]:
    if uncertainty is None or not policy.use_uncertainty_when_available:
        return {
            "periselene_spread": nominal.periselene_altitude_peak_to_peak_m,
            "aposelene_spread": nominal.aposelene_altitude_peak_to_peak_m,
            "eccentricity_vector_drift": nominal.eccentricity_vector_linear_drift_norm,
            "eccentricity_variation": nominal.eccentricity_peak_to_peak,
            "apsidal_change": nominal.apsidal_direction_max_change_rad,
            "plane_change": nominal.orbital_plane_max_change_rad,
            "survival_fraction": nominal.survived_duration_fraction,
            "minimum_terrain_clearance": nominal.minimum_terrain_clearance_m,
        }
    return {
        "periselene_spread": max(
            nominal.periselene_altitude_peak_to_peak_m,
            uncertainty.periselene_altitude_peak_to_peak_m,
        ),
        "aposelene_spread": max(
            nominal.aposelene_altitude_peak_to_peak_m,
            uncertainty.aposelene_altitude_peak_to_peak_m,
        ),
        "eccentricity_vector_drift": max(
            nominal.eccentricity_vector_linear_drift_norm,
            uncertainty.eccentricity_vector_linear_drift_norm,
        ),
        "eccentricity_variation": max(
            nominal.eccentricity_peak_to_peak,
            uncertainty.eccentricity_peak_to_peak,
        ),
        "apsidal_change": (
            nominal.apsidal_direction_max_change_rad
            if uncertainty.apsidal_direction_max_change_rad is None
            else (
                uncertainty.apsidal_direction_max_change_rad
                if nominal.apsidal_direction_max_change_rad is None
                else max(
                    nominal.apsidal_direction_max_change_rad,
                    uncertainty.apsidal_direction_max_change_rad,
                )
            )
        ),
        "plane_change": max(
            nominal.orbital_plane_max_change_rad,
            uncertainty.orbital_plane_max_change_rad,
        ),
        "survival_fraction": min(
            nominal.survived_duration_fraction,
            uncertainty.minimum_lifetime_s / nominal.duration_s,
        ),
        "minimum_terrain_clearance": (
            nominal.minimum_terrain_clearance_m
            if uncertainty.minimum_terrain_clearance_m is None
            else (
                uncertainty.minimum_terrain_clearance_m
                if nominal.minimum_terrain_clearance_m is None
                else min(
                    nominal.minimum_terrain_clearance_m,
                    uncertainty.minimum_terrain_clearance_m,
                )
            )
        ),
    }


def _ranking_breakdown(
    nominal: StabilityMetrics,
    uncertainty: UncertaintyStabilitySummary | None,
    policy: StabilityRankingPolicy,
) -> RankingBreakdown:
    value = _effective_metric_values(nominal, uncertainty, policy)
    terms: dict[str, float] = {
        "periselene_spread": float(value["periselene_spread"]) / policy.periselene_spread_scale_m,
        "aposelene_spread": float(value["aposelene_spread"]) / policy.aposelene_spread_scale_m,
        "eccentricity_vector_drift": float(value["eccentricity_vector_drift"]) / policy.eccentricity_vector_drift_scale,
        "eccentricity_variation": float(value["eccentricity_variation"]) / policy.eccentricity_variation_scale,
        "plane_change": float(value["plane_change"]) / policy.plane_change_scale_rad,
        "lifetime_shortfall": max(0.0, 1.0 - float(value["survival_fraction"])),
    }
    weights: dict[str, float] = {
        "periselene_spread": policy.periselene_weight,
        "aposelene_spread": policy.aposelene_weight,
        "eccentricity_vector_drift": policy.eccentricity_vector_drift_weight,
        "eccentricity_variation": policy.eccentricity_variation_weight,
        "plane_change": policy.plane_change_weight,
        "lifetime_shortfall": policy.lifetime_shortfall_weight,
    }
    if value["apsidal_change"] is not None:
        terms["apsidal_change"] = float(value["apsidal_change"]) / policy.apsidal_change_scale_rad
        weights["apsidal_change"] = policy.apsidal_change_weight
    if policy.minimum_clearance_target_m is not None and value["minimum_terrain_clearance"] is not None:
        terms["terrain_clearance_shortfall"] = max(
            0.0,
            (policy.minimum_clearance_target_m - float(value["minimum_terrain_clearance"]))
            / policy.minimum_clearance_target_m,
        )
        weights["terrain_clearance_shortfall"] = policy.clearance_shortfall_weight
    weighted = {name: terms[name] * weights[name] for name in terms}
    weight_sum = float(sum(weights[name] for name in terms))
    penalty = float(sum(weighted.values()) / weight_sum) if weight_sum > 0.0 else 0.0
    return RankingBreakdown(
        normalised_terms=terms,
        weighted_contributions=weighted,
        applicable_weight_sum=weight_sum,
        penalty=penalty,
    )


def _constraint_violations(
    nominal: StabilityMetrics,
    uncertainty: UncertaintyStabilitySummary | None,
    constraints: StabilityConstraints,
) -> tuple[str, ...]:
    violations: list[str] = []
    if constraints.require_full_duration and nominal.survived_duration_fraction < 1.0 - 1.0e-12:
        violations.append("nominal trajectory did not survive the requested duration")
    if (
        constraints.minimum_reference_altitude_m is not None
        and nominal.minimum_reference_altitude_m < constraints.minimum_reference_altitude_m
    ):
        violations.append("minimum reference-radius altitude below constraint")
    if constraints.minimum_terrain_clearance_m is not None:
        if nominal.minimum_terrain_clearance_m is None:
            violations.append("terrain-clearance constraint requested without terrain evaluation")
        elif nominal.minimum_terrain_clearance_m < constraints.minimum_terrain_clearance_m:
            violations.append("minimum terrain clearance below constraint")
    if (
        constraints.maximum_periselene_spread_m is not None
        and nominal.periselene_altitude_peak_to_peak_m > constraints.maximum_periselene_spread_m
    ):
        violations.append("periselene-altitude spread above constraint")
    if (
        constraints.maximum_aposelene_spread_m is not None
        and nominal.aposelene_altitude_peak_to_peak_m > constraints.maximum_aposelene_spread_m
    ):
        violations.append("aposelene-altitude spread above constraint")
    if (
        constraints.maximum_eccentricity_variation is not None
        and nominal.eccentricity_peak_to_peak > constraints.maximum_eccentricity_variation
    ):
        violations.append("eccentricity variation above constraint")
    if (
        constraints.maximum_eccentricity_vector_drift is not None
        and nominal.eccentricity_vector_linear_drift_norm > constraints.maximum_eccentricity_vector_drift
    ):
        violations.append("eccentricity-vector drift above constraint")
    if (
        constraints.maximum_orbital_plane_change_rad is not None
        and nominal.orbital_plane_max_change_rad > constraints.maximum_orbital_plane_change_rad
    ):
        violations.append("orbital-plane change above constraint")

    if uncertainty is not None and constraints.apply_to_uncertainty:
        if constraints.require_full_duration and uncertainty.minimum_lifetime_s < nominal.duration_s - 1.0e-9:
            violations.append("at least one uncertainty realization did not survive the requested duration")
        if constraints.minimum_terrain_clearance_m is not None:
            if uncertainty.minimum_terrain_clearance_m is None:
                violations.append("uncertainty terrain clearance unavailable")
            elif uncertainty.minimum_terrain_clearance_m < constraints.minimum_terrain_clearance_m:
                violations.append("uncertainty minimum terrain clearance below constraint")
        if (
            constraints.maximum_periselene_spread_m is not None
            and uncertainty.periselene_altitude_peak_to_peak_m > constraints.maximum_periselene_spread_m
        ):
            violations.append("uncertainty periselene-altitude spread above constraint")
        if (
            constraints.maximum_aposelene_spread_m is not None
            and uncertainty.aposelene_altitude_peak_to_peak_m > constraints.maximum_aposelene_spread_m
        ):
            violations.append("uncertainty aposelene-altitude spread above constraint")
        if (
            constraints.maximum_eccentricity_variation is not None
            and uncertainty.eccentricity_peak_to_peak > constraints.maximum_eccentricity_variation
        ):
            violations.append("uncertainty eccentricity variation above constraint")
        if (
            constraints.maximum_eccentricity_vector_drift is not None
            and uncertainty.eccentricity_vector_linear_drift_norm > constraints.maximum_eccentricity_vector_drift
        ):
            violations.append("uncertainty eccentricity-vector drift above constraint")
        if (
            constraints.maximum_orbital_plane_change_rad is not None
            and uncertainty.orbital_plane_max_change_rad > constraints.maximum_orbital_plane_change_rad
        ):
            violations.append("uncertainty orbital-plane change above constraint")
    return tuple(dict.fromkeys(violations))


def _evaluate_candidate(
    candidate_id: str,
    point: OrbitSearchPoint,
    dynamics: SearchDynamics,
    settings: StabilitySearchSettings,
    *,
    terrain: TerrainShapeModel | None,
    terrain_body_fixed_from_inertial: RotationProvider | None,
    terrain_frame: str | None,
    uncertainty_dynamics: Sequence[SearchDynamics],
) -> StabilityCandidate:
    state = point.initial_state(dynamics.mu_m3_s2)
    nominal = _propagate_candidate(
        state,
        dynamics,
        settings,
        terrain=terrain,
        terrain_body_fixed_from_inertial=terrain_body_fixed_from_inertial,
        terrain_frame=terrain_frame,
    )
    realization_metrics = tuple(
        _propagate_candidate(
            state,
            realization,
            settings,
            terrain=terrain,
            terrain_body_fixed_from_inertial=terrain_body_fixed_from_inertial,
            terrain_frame=terrain_frame,
        )
        for realization in uncertainty_dynamics
    )
    uncertainty = (
        None
        if not realization_metrics
        else _uncertainty_summary(realization_metrics, settings.uncertainty_adverse_percentile)
    )
    violations = _constraint_violations(nominal, uncertainty, settings.constraints)
    ranking = _ranking_breakdown(nominal, uncertainty, settings.ranking)
    return StabilityCandidate(
        candidate_id=candidate_id,
        parameters=point,
        initial_state=tuple(float(value) for value in state),  # type: ignore[arg-type]
        metrics=nominal,
        uncertainty_metrics=realization_metrics,
        uncertainty_summary=uncertainty,
        passed_constraints=not violations,
        constraint_violations=violations,
        ranking=ranking,
    )


def _rank_candidates(candidates: Sequence[StabilityCandidate]) -> tuple[StabilityCandidate, ...]:
    ordered = sorted(
        candidates,
        key=lambda candidate: (
            not candidate.passed_constraints,
            candidate.ranking.penalty,
            candidate.candidate_id,
        ),
    )
    return tuple(replace(candidate, rank=index + 1) for index, candidate in enumerate(ordered))


def _run_points_search(
    points: Sequence[OrbitSearchPoint],
    dynamics: SearchDynamics,
    settings: StabilitySearchSettings,
    *,
    raw_grid_size: int,
    search_definition: Mapping[str, object],
    candidate_prefix: str,
    terrain: TerrainShapeModel | None,
    terrain_body_fixed_from_inertial: RotationProvider | None,
    terrain_frame: str | None,
    uncertainty_dynamics: Sequence[SearchDynamics],
) -> StabilitySearchResult:
    if (terrain is None) != (terrain_body_fixed_from_inertial is None) or (terrain is None) != (terrain_frame is None):
        raise ValueError(
            "terrain, terrain_body_fixed_from_inertial, and terrain_frame must be supplied together"
        )
    uncertainty_models = tuple(uncertainty_dynamics)
    unique = _unique_points(tuple(points), dynamics.mu_m3_s2)
    if not unique:
        raise ValueError("search contains no physical candidate states")
    if settings.workers > 1:
        models = (dynamics,) + uncertainty_models
        if any(not model.parallel_safe for model in models):
            raise ValueError(
                "parallel search requested with dynamics marked parallel_safe=False; use workers=1 or an explicitly parallel-safe model"
            )

    tasks = tuple(
        (f"{candidate_prefix}{index + 1:06d}", point)
        for index, point in enumerate(unique)
    )

    def evaluate(task: tuple[str, OrbitSearchPoint]) -> StabilityCandidate:
        candidate_id, point = task
        return _evaluate_candidate(
            candidate_id,
            point,
            dynamics,
            settings,
            terrain=terrain,
            terrain_body_fixed_from_inertial=terrain_body_fixed_from_inertial,
            terrain_frame=terrain_frame,
            uncertainty_dynamics=uncertainty_models,
        )

    if settings.workers == 1:
        evaluated = tuple(evaluate(task) for task in tasks)
    else:
        with ThreadPoolExecutor(max_workers=settings.workers) as executor:
            evaluated = tuple(executor.map(evaluate, tasks))

    return StabilitySearchResult(
        candidates=_rank_candidates(evaluated),
        raw_grid_size=int(raw_grid_size),
        unique_candidate_count=len(unique),
        dynamics_provenance=dynamics.provenance(),
        uncertainty_dynamics_provenance=tuple(model.provenance() for model in uncertainty_models),
        search_definition=dict(search_definition),
        settings=settings,
    )


def run_stability_search(
    search_space: StabilitySearchSpace,
    dynamics: SearchDynamics,
    *,
    settings: StabilitySearchSettings = StabilitySearchSettings(),
    terrain: TerrainShapeModel | None = None,
    terrain_body_fixed_from_inertial: RotationProvider | None = None,
    terrain_frame: str | None = None,
    uncertainty_dynamics: Sequence[SearchDynamics] = (),
) -> StabilitySearchResult:
    """Evaluate and rank a deterministic low-lunar-orbit parameter grid."""
    points = search_space.points(dynamics.analysis_reference_radius_m)
    return _run_points_search(
        points,
        dynamics,
        settings,
        raw_grid_size=search_space.raw_grid_size,
        search_definition={"mode": "grid", "space": search_space.as_dict()},
        candidate_prefix="C",
        terrain=terrain,
        terrain_body_fixed_from_inertial=terrain_body_fixed_from_inertial,
        terrain_frame=terrain_frame,
        uncertainty_dynamics=uncertainty_dynamics,
    )


def _axis_step(values: Sequence[float]) -> float | None:
    unique = np.unique(np.asarray(values, dtype=float))
    if unique.size < 2:
        return None
    differences = np.diff(np.sort(unique))
    positive = differences[differences > 0.0]
    return None if not positive.size else float(np.min(positive))


def _refined_values(
    center: float,
    *,
    step: float | None,
    count: int,
    fraction: float,
    lower: float | None,
    upper: float | None,
    periodic: bool,
) -> tuple[float, ...]:
    if step is None or count == 1:
        return (float(center),)
    offsets = np.linspace(-1.0, 1.0, count) * step * fraction
    values = center + offsets
    if periodic:
        values = np.mod(values, _TWO_PI)
    else:
        if lower is not None:
            values = np.maximum(values, lower)
        if upper is not None:
            values = np.minimum(values, upper)
    return tuple(float(value) for value in np.unique(np.round(values, 15)))


def _refinement_points(
    search_space: StabilitySearchSpace,
    seeds: Sequence[StabilityCandidate],
    refinement: RefinementSettings,
    reference_radius_m: float,
) -> tuple[OrbitSearchPoint, ...]:
    original_points = search_space.points(reference_radius_m)
    axis_values: dict[str, tuple[float, ...]] = {
        "semi_major_axis_m": tuple(point.semi_major_axis_m for point in original_points),
        "eccentricity": tuple(point.eccentricity for point in original_points),
        "inclination_rad": tuple(point.inclination_rad for point in original_points),
        "raan_rad": tuple(point.raan_rad for point in original_points),
        "periapsis_parameter_rad": tuple(point.periapsis_parameter_rad for point in original_points),
        "initial_anomaly_rad": tuple(point.initial_anomaly_rad for point in original_points),
    }
    bounds: dict[str, tuple[float | None, float | None, bool]] = {
        "semi_major_axis_m": (
            min(axis_values["semi_major_axis_m"]),
            max(axis_values["semi_major_axis_m"]),
            False,
        ),
        "eccentricity": (0.0, min(0.999999999999, max(axis_values["eccentricity"])), False),
        "inclination_rad": (0.0, min(np.pi - 1.0e-12, max(axis_values["inclination_rad"])), False),
        "raan_rad": (None, None, True),
        "periapsis_parameter_rad": (None, None, True),
        "initial_anomaly_rad": (None, None, True),
    }
    result: list[OrbitSearchPoint] = []
    for seed in seeds:
        center = seed.parameters
        local: dict[str, tuple[float, ...]] = {}
        for axis in axis_values:
            current = float(getattr(center, axis))
            if axis not in refinement.refine_axes:
                local[axis] = (current,)
                continue
            lower, upper, periodic = bounds[axis]
            local[axis] = _refined_values(
                current,
                step=_axis_step(axis_values[axis]),
                count=refinement.points_per_axis,
                fraction=refinement.spacing_fraction,
                lower=lower,
                upper=upper,
                periodic=periodic,
            )
        for axis, eccentricity, inclination, raan, periapsis, anomaly in product(
            local["semi_major_axis_m"],
            local["eccentricity"],
            local["inclination_rad"],
            local["raan_rad"],
            local["periapsis_parameter_rad"],
            local["initial_anomaly_rad"],
        ):
            result.append(
                OrbitSearchPoint(
                    semi_major_axis_m=float(axis),
                    semi_major_axis_altitude_m=float(axis - reference_radius_m),
                    eccentricity=float(eccentricity),
                    inclination_rad=float(inclination),
                    raan_rad=_wrap(raan),
                    periapsis_parameter_rad=_wrap(periapsis),
                    initial_anomaly_rad=_wrap(anomaly),
                    periapsis_parameterization=search_space.periapsis_parameterization,
                )
            )
    return tuple(result)


def run_coarse_to_fine_search(
    search_space: StabilitySearchSpace,
    dynamics: SearchDynamics,
    *,
    settings: StabilitySearchSettings = StabilitySearchSettings(),
    refinement: RefinementSettings = RefinementSettings(),
    terrain: TerrainShapeModel | None = None,
    terrain_body_fixed_from_inertial: RotationProvider | None = None,
    terrain_frame: str | None = None,
    uncertainty_dynamics: Sequence[SearchDynamics] = (),
) -> CoarseToFineSearchResult:
    """Run a deterministic coarse grid, then locally refine its best candidates."""
    coarse = run_stability_search(
        search_space,
        dynamics,
        settings=settings,
        terrain=terrain,
        terrain_body_fixed_from_inertial=terrain_body_fixed_from_inertial,
        terrain_frame=terrain_frame,
        uncertainty_dynamics=uncertainty_dynamics,
    )
    feasible = coarse.best_candidates(refinement.top_candidates, feasible_only=True)
    seeds = feasible or coarse.best_candidates(refinement.top_candidates, feasible_only=False)
    points = _refinement_points(
        search_space,
        seeds,
        refinement,
        dynamics.analysis_reference_radius_m,
    )
    refined = _run_points_search(
        points,
        dynamics,
        settings,
        raw_grid_size=len(points),
        search_definition={
            "mode": "refinement",
            "seed_candidate_ids": [candidate.candidate_id for candidate in seeds],
            "refinement": _jsonable(asdict(refinement)),
        },
        candidate_prefix="R",
        terrain=terrain,
        terrain_body_fixed_from_inertial=terrain_body_fixed_from_inertial,
        terrain_frame=terrain_frame,
        uncertainty_dynamics=uncertainty_dynamics,
    )
    return CoarseToFineSearchResult(
        coarse=coarse,
        refined=refined,
        seed_candidate_ids=tuple(candidate.candidate_id for candidate in seeds),
    )


def _parameter_value(candidate: StabilityCandidate, name: str) -> float:
    parameters = candidate.parameters.as_dict()
    if name not in parameters or not isinstance(parameters[name], (int, float)):
        raise ValueError(f"unknown numeric candidate parameter '{name}'")
    return float(parameters[name])


def _metric_value(candidate: StabilityCandidate, name: str) -> float:
    if name == "ranking_penalty":
        return float(candidate.ranking.penalty)
    if hasattr(candidate.metrics, name):
        value = getattr(candidate.metrics, name)
        if value is None:
            return float("nan")
        return float(value)
    if name.startswith("uncertainty_") and candidate.uncertainty_summary is not None:
        field_name = name.removeprefix("uncertainty_")
        if hasattr(candidate.uncertainty_summary, field_name):
            value = getattr(candidate.uncertainty_summary, field_name)
            if value is None:
                return float("nan")
            return float(value)
    raise ValueError(f"unknown stability-map metric '{name}'")


def make_stability_map(
    result: StabilitySearchResult,
    x_parameter: str,
    y_parameter: str,
    *,
    metric: str = "ranking_penalty",
    fixed_parameters: Mapping[str, float] | None = None,
) -> StabilityMap:
    """Collapse a search to a 2-D slice suitable for plotting.

    When several candidates occupy the same x/y cell because other parameters
    remain free, the lowest-ranked candidate in that cell is selected.  The map
    stores that candidate id so the reduction is auditable.
    """
    fixed = dict(fixed_parameters or {})
    selected: list[StabilityCandidate] = []
    for candidate in result.candidates:
        keep = True
        for name, target in fixed.items():
            value = _parameter_value(candidate, name)
            if not np.isclose(value, float(target), rtol=0.0, atol=1.0e-12 * max(1.0, abs(float(target)))):
                keep = False
                break
        if keep:
            selected.append(candidate)
    if not selected:
        raise ValueError("stability-map filter selected no candidates")
    x_values = tuple(sorted({_parameter_value(candidate, x_parameter) for candidate in selected}))
    y_values = tuple(sorted({_parameter_value(candidate, y_parameter) for candidate in selected}))
    value_rows: list[tuple[float | None, ...]] = []
    id_rows: list[tuple[str | None, ...]] = []
    for y_value in y_values:
        values: list[float | None] = []
        ids: list[str | None] = []
        for x_value in x_values:
            cell = [
                candidate
                for candidate in selected
                if np.isclose(_parameter_value(candidate, x_parameter), x_value, rtol=0.0, atol=1.0e-12 * max(1.0, abs(x_value)))
                and np.isclose(_parameter_value(candidate, y_parameter), y_value, rtol=0.0, atol=1.0e-12 * max(1.0, abs(y_value)))
            ]
            if not cell:
                values.append(None)
                ids.append(None)
                continue
            best = min(cell, key=lambda candidate: (candidate.rank or 10**12, candidate.candidate_id))
            metric_value = _metric_value(best, metric)
            values.append(None if not np.isfinite(metric_value) else float(metric_value))
            ids.append(best.candidate_id)
        value_rows.append(tuple(values))
        id_rows.append(tuple(ids))
    return StabilityMap(
        x_parameter=x_parameter,
        y_parameter=y_parameter,
        metric=metric,
        x_values=x_values,
        y_values=y_values,
        values=tuple(value_rows),
        candidate_ids=tuple(id_rows),
        fixed_parameters=fixed,
    )
