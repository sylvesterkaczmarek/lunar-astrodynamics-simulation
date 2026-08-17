"""Gravity truncation convergence and force-model fidelity analysis.

The routines in this module compare explicit model choices against an explicit
reference.  They do not infer a universal harmonic degree from altitude.  A
truncation decision is therefore tied to the supplied positions, trajectory,
epoch/frame rotation, propagation horizon, terrain model and tolerance policy.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Mapping, Sequence
import csv
import json

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .analysis import orbit_history
from .constants import MOON_MEAN_RADIUS_M
from .forces import ForceComponent
from .frames import RotationProvider
from .harmonics import SphericalHarmonicModel, gravity_acceleration_inertial
from .propagation import PropagationSettings, propagate_with_acceleration
from .stability import SearchDynamics, harmonic_search_dynamics
from .terrain import TerrainShapeModel, propagate_with_terrain

FloatArray = NDArray[np.float64]


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


def _vectors3(values: ArrayLike, *, name: str) -> FloatArray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 1 and array.shape == (3,):
        array = array.reshape(1, 3)
    if array.ndim != 2 or array.shape[1] != 3 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite (N,3) array or three-vector")
    return array


def _times(values: ArrayLike | None, count: int) -> FloatArray:
    if values is None:
        return np.zeros(count, dtype=float)
    array = np.asarray(values, dtype=float)
    if array.ndim == 0:
        array = np.full(count, float(array), dtype=float)
    if array.shape != (count,) or not np.all(np.isfinite(array)):
        raise ValueError("times_s must be a finite scalar or length-N vector")
    return array


@dataclass(frozen=True, order=True)
class HarmonicTruncation:
    degree: int
    order: int

    def __post_init__(self) -> None:
        if self.degree < 0 or self.order < 0 or self.order > self.degree:
            raise ValueError("harmonic truncation must satisfy 0 <= order <= degree")

    @property
    def label(self) -> str:
        return f"{self.degree}x{self.order}"

    def validate_for(self, model: SphericalHarmonicModel) -> None:
        if self.degree > model.max_degree:
            raise ValueError(
                f"truncation {self.label} exceeds model maximum degree {model.max_degree}"
            )

    def as_dict(self) -> dict[str, int | str]:
        return {"degree": self.degree, "order": self.order, "label": self.label}


def default_harmonic_truncations(
    model: SphericalHarmonicModel,
    *,
    degrees: Sequence[int] = (2, 10, 20, 40, 60, 120, 300),
    include_model_maximum: bool = True,
) -> tuple[HarmonicTruncation, ...]:
    """Return the standard convergence ladder clipped to the available model."""
    values: list[HarmonicTruncation] = []
    seen: set[tuple[int, int]] = set()
    for raw_degree in degrees:
        degree = int(raw_degree)
        if degree < 0:
            raise ValueError("default convergence degrees must be non-negative")
        if degree <= model.max_degree:
            item = HarmonicTruncation(degree, degree)
            if (item.degree, item.order) not in seen:
                values.append(item)
                seen.add((item.degree, item.order))
    if include_model_maximum:
        reference_order = min(model.max_degree, max(model.max_order, 0))
        item = HarmonicTruncation(model.max_degree, reference_order)
        if (item.degree, item.order) not in seen:
            values.append(item)
    return tuple(values)


def _validated_truncations(
    truncations: Sequence[HarmonicTruncation | tuple[int, int]],
    model: SphericalHarmonicModel,
    reference: HarmonicTruncation,
) -> tuple[HarmonicTruncation, ...]:
    items: list[HarmonicTruncation] = []
    seen: set[tuple[int, int]] = set()
    for value in truncations:
        item = value if isinstance(value, HarmonicTruncation) else HarmonicTruncation(*value)
        item.validate_for(model)
        key = (item.degree, item.order)
        if key not in seen:
            items.append(item)
            seen.add(key)
    reference.validate_for(model)
    key = (reference.degree, reference.order)
    if key not in seen:
        items.append(reference)
    if not items:
        raise ValueError("at least one harmonic truncation is required")
    return tuple(items)


def _default_reference(model: SphericalHarmonicModel) -> HarmonicTruncation:
    return HarmonicTruncation(model.max_degree, min(model.max_degree, max(model.max_order, 0)))


def _rtn_components(
    vector: FloatArray,
    position: FloatArray,
    velocity: FloatArray | None,
) -> tuple[float, float | None, float | None]:
    radius = float(np.linalg.norm(position))
    if radius == 0.0:
        raise ValueError("RTN decomposition is undefined at the origin")
    radial = position / radius
    radial_component = float(np.dot(vector, radial))
    if velocity is None:
        return radial_component, None, None
    angular_momentum = np.cross(position, velocity)
    h_norm = float(np.linalg.norm(angular_momentum))
    if h_norm <= np.finfo(float).eps * max(radius * float(np.linalg.norm(velocity)), 1.0):
        return radial_component, None, None
    normal = angular_momentum / h_norm
    along = np.cross(normal, radial)
    return (
        radial_component,
        float(np.dot(vector, along)),
        float(np.dot(vector, normal)),
    )


@dataclass(frozen=True)
class AccelerationSampleError:
    sample_index: int
    time_s: float
    position_m: tuple[float, float, float]
    reference_acceleration_m_s2: tuple[float, float, float]
    acceleration_error_m_s2: tuple[float, float, float]
    absolute_error_m_s2: float
    relative_error: float
    radial_error_m_s2: float
    along_track_error_m_s2: float | None
    cross_track_error_m_s2: float | None

    def as_dict(self) -> dict[str, object]:
        return _jsonable(asdict(self))  # type: ignore[return-value]


@dataclass(frozen=True)
class AccelerationFidelityEntry:
    truncation: HarmonicTruncation
    samples: tuple[AccelerationSampleError, ...]
    maximum_absolute_error_m_s2: float
    rms_absolute_error_m_s2: float
    maximum_relative_error: float
    rms_relative_error: float
    runtime_seconds_median: float
    runtime_seconds_per_evaluation: float

    def as_dict(self) -> dict[str, object]:
        return {
            "truncation": self.truncation.as_dict(),
            "maximum_absolute_error_m_s2": self.maximum_absolute_error_m_s2,
            "rms_absolute_error_m_s2": self.rms_absolute_error_m_s2,
            "maximum_relative_error": self.maximum_relative_error,
            "rms_relative_error": self.rms_relative_error,
            "runtime_seconds_median": self.runtime_seconds_median,
            "runtime_seconds_per_evaluation": self.runtime_seconds_per_evaluation,
            "samples": [sample.as_dict() for sample in self.samples],
        }


@dataclass(frozen=True)
class AccelerationFidelityReport:
    model_name: str
    model_frame: str
    reference: HarmonicTruncation
    sample_count: int
    benchmark_repetitions: int
    entries: tuple[AccelerationFidelityEntry, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "model_name": self.model_name,
            "model_frame": self.model_frame,
            "reference": self.reference.as_dict(),
            "sample_count": self.sample_count,
            "benchmark_repetitions": self.benchmark_repetitions,
            "entries": [entry.as_dict() for entry in self.entries],
        }

    def write_json(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

    def write_csv(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        fields = [
            "degree",
            "order",
            "maximum_absolute_error_m_s2",
            "rms_absolute_error_m_s2",
            "maximum_relative_error",
            "rms_relative_error",
            "runtime_seconds_median",
            "runtime_seconds_per_evaluation",
        ]
        with destination.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for entry in self.entries:
                writer.writerow(
                    {
                        "degree": entry.truncation.degree,
                        "order": entry.truncation.order,
                        "maximum_absolute_error_m_s2": entry.maximum_absolute_error_m_s2,
                        "rms_absolute_error_m_s2": entry.rms_absolute_error_m_s2,
                        "maximum_relative_error": entry.maximum_relative_error,
                        "rms_relative_error": entry.rms_relative_error,
                        "runtime_seconds_median": entry.runtime_seconds_median,
                        "runtime_seconds_per_evaluation": entry.runtime_seconds_per_evaluation,
                    }
                )


def _accelerations(
    model: SphericalHarmonicModel,
    rotation: RotationProvider,
    truncation: HarmonicTruncation,
    times_s: FloatArray,
    positions_m: FloatArray,
) -> FloatArray:
    return np.vstack(
        [
            gravity_acceleration_inertial(
                float(time_s),
                position,
                model,
                rotation,
                max_degree=truncation.degree,
                max_order=truncation.order,
            )
            for time_s, position in zip(times_s, positions_m, strict=True)
        ]
    )


def _benchmark_accelerations(
    model: SphericalHarmonicModel,
    rotation: RotationProvider,
    truncation: HarmonicTruncation,
    times_s: FloatArray,
    positions_m: FloatArray,
    repetitions: int,
) -> tuple[float, float]:
    timings: list[float] = []
    for _ in range(repetitions):
        start = perf_counter()
        _accelerations(model, rotation, truncation, times_s, positions_m)
        timings.append(perf_counter() - start)
    median = float(np.median(timings))
    return median, median / positions_m.shape[0]


def compare_harmonic_accelerations(
    model: SphericalHarmonicModel,
    body_fixed_from_inertial: RotationProvider,
    positions_m: ArrayLike,
    *,
    times_s: ArrayLike | None = None,
    velocities_m_s: ArrayLike | None = None,
    truncations: Sequence[HarmonicTruncation | tuple[int, int]] | None = None,
    reference: HarmonicTruncation | tuple[int, int] | None = None,
    benchmark_repetitions: int = 3,
    relative_floor_m_s2: float = 1.0e-30,
) -> AccelerationFidelityReport:
    """Compare harmonic accelerations against an explicit high-degree reference.

    Radial error is always reported. Along-track and cross-track components are
    reported only when a non-degenerate velocity is supplied for that sample.
    """
    positions = _vectors3(positions_m, name="positions_m")
    times = _times(times_s, positions.shape[0])
    velocities = None if velocities_m_s is None else _vectors3(velocities_m_s, name="velocities_m_s")
    if velocities is not None and velocities.shape[0] != positions.shape[0]:
        raise ValueError("velocities_m_s must have the same sample count as positions_m")
    if benchmark_repetitions < 1:
        raise ValueError("benchmark_repetitions must be at least one")
    if not np.isfinite(relative_floor_m_s2) or relative_floor_m_s2 <= 0.0:
        raise ValueError("relative_floor_m_s2 must be finite and positive")

    ref = (
        _default_reference(model)
        if reference is None
        else reference
        if isinstance(reference, HarmonicTruncation)
        else HarmonicTruncation(*reference)
    )
    candidates = _validated_truncations(
        default_harmonic_truncations(model) if truncations is None else truncations,
        model,
        ref,
    )
    reference_acceleration = _accelerations(model, body_fixed_from_inertial, ref, times, positions)
    reference_runtime = _benchmark_accelerations(
        model, body_fixed_from_inertial, ref, times, positions, benchmark_repetitions
    )

    entries: list[AccelerationFidelityEntry] = []
    for truncation in candidates:
        if truncation == ref:
            acceleration = reference_acceleration
            runtime_total, runtime_per = reference_runtime
        else:
            acceleration = _accelerations(
                model, body_fixed_from_inertial, truncation, times, positions
            )
            runtime_total, runtime_per = _benchmark_accelerations(
                model,
                body_fixed_from_inertial,
                truncation,
                times,
                positions,
                benchmark_repetitions,
            )
        error = acceleration - reference_acceleration
        absolute = np.linalg.norm(error, axis=1)
        reference_norm = np.linalg.norm(reference_acceleration, axis=1)
        relative = absolute / np.maximum(reference_norm, relative_floor_m_s2)
        samples: list[AccelerationSampleError] = []
        for index in range(positions.shape[0]):
            velocity = None if velocities is None else velocities[index]
            radial, along, cross = _rtn_components(error[index], positions[index], velocity)
            samples.append(
                AccelerationSampleError(
                    sample_index=index,
                    time_s=float(times[index]),
                    position_m=tuple(float(value) for value in positions[index]),
                    reference_acceleration_m_s2=tuple(
                        float(value) for value in reference_acceleration[index]
                    ),
                    acceleration_error_m_s2=tuple(float(value) for value in error[index]),
                    absolute_error_m_s2=float(absolute[index]),
                    relative_error=float(relative[index]),
                    radial_error_m_s2=radial,
                    along_track_error_m_s2=along,
                    cross_track_error_m_s2=cross,
                )
            )
        entries.append(
            AccelerationFidelityEntry(
                truncation=truncation,
                samples=tuple(samples),
                maximum_absolute_error_m_s2=float(np.max(absolute)),
                rms_absolute_error_m_s2=float(np.sqrt(np.mean(absolute**2))),
                maximum_relative_error=float(np.max(relative)),
                rms_relative_error=float(np.sqrt(np.mean(relative**2))),
                runtime_seconds_median=runtime_total,
                runtime_seconds_per_evaluation=runtime_per,
            )
        )
    entries.sort(key=lambda entry: (entry.truncation.degree, entry.truncation.order))
    return AccelerationFidelityReport(
        model_name=model.name,
        model_frame=model.frame,
        reference=ref,
        sample_count=positions.shape[0],
        benchmark_repetitions=benchmark_repetitions,
        entries=tuple(entries),
    )


@dataclass(frozen=True)
class TrajectoryOutcome:
    achieved_duration_s: float
    impacted: bool
    impact_time_s: float | None
    final_position_m: tuple[float, float, float]
    final_velocity_m_s: tuple[float, float, float]
    periselene_altitude_peak_to_peak_m: float
    eccentricity_peak_to_peak: float
    minimum_reference_altitude_m: float
    minimum_terrain_clearance_m: float | None

    def as_dict(self) -> dict[str, object]:
        return _jsonable(asdict(self))  # type: ignore[return-value]


@dataclass(frozen=True)
class TrajectoryFidelityEntry:
    truncation: HarmonicTruncation
    outcome: TrajectoryOutcome
    runtime_seconds: float
    final_position_difference_m: float | None
    final_velocity_difference_m_s: float | None
    maximum_common_sample_position_difference_m: float
    maximum_common_sample_velocity_difference_m_s: float
    periselene_variation_difference_m: float
    eccentricity_variation_difference: float
    minimum_terrain_clearance_difference_m: float | None
    impact_matches_reference: bool
    lifetime_difference_s: float

    def as_dict(self) -> dict[str, object]:
        return {
            "truncation": self.truncation.as_dict(),
            "outcome": self.outcome.as_dict(),
            "runtime_seconds": self.runtime_seconds,
            "final_position_difference_m": self.final_position_difference_m,
            "final_velocity_difference_m_s": self.final_velocity_difference_m_s,
            "maximum_common_sample_position_difference_m": self.maximum_common_sample_position_difference_m,
            "maximum_common_sample_velocity_difference_m_s": self.maximum_common_sample_velocity_difference_m_s,
            "periselene_variation_difference_m": self.periselene_variation_difference_m,
            "eccentricity_variation_difference": self.eccentricity_variation_difference,
            "minimum_terrain_clearance_difference_m": self.minimum_terrain_clearance_difference_m,
            "impact_matches_reference": self.impact_matches_reference,
            "lifetime_difference_s": self.lifetime_difference_s,
        }


@dataclass(frozen=True)
class TrajectoryFidelityReport:
    model_name: str
    model_frame: str
    reference: HarmonicTruncation
    duration_s: float
    sample_count: int
    reference_outcome: TrajectoryOutcome
    reference_runtime_seconds: float
    entries: tuple[TrajectoryFidelityEntry, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "model_name": self.model_name,
            "model_frame": self.model_frame,
            "reference": self.reference.as_dict(),
            "duration_s": self.duration_s,
            "sample_count": self.sample_count,
            "reference_outcome": self.reference_outcome.as_dict(),
            "reference_runtime_seconds": self.reference_runtime_seconds,
            "entries": [entry.as_dict() for entry in self.entries],
        }

    def write_json(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

    def write_csv(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        fields = [
            "degree",
            "order",
            "runtime_seconds",
            "final_position_difference_m",
            "final_velocity_difference_m_s",
            "maximum_common_sample_position_difference_m",
            "maximum_common_sample_velocity_difference_m_s",
            "periselene_variation_difference_m",
            "eccentricity_variation_difference",
            "minimum_terrain_clearance_difference_m",
            "impact_matches_reference",
            "lifetime_difference_s",
            "impacted",
            "impact_time_s",
            "minimum_reference_altitude_m",
            "minimum_terrain_clearance_m",
        ]
        with destination.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for entry in self.entries:
                writer.writerow(
                    {
                        "degree": entry.truncation.degree,
                        "order": entry.truncation.order,
                        "runtime_seconds": entry.runtime_seconds,
                        "final_position_difference_m": entry.final_position_difference_m,
                        "final_velocity_difference_m_s": entry.final_velocity_difference_m_s,
                        "maximum_common_sample_position_difference_m": entry.maximum_common_sample_position_difference_m,
                        "maximum_common_sample_velocity_difference_m_s": entry.maximum_common_sample_velocity_difference_m_s,
                        "periselene_variation_difference_m": entry.periselene_variation_difference_m,
                        "eccentricity_variation_difference": entry.eccentricity_variation_difference,
                        "minimum_terrain_clearance_difference_m": entry.minimum_terrain_clearance_difference_m,
                        "impact_matches_reference": entry.impact_matches_reference,
                        "lifetime_difference_s": entry.lifetime_difference_s,
                        "impacted": entry.outcome.impacted,
                        "impact_time_s": entry.outcome.impact_time_s,
                        "minimum_reference_altitude_m": entry.outcome.minimum_reference_altitude_m,
                        "minimum_terrain_clearance_m": entry.outcome.minimum_terrain_clearance_m,
                    }
                )


def _trajectory_arrays(solution: object) -> tuple[FloatArray, FloatArray, bool, float | None]:
    time = np.asarray(solution.t, dtype=float)  # type: ignore[attr-defined]
    states = np.asarray(solution.y, dtype=float)  # type: ignore[attr-defined]
    event_times = solution.t_events[0]  # type: ignore[attr-defined]
    impacted = bool(len(event_times))
    impact_time = float(event_times[0]) if impacted else None
    if impacted:
        event_state = np.asarray(solution.y_events[0][0], dtype=float)  # type: ignore[attr-defined]
        if time.size == 0 or abs(float(time[-1]) - float(impact_time)) > 1.0e-9:
            time = np.concatenate((time, np.array([impact_time])))
            states = np.column_stack((states, event_state))
        else:
            states[:, -1] = event_state
    if time.size < 2:
        raise ValueError("trajectory fidelity analysis requires at least two propagated samples")
    return time, states, impacted, impact_time


def _propagate_outcome(
    dynamics: SearchDynamics,
    initial_state: FloatArray,
    duration_s: float,
    sample_times: FloatArray,
    settings: PropagationSettings,
    *,
    terrain: TerrainShapeModel | None,
    terrain_body_fixed_from_inertial: RotationProvider | None,
    terrain_frame: str | None,
    terrain_clearance_search_samples: int,
) -> tuple[TrajectoryOutcome, FloatArray, FloatArray, float]:
    start = perf_counter()
    if terrain is None:
        if terrain_body_fixed_from_inertial is not None or terrain_frame is not None:
            raise ValueError("terrain frame/rotation cannot be supplied without terrain")
        solution = propagate_with_acceleration(
            initial_state,
            duration_s,
            dynamics.acceleration,
            collision_radius_m=dynamics.collision_radius_m,
            sample_times_s=sample_times,
            settings=settings,
        )
        minimum_clearance = None
    else:
        if terrain_body_fixed_from_inertial is None or terrain_frame is None:
            raise ValueError("terrain fidelity analysis requires rotation and explicit terrain_frame")
        terrain_result = propagate_with_terrain(
            initial_state,
            duration_s,
            dynamics.acceleration,
            terrain,
            terrain_body_fixed_from_inertial,
            terrain_frame=terrain_frame,
            sample_times_s=sample_times,
            settings=settings,
            clearance_search_samples=terrain_clearance_search_samples,
        )
        solution = terrain_result.solution
        minimum_clearance = float(terrain_result.clearance.minimum_clearance_m)
    elapsed = perf_counter() - start
    if not solution.success:  # type: ignore[attr-defined]
        raise ValueError(f"trajectory fidelity propagation failed: {solution.message}")  # type: ignore[attr-defined]
    time, states, impacted, impact_time = _trajectory_arrays(solution)
    history = orbit_history(
        time,
        states,
        dynamics.mu_m3_s2,
        reference_radius_m=dynamics.analysis_reference_radius_m,
    )
    achieved = float(impact_time if impacted and impact_time is not None else time[-1])
    outcome = TrajectoryOutcome(
        achieved_duration_s=achieved,
        impacted=impacted,
        impact_time_s=impact_time,
        final_position_m=tuple(float(value) for value in states[:3, -1]),
        final_velocity_m_s=tuple(float(value) for value in states[3:, -1]),
        periselene_altitude_peak_to_peak_m=float(np.ptp(history.periselene_altitude_m)),
        eccentricity_peak_to_peak=float(np.ptp(history.eccentricity)),
        minimum_reference_altitude_m=history.minimum_reference_altitude_m,
        minimum_terrain_clearance_m=minimum_clearance,
    )
    return outcome, time, states, float(elapsed)


def _common_sample_differences(
    candidate_time: FloatArray,
    candidate_states: FloatArray,
    reference_time: FloatArray,
    reference_states: FloatArray,
) -> tuple[float, float]:
    count = min(candidate_time.size, reference_time.size)
    position: list[float] = []
    velocity: list[float] = []
    for index in range(count):
        if abs(float(candidate_time[index]) - float(reference_time[index])) > 1.0e-8:
            break
        position.append(float(np.linalg.norm(candidate_states[:3, index] - reference_states[:3, index])))
        velocity.append(float(np.linalg.norm(candidate_states[3:, index] - reference_states[3:, index])))
    if not position:
        raise ValueError("candidate and reference trajectories have no common sample epoch")
    return max(position), max(velocity)


def _trajectory_entry(
    truncation: HarmonicTruncation,
    outcome: TrajectoryOutcome,
    runtime: float,
    time: FloatArray,
    states: FloatArray,
    reference_outcome: TrajectoryOutcome,
    reference_time: FloatArray,
    reference_states: FloatArray,
) -> TrajectoryFidelityEntry:
    same_finish = abs(outcome.achieved_duration_s - reference_outcome.achieved_duration_s) <= 1.0e-6
    if same_finish:
        final_position_difference = float(
            np.linalg.norm(np.asarray(outcome.final_position_m) - np.asarray(reference_outcome.final_position_m))
        )
        final_velocity_difference = float(
            np.linalg.norm(np.asarray(outcome.final_velocity_m_s) - np.asarray(reference_outcome.final_velocity_m_s))
        )
    else:
        final_position_difference = None
        final_velocity_difference = None
    max_position, max_velocity = _common_sample_differences(
        time, states, reference_time, reference_states
    )
    if (
        outcome.minimum_terrain_clearance_m is not None
        and reference_outcome.minimum_terrain_clearance_m is not None
    ):
        clearance_difference = abs(
            outcome.minimum_terrain_clearance_m - reference_outcome.minimum_terrain_clearance_m
        )
    else:
        clearance_difference = None
    return TrajectoryFidelityEntry(
        truncation=truncation,
        outcome=outcome,
        runtime_seconds=runtime,
        final_position_difference_m=final_position_difference,
        final_velocity_difference_m_s=final_velocity_difference,
        maximum_common_sample_position_difference_m=max_position,
        maximum_common_sample_velocity_difference_m_s=max_velocity,
        periselene_variation_difference_m=abs(
            outcome.periselene_altitude_peak_to_peak_m
            - reference_outcome.periselene_altitude_peak_to_peak_m
        ),
        eccentricity_variation_difference=abs(
            outcome.eccentricity_peak_to_peak - reference_outcome.eccentricity_peak_to_peak
        ),
        minimum_terrain_clearance_difference_m=clearance_difference,
        impact_matches_reference=outcome.impacted == reference_outcome.impacted,
        lifetime_difference_s=abs(
            outcome.achieved_duration_s - reference_outcome.achieved_duration_s
        ),
    )


def compare_harmonic_trajectories(
    model: SphericalHarmonicModel,
    body_fixed_from_inertial: RotationProvider,
    initial_state: ArrayLike,
    duration_s: float,
    *,
    truncations: Sequence[HarmonicTruncation | tuple[int, int]] | None = None,
    reference: HarmonicTruncation | tuple[int, int] | None = None,
    additional_forces: Sequence[ForceComponent] = (),
    analysis_reference_radius_m: float = MOON_MEAN_RADIUS_M,
    collision_radius_m: float = MOON_MEAN_RADIUS_M,
    sample_count: int = 257,
    propagation: PropagationSettings = PropagationSettings(),
    terrain: TerrainShapeModel | None = None,
    terrain_body_fixed_from_inertial: RotationProvider | None = None,
    terrain_frame: str | None = None,
    terrain_clearance_search_samples: int = 513,
) -> TrajectoryFidelityReport:
    """Compare propagated truncations using orbital outcomes and runtime."""
    state0 = np.asarray(initial_state, dtype=float)
    if state0.shape != (6,) or not np.all(np.isfinite(state0)):
        raise ValueError("initial_state must be a finite six-vector")
    if not np.isfinite(duration_s) or duration_s <= 0.0:
        raise ValueError("duration_s must be finite and positive")
    if sample_count < 3:
        raise ValueError("sample_count must be at least three")
    if terrain_clearance_search_samples < 3:
        raise ValueError("terrain_clearance_search_samples must be at least three")
    ref = (
        _default_reference(model)
        if reference is None
        else reference
        if isinstance(reference, HarmonicTruncation)
        else HarmonicTruncation(*reference)
    )
    candidates = _validated_truncations(
        default_harmonic_truncations(model) if truncations is None else truncations,
        model,
        ref,
    )
    sample_times = np.linspace(0.0, float(duration_s), int(sample_count))

    def dynamics_for(truncation: HarmonicTruncation) -> SearchDynamics:
        return harmonic_search_dynamics(
            model,
            body_fixed_from_inertial,
            max_degree=truncation.degree,
            max_order=truncation.order,
            additional_forces=additional_forces,
            analysis_reference_radius_m=analysis_reference_radius_m,
            collision_radius_m=collision_radius_m,
            parallel_safe=False,
            name=f"{model.name} {truncation.label}",
        )

    reference_dynamics = dynamics_for(ref)
    reference_outcome, reference_time, reference_states, reference_runtime = _propagate_outcome(
        reference_dynamics,
        state0,
        duration_s,
        sample_times,
        propagation,
        terrain=terrain,
        terrain_body_fixed_from_inertial=terrain_body_fixed_from_inertial,
        terrain_frame=terrain_frame,
        terrain_clearance_search_samples=terrain_clearance_search_samples,
    )
    entries: list[TrajectoryFidelityEntry] = []
    for truncation in candidates:
        if truncation == ref:
            outcome, time, states, runtime = (
                reference_outcome,
                reference_time,
                reference_states,
                reference_runtime,
            )
        else:
            outcome, time, states, runtime = _propagate_outcome(
                dynamics_for(truncation),
                state0,
                duration_s,
                sample_times,
                propagation,
                terrain=terrain,
                terrain_body_fixed_from_inertial=terrain_body_fixed_from_inertial,
                terrain_frame=terrain_frame,
                terrain_clearance_search_samples=terrain_clearance_search_samples,
            )
        entries.append(
            _trajectory_entry(
                truncation,
                outcome,
                runtime,
                time,
                states,
                reference_outcome,
                reference_time,
                reference_states,
            )
        )
    entries.sort(key=lambda entry: (entry.truncation.degree, entry.truncation.order))
    return TrajectoryFidelityReport(
        model_name=model.name,
        model_frame=model.frame,
        reference=ref,
        duration_s=float(duration_s),
        sample_count=int(sample_count),
        reference_outcome=reference_outcome,
        reference_runtime_seconds=reference_runtime,
        entries=tuple(entries),
    )


@dataclass(frozen=True)
class FidelityTolerance:
    maximum_absolute_acceleration_error_m_s2: float | None = None
    maximum_relative_acceleration_error: float | None = None
    maximum_final_position_difference_m: float | None = None
    maximum_final_velocity_difference_m_s: float | None = None
    maximum_periselene_variation_difference_m: float | None = None
    maximum_eccentricity_variation_difference: float | None = None
    maximum_minimum_terrain_clearance_difference_m: float | None = None
    maximum_lifetime_difference_s: float | None = None
    require_impact_match: bool = True

    def __post_init__(self) -> None:
        values = (
            self.maximum_absolute_acceleration_error_m_s2,
            self.maximum_relative_acceleration_error,
            self.maximum_final_position_difference_m,
            self.maximum_final_velocity_difference_m_s,
            self.maximum_periselene_variation_difference_m,
            self.maximum_eccentricity_variation_difference,
            self.maximum_minimum_terrain_clearance_difference_m,
            self.maximum_lifetime_difference_s,
        )
        if not any(value is not None for value in values):
            raise ValueError("at least one fidelity tolerance must be supplied")
        if any(value is not None and (not np.isfinite(value) or value < 0.0) for value in values):
            raise ValueError("fidelity tolerances must be finite and non-negative")


@dataclass(frozen=True)
class FidelitySelectionResult:
    selected_truncation: HarmonicTruncation | None
    satisfied: bool
    reason: str
    evaluated_truncations: tuple[HarmonicTruncation, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "selected_truncation": (
                None if self.selected_truncation is None else self.selected_truncation.as_dict()
            ),
            "satisfied": self.satisfied,
            "reason": self.reason,
            "evaluated_truncations": [item.as_dict() for item in self.evaluated_truncations],
        }


def _acceleration_passes(entry: AccelerationFidelityEntry, tolerance: FidelityTolerance) -> bool:
    if (
        tolerance.maximum_absolute_acceleration_error_m_s2 is not None
        and entry.maximum_absolute_error_m_s2
        > tolerance.maximum_absolute_acceleration_error_m_s2
    ):
        return False
    if (
        tolerance.maximum_relative_acceleration_error is not None
        and entry.maximum_relative_error > tolerance.maximum_relative_acceleration_error
    ):
        return False
    return True


def _trajectory_passes(entry: TrajectoryFidelityEntry, tolerance: FidelityTolerance) -> bool:
    checks: tuple[tuple[float | None, float | None], ...] = (
        (entry.final_position_difference_m, tolerance.maximum_final_position_difference_m),
        (entry.final_velocity_difference_m_s, tolerance.maximum_final_velocity_difference_m_s),
        (
            entry.periselene_variation_difference_m,
            tolerance.maximum_periselene_variation_difference_m,
        ),
        (
            entry.eccentricity_variation_difference,
            tolerance.maximum_eccentricity_variation_difference,
        ),
        (
            entry.minimum_terrain_clearance_difference_m,
            tolerance.maximum_minimum_terrain_clearance_difference_m,
        ),
        (entry.lifetime_difference_s, tolerance.maximum_lifetime_difference_s),
    )
    for value, limit in checks:
        if limit is not None and (value is None or value > limit):
            return False
    if tolerance.require_impact_match and not entry.impact_matches_reference:
        return False
    return True


def select_lowest_harmonic_truncation(
    report: AccelerationFidelityReport | TrajectoryFidelityReport,
    tolerance: FidelityTolerance,
) -> FidelitySelectionResult:
    """Return the lowest tested degree/order meeting all applicable tolerances.

    The result applies only to the samples or trajectory represented by
    ``report``. It is not an altitude-only or mission-independent rule.
    """
    ordered = tuple(
        sorted(report.entries, key=lambda entry: (entry.truncation.degree, entry.truncation.order))
    )
    if isinstance(report, AccelerationFidelityReport):
        if (
            tolerance.maximum_absolute_acceleration_error_m_s2 is None
            and tolerance.maximum_relative_acceleration_error is None
        ):
            raise ValueError("acceleration selection requires an acceleration-error tolerance")
        passes = _acceleration_passes
    else:
        if all(
            value is None
            for value in (
                tolerance.maximum_final_position_difference_m,
                tolerance.maximum_final_velocity_difference_m_s,
                tolerance.maximum_periselene_variation_difference_m,
                tolerance.maximum_eccentricity_variation_difference,
                tolerance.maximum_minimum_terrain_clearance_difference_m,
                tolerance.maximum_lifetime_difference_s,
            )
        ):
            raise ValueError("trajectory selection requires at least one trajectory tolerance")
        passes = _trajectory_passes
    for entry in ordered:
        if passes(entry, tolerance):
            return FidelitySelectionResult(
                selected_truncation=entry.truncation,
                satisfied=True,
                reason=(
                    "lowest tested truncation satisfying the requested tolerances over the "
                    "supplied samples/trajectory"
                ),
                evaluated_truncations=tuple(item.truncation for item in ordered),
            )
    return FidelitySelectionResult(
        selected_truncation=None,
        satisfied=False,
        reason="no tested truncation satisfies the requested tolerances",
        evaluated_truncations=tuple(item.truncation for item in ordered),
    )


@dataclass(frozen=True)
class GravityFidelityStudy:
    acceleration: AccelerationFidelityReport
    trajectory: TrajectoryFidelityReport | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "acceleration": self.acceleration.to_dict(),
            "trajectory": None if self.trajectory is None else self.trajectory.to_dict(),
            "interpretation": (
                "Fidelity conclusions apply to the supplied positions/trajectory, epoch/frame, "
                "duration, force model and tolerances; they are not universal altitude-to-degree rules."
            ),
        }

    def write_json(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

    def write_runtime_csv(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        trajectory_by_key = (
            {}
            if self.trajectory is None
            else {
                (entry.truncation.degree, entry.truncation.order): entry
                for entry in self.trajectory.entries
            }
        )
        fields = [
            "degree",
            "order",
            "acceleration_runtime_seconds_per_evaluation",
            "maximum_absolute_acceleration_error_m_s2",
            "maximum_relative_acceleration_error",
            "trajectory_runtime_seconds",
            "final_position_difference_m",
            "final_velocity_difference_m_s",
            "periselene_variation_difference_m",
            "eccentricity_variation_difference",
            "minimum_terrain_clearance_difference_m",
            "impact_matches_reference",
            "lifetime_difference_s",
        ]
        with destination.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for acceleration in self.acceleration.entries:
                trajectory = trajectory_by_key.get(
                    (acceleration.truncation.degree, acceleration.truncation.order)
                )
                writer.writerow(
                    {
                        "degree": acceleration.truncation.degree,
                        "order": acceleration.truncation.order,
                        "acceleration_runtime_seconds_per_evaluation": acceleration.runtime_seconds_per_evaluation,
                        "maximum_absolute_acceleration_error_m_s2": acceleration.maximum_absolute_error_m_s2,
                        "maximum_relative_acceleration_error": acceleration.maximum_relative_error,
                        "trajectory_runtime_seconds": None if trajectory is None else trajectory.runtime_seconds,
                        "final_position_difference_m": None if trajectory is None else trajectory.final_position_difference_m,
                        "final_velocity_difference_m_s": None if trajectory is None else trajectory.final_velocity_difference_m_s,
                        "periselene_variation_difference_m": None if trajectory is None else trajectory.periselene_variation_difference_m,
                        "eccentricity_variation_difference": None if trajectory is None else trajectory.eccentricity_variation_difference,
                        "minimum_terrain_clearance_difference_m": None if trajectory is None else trajectory.minimum_terrain_clearance_difference_m,
                        "impact_matches_reference": None if trajectory is None else trajectory.impact_matches_reference,
                        "lifetime_difference_s": None if trajectory is None else trajectory.lifetime_difference_s,
                    }
                )


@dataclass(frozen=True)
class ForceModelCase:
    name: str
    dynamics: SearchDynamics

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("force-model case name must be non-empty")


@dataclass(frozen=True)
class ForceModelFidelityEntry:
    name: str
    dynamics_provenance: Mapping[str, object]
    outcome: TrajectoryOutcome
    runtime_seconds: float
    final_position_difference_m: float | None
    final_velocity_difference_m_s: float | None
    maximum_common_sample_position_difference_m: float
    maximum_common_sample_velocity_difference_m_s: float
    periselene_variation_difference_m: float
    eccentricity_variation_difference: float
    minimum_terrain_clearance_difference_m: float | None
    impact_matches_reference: bool
    lifetime_difference_s: float

    def as_dict(self) -> dict[str, object]:
        return _jsonable(asdict(self))  # type: ignore[return-value]


@dataclass(frozen=True)
class ForceModelFidelityReport:
    reference_case: str
    duration_s: float
    entries: tuple[ForceModelFidelityEntry, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "reference_case": self.reference_case,
            "duration_s": self.duration_s,
            "entries": [entry.as_dict() for entry in self.entries],
        }

    def write_json(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")


def build_force_model_ladder(
    model: SphericalHarmonicModel,
    body_fixed_from_inertial: RotationProvider,
    *,
    truncated: HarmonicTruncation = HarmonicTruncation(60, 60),
    high_degree: HarmonicTruncation | None = None,
    third_body_forces: Sequence[ForceComponent] = (),
    srp: ForceComponent | None = None,
    analysis_reference_radius_m: float = MOON_MEAN_RADIUS_M,
    collision_radius_m: float = MOON_MEAN_RADIUS_M,
) -> tuple[ForceModelCase, ...]:
    """Build the optional central/J2/GRAIL/third-body/SRP fidelity ladder."""
    high = _default_reference(model) if high_degree is None else high_degree
    truncated.validate_for(model)
    high.validate_for(model)
    if model.max_degree < 2:
        raise ValueError("force-model ladder requires a gravity model containing degree 2")

    def harmonic_case(
        name: str,
        truncation: HarmonicTruncation,
        extras: Sequence[ForceComponent] = (),
    ) -> ForceModelCase:
        return ForceModelCase(
            name,
            harmonic_search_dynamics(
                model,
                body_fixed_from_inertial,
                max_degree=truncation.degree,
                max_order=truncation.order,
                additional_forces=extras,
                analysis_reference_radius_m=analysis_reference_radius_m,
                collision_radius_m=collision_radius_m,
                parallel_safe=False,
                name=name,
            ),
        )

    cases = [
        harmonic_case("central gravity", HarmonicTruncation(0, 0)),
        harmonic_case("J2 zonal gravity", HarmonicTruncation(2, 0)),
        harmonic_case(f"truncated GRAIL {truncated.label}", truncated),
        harmonic_case(f"high-degree GRAIL {high.label}", high),
    ]
    third = tuple(third_body_forces)
    if third:
        cases.append(harmonic_case(f"high-degree GRAIL {high.label} + third bodies", high, third))
    if srp is not None:
        cases.append(
            harmonic_case(
                f"high-degree GRAIL {high.label} + third bodies + SRP",
                high,
                third + (srp,),
            )
        )
    return tuple(cases)


def compare_force_model_ladder(
    initial_state: ArrayLike,
    duration_s: float,
    cases: Sequence[ForceModelCase],
    *,
    reference_case: str | None = None,
    sample_count: int = 257,
    propagation: PropagationSettings = PropagationSettings(),
    terrain: TerrainShapeModel | None = None,
    terrain_body_fixed_from_inertial: RotationProvider | None = None,
    terrain_frame: str | None = None,
    terrain_clearance_search_samples: int = 513,
) -> ForceModelFidelityReport:
    """Compare a force-model ladder against one selected reference case."""
    state0 = np.asarray(initial_state, dtype=float)
    if state0.shape != (6,) or not np.all(np.isfinite(state0)):
        raise ValueError("initial_state must be a finite six-vector")
    values = tuple(cases)
    if not values:
        raise ValueError("at least one force-model case is required")
    names = [case.name for case in values]
    if len(set(names)) != len(names):
        raise ValueError("force-model case names must be unique")
    selected_name = names[-1] if reference_case is None else reference_case
    try:
        reference = next(case for case in values if case.name == selected_name)
    except StopIteration as exc:
        raise ValueError("reference_case is not present in the supplied force-model cases") from exc
    if sample_count < 3:
        raise ValueError("sample_count must be at least three")
    sample_times = np.linspace(0.0, float(duration_s), int(sample_count))
    reference_outcome, reference_time, reference_states, _ = _propagate_outcome(
        reference.dynamics,
        state0,
        duration_s,
        sample_times,
        propagation,
        terrain=terrain,
        terrain_body_fixed_from_inertial=terrain_body_fixed_from_inertial,
        terrain_frame=terrain_frame,
        terrain_clearance_search_samples=terrain_clearance_search_samples,
    )
    entries: list[ForceModelFidelityEntry] = []
    for case in values:
        if case.name == selected_name:
            outcome, time, states = reference_outcome, reference_time, reference_states
            start = perf_counter()
            _propagate_outcome(
                case.dynamics,
                state0,
                duration_s,
                sample_times,
                propagation,
                terrain=terrain,
                terrain_body_fixed_from_inertial=terrain_body_fixed_from_inertial,
                terrain_frame=terrain_frame,
                terrain_clearance_search_samples=terrain_clearance_search_samples,
            )
            runtime = perf_counter() - start
        else:
            outcome, time, states, runtime = _propagate_outcome(
                case.dynamics,
                state0,
                duration_s,
                sample_times,
                propagation,
                terrain=terrain,
                terrain_body_fixed_from_inertial=terrain_body_fixed_from_inertial,
                terrain_frame=terrain_frame,
                terrain_clearance_search_samples=terrain_clearance_search_samples,
            )
        same_finish = abs(outcome.achieved_duration_s - reference_outcome.achieved_duration_s) <= 1.0e-6
        if same_finish:
            final_position = float(
                np.linalg.norm(np.asarray(outcome.final_position_m) - np.asarray(reference_outcome.final_position_m))
            )
            final_velocity = float(
                np.linalg.norm(np.asarray(outcome.final_velocity_m_s) - np.asarray(reference_outcome.final_velocity_m_s))
            )
        else:
            final_position = None
            final_velocity = None
        max_position, max_velocity = _common_sample_differences(
            time, states, reference_time, reference_states
        )
        if (
            outcome.minimum_terrain_clearance_m is not None
            and reference_outcome.minimum_terrain_clearance_m is not None
        ):
            clearance_difference = abs(
                outcome.minimum_terrain_clearance_m - reference_outcome.minimum_terrain_clearance_m
            )
        else:
            clearance_difference = None
        entries.append(
            ForceModelFidelityEntry(
                name=case.name,
                dynamics_provenance=case.dynamics.provenance(),
                outcome=outcome,
                runtime_seconds=float(runtime),
                final_position_difference_m=final_position,
                final_velocity_difference_m_s=final_velocity,
                maximum_common_sample_position_difference_m=max_position,
                maximum_common_sample_velocity_difference_m_s=max_velocity,
                periselene_variation_difference_m=abs(
                    outcome.periselene_altitude_peak_to_peak_m
                    - reference_outcome.periselene_altitude_peak_to_peak_m
                ),
                eccentricity_variation_difference=abs(
                    outcome.eccentricity_peak_to_peak - reference_outcome.eccentricity_peak_to_peak
                ),
                minimum_terrain_clearance_difference_m=clearance_difference,
                impact_matches_reference=outcome.impacted == reference_outcome.impacted,
                lifetime_difference_s=abs(
                    outcome.achieved_duration_s - reference_outcome.achieved_duration_s
                ),
            )
        )
    return ForceModelFidelityReport(selected_name, float(duration_s), tuple(entries))
