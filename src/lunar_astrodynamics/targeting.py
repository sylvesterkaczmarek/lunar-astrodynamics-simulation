"""Orbit sensitivity, local targeting, and preliminary impulsive station-keeping.

This module intentionally keeps the numerical machinery explicit. Sensitivities
use central finite differences and compare half/base/double perturbation sizes.
The differential corrector uses scaled least-squares Newton steps plus a line
search and always reports convergence or failure. Station-keeping is an
impulsive osculating-orbit restore model for preliminary mission analysis, not
flight guidance software.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Mapping, Sequence
import json

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .analysis import OrbitHistory, orbit_history
from .elements import orbital_vectors_from_state
from .frames import RotationProvider, validate_rotation_matrix
from .propagation import PropagationSettings, propagate_with_acceleration
from .stability import OrbitSearchPoint, SearchDynamics
from .terrain import TerrainShapeModel, propagate_with_terrain

FloatArray = NDArray[np.float64]
VectorFunction = Callable[[FloatArray], FloatArray]
_TWO_PI = 2.0 * np.pi


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


def _wrap(angle_rad: float) -> float:
    return float(angle_rad % _TWO_PI)


def _signed_angle_difference(angle_rad: float, target_rad: float) -> float:
    return float((angle_rad - target_rad + np.pi) % _TWO_PI - np.pi)


def _finite_vector(value: ArrayLike, *, name: str, length: int | None = None) -> FloatArray:
    array = np.asarray(value, dtype=float)
    if array.ndim != 1 or (length is not None and array.size != length) or not np.all(np.isfinite(array)):
        suffix = "" if length is None else f" of length {length}"
        raise ValueError(f"{name} must be a finite one-dimensional vector{suffix}")
    return array


@dataclass(frozen=True)
class FiniteDifferenceSettings:
    """Central finite-difference step validation settings.

    A derivative column is evaluated at ``0.5 h``, ``h`` and ``2 h``. The
    derivative from the tighter of the half/base or base/double pair is kept,
    while the disagreement is retained as a numerical diagnostic.
    """

    max_relative_disagreement: float = 5.0e-2
    comparison_floor: float = 1.0e-12

    def __post_init__(self) -> None:
        if not np.isfinite(self.max_relative_disagreement) or self.max_relative_disagreement <= 0.0:
            raise ValueError("max_relative_disagreement must be finite and positive")
        if not np.isfinite(self.comparison_floor) or self.comparison_floor <= 0.0:
            raise ValueError("comparison_floor must be finite and positive")


@dataclass(frozen=True)
class DerivativeColumnDiagnostic:
    label: str
    nominal_step: float
    selected_step: float
    half_base_relative_disagreement: float
    base_double_relative_disagreement: float
    selected_pair_relative_disagreement: float
    stable: bool

    def as_dict(self) -> dict[str, object]:
        return _jsonable(asdict(self))  # type: ignore[return-value]


@dataclass(frozen=True)
class FiniteDifferenceJacobianResult:
    nominal_output: FloatArray
    jacobian: FloatArray
    diagnostics: tuple[DerivativeColumnDiagnostic, ...]

    @property
    def all_columns_stable(self) -> bool:
        return all(item.stable for item in self.diagnostics)

    def as_dict(self) -> dict[str, object]:
        return _jsonable(asdict(self))  # type: ignore[return-value]


def finite_difference_jacobian(
    evaluate: VectorFunction,
    variables: ArrayLike,
    steps: ArrayLike,
    *,
    labels: Sequence[str] | None = None,
    settings: FiniteDifferenceSettings = FiniteDifferenceSettings(),
) -> FiniteDifferenceJacobianResult:
    """Return a central finite-difference Jacobian with explicit step sweep.

    The function is intentionally generic so the same numerical derivative
    machinery is used for Cartesian state sensitivities, orbital-parameter
    targeting and impulsive burn correction.
    """
    x = _finite_vector(variables, name="variables")
    h = _finite_vector(steps, name="steps", length=x.size)
    if np.any(h <= 0.0):
        raise ValueError("finite-difference steps must be positive")
    column_labels = tuple(labels) if labels is not None else tuple(f"x{index}" for index in range(x.size))
    if len(column_labels) != x.size or any(not label for label in column_labels):
        raise ValueError("labels must match the variable vector length")

    y0 = _finite_vector(evaluate(x.copy()), name="nominal output")
    jacobian = np.empty((y0.size, x.size), dtype=float)
    diagnostics: list[DerivativeColumnDiagnostic] = []

    for column, (step, label) in enumerate(zip(h, column_labels, strict=True)):
        derivatives: dict[float, FloatArray] = {}
        for factor in (0.5, 1.0, 2.0):
            delta = float(step * factor)
            plus = x.copy()
            minus = x.copy()
            plus[column] += delta
            minus[column] -= delta
            y_plus = _finite_vector(evaluate(plus), name=f"{label} positive perturbation output", length=y0.size)
            y_minus = _finite_vector(evaluate(minus), name=f"{label} negative perturbation output", length=y0.size)
            derivatives[factor] = (y_plus - y_minus) / (2.0 * delta)

        half = derivatives[0.5]
        base = derivatives[1.0]
        double = derivatives[2.0]

        def relative_difference(left: FloatArray, right: FloatArray) -> float:
            denominator = max(
                float(np.linalg.norm(left)),
                float(np.linalg.norm(right)),
                settings.comparison_floor,
            )
            return float(np.linalg.norm(left - right) / denominator)

        half_base = relative_difference(half, base)
        base_double = relative_difference(base, double)
        if half_base <= base_double:
            selected = half
            selected_step = 0.5 * float(step)
            selected_difference = half_base
        else:
            selected = base
            selected_step = float(step)
            selected_difference = base_double
        jacobian[:, column] = selected
        diagnostics.append(
            DerivativeColumnDiagnostic(
                label=label,
                nominal_step=float(step),
                selected_step=selected_step,
                half_base_relative_disagreement=half_base,
                base_double_relative_disagreement=base_double,
                selected_pair_relative_disagreement=selected_difference,
                stable=selected_difference <= settings.max_relative_disagreement,
            )
        )

    return FiniteDifferenceJacobianResult(y0.copy(), jacobian, tuple(diagnostics))


@dataclass(frozen=True)
class StateTransitionResult:
    duration_s: float
    initial_state: FloatArray
    final_state: FloatArray
    state_transition_matrix: FloatArray
    diagnostics: tuple[DerivativeColumnDiagnostic, ...]
    dynamics_provenance: Mapping[str, object]

    @property
    def all_columns_stable(self) -> bool:
        return all(item.stable for item in self.diagnostics)

    def as_dict(self) -> dict[str, object]:
        return _jsonable(asdict(self))  # type: ignore[return-value]


def finite_difference_state_transition(
    initial_state: ArrayLike,
    duration_s: float,
    dynamics: SearchDynamics,
    *,
    position_step_m: float = 1.0,
    velocity_step_m_s: float = 1.0e-3,
    propagation: PropagationSettings = PropagationSettings(),
    finite_difference: FiniteDifferenceSettings = FiniteDifferenceSettings(),
) -> StateTransitionResult:
    """Numerically approximate ``d x(tf) / d x(t0)`` by central differences."""
    state0 = _finite_vector(initial_state, name="initial_state", length=6)
    if not np.isfinite(duration_s) or duration_s <= 0.0:
        raise ValueError("duration_s must be finite and positive")
    if not np.isfinite(position_step_m) or position_step_m <= 0.0:
        raise ValueError("position_step_m must be finite and positive")
    if not np.isfinite(velocity_step_m_s) or velocity_step_m_s <= 0.0:
        raise ValueError("velocity_step_m_s must be finite and positive")

    def final_state(state: FloatArray) -> FloatArray:
        solution = propagate_with_acceleration(
            state,
            duration_s,
            dynamics.acceleration,
            collision_radius_m=dynamics.collision_radius_m,
            sample_times_s=np.array([duration_s]),
            settings=propagation,
        )
        if bool(len(solution.t_events[0])):
            raise ValueError("state-transition perturbation impacted the lunar surface")
        if not solution.success or solution.y.shape[1] != 1:
            raise ValueError("state-transition perturbation propagation failed")
        return np.asarray(solution.y[:, -1], dtype=float)

    steps = np.array([position_step_m] * 3 + [velocity_step_m_s] * 3, dtype=float)
    labels = ("x0", "y0", "z0", "vx0", "vy0", "vz0")
    result = finite_difference_jacobian(
        final_state,
        state0,
        steps,
        labels=labels,
        settings=finite_difference,
    )
    return StateTransitionResult(
        duration_s=float(duration_s),
        initial_state=state0.copy(),
        final_state=result.nominal_output.copy(),
        state_transition_matrix=result.jacobian.copy(),
        diagnostics=result.diagnostics,
        dynamics_provenance=dynamics.provenance(),
    )


@dataclass(frozen=True)
class CorrectorVariable:
    label: str
    finite_difference_step: float
    correction_scale: float
    lower_bound: float | None = None
    upper_bound: float | None = None
    periodic: bool = False
    period: float = _TWO_PI

    def __post_init__(self) -> None:
        if not self.label:
            raise ValueError("corrector variable label must be non-empty")
        if not np.isfinite(self.finite_difference_step) or self.finite_difference_step <= 0.0:
            raise ValueError("finite_difference_step must be finite and positive")
        if not np.isfinite(self.correction_scale) or self.correction_scale <= 0.0:
            raise ValueError("correction_scale must be finite and positive")
        if self.lower_bound is not None and not np.isfinite(self.lower_bound):
            raise ValueError("lower_bound must be finite")
        if self.upper_bound is not None and not np.isfinite(self.upper_bound):
            raise ValueError("upper_bound must be finite")
        if self.lower_bound is not None and self.upper_bound is not None and self.lower_bound >= self.upper_bound:
            raise ValueError("lower_bound must be less than upper_bound")
        if self.periodic and (not np.isfinite(self.period) or self.period <= 0.0):
            raise ValueError("period must be finite and positive for periodic variables")


@dataclass(frozen=True)
class DifferentialCorrectionSettings:
    max_iterations: int = 10
    residual_tolerance: float = 1.0e-5
    damping: float = 1.0e-8
    max_scaled_step: float = 2.0
    line_search_factors: tuple[float, ...] = (1.0, 0.5, 0.25, 0.125, 0.0625)
    maximum_condition_number: float = 1.0e14
    require_stable_jacobian: bool = True
    finite_difference: FiniteDifferenceSettings = FiniteDifferenceSettings()

    def __post_init__(self) -> None:
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be at least one")
        if not np.isfinite(self.residual_tolerance) or self.residual_tolerance <= 0.0:
            raise ValueError("residual_tolerance must be finite and positive")
        if not np.isfinite(self.damping) or self.damping < 0.0:
            raise ValueError("damping must be finite and non-negative")
        if not np.isfinite(self.max_scaled_step) or self.max_scaled_step <= 0.0:
            raise ValueError("max_scaled_step must be finite and positive")
        if not self.line_search_factors or any(
            not np.isfinite(value) or value <= 0.0 or value > 1.0 for value in self.line_search_factors
        ):
            raise ValueError("line_search_factors must lie within (0, 1]")
        if not np.isfinite(self.maximum_condition_number) or self.maximum_condition_number <= 1.0:
            raise ValueError("maximum_condition_number must be finite and greater than one")


@dataclass(frozen=True)
class DifferentialCorrectionIteration:
    iteration: int
    residual_norm_before: float
    residual_norm_after: float | None
    jacobian_rank: int
    jacobian_condition_number: float
    unstable_derivative_columns: tuple[str, ...]
    scaled_step_norm: float | None
    line_search_factor: float | None
    accepted: bool
    note: str

    def as_dict(self) -> dict[str, object]:
        return _jsonable(asdict(self))  # type: ignore[return-value]


@dataclass(frozen=True)
class DifferentialCorrectionResult:
    converged: bool
    reason: str
    initial_variables: FloatArray
    final_variables: FloatArray
    initial_residual: FloatArray
    final_residual: FloatArray
    iterations: tuple[DifferentialCorrectionIteration, ...]
    last_derivative_diagnostics: tuple[DerivativeColumnDiagnostic, ...]

    @property
    def initial_residual_norm(self) -> float:
        return float(np.linalg.norm(self.initial_residual))

    @property
    def final_residual_norm(self) -> float:
        return float(np.linalg.norm(self.final_residual))

    def as_dict(self) -> dict[str, object]:
        return _jsonable(asdict(self))  # type: ignore[return-value]


def _bounded_variables(values: FloatArray, variables: Sequence[CorrectorVariable]) -> FloatArray:
    bounded = np.asarray(values, dtype=float).copy()
    for index, definition in enumerate(variables):
        if definition.periodic:
            bounded[index] = bounded[index] % definition.period
        if definition.lower_bound is not None:
            bounded[index] = max(bounded[index], definition.lower_bound)
        if definition.upper_bound is not None:
            bounded[index] = min(bounded[index], definition.upper_bound)
    return bounded


def differential_correct(
    initial_variables: ArrayLike,
    residual_function: VectorFunction,
    variables: Sequence[CorrectorVariable],
    *,
    settings: DifferentialCorrectionSettings = DifferentialCorrectionSettings(),
) -> DifferentialCorrectionResult:
    """Solve a local nonlinear targeting problem with explicit diagnostics."""
    x = _finite_vector(initial_variables, name="initial_variables")
    definitions = tuple(variables)
    if len(definitions) != x.size:
        raise ValueError("one CorrectorVariable is required for each variable")
    x = _bounded_variables(x, definitions)
    residual = _finite_vector(residual_function(x.copy()), name="initial residual")
    initial_residual = residual.copy()
    history: list[DifferentialCorrectionIteration] = []
    last_diagnostics: tuple[DerivativeColumnDiagnostic, ...] = ()

    if float(np.linalg.norm(residual)) <= settings.residual_tolerance:
        return DifferentialCorrectionResult(
            True,
            "initial guess already satisfies the normalized target tolerance",
            x.copy(),
            x.copy(),
            initial_residual,
            residual.copy(),
            (),
            (),
        )

    steps = np.array([definition.finite_difference_step for definition in definitions], dtype=float)
    scales = np.array([definition.correction_scale for definition in definitions], dtype=float)
    labels = tuple(definition.label for definition in definitions)

    for iteration in range(1, settings.max_iterations + 1):
        before_norm = float(np.linalg.norm(residual))
        jacobian_result = finite_difference_jacobian(
            residual_function,
            x,
            steps,
            labels=labels,
            settings=settings.finite_difference,
        )
        last_diagnostics = jacobian_result.diagnostics
        unstable = tuple(item.label for item in last_diagnostics if not item.stable)
        scaled_jacobian = jacobian_result.jacobian * scales[None, :]
        rank = int(np.linalg.matrix_rank(scaled_jacobian))
        condition = float(np.linalg.cond(scaled_jacobian)) if rank > 0 else float(np.inf)

        if settings.require_stable_jacobian and unstable:
            history.append(
                DifferentialCorrectionIteration(
                    iteration, before_norm, None, rank, condition, unstable, None, None, False,
                    "finite-difference derivative failed the configured half/base/double step consistency test",
                )
            )
            return DifferentialCorrectionResult(
                False,
                "finite-difference Jacobian is step-size sensitive",
                _finite_vector(initial_variables, name="initial_variables"),
                x.copy(),
                initial_residual,
                residual.copy(),
                tuple(history),
                last_diagnostics,
            )
        if rank == 0:
            history.append(
                DifferentialCorrectionIteration(
                    iteration, before_norm, None, 0, condition, unstable, None, None, False,
                    "Jacobian has zero rank",
                )
            )
            return DifferentialCorrectionResult(
                False,
                "target residual is locally insensitive to the selected variables",
                _finite_vector(initial_variables, name="initial_variables"),
                x.copy(),
                initial_residual,
                residual.copy(),
                tuple(history),
                last_diagnostics,
            )
        if not np.isfinite(condition) or condition > settings.maximum_condition_number:
            history.append(
                DifferentialCorrectionIteration(
                    iteration, before_norm, None, rank, condition, unstable, None, None, False,
                    "Jacobian exceeds the configured condition-number limit",
                )
            )
            return DifferentialCorrectionResult(
                False,
                "target Jacobian is too ill-conditioned for the configured corrector",
                _finite_vector(initial_variables, name="initial_variables"),
                x.copy(),
                initial_residual,
                residual.copy(),
                tuple(history),
                last_diagnostics,
            )

        if settings.damping > 0.0:
            augmented_matrix = np.vstack(
                (scaled_jacobian, np.sqrt(settings.damping) * np.eye(x.size))
            )
            augmented_rhs = np.concatenate((-residual, np.zeros(x.size)))
            delta_scaled, *_ = np.linalg.lstsq(augmented_matrix, augmented_rhs, rcond=None)
        else:
            delta_scaled, *_ = np.linalg.lstsq(scaled_jacobian, -residual, rcond=None)
        scaled_norm = float(np.linalg.norm(delta_scaled))
        infinity_norm = float(np.max(np.abs(delta_scaled))) if delta_scaled.size else 0.0
        if infinity_norm > settings.max_scaled_step:
            delta_scaled *= settings.max_scaled_step / infinity_norm
            scaled_norm = float(np.linalg.norm(delta_scaled))
        delta = delta_scaled * scales

        accepted = False
        accepted_factor: float | None = None
        accepted_residual: FloatArray | None = None
        accepted_x: FloatArray | None = None
        for factor in settings.line_search_factors:
            trial_x = _bounded_variables(x + float(factor) * delta, definitions)
            try:
                trial_residual = _finite_vector(
                    residual_function(trial_x.copy()),
                    name="trial residual",
                    length=residual.size,
                )
            except (ValueError, FloatingPointError):
                continue
            if float(np.linalg.norm(trial_residual)) < before_norm:
                accepted = True
                accepted_factor = float(factor)
                accepted_residual = trial_residual
                accepted_x = trial_x
                break

        if not accepted or accepted_residual is None or accepted_x is None:
            history.append(
                DifferentialCorrectionIteration(
                    iteration, before_norm, None, rank, condition, unstable, scaled_norm, None, False,
                    "line search could not find a residual-reducing update",
                )
            )
            return DifferentialCorrectionResult(
                False,
                "differential correction stalled without a residual-reducing step",
                _finite_vector(initial_variables, name="initial_variables"),
                x.copy(),
                initial_residual,
                residual.copy(),
                tuple(history),
                last_diagnostics,
            )

        x = accepted_x
        residual = accepted_residual
        after_norm = float(np.linalg.norm(residual))
        history.append(
            DifferentialCorrectionIteration(
                iteration, before_norm, after_norm, rank, condition, unstable, scaled_norm,
                accepted_factor, True, "accepted residual-reducing Newton/least-squares step",
            )
        )
        if after_norm <= settings.residual_tolerance:
            return DifferentialCorrectionResult(
                True,
                "normalized target residual converged",
                _finite_vector(initial_variables, name="initial_variables"),
                x.copy(),
                initial_residual,
                residual.copy(),
                tuple(history),
                last_diagnostics,
            )

    return DifferentialCorrectionResult(
        False,
        "maximum differential-correction iterations reached without convergence",
        _finite_vector(initial_variables, name="initial_variables"),
        x.copy(),
        initial_residual,
        residual.copy(),
        tuple(history),
        last_diagnostics,
    )


@dataclass(frozen=True)
class TerminalStateTarget:
    indices: tuple[int, ...]
    desired_values: tuple[float, ...]
    scales: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.indices or len(self.indices) != len(self.desired_values) or len(self.indices) != len(self.scales):
            raise ValueError("terminal-state target indices, desired_values and scales must have matching non-zero length")
        if any(index < 0 or index >= 6 for index in self.indices) or len(set(self.indices)) != len(self.indices):
            raise ValueError("terminal-state indices must be unique values within [0, 5]")
        if not np.all(np.isfinite(self.desired_values)):
            raise ValueError("desired terminal-state values must be finite")
        if any(not np.isfinite(scale) or scale <= 0.0 for scale in self.scales):
            raise ValueError("terminal-state scales must be finite and positive")


@dataclass(frozen=True)
class InitialStateTargetingResult:
    correction: DifferentialCorrectionResult
    initial_state: FloatArray
    corrected_initial_state: FloatArray
    corrected_final_state: FloatArray | None

    @property
    def converged(self) -> bool:
        return self.correction.converged

    def as_dict(self) -> dict[str, object]:
        return _jsonable(asdict(self))  # type: ignore[return-value]


def target_initial_state(
    initial_state: ArrayLike,
    duration_s: float,
    dynamics: SearchDynamics,
    target: TerminalStateTarget,
    *,
    decision_indices: Sequence[int] = (3, 4, 5),
    propagation: PropagationSettings = PropagationSettings(),
    correction: DifferentialCorrectionSettings = DifferentialCorrectionSettings(),
    position_finite_difference_step_m: float = 1.0,
    velocity_finite_difference_step_m_s: float = 1.0e-3,
    position_correction_scale_m: float = 1_000.0,
    velocity_correction_scale_m_s: float = 1.0,
) -> InitialStateTargetingResult:
    """Locally target terminal Cartesian state components by changing x(t0)."""
    state0 = _finite_vector(initial_state, name="initial_state", length=6)
    indices = tuple(int(index) for index in decision_indices)
    if not indices or any(index < 0 or index >= 6 for index in indices) or len(set(indices)) != len(indices):
        raise ValueError("decision_indices must be unique state indices within [0, 5]")

    initial_variables = state0[list(indices)]
    definitions = tuple(
        CorrectorVariable(
            label=("x0", "y0", "z0", "vx0", "vy0", "vz0")[index],
            finite_difference_step=(position_finite_difference_step_m if index < 3 else velocity_finite_difference_step_m_s),
            correction_scale=(position_correction_scale_m if index < 3 else velocity_correction_scale_m_s),
        )
        for index in indices
    )

    def state_for_variables(values: FloatArray) -> FloatArray:
        state = state0.copy()
        state[list(indices)] = values
        return state

    def propagate_final(values: FloatArray) -> FloatArray:
        state = state_for_variables(values)
        solution = propagate_with_acceleration(
            state,
            duration_s,
            dynamics.acceleration,
            collision_radius_m=dynamics.collision_radius_m,
            sample_times_s=np.array([duration_s]),
            settings=propagation,
        )
        if bool(len(solution.t_events[0])) or not solution.success or solution.y.shape[1] != 1:
            raise ValueError("targeting trial did not reach the requested terminal epoch")
        return np.asarray(solution.y[:, -1], dtype=float)

    target_indices = np.array(target.indices, dtype=int)
    desired = np.array(target.desired_values, dtype=float)
    scales = np.array(target.scales, dtype=float)

    def residual(values: FloatArray) -> FloatArray:
        final_state = propagate_final(values)
        return (final_state[target_indices] - desired) / scales

    result = differential_correct(initial_variables, residual, definitions, settings=correction)
    corrected_initial = state_for_variables(result.final_variables)
    corrected_final: FloatArray | None
    if result.converged:
        corrected_final = propagate_final(result.final_variables)
    else:
        corrected_final = None
    return InitialStateTargetingResult(result, state0.copy(), corrected_initial, corrected_final)


@dataclass(frozen=True)
class OrbitTargetScales:
    apsis_altitude_m: float = 1_000.0
    eccentricity_vector_drift: float = 1.0e-4
    terrain_clearance_m: float = 1_000.0
    periapsis_location_rad: float = float(np.deg2rad(1.0))

    def __post_init__(self) -> None:
        values = (
            self.apsis_altitude_m,
            self.eccentricity_vector_drift,
            self.terrain_clearance_m,
            self.periapsis_location_rad,
        )
        if any(not np.isfinite(value) or value <= 0.0 for value in values):
            raise ValueError("orbit target scales must be finite and positive")


@dataclass(frozen=True)
class OrbitTargetSpecification:
    desired_final_periselene_altitude_m: float | None = None
    desired_final_aposelene_altitude_m: float | None = None
    maximum_eccentricity_vector_linear_drift_norm: float | None = None
    desired_periapsis_latitude_deg: float | None = None
    desired_periapsis_longitude_deg_east: float | None = None
    minimum_terrain_clearance_m: float | None = None
    scales: OrbitTargetScales = OrbitTargetScales()

    def __post_init__(self) -> None:
        scalar_values = (
            self.desired_final_periselene_altitude_m,
            self.desired_final_aposelene_altitude_m,
            self.maximum_eccentricity_vector_linear_drift_norm,
            self.minimum_terrain_clearance_m,
        )
        if all(value is None for value in scalar_values) and self.desired_periapsis_latitude_deg is None and self.desired_periapsis_longitude_deg_east is None:
            raise ValueError("at least one orbit target constraint is required")
        if any(value is not None and not np.isfinite(value) for value in scalar_values):
            raise ValueError("orbit target values must be finite when supplied")
        if self.maximum_eccentricity_vector_linear_drift_norm is not None and self.maximum_eccentricity_vector_linear_drift_norm < 0.0:
            raise ValueError("maximum eccentricity-vector drift must be non-negative")
        if self.minimum_terrain_clearance_m is not None and self.minimum_terrain_clearance_m < 0.0:
            raise ValueError("minimum terrain clearance must be non-negative")
        if self.desired_periapsis_latitude_deg is not None and not -90.0 <= self.desired_periapsis_latitude_deg <= 90.0:
            raise ValueError("desired periapsis latitude must lie within [-90, 90] deg")
        if self.desired_periapsis_longitude_deg_east is not None and not np.isfinite(self.desired_periapsis_longitude_deg_east):
            raise ValueError("desired periapsis longitude must be finite")


@dataclass(frozen=True)
class OrbitTargetEvaluation:
    final_periselene_altitude_m: float
    final_aposelene_altitude_m: float
    eccentricity_vector_linear_drift_norm: float
    minimum_reference_altitude_m: float
    minimum_terrain_clearance_m: float | None
    periapsis_sample_time_s: float
    periapsis_latitude_deg: float | None
    periapsis_longitude_deg_east: float | None
    impacted: bool
    lifetime_s: float

    def as_dict(self) -> dict[str, object]:
        return _jsonable(asdict(self))  # type: ignore[return-value]


@dataclass(frozen=True)
class OrbitParameterSensitivityResult:
    parameter_names: tuple[str, ...]
    output_names: tuple[str, ...]
    baseline_parameters: OrbitSearchPoint
    baseline_outputs: FloatArray
    jacobian: FloatArray
    diagnostics: tuple[DerivativeColumnDiagnostic, ...]

    @property
    def all_columns_stable(self) -> bool:
        return all(item.stable for item in self.diagnostics)

    def as_dict(self) -> dict[str, object]:
        return {
            "parameter_names": list(self.parameter_names),
            "output_names": list(self.output_names),
            "baseline_parameters": self.baseline_parameters.as_dict(),
            "baseline_outputs": _jsonable(self.baseline_outputs),
            "jacobian": _jsonable(self.jacobian),
            "diagnostics": [item.as_dict() for item in self.diagnostics],
        }


_ALLOWED_ORBIT_VARIABLES = {
    "semi_major_axis_m",
    "eccentricity",
    "inclination_rad",
    "raan_rad",
    "periapsis_parameter_rad",
    "initial_anomaly_rad",
}


def default_orbit_target_variables() -> tuple[CorrectorVariable, ...]:
    return (
        CorrectorVariable("semi_major_axis_m", 10.0, 10_000.0, lower_bound=1.0),
        CorrectorVariable("eccentricity", 1.0e-6, 1.0e-2, lower_bound=0.0, upper_bound=0.95),
        CorrectorVariable("periapsis_parameter_rad", 1.0e-5, float(np.deg2rad(5.0)), periodic=True),
    )


def _point_variable_vector(point: OrbitSearchPoint, variables: Sequence[CorrectorVariable]) -> FloatArray:
    values = []
    for definition in variables:
        if definition.label not in _ALLOWED_ORBIT_VARIABLES:
            raise ValueError(f"unsupported orbit targeting variable '{definition.label}'")
        values.append(float(getattr(point, definition.label)))
    return np.array(values, dtype=float)


def _point_with_variables(
    point: OrbitSearchPoint,
    variables: Sequence[CorrectorVariable],
    values: FloatArray,
    *,
    analysis_reference_radius_m: float,
) -> OrbitSearchPoint:
    updates: dict[str, float] = {}
    for definition, value in zip(variables, values, strict=True):
        if definition.label in {"raan_rad", "periapsis_parameter_rad", "initial_anomaly_rad"}:
            updates[definition.label] = _wrap(float(value))
        else:
            updates[definition.label] = float(value)
    updated = replace(point, **updates)
    if "semi_major_axis_m" in updates:
        updated = replace(
            updated,
            semi_major_axis_altitude_m=float(updated.semi_major_axis_m - analysis_reference_radius_m),
        )
    if updated.semi_major_axis_m <= analysis_reference_radius_m:
        raise ValueError("targeting trial semimajor axis is at or below the analysis reference radius")
    if updated.eccentricity < 0.0 or updated.eccentricity >= 1.0:
        raise ValueError("targeting trial eccentricity is outside [0, 1)")
    if updated.inclination_rad < 0.0 or updated.inclination_rad >= np.pi:
        raise ValueError("targeting trial inclination is outside [0, pi)")
    return updated


def _trajectory_with_event_sample(solution: object) -> tuple[FloatArray, FloatArray, bool, float | None]:
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
    return time, states, impacted, impact_time


def evaluate_orbit_target(
    point: OrbitSearchPoint,
    duration_s: float,
    dynamics: SearchDynamics,
    *,
    sample_count: int = 257,
    propagation: PropagationSettings = PropagationSettings(),
    terrain: TerrainShapeModel | None = None,
    terrain_body_fixed_from_inertial: RotationProvider | None = None,
    terrain_frame: str | None = None,
    periapsis_body_fixed_from_inertial: RotationProvider | None = None,
    apsis_eccentricity_threshold: float = 1.0e-8,
) -> OrbitTargetEvaluation:
    """Propagate one candidate and return quantities used by local targeting."""
    if sample_count < 3:
        raise ValueError("sample_count must be at least three")
    initial_state = point.initial_state(dynamics.mu_m3_s2)
    sample_times = np.linspace(0.0, float(duration_s), int(sample_count))
    if terrain is None:
        if terrain_body_fixed_from_inertial is not None or terrain_frame is not None:
            raise ValueError("terrain transform/frame cannot be supplied without terrain")
        solution = propagate_with_acceleration(
            initial_state,
            duration_s,
            dynamics.acceleration,
            collision_radius_m=dynamics.collision_radius_m,
            sample_times_s=sample_times,
            settings=propagation,
        )
        minimum_terrain_clearance = None
    else:
        if terrain_body_fixed_from_inertial is None or terrain_frame is None:
            raise ValueError("terrain targeting requires terrain transform and explicit terrain_frame")
        terrain_result = propagate_with_terrain(
            initial_state,
            duration_s,
            dynamics.acceleration,
            terrain,
            terrain_body_fixed_from_inertial,
            terrain_frame=terrain_frame,
            sample_times_s=sample_times,
            settings=propagation,
            clearance_search_samples=max(129, min(1025, sample_count * 2 + 1)),
        )
        solution = terrain_result.solution
        minimum_terrain_clearance = float(terrain_result.clearance.minimum_clearance_m)

    time, states, impacted, impact_time = _trajectory_with_event_sample(solution)
    if time.size < 2:
        raise ValueError("targeting trajectory contains fewer than two samples")
    history = orbit_history(
        time,
        states,
        dynamics.mu_m3_s2,
        reference_radius_m=dynamics.analysis_reference_radius_m,
        apsis_eccentricity_threshold=apsis_eccentricity_threshold,
    )
    drift = float(np.linalg.norm(history.statistics.eccentricity_vector.linear_drift_over_span))
    periapsis_index = int(np.argmin(history.reference_radius_altitude_m))
    periapsis_time = float(time[periapsis_index])
    periapsis_latitude: float | None = None
    periapsis_longitude: float | None = None
    if periapsis_body_fixed_from_inertial is not None:
        rotation = validate_rotation_matrix(periapsis_body_fixed_from_inertial(periapsis_time))
        body_position = rotation @ states[:3, periapsis_index]
        radius = float(np.linalg.norm(body_position))
        if radius == 0.0:
            raise ValueError("periapsis sample is at the lunar center")
        periapsis_latitude = float(np.rad2deg(np.arctan2(body_position[2], np.hypot(body_position[0], body_position[1]))))
        periapsis_longitude = float(np.rad2deg(np.arctan2(body_position[1], body_position[0])) % 360.0)

    final_periselene = float(history.periselene_altitude_m[-1])
    final_aposelene = float(history.aposelene_altitude_m[-1])
    lifetime = float(impact_time if impacted and impact_time is not None else duration_s)
    return OrbitTargetEvaluation(
        final_periselene_altitude_m=final_periselene,
        final_aposelene_altitude_m=final_aposelene,
        eccentricity_vector_linear_drift_norm=drift,
        minimum_reference_altitude_m=history.minimum_reference_altitude_m,
        minimum_terrain_clearance_m=minimum_terrain_clearance,
        periapsis_sample_time_s=periapsis_time,
        periapsis_latitude_deg=periapsis_latitude,
        periapsis_longitude_deg_east=periapsis_longitude,
        impacted=impacted,
        lifetime_s=lifetime,
    )


def _orbit_target_residual(
    evaluation: OrbitTargetEvaluation,
    specification: OrbitTargetSpecification,
) -> tuple[tuple[str, ...], FloatArray]:
    names: list[str] = []
    residuals: list[float] = []
    scales = specification.scales
    if evaluation.impacted:
        raise ValueError("targeting trial impacted before the requested duration")
    if specification.desired_final_periselene_altitude_m is not None:
        names.append("final_periselene_altitude")
        residuals.append(
            (evaluation.final_periselene_altitude_m - specification.desired_final_periselene_altitude_m)
            / scales.apsis_altitude_m
        )
    if specification.desired_final_aposelene_altitude_m is not None:
        names.append("final_aposelene_altitude")
        residuals.append(
            (evaluation.final_aposelene_altitude_m - specification.desired_final_aposelene_altitude_m)
            / scales.apsis_altitude_m
        )
    if specification.maximum_eccentricity_vector_linear_drift_norm is not None:
        names.append("eccentricity_vector_drift_upper_bound")
        residuals.append(
            max(
                0.0,
                evaluation.eccentricity_vector_linear_drift_norm
                - specification.maximum_eccentricity_vector_linear_drift_norm,
            )
            / scales.eccentricity_vector_drift
        )
    if specification.minimum_terrain_clearance_m is not None:
        if evaluation.minimum_terrain_clearance_m is None:
            raise ValueError("minimum terrain-clearance target requires a terrain model")
        names.append("terrain_clearance_lower_bound")
        residuals.append(
            max(0.0, specification.minimum_terrain_clearance_m - evaluation.minimum_terrain_clearance_m)
            / scales.terrain_clearance_m
        )
    if specification.desired_periapsis_latitude_deg is not None:
        if evaluation.periapsis_latitude_deg is None:
            raise ValueError("periapsis latitude target requires a body-fixed rotation provider")
        names.append("periapsis_latitude")
        residuals.append(
            np.deg2rad(evaluation.periapsis_latitude_deg - specification.desired_periapsis_latitude_deg)
            / scales.periapsis_location_rad
        )
    if specification.desired_periapsis_longitude_deg_east is not None:
        if evaluation.periapsis_longitude_deg_east is None:
            raise ValueError("periapsis longitude target requires a body-fixed rotation provider")
        names.append("periapsis_longitude")
        residuals.append(
            _signed_angle_difference(
                np.deg2rad(evaluation.periapsis_longitude_deg_east),
                np.deg2rad(specification.desired_periapsis_longitude_deg_east),
            )
            / scales.periapsis_location_rad
        )
    return tuple(names), np.asarray(residuals, dtype=float)


def orbit_parameter_sensitivity(
    point: OrbitSearchPoint,
    duration_s: float,
    dynamics: SearchDynamics,
    *,
    variables: Sequence[CorrectorVariable] = default_orbit_target_variables(),
    outputs: Sequence[str] = (
        "final_periselene_altitude_m",
        "final_aposelene_altitude_m",
        "eccentricity_vector_linear_drift_norm",
        "minimum_reference_altitude_m",
    ),
    sample_count: int = 257,
    propagation: PropagationSettings = PropagationSettings(),
    finite_difference: FiniteDifferenceSettings = FiniteDifferenceSettings(),
    terrain: TerrainShapeModel | None = None,
    terrain_body_fixed_from_inertial: RotationProvider | None = None,
    terrain_frame: str | None = None,
) -> OrbitParameterSensitivityResult:
    """Differentiate propagated lunar-orbit analysis outputs w.r.t. initial parameters."""
    definitions = tuple(variables)
    output_names = tuple(outputs)
    allowed_outputs = {
        "final_periselene_altitude_m",
        "final_aposelene_altitude_m",
        "eccentricity_vector_linear_drift_norm",
        "minimum_reference_altitude_m",
        "minimum_terrain_clearance_m",
    }
    if not output_names or any(name not in allowed_outputs for name in output_names):
        raise ValueError("outputs contains an unsupported orbit sensitivity quantity")
    initial_variables = _point_variable_vector(point, definitions)

    def evaluate(values: FloatArray) -> FloatArray:
        trial = _point_with_variables(
            point,
            definitions,
            values,
            analysis_reference_radius_m=dynamics.analysis_reference_radius_m,
        )
        result = evaluate_orbit_target(
            trial,
            duration_s,
            dynamics,
            sample_count=sample_count,
            propagation=propagation,
            terrain=terrain,
            terrain_body_fixed_from_inertial=terrain_body_fixed_from_inertial,
            terrain_frame=terrain_frame,
        )
        values_out: list[float] = []
        for name in output_names:
            value = getattr(result, name)
            if value is None:
                raise ValueError(f"output '{name}' is unavailable for this analysis")
            values_out.append(float(value))
        return np.asarray(values_out, dtype=float)

    steps = np.array([item.finite_difference_step for item in definitions], dtype=float)
    jacobian = finite_difference_jacobian(
        evaluate,
        initial_variables,
        steps,
        labels=tuple(item.label for item in definitions),
        settings=finite_difference,
    )
    return OrbitParameterSensitivityResult(
        parameter_names=tuple(item.label for item in definitions),
        output_names=output_names,
        baseline_parameters=point,
        baseline_outputs=jacobian.nominal_output.copy(),
        jacobian=jacobian.jacobian.copy(),
        diagnostics=jacobian.diagnostics,
    )


@dataclass(frozen=True)
class OrbitTargetingResult:
    converged: bool
    correction: DifferentialCorrectionResult
    initial_parameters: OrbitSearchPoint
    final_parameters: OrbitSearchPoint
    initial_evaluation: OrbitTargetEvaluation
    final_evaluation: OrbitTargetEvaluation
    residual_names: tuple[str, ...]
    dynamics_provenance: Mapping[str, object]

    def as_dict(self) -> dict[str, object]:
        return {
            "converged": self.converged,
            "correction": self.correction.as_dict(),
            "initial_parameters": self.initial_parameters.as_dict(),
            "final_parameters": self.final_parameters.as_dict(),
            "initial_evaluation": self.initial_evaluation.as_dict(),
            "final_evaluation": self.final_evaluation.as_dict(),
            "residual_names": list(self.residual_names),
            "dynamics": _jsonable(self.dynamics_provenance),
        }

    def write_json(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.as_dict(), indent=2) + "\n", encoding="utf-8")


def target_orbit_parameters(
    point: OrbitSearchPoint,
    duration_s: float,
    dynamics: SearchDynamics,
    specification: OrbitTargetSpecification,
    *,
    variables: Sequence[CorrectorVariable] = default_orbit_target_variables(),
    sample_count: int = 257,
    propagation: PropagationSettings = PropagationSettings(),
    correction: DifferentialCorrectionSettings = DifferentialCorrectionSettings(),
    terrain: TerrainShapeModel | None = None,
    terrain_body_fixed_from_inertial: RotationProvider | None = None,
    terrain_frame: str | None = None,
    periapsis_body_fixed_from_inertial: RotationProvider | None = None,
) -> OrbitTargetingResult:
    """Locally refine one stability-search candidate toward explicit constraints."""
    definitions = tuple(variables)
    initial_variables = _point_variable_vector(point, definitions)

    def point_for(values: FloatArray) -> OrbitSearchPoint:
        return _point_with_variables(
            point,
            definitions,
            values,
            analysis_reference_radius_m=dynamics.analysis_reference_radius_m,
        )

    def evaluate(values: FloatArray) -> OrbitTargetEvaluation:
        return evaluate_orbit_target(
            point_for(values),
            duration_s,
            dynamics,
            sample_count=sample_count,
            propagation=propagation,
            terrain=terrain,
            terrain_body_fixed_from_inertial=terrain_body_fixed_from_inertial,
            terrain_frame=terrain_frame,
            periapsis_body_fixed_from_inertial=periapsis_body_fixed_from_inertial,
        )

    initial_evaluation = evaluate(initial_variables)
    residual_names, _ = _orbit_target_residual(initial_evaluation, specification)

    def residual(values: FloatArray) -> FloatArray:
        _, vector = _orbit_target_residual(evaluate(values), specification)
        return vector

    correction_result = differential_correct(
        initial_variables,
        residual,
        definitions,
        settings=correction,
    )
    final_point = point_for(correction_result.final_variables)
    final_evaluation = evaluate(correction_result.final_variables)
    return OrbitTargetingResult(
        converged=correction_result.converged,
        correction=correction_result,
        initial_parameters=point,
        final_parameters=final_point,
        initial_evaluation=initial_evaluation,
        final_evaluation=final_evaluation,
        residual_names=residual_names,
        dynamics_provenance=dynamics.provenance(),
    )


@dataclass(frozen=True)
class StationKeepingPolicy:
    """Transparent threshold-based impulsive orbit-maintenance policy."""

    check_interval_s: float = 6.0 * 3600.0
    minimum_periselene_altitude_m: float | None = None
    maximum_semi_major_axis_deviation_m: float | None = None
    maximum_eccentricity_vector_deviation: float | None = None
    target_periselene_altitude_m: float | None = None
    target_aposelene_altitude_m: float | None = None
    correction_components: tuple[str, ...] = ("radial", "transverse")
    correct_eccentricity_vector: bool = False
    correction_apsis_scale_m: float = 500.0
    correction_eccentricity_vector_scale: float = 1.0e-4
    correction_finite_difference_step_m_s: float = 1.0e-2
    correction_variable_scale_m_s: float = 1.0
    correction_tolerance: float = 5.0e-2
    maximum_delta_v_per_maneuver_m_s: float = 20.0
    maximum_maneuvers: int = 100
    samples_per_interval: int = 17

    def __post_init__(self) -> None:
        if not np.isfinite(self.check_interval_s) or self.check_interval_s <= 0.0:
            raise ValueError("check_interval_s must be finite and positive")
        for name in (
            "minimum_periselene_altitude_m",
            "maximum_semi_major_axis_deviation_m",
            "maximum_eccentricity_vector_deviation",
            "target_periselene_altitude_m",
            "target_aposelene_altitude_m",
        ):
            value = getattr(self, name)
            if value is not None and (not np.isfinite(value) or value < 0.0):
                raise ValueError(f"{name} must be finite and non-negative when supplied")
        allowed = {"radial", "transverse", "normal"}
        if not self.correction_components or any(item not in allowed for item in self.correction_components):
            raise ValueError("correction_components must use radial/transverse/normal")
        if len(set(self.correction_components)) != len(self.correction_components):
            raise ValueError("correction_components must be unique")
        positive = (
            self.correction_apsis_scale_m,
            self.correction_eccentricity_vector_scale,
            self.correction_finite_difference_step_m_s,
            self.correction_variable_scale_m_s,
            self.correction_tolerance,
            self.maximum_delta_v_per_maneuver_m_s,
        )
        if any(not np.isfinite(value) or value <= 0.0 for value in positive):
            raise ValueError("station-keeping numerical scales and delta-v limit must be finite and positive")
        if self.maximum_maneuvers < 1:
            raise ValueError("maximum_maneuvers must be at least one")
        if self.samples_per_interval < 2:
            raise ValueError("samples_per_interval must be at least two")
        if (
            self.minimum_periselene_altitude_m is None
            and self.maximum_semi_major_axis_deviation_m is None
            and self.maximum_eccentricity_vector_deviation is None
        ):
            raise ValueError("at least one station-keeping trigger threshold is required")


@dataclass(frozen=True)
class StationKeepingManeuver:
    index: int
    time_s: float
    utc_time: str | None
    trigger_reasons: tuple[str, ...]
    delta_v_inertial_m_s: FloatArray
    delta_v_rtn_m_s: FloatArray
    delta_v_magnitude_m_s: float
    pre_periselene_altitude_m: float
    post_periselene_altitude_m: float
    pre_aposelene_altitude_m: float
    post_aposelene_altitude_m: float
    pre_eccentricity_vector: FloatArray
    post_eccentricity_vector: FloatArray
    correction: DifferentialCorrectionResult

    def as_dict(self) -> dict[str, object]:
        return _jsonable(asdict(self))  # type: ignore[return-value]


@dataclass(frozen=True)
class StationKeepingResult:
    time_s: FloatArray
    states: FloatArray
    maneuvers: tuple[StationKeepingManeuver, ...]
    requested_duration_s: float
    achieved_duration_s: float
    impacted: bool
    impact_time_s: float | None
    terminated_early: bool
    termination_reason: str
    start_epoch_utc: str | None
    dynamics_provenance: Mapping[str, object]

    @property
    def maneuver_count(self) -> int:
        return len(self.maneuvers)

    @property
    def total_delta_v_m_s(self) -> float:
        return float(sum(item.delta_v_magnitude_m_s for item in self.maneuvers))

    @property
    def maximum_delta_v_m_s(self) -> float:
        return float(max((item.delta_v_magnitude_m_s for item in self.maneuvers), default=0.0))

    def as_dict(self) -> dict[str, object]:
        return {
            "requested_duration_s": self.requested_duration_s,
            "achieved_duration_s": self.achieved_duration_s,
            "impacted": self.impacted,
            "impact_time_s": self.impact_time_s,
            "terminated_early": self.terminated_early,
            "termination_reason": self.termination_reason,
            "start_epoch_utc": self.start_epoch_utc,
            "maneuver_count": self.maneuver_count,
            "total_delta_v_m_s": self.total_delta_v_m_s,
            "maximum_delta_v_m_s": self.maximum_delta_v_m_s,
            "maneuvers": [item.as_dict() for item in self.maneuvers],
            "time_s": _jsonable(self.time_s),
            "states": _jsonable(self.states),
            "dynamics": _jsonable(self.dynamics_provenance),
        }

    def write_json(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.as_dict(), indent=2) + "\n", encoding="utf-8")


def _osculating_summary(state: FloatArray, dynamics: SearchDynamics) -> tuple[float, float, float, FloatArray]:
    vectors = orbital_vectors_from_state(state, dynamics.mu_m3_s2)
    if not np.isfinite(vectors.semi_major_axis_m) or vectors.semi_major_axis_m <= 0.0 or vectors.eccentricity >= 1.0:
        raise ValueError("station-keeping controller requires a bound elliptic osculating state")
    periselene = float(vectors.periselene_radius_m - dynamics.analysis_reference_radius_m)
    aposelene = float(vectors.aposelene_radius_m - dynamics.analysis_reference_radius_m)
    return periselene, aposelene, float(vectors.semi_major_axis_m), vectors.eccentricity_vector.copy()


def _rtn_basis(state: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray]:
    position = state[:3]
    velocity = state[3:]
    radius = float(np.linalg.norm(position))
    angular_momentum = np.cross(position, velocity)
    h_norm = float(np.linalg.norm(angular_momentum))
    if radius == 0.0 or h_norm == 0.0:
        raise ValueError("RTN basis requires non-zero radius and angular momentum")
    radial = position / radius
    normal = angular_momentum / h_norm
    transverse = np.cross(normal, radial)
    return radial, transverse, normal


def _parse_epoch_utc(value: str | None) -> datetime | None:
    if value is None:
        return None
    text = value.strip().replace("Z", "+00:00")
    try:
        epoch = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValueError("start_epoch_utc must be ISO-8601 when supplied") from exc
    if epoch.tzinfo is None:
        raise ValueError("start_epoch_utc must include a UTC offset or Z")
    return epoch


def _maneuver_utc(epoch: datetime | None, time_s: float) -> str | None:
    if epoch is None:
        return None
    return (epoch + timedelta(seconds=float(time_s))).isoformat()


def _stationkeeping_trigger_reasons(
    state: FloatArray,
    dynamics: SearchDynamics,
    policy: StationKeepingPolicy,
    *,
    reference_semi_major_axis_m: float,
    reference_eccentricity_vector: FloatArray,
) -> tuple[str, ...]:
    periselene, _, semimajor, eccentricity_vector = _osculating_summary(state, dynamics)
    reasons: list[str] = []
    if policy.minimum_periselene_altitude_m is not None and periselene < policy.minimum_periselene_altitude_m:
        reasons.append("periselene below threshold")
    if (
        policy.maximum_semi_major_axis_deviation_m is not None
        and abs(semimajor - reference_semi_major_axis_m) > policy.maximum_semi_major_axis_deviation_m
    ):
        reasons.append("semimajor-axis deviation above threshold")
    if (
        policy.maximum_eccentricity_vector_deviation is not None
        and float(np.linalg.norm(eccentricity_vector - reference_eccentricity_vector))
        > policy.maximum_eccentricity_vector_deviation
    ):
        reasons.append("eccentricity-vector deviation above threshold")
    return tuple(reasons)


def _compute_stationkeeping_burn(
    state: FloatArray,
    dynamics: SearchDynamics,
    policy: StationKeepingPolicy,
    *,
    target_periselene_altitude_m: float,
    target_aposelene_altitude_m: float,
    reference_eccentricity_vector: FloatArray,
) -> tuple[FloatArray, FloatArray, DifferentialCorrectionResult]:
    radial, transverse, normal = _rtn_basis(state)
    basis_map = {"radial": radial, "transverse": transverse, "normal": normal}
    basis = np.column_stack([basis_map[name] for name in policy.correction_components])
    variable_count = basis.shape[1]
    definitions = tuple(
        CorrectorVariable(
            label=f"delta_v_{name}_m_s",
            finite_difference_step=policy.correction_finite_difference_step_m_s,
            correction_scale=policy.correction_variable_scale_m_s,
        )
        for name in policy.correction_components
    )

    def residual(delta_components: FloatArray) -> FloatArray:
        trial = state.copy()
        trial[3:] += basis @ delta_components
        periselene, aposelene, _, eccentricity_vector = _osculating_summary(trial, dynamics)
        values = [
            (periselene - target_periselene_altitude_m) / policy.correction_apsis_scale_m,
            (aposelene - target_aposelene_altitude_m) / policy.correction_apsis_scale_m,
        ]
        if policy.correct_eccentricity_vector:
            values.extend(
                ((eccentricity_vector - reference_eccentricity_vector) / policy.correction_eccentricity_vector_scale).tolist()
            )
        return np.asarray(values, dtype=float)

    settings = DifferentialCorrectionSettings(
        max_iterations=8,
        residual_tolerance=policy.correction_tolerance,
        damping=1.0e-8,
        max_scaled_step=5.0,
        line_search_factors=(1.0, 0.5, 0.25, 0.125, 0.0625),
        maximum_condition_number=1.0e14,
        require_stable_jacobian=True,
        finite_difference=FiniteDifferenceSettings(max_relative_disagreement=0.10),
    )
    correction = differential_correct(
        np.zeros(variable_count, dtype=float),
        residual,
        definitions,
        settings=settings,
    )
    delta_components = correction.final_variables.copy()
    delta_inertial = basis @ delta_components
    return delta_inertial.astype(float), delta_components.astype(float), correction


def simulate_impulsive_stationkeeping(
    initial_state: ArrayLike,
    duration_s: float,
    dynamics: SearchDynamics,
    policy: StationKeepingPolicy,
    *,
    propagation: PropagationSettings = PropagationSettings(),
    start_epoch_utc: str | None = None,
) -> StationKeepingResult:
    """Simulate threshold-triggered impulsive orbit maintenance.

    At each control epoch the current osculating orbit is checked. If a trigger
    fires, an instantaneous RTN burn is locally corrected to restore target
    osculating periselene/aposelene (and optionally the eccentricity vector).
    The model ignores navigation error, execution error, finite burn duration,
    attitude constraints and operational maneuver windows.
    """
    state = _finite_vector(initial_state, name="initial_state", length=6).copy()
    if not np.isfinite(duration_s) or duration_s <= 0.0:
        raise ValueError("duration_s must be finite and positive")
    epoch = _parse_epoch_utc(start_epoch_utc)
    initial_periselene, initial_aposelene, reference_a, reference_e = _osculating_summary(state, dynamics)
    target_periselene = (
        initial_periselene if policy.target_periselene_altitude_m is None else policy.target_periselene_altitude_m
    )
    target_aposelene = (
        initial_aposelene if policy.target_aposelene_altitude_m is None else policy.target_aposelene_altitude_m
    )

    times: list[float] = [0.0]
    states: list[FloatArray] = [state.copy()]
    maneuvers: list[StationKeepingManeuver] = []
    current_time = 0.0
    impacted = False
    impact_time: float | None = None
    termination_reason = "requested duration completed"
    terminated_early = False

    while current_time < duration_s - 1.0e-12:
        reasons = _stationkeeping_trigger_reasons(
            state,
            dynamics,
            policy,
            reference_semi_major_axis_m=reference_a,
            reference_eccentricity_vector=reference_e,
        )
        if reasons:
            if len(maneuvers) >= policy.maximum_maneuvers:
                terminated_early = True
                termination_reason = "maximum maneuver count reached"
                break
            pre_peri, pre_apo, _, pre_e = _osculating_summary(state, dynamics)
            delta_inertial, delta_components, correction = _compute_stationkeeping_burn(
                state,
                dynamics,
                policy,
                target_periselene_altitude_m=target_periselene,
                target_aposelene_altitude_m=target_aposelene,
                reference_eccentricity_vector=reference_e,
            )
            delta_v = float(np.linalg.norm(delta_inertial))
            if not correction.converged:
                terminated_early = True
                termination_reason = f"station-keeping corrector failed: {correction.reason}"
                break
            if delta_v > policy.maximum_delta_v_per_maneuver_m_s:
                terminated_early = True
                termination_reason = (
                    f"required station-keeping burn {delta_v:.6g} m/s exceeds configured "
                    f"limit {policy.maximum_delta_v_per_maneuver_m_s:.6g} m/s"
                )
                break
            state = state.copy()
            state[3:] += delta_inertial
            post_peri, post_apo, _, post_e = _osculating_summary(state, dynamics)
            if times and abs(times[-1] - current_time) <= 1.0e-9:
                states[-1] = state.copy()
            maneuvers.append(
                StationKeepingManeuver(
                    index=len(maneuvers) + 1,
                    time_s=float(current_time),
                    utc_time=_maneuver_utc(epoch, current_time),
                    trigger_reasons=reasons,
                    delta_v_inertial_m_s=delta_inertial.copy(),
                    delta_v_rtn_m_s=delta_components.copy(),
                    delta_v_magnitude_m_s=delta_v,
                    pre_periselene_altitude_m=pre_peri,
                    post_periselene_altitude_m=post_peri,
                    pre_aposelene_altitude_m=pre_apo,
                    post_aposelene_altitude_m=post_apo,
                    pre_eccentricity_vector=pre_e.copy(),
                    post_eccentricity_vector=post_e.copy(),
                    correction=correction,
                )
            )

        segment_duration = min(policy.check_interval_s, duration_s - current_time)
        local_times = np.linspace(0.0, segment_duration, policy.samples_per_interval)
        segment_start = float(current_time)

        def segment_acceleration(local_time_s: float, position_m: FloatArray) -> FloatArray:
            return dynamics.acceleration(segment_start + float(local_time_s), position_m)

        solution = propagate_with_acceleration(
            state,
            segment_duration,
            segment_acceleration,
            collision_radius_m=dynamics.collision_radius_m,
            sample_times_s=local_times,
            settings=propagation,
        )
        if not solution.success:
            terminated_early = True
            termination_reason = f"propagation failed during station-keeping simulation: {solution.message}"
            break
        for index in range(1, solution.t.size):
            times.append(segment_start + float(solution.t[index]))
            states.append(np.asarray(solution.y[:, index], dtype=float).copy())
        event_times = solution.t_events[0]
        if len(event_times):
            impacted = True
            local_impact_time = float(event_times[0])
            impact_time = segment_start + local_impact_time
            event_state = np.asarray(solution.y_events[0][0], dtype=float)
            if not times or abs(times[-1] - impact_time) > 1.0e-9:
                times.append(impact_time)
                states.append(event_state.copy())
            else:
                states[-1] = event_state.copy()
            current_time = impact_time
            state = event_state.copy()
            terminated_early = impact_time < duration_s - 1.0e-9
            termination_reason = "lunar-surface impact"
            break
        current_time = segment_start + segment_duration
        state = np.asarray(solution.y[:, -1], dtype=float).copy()

    achieved = float(times[-1]) if times else 0.0
    return StationKeepingResult(
        time_s=np.asarray(times, dtype=float),
        states=np.column_stack(states),
        maneuvers=tuple(maneuvers),
        requested_duration_s=float(duration_s),
        achieved_duration_s=achieved,
        impacted=impacted,
        impact_time_s=impact_time,
        terminated_early=terminated_early,
        termination_reason=termination_reason,
        start_epoch_utc=start_epoch_utc,
        dynamics_provenance=dynamics.provenance(),
    )
