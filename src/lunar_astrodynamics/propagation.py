"""Numerical propagation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import solve_ivp

from .constants import GRGM1200A_J2, LunarJ2Model
from .dynamics import equations_of_motion

FloatArray = NDArray[np.float64]
AccelerationFunction = Callable[[float, FloatArray], FloatArray]


@dataclass(frozen=True)
class PropagationSettings:
    method: str = "DOP853"
    rtol: float = 1e-12
    position_atol_m: float = 1e-6
    velocity_atol_m_s: float = 1e-9
    max_step_s: float = np.inf

    @property
    def atol(self) -> FloatArray:
        return np.array([self.position_atol_m] * 3 + [self.velocity_atol_m_s] * 3, dtype=float)


def make_surface_event(collision_radius_m: float):
    if collision_radius_m <= 0.0:
        raise ValueError("collision_radius_m must be positive")

    def surface_event(_time_s: float, state: ArrayLike, *_args: object) -> float:
        return float(np.linalg.norm(np.asarray(state, dtype=float)[:3]) - collision_radius_m)

    surface_event.terminal = True  # type: ignore[attr-defined]
    surface_event.direction = -1.0  # type: ignore[attr-defined]
    return surface_event


def _validate_common(
    initial_state: ArrayLike,
    duration_s: float,
    collision_radius_m: float,
    sample_times_s: ArrayLike | None,
) -> tuple[FloatArray, FloatArray | None]:
    y0 = np.asarray(initial_state, dtype=float)
    if y0.shape != (6,) or not np.all(np.isfinite(y0)):
        raise ValueError("initial_state must be a finite six-vector")
    if duration_s <= 0.0 or not np.isfinite(duration_s):
        raise ValueError("duration_s must be finite and positive")
    if float(np.linalg.norm(y0[:3])) <= collision_radius_m:
        raise ValueError("initial state is at or below the mean lunar surface")

    t_eval = None
    if sample_times_s is not None:
        t_eval = np.asarray(sample_times_s, dtype=float)
        if t_eval.ndim != 1 or t_eval.size == 0:
            raise ValueError("sample_times_s must be a non-empty one-dimensional array")
        if t_eval[0] < 0.0 or t_eval[-1] > duration_s or np.any(np.diff(t_eval) <= 0.0):
            raise ValueError("sample_times_s must be strictly increasing within [0, duration_s]")
    return y0, t_eval


def propagate(
    initial_state: ArrayLike,
    duration_s: float,
    *,
    model: LunarJ2Model = GRGM1200A_J2,
    include_j2: bool = True,
    sample_times_s: ArrayLike | None = None,
    settings: PropagationSettings = PropagationSettings(),
) -> Any:
    """Propagate a lunar orbit with central gravity and optional J2."""
    y0, t_eval = _validate_common(
        initial_state, duration_s, model.collision_radius_m, sample_times_s
    )
    j2 = model.j2 if include_j2 else 0.0
    return solve_ivp(
        equations_of_motion,
        (0.0, float(duration_s)),
        y0,
        args=(model.mu_m3_s2, model.reference_radius_m, j2),
        method=settings.method,
        t_eval=t_eval,
        rtol=settings.rtol,
        atol=settings.atol,
        max_step=settings.max_step_s,
        events=make_surface_event(model.collision_radius_m),
    )


def propagate_with_acceleration(
    initial_state: ArrayLike,
    duration_s: float,
    acceleration: AccelerationFunction,
    *,
    collision_radius_m: float,
    sample_times_s: ArrayLike | None = None,
    settings: PropagationSettings = PropagationSettings(),
) -> Any:
    """Propagate in an inertial frame using a caller-supplied acceleration law."""
    y0, t_eval = _validate_common(
        initial_state, duration_s, collision_radius_m, sample_times_s
    )

    def rhs(time_s: float, state: FloatArray) -> FloatArray:
        derivative = np.empty(6, dtype=float)
        derivative[:3] = state[3:]
        derivative[3:] = acceleration(time_s, state[:3])
        return derivative

    return solve_ivp(
        rhs,
        (0.0, float(duration_s)),
        y0,
        method=settings.method,
        t_eval=t_eval,
        rtol=settings.rtol,
        atol=settings.atol,
        max_step=settings.max_step_s,
        events=make_surface_event(collision_radius_m),
    )
