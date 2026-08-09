"""Quantitative analysis helpers for propagated trajectories."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .elements import elements_from_state

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class ElementHistory:
    time_s: FloatArray
    semi_major_axis_m: FloatArray
    eccentricity: FloatArray
    inclination_rad: FloatArray
    raan_rad_unwrapped: FloatArray
    argument_of_periapsis_rad_unwrapped: FloatArray


def element_history(time_s: FloatArray, states: FloatArray, mu_m3_s2: float) -> ElementHistory:
    """Convert a state history to non-singular osculating element histories."""
    t = np.asarray(time_s, dtype=float)
    y = np.asarray(states, dtype=float)
    if y.shape != (6, t.size):
        raise ValueError("states must have shape (6, len(time_s))")

    elements = [elements_from_state(y[:, idx], mu_m3_s2) for idx in range(t.size)]
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


def linear_rate(time_s: FloatArray, angle_rad_unwrapped: FloatArray) -> float:
    """Fit and return a linear angular rate in rad/s."""
    t = np.asarray(time_s, dtype=float)
    angle = np.asarray(angle_rad_unwrapped, dtype=float)
    if t.shape != angle.shape or t.ndim != 1 or t.size < 2:
        raise ValueError("time and angle histories must be matching one-dimensional arrays")
    centered_t = t - np.mean(t)
    centered_a = angle - np.mean(angle)
    denominator = float(np.dot(centered_t, centered_t))
    if denominator == 0.0:
        raise ValueError("time history has zero span")
    return float(np.dot(centered_t, centered_a) / denominator)
