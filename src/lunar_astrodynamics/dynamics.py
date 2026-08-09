"""Lunar gravity accelerations and equations of motion."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


def _position(position_m: ArrayLike) -> FloatArray:
    r = np.asarray(position_m, dtype=float)
    if r.shape != (3,):
        raise ValueError("position must be a 3-vector")
    if not np.all(np.isfinite(r)):
        raise ValueError("position must contain finite values")
    return r


def central_acceleration(position_m: ArrayLike, mu_m3_s2: float) -> FloatArray:
    """Return point-mass gravitational acceleration in m/s^2."""
    r = _position(position_m)
    radius = float(np.linalg.norm(r))
    if radius == 0.0:
        raise ValueError("position cannot be the central-body origin")
    if mu_m3_s2 <= 0.0 or not np.isfinite(mu_m3_s2):
        raise ValueError("mu_m3_s2 must be finite and positive")
    return -mu_m3_s2 * r / radius**3


def j2_acceleration(
    position_m: ArrayLike,
    mu_m3_s2: float,
    reference_radius_m: float,
    j2: float,
) -> FloatArray:
    """Return the axisymmetric J2 perturbing acceleration in m/s^2.

    The z-axis is the symmetry axis of the low-degree model. This function does
    not implement tesseral/sectoral spherical harmonics and therefore does not
    require lunar rotation about that axis. A full lunar gravity field must be
    evaluated in the gravity model's body-fixed principal-axes frame.
    """
    r = _position(position_m)
    radius = float(np.linalg.norm(r))
    if radius == 0.0:
        raise ValueError("position cannot be the central-body origin")
    if mu_m3_s2 <= 0.0 or reference_radius_m <= 0.0:
        raise ValueError("mu_m3_s2 and reference_radius_m must be positive")
    if not np.isfinite(j2):
        raise ValueError("j2 must be finite")

    x, y, z = r
    r2 = radius * radius
    z_ratio = z * z / r2
    factor = 1.5 * j2 * mu_m3_s2 * reference_radius_m**2 / radius**5

    return np.array(
        [
            factor * x * (5.0 * z_ratio - 1.0),
            factor * y * (5.0 * z_ratio - 1.0),
            factor * z * (5.0 * z_ratio - 3.0),
        ],
        dtype=float,
    )


def total_acceleration(
    position_m: ArrayLike,
    mu_m3_s2: float,
    reference_radius_m: float,
    j2: float = 0.0,
) -> FloatArray:
    """Return central plus J2 acceleration in m/s^2."""
    return central_acceleration(position_m, mu_m3_s2) + j2_acceleration(
        position_m, mu_m3_s2, reference_radius_m, j2
    )


def equations_of_motion(
    _time_s: float,
    state: ArrayLike,
    mu_m3_s2: float,
    reference_radius_m: float,
    j2: float,
) -> FloatArray:
    """Return [vx, vy, vz, ax, ay, az] for the six-state propagator."""
    y = np.asarray(state, dtype=float)
    if y.shape != (6,):
        raise ValueError("state must contain six values")
    if not np.all(np.isfinite(y)):
        raise ValueError("state must contain finite values")

    derivative = np.empty(6, dtype=float)
    derivative[:3] = y[3:]
    derivative[3:] = total_acceleration(
        y[:3], mu_m3_s2, reference_radius_m, j2
    )
    return derivative
