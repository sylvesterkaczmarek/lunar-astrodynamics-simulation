"""Frame transformations for body-fixed lunar gravity evaluation."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]
RotationProvider = Callable[[float], FloatArray]


def validate_rotation_matrix(matrix: ArrayLike, *, atol: float = 1e-12) -> FloatArray:
    """Validate and return a proper 3x3 rotation matrix."""
    rotation = np.asarray(matrix, dtype=float)
    if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
        raise ValueError("rotation must be a finite 3x3 matrix")
    if not np.allclose(rotation @ rotation.T, np.eye(3), atol=atol, rtol=0.0):
        raise ValueError("rotation matrix must be orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=atol, rtol=0.0):
        raise ValueError("rotation matrix must have determinant +1")
    return rotation


def rotation_z(angle_rad: float) -> FloatArray:
    """Return a passive rotation from an inertial frame into a z-rotated frame."""
    c = float(np.cos(angle_rad))
    s = float(np.sin(angle_rad))
    return np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)


def constant_rate_z_rotation(
    rotation_rate_rad_s: float, *, phase_rad: float = 0.0
) -> RotationProvider:
    """Return a simple z-axis body-fixed-from-inertial rotation provider.

    This helper is useful for tests and demonstrations. It is not a lunar
    libration or principal-axes orientation model.
    """

    def provider(time_s: float) -> FloatArray:
        return rotation_z(phase_rad + rotation_rate_rad_s * time_s)

    return provider


def spice_rotation_provider(
    inertial_frame: str,
    body_fixed_frame: str,
    *,
    et_offset_s: float = 0.0,
) -> RotationProvider:
    """Build a SPICE rotation provider after the caller has loaded kernels.

    Requires the optional ``spiceypy`` dependency. ``time_s`` is added to
    ``et_offset_s`` and passed to ``spiceypy.pxform`` as ephemeris time.
    Frame names are intentionally caller-supplied because they depend on the
    loaded kernel set.
    """
    try:
        import spiceypy as spice  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "SPICE support requires 'pip install lunar-astrodynamics-simulation[spice]'"
        ) from exc

    def provider(time_s: float) -> FloatArray:
        matrix = spice.pxform(inertial_frame, body_fixed_frame, et_offset_s + time_s)
        return validate_rotation_matrix(matrix, atol=1e-10)

    return provider
