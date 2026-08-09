import numpy as np
import pytest

from lunar_astrodynamics import (
    SphericalHarmonicModel,
    gravity_acceleration_body_fixed,
    gravity_acceleration_inertial,
    rotation_z,
    validate_rotation_matrix,
)


def _tesseral_model() -> SphericalHarmonicModel:
    c = np.zeros((3, 3))
    s = np.zeros_like(c)
    c[0, 0] = 1.0
    c[2, 2] = 4.0e-5
    s[2, 2] = -2.0e-5
    return SphericalHarmonicModel(4.9028e12, 1.738e6, c, s)


def test_rotation_z_is_proper_rotation() -> None:
    rotation = rotation_z(0.73)
    assert validate_rotation_matrix(rotation) == pytest.approx(rotation)


def test_invalid_rotation_is_rejected() -> None:
    with pytest.raises(ValueError, match="orthonormal"):
        validate_rotation_matrix(np.diag([1.0, 1.0, 2.0]))


def test_inertial_wrapper_rotates_position_and_acceleration_consistently() -> None:
    model = _tesseral_model()
    rotation = rotation_z(0.6)
    provider = lambda _time: rotation
    r_i = np.array([1.87e6, 0.22e6, 0.31e6])
    expected = rotation.T @ gravity_acceleration_body_fixed(rotation @ r_i, model)
    actual = gravity_acceleration_inertial(123.0, r_i, model, provider)
    assert actual == pytest.approx(expected, rel=1e-14, abs=1e-14)
