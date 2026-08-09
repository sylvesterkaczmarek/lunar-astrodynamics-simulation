from io import StringIO

import numpy as np
import pytest

from lunar_astrodynamics import (
    GRGM1200A_J2,
    SphericalHarmonicModel,
    central_acceleration,
    gravity_acceleration_body_fixed,
    gravity_potential_body_fixed,
    j2_acceleration,
    normalized_legendre_4pi,
    read_shadr,
)


def _central_model(degree: int = 0) -> SphericalHarmonicModel:
    c = np.zeros((degree + 1, degree + 1))
    s = np.zeros_like(c)
    c[0, 0] = 1.0
    return SphericalHarmonicModel(
        GRGM1200A_J2.mu_m3_s2,
        GRGM1200A_J2.reference_radius_m,
        c,
        s,
        name="test",
    )


def _j2_harmonic_model() -> SphericalHarmonicModel:
    c = np.zeros((3, 3))
    s = np.zeros_like(c)
    c[0, 0] = 1.0
    c[2, 0] = -GRGM1200A_J2.j2 / np.sqrt(5.0)
    return SphericalHarmonicModel(
        GRGM1200A_J2.mu_m3_s2,
        GRGM1200A_J2.reference_radius_m,
        c,
        s,
        name="J2 as C20",
    )


def test_legendre_low_degree_values() -> None:
    latitude = np.deg2rad(30.0)
    x = np.sin(latitude)
    p, dp = normalized_legendre_4pi(latitude, 2)
    assert p[0, 0] == pytest.approx(1.0)
    assert p[1, 0] == pytest.approx(np.sqrt(3.0) * x)
    assert p[2, 0] == pytest.approx(np.sqrt(5.0) * 0.5 * (3.0 * x * x - 1.0))
    expected_derivative = np.sqrt(5.0) * 3.0 * x * np.cos(latitude)
    assert dp[2, 0] == pytest.approx(expected_derivative)


def test_central_harmonic_potential_and_acceleration() -> None:
    model = _central_model()
    r = np.array([1.9e6, -0.3e6, 0.2e6])
    potential = gravity_potential_body_fixed(r, model)
    acceleration = gravity_acceleration_body_fixed(r, model)
    assert potential == pytest.approx(model.mu_m3_s2 / np.linalg.norm(r), rel=1e-14)
    assert acceleration == pytest.approx(central_acceleration(r, model.mu_m3_s2), rel=2e-14)


def test_c20_exactly_reproduces_j2_acceleration() -> None:
    model = _j2_harmonic_model()
    r = np.array([1.91e6, -0.42e6, 0.31e6])
    expected = central_acceleration(r, model.mu_m3_s2) + j2_acceleration(
        r, model.mu_m3_s2, model.reference_radius_m, GRGM1200A_J2.j2
    )
    actual = gravity_acceleration_body_fixed(r, model)
    assert actual == pytest.approx(expected, rel=2e-13, abs=1e-13)


def test_tesseral_acceleration_matches_potential_gradient() -> None:
    c = np.zeros((4, 4))
    s = np.zeros_like(c)
    c[0, 0] = 1.0
    c[2, 0] = -9.0e-5
    c[2, 2] = 2.2e-5
    s[2, 2] = -1.7e-5
    c[3, 1] = 8.0e-6
    s[3, 1] = 5.0e-6
    model = SphericalHarmonicModel(4.9028e12, 1.738e6, c, s)
    r = np.array([1.82e6, 0.51e6, 0.37e6])
    analytic = gravity_acceleration_body_fixed(r, model)
    h = 0.25
    finite = np.empty(3)
    for axis in range(3):
        delta = np.zeros(3)
        delta[axis] = h
        finite[axis] = (
            gravity_potential_body_fixed(r + delta, model)
            - gravity_potential_body_fixed(r - delta, model)
        ) / (2.0 * h)
    assert analytic == pytest.approx(finite, rel=2e-8, abs=2e-9)


def test_tesseral_terms_depend_on_longitude() -> None:
    c = np.zeros((3, 3))
    s = np.zeros_like(c)
    c[0, 0] = 1.0
    c[2, 2] = 5e-5
    model = SphericalHarmonicModel(4.9028e12, 1.738e6, c, s)
    radius = 1.9e6
    a_x = gravity_acceleration_body_fixed([radius, 0.0, 0.0], model)
    a_y = gravity_acceleration_body_fixed([0.0, radius, 0.0], model)
    assert not np.isclose(np.linalg.norm(a_x), np.linalg.norm(a_y), rtol=1e-8)


def test_shadr_parser_inserts_c00_and_truncates() -> None:
    text = (
        " 1.7380000000000000E+03 4.9028001152632300E+03 0.0000000000000000E+00 1200\n"
        " 1200 1 0.0000000000000000E+00 0.0000000000000000E+00\n"
        "    1     0 0.0000000000000000E+00 0.0000000000000000E+00 1.0E-12 1.0E-12\n"
        "    2     0 -9.0000000000000000E-05 0.0000000000000000E+00 1.0E-12 1.0E-12\n"
        "    2     2 2.0000000000000000E-05 -3.0000000000000000E-05 1.0E-12 1.0E-12\n"
        "    3     0 1.0000000000000000E-06 0.0000000000000000E+00 1.0E-12 1.0E-12\n"
    )
    model = read_shadr(StringIO(text), max_degree=2, name="synthetic")
    assert model.max_degree == 2
    assert model.c[0, 0] == 1.0
    assert model.c[2, 0] == pytest.approx(-9e-5)
    assert model.s[2, 2] == pytest.approx(-3e-5)
    assert model.mu_m3_s2 == pytest.approx(4.90280011526323e12)
    assert model.reference_radius_m == pytest.approx(1.738e6)


def test_shadr_parser_rejects_unnormalized_coefficients() -> None:
    text = (
        " 1.7380000000000000E+03 4.9028001152632300E+03 0.0000000000000000E+00 2\n"
        " 2 0 0.0 0.0\n"
        " 2 0 -9.0E-5 0.0 1E-12 1E-12\n"
    )
    with pytest.raises(ValueError, match="normalized"):
        read_shadr(StringIO(text))


def test_degree_1200_legendre_recursion_stays_finite() -> None:
    p, dp = normalized_legendre_4pi(np.deg2rad(37.0), 1200)
    assert np.all(np.isfinite(p))
    assert np.all(np.isfinite(dp))
    assert p.shape == (1201, 1201)


def test_degree_1200_acceleration_evaluation_stays_finite() -> None:
    n = 1200
    c = np.zeros((n + 1, n + 1))
    s = np.zeros_like(c)
    c[0, 0] = 1.0
    c[2, 0] = -9.0e-5
    c[600, 13] = 1.0e-10
    s[1200, 1200] = 1.0e-12
    model = SphericalHarmonicModel(4.9028e12, 1.738e6, c, s)
    acceleration = gravity_acceleration_body_fixed([1.9e6, 2.0e5, 3.0e5], model)
    assert np.all(np.isfinite(acceleration))
