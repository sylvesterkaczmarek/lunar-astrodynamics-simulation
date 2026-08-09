import numpy as np
import pytest

from lunar_astrodynamics import GRGM1200A_J2, central_acceleration, j2_acceleration


def test_central_acceleration_points_inward() -> None:
    r = np.array([1.9e6, -2.0e5, 3.0e5])
    a = central_acceleration(r, GRGM1200A_J2.mu_m3_s2)
    assert np.dot(r, a) < 0.0
    expected = GRGM1200A_J2.mu_m3_s2 / np.linalg.norm(r) ** 2
    assert np.linalg.norm(a) == pytest.approx(expected, rel=1e-14)


def test_j2_equatorial_direction_matches_closed_form() -> None:
    r = np.array([1.9e6, 0.0, 0.0])
    a = j2_acceleration(
        r,
        GRGM1200A_J2.mu_m3_s2,
        GRGM1200A_J2.reference_radius_m,
        GRGM1200A_J2.j2,
    )
    expected_x = (
        -1.5
        * GRGM1200A_J2.j2
        * GRGM1200A_J2.mu_m3_s2
        * GRGM1200A_J2.reference_radius_m**2
        / r[0] ** 4
    )
    assert a[0] == pytest.approx(expected_x, rel=1e-14)
    assert a[1] == pytest.approx(0.0, abs=0.0)
    assert a[2] == pytest.approx(0.0, abs=0.0)


def test_origin_is_rejected() -> None:
    with pytest.raises(ValueError):
        central_acceleration([0.0, 0.0, 0.0], GRGM1200A_J2.mu_m3_s2)
