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


def _pds_header_record(
    *,
    degree: int = 3,
    order: int = 3,
    normalization: int = 1,
    reference_longitude_deg: float = 0.0,
    reference_latitude_deg: float = 0.0,
) -> str:
    data = ",".join(
        [
            f"{1738.0:23.16E}",
            f"{4902.80011526323:23.16E}",
            f"{0.0:23.16E}",
            f"{degree:5d}",
            f"{order:5d}",
            f"{normalization:5d}",
            f"{reference_longitude_deg:23.16E}",
            f"{reference_latitude_deg:23.16E}",
        ]
    )
    assert len(data) == 137
    record = data + " " * 105 + "\r\n"
    assert len(record.encode("ascii")) == 244
    return record


def _pds_coefficient_record(
    n: int,
    m: int,
    c: float,
    s: float,
    sigma_c: float = 1e-12,
    sigma_s: float = 1e-12,
) -> str:
    data = ",".join(
        [
            f"{n:5d}",
            f"{m:5d}",
            f"{c:23.16E}",
            f"{s:23.16E}",
            f"{sigma_c:23.16E}",
            f"{sigma_s:23.16E}",
        ]
    )
    assert len(data) == 107
    record = data + " " * 13 + "\r\n"
    assert len(record.encode("ascii")) == 122
    return record


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


def _mixed_harmonic_model() -> SphericalHarmonicModel:
    """Deterministic zonal/tesseral/sectoral field for independent checks."""
    c = np.zeros((9, 9))
    s = np.zeros_like(c)
    c[0, 0] = 1.0
    terms = [
        (2, 0, -9.0e-5, 0.0),
        (2, 2, 2.2e-5, -1.7e-5),
        (3, 1, 8.0e-6, 5.0e-6),
        (4, 4, -3.5e-6, 2.0e-6),
        (5, 3, 1.2e-6, -1.1e-6),
        (6, 2, -8.0e-7, 9.0e-7),
        (7, 1, 4.0e-7, -3.0e-7),
        (8, 8, 2.0e-7, 1.0e-7),
    ]
    for n, m, c_nm, s_nm in terms:
        c[n, m] = c_nm
        s[n, m] = s_nm
    return SphericalHarmonicModel(4.9028e12, 1.738e6, c, s, name="mixed test field")


def _finite_difference_gradient(
    position_m: np.ndarray,
    model: SphericalHarmonicModel,
    *,
    step_m: float = 0.25,
    max_degree: int | None = None,
    max_order: int | None = None,
) -> np.ndarray:
    """Independent Cartesian central-difference gradient of the potential."""
    gradient = np.empty(3)
    for axis in range(3):
        delta = np.zeros(3)
        delta[axis] = step_m
        gradient[axis] = (
            gravity_potential_body_fixed(
                position_m + delta,
                model,
                max_degree=max_degree,
                max_order=max_order,
            )
            - gravity_potential_body_fixed(
                position_m - delta,
                model,
                max_degree=max_degree,
                max_order=max_order,
            )
        ) / (2.0 * step_m)
    return gradient


def test_legendre_low_degree_values() -> None:
    latitude = np.deg2rad(30.0)
    x = np.sin(latitude)
    p, dp = normalized_legendre_4pi(latitude, 2)
    assert p[0, 0] == pytest.approx(1.0)
    assert p[1, 0] == pytest.approx(np.sqrt(3.0) * x)
    assert p[2, 0] == pytest.approx(np.sqrt(5.0) * 0.5 * (3.0 * x * x - 1.0))
    expected_derivative = np.sqrt(5.0) * 3.0 * x * np.cos(latitude)
    assert dp[2, 0] == pytest.approx(expected_derivative)


@pytest.mark.parametrize("latitude_deg", [-90.0, -89.999999, 0.0, 37.0, 89.999999, 90.0])
def test_legendre_degree_1200_stays_finite_across_latitudes(latitude_deg: float) -> None:
    p, dp = normalized_legendre_4pi(np.deg2rad(latitude_deg), 1200)
    assert np.all(np.isfinite(p))
    assert np.all(np.isfinite(dp))
    assert p.shape == (1201, 1201)


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
    finite = _finite_difference_gradient(r, model)
    assert analytic == pytest.approx(finite, rel=2e-8, abs=2e-9)


@pytest.mark.parametrize(
    "position_m",
    [
        np.array([1.78e6, 0.0, 0.0]),
        np.array([1.55e6, 0.70e6, 0.80e6]),
        np.array([2.45e6, -0.90e6, 0.55e6]),
        np.array([2.0e-2, -1.0e-2, 1.90e6]),
        np.array([-2.0e-2, 1.0e-2, -1.90e6]),
        np.array([0.0, 0.0, 1.90e6]),
        np.array([0.0, 0.0, -1.90e6]),
    ],
    ids=[
        "low-altitude-equator",
        "mid-latitude",
        "higher-altitude",
        "near-north-pole",
        "near-south-pole",
        "north-axis",
        "south-axis",
    ],
)
@pytest.mark.parametrize("max_degree", [2, 4, 8])
def test_cartesian_acceleration_matches_independent_gradient_across_geometry(
    position_m: np.ndarray,
    max_degree: int,
) -> None:
    model = _mixed_harmonic_model()
    analytic = gravity_acceleration_body_fixed(position_m, model, max_degree=max_degree)
    finite = _finite_difference_gradient(position_m, model, max_degree=max_degree)
    assert analytic == pytest.approx(finite, rel=3e-8, abs=3e-9)


def test_non_zonal_gravity_has_finite_exact_pole_limit() -> None:
    c = np.zeros((5, 5))
    s = np.zeros_like(c)
    c[0, 0] = 1.0
    c[2, 1] = 2.5e-5
    s[2, 1] = -1.2e-5
    c[4, 1] = -6.0e-6
    s[4, 1] = 4.0e-6
    c[4, 4] = 3.0e-6
    model = SphericalHarmonicModel(4.9028e12, 1.738e6, c, s)

    for z in (1.90e6, -1.90e6):
        r = np.array([0.0, 0.0, z])
        analytic = gravity_acceleration_body_fixed(r, model)
        finite = _finite_difference_gradient(r, model)
        assert np.linalg.norm(analytic[:2]) > 1e-7
        assert analytic == pytest.approx(finite, rel=3e-8, abs=3e-9)


def test_acceleration_converges_continuously_when_crossing_north_pole() -> None:
    model = _mixed_harmonic_model()
    radius = 1.90e6
    pole = gravity_acceleration_body_fixed([0.0, 0.0, radius], model)
    offsets_m = [100.0, 10.0, 1.0, 0.1, 0.01]
    previous_error = np.inf

    for offset in offsets_m:
        z = np.sqrt(radius * radius - offset * offset)
        plus = gravity_acceleration_body_fixed([offset, 0.0, z], model)
        minus = gravity_acceleration_body_fixed([-offset, 0.0, z], model)
        error = max(np.linalg.norm(plus - pole), np.linalg.norm(minus - pole))
        assert error < previous_error
        previous_error = error

    assert previous_error < 1e-8


def test_exact_pole_limit_is_independent_of_approach_azimuth() -> None:
    model = _mixed_harmonic_model()
    radius = 1.90e6
    transverse = 0.01
    pole = gravity_acceleration_body_fixed([0.0, 0.0, radius], model)

    for azimuth in np.deg2rad([0.0, 45.0, 90.0, 180.0, 270.0]):
        x = transverse * np.cos(azimuth)
        y = transverse * np.sin(azimuth)
        z = np.sqrt(radius * radius - transverse * transverse)
        nearby = gravity_acceleration_body_fixed([x, y, z], model)
        assert np.linalg.norm(nearby - pole) < 1e-8


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


def test_degree_and_order_truncation_remain_consistent() -> None:
    model = _mixed_harmonic_model()
    position = np.array([1.72e6, 0.44e6, 0.61e6])
    truncated = model.truncated(4, 2)
    expected = gravity_acceleration_body_fixed(position, truncated)
    actual = gravity_acceleration_body_fixed(position, model, max_degree=4, max_order=2)
    assert actual == pytest.approx(expected, rel=2e-14, abs=2e-14)


def test_shadr_parser_reads_byte_faithful_pds_records_and_keeps_first_coefficient() -> None:
    text = (
        _pds_header_record(degree=3, order=3)
        + _pds_coefficient_record(2, 2, 2.0e-5, -3.0e-5)
        + _pds_coefficient_record(1, 0, 7.0e-8, 0.0)
        + _pds_coefficient_record(3, 1, 4.0e-6, 5.0e-6)
        + _pds_coefficient_record(2, 0, -9.0e-5, 0.0)
    )
    model = read_shadr(StringIO(text), max_degree=2, name="synthetic")
    assert model.max_degree == 2
    assert model.c[0, 0] == 1.0
    assert model.c[1, 0] == pytest.approx(7e-8)
    assert model.c[2, 0] == pytest.approx(-9e-5)
    assert model.s[2, 2] == pytest.approx(-3e-5)
    assert model.mu_m3_s2 == pytest.approx(4.90280011526323e12)
    assert model.reference_radius_m == pytest.approx(1.738e6)


def test_shadr_parser_does_not_require_coefficient_ordering_or_completeness() -> None:
    text = (
        _pds_header_record(degree=4, order=4)
        + _pds_coefficient_record(4, 2, 1.0e-7, 2.0e-7)
        + _pds_coefficient_record(2, 0, -9.0e-5, 0.0)
    )
    model = read_shadr(StringIO(text))
    assert model.c[4, 2] == pytest.approx(1e-7)
    assert model.s[4, 2] == pytest.approx(2e-7)
    assert model.c[3, 0] == 0.0


def test_shadr_parser_rejects_unnormalized_coefficients() -> None:
    text = _pds_header_record(degree=2, order=2, normalization=0)
    with pytest.raises(ValueError, match="normalized"):
        read_shadr(StringIO(text))


def test_shadr_parser_rejects_nonzero_reference_origin() -> None:
    text = _pds_header_record(degree=2, order=2, reference_longitude_deg=1.0)
    with pytest.raises(ValueError, match="reference longitude/latitude"):
        read_shadr(StringIO(text))


def test_shadr_parser_rejects_malformed_coefficient_rows() -> None:
    malformed = "bad row" + " " * 113 + "\r\n"
    text = _pds_header_record(degree=2, order=2) + malformed
    with pytest.raises(ValueError, match="coefficient record"):
        read_shadr(StringIO(text))


def test_shadr_parser_rejects_duplicate_coefficients() -> None:
    row = _pds_coefficient_record(2, 0, -9e-5, 0.0)
    text = _pds_header_record(degree=2, order=2) + row + row
    with pytest.raises(ValueError, match="duplicate"):
        read_shadr(StringIO(text))


def test_model_copies_and_write_protects_coefficients() -> None:
    c = np.zeros((3, 3))
    s = np.zeros_like(c)
    c[0, 0] = 1.0
    model = SphericalHarmonicModel(4.9028e12, 1.738e6, c, s)
    c[2, 0] = 123.0
    assert model.c[2, 0] == 0.0
    with pytest.raises(ValueError):
        model.c[2, 0] = 1.0


def test_degree_1200_acceleration_evaluation_stays_finite_at_poles_and_mid_latitude() -> None:
    n = 1200
    c = np.zeros((n + 1, n + 1))
    s = np.zeros_like(c)
    c[0, 0] = 1.0
    c[2, 0] = -9.0e-5
    c[600, 13] = 1.0e-10
    c[1199, 1] = -2.0e-14
    s[1199, 1] = 1.0e-14
    c[1200, 1] = 1.0e-14
    s[1200, 1200] = 1.0e-12
    model = SphericalHarmonicModel(4.9028e12, 1.738e6, c, s)

    positions = [
        [1.9e6, 2.0e5, 3.0e5],
        [0.0, 0.0, 1.9e6],
        [0.0, 0.0, -1.9e6],
        [1.0e-3, -2.0e-3, 1.9e6],
    ]
    for position in positions:
        acceleration = gravity_acceleration_body_fixed(position, model)
        assert np.all(np.isfinite(acceleration))


def test_shadr_parser_reads_fixed_record_file_from_disk(tmp_path) -> None:
    text = (
        _pds_header_record(degree=2, order=2)
        + _pds_coefficient_record(1, 0, 7.0e-8, 0.0)
        + _pds_coefficient_record(2, 0, -9.0e-5, 0.0)
    )
    path = tmp_path / "fixture_sha.tab"
    path.write_bytes(text.encode("ascii"))
    assert path.read_bytes()[:244].endswith(b"\r\n")
    model = read_shadr(path)
    assert model.c[1, 0] == pytest.approx(7e-8)
    assert model.c[2, 0] == pytest.approx(-9e-5)
