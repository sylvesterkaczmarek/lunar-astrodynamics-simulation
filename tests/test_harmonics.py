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
