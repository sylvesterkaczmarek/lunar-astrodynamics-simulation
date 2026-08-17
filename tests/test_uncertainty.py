from io import StringIO

import numpy as np
import pytest

from lunar_astrodynamics import (
    GRGM1200A_CLONE_COUNT,
    GRGM1200A_CLONE_EXPECTED_SIZE_BYTES,
    MOON_MEAN_RADIUS_M,
    OrbitUncertaintySample,
    PropagationSettings,
    SphericalHarmonicModel,
    grgm1200a_clone_url,
    propagate_gravity_ensemble,
    read_grgm1200a_clone,
    read_shadr,
    sample_independent_coefficient_uncertainty,
    summarize_ensemble,
)


def _pds_header_record(degree: int = 3, order: int = 3, mu_sigma: float = 2.5e-8) -> str:
    data = ",".join(
        [
            f"{1738.0:23.16E}",
            f"{4902.80011526323:23.16E}",
            f"{mu_sigma:23.16E}",
            f"{degree:5d}",
            f"{order:5d}",
            f"{1:5d}",
            f"{0.0:23.16E}",
            f"{0.0:23.16E}",
        ]
    )
    assert len(data) == 137
    return data + " " * 105 + "\r\n"


def _pds_coefficient_record(
    n: int,
    m: int,
    c: float,
    s: float,
    sigma_c: float,
    sigma_s: float,
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
    return data + " " * 13 + "\r\n"


def _uncertain_model() -> SphericalHarmonicModel:
    c = np.zeros((4, 4))
    s = np.zeros_like(c)
    sigma_c = np.zeros_like(c)
    sigma_s = np.zeros_like(c)
    c[0, 0] = 1.0
    c[2, 0] = -9.0e-5
    c[2, 2] = 2.2e-5
    s[2, 2] = -1.7e-5
    c[3, 1] = 8.0e-6
    s[3, 1] = 5.0e-6
    sigma_c[2, 0] = 2.0e-7
    sigma_c[2, 2] = 3.0e-7
    sigma_s[2, 2] = 4.0e-7
    sigma_c[3, 1] = 1.0e-7
    sigma_s[3, 1] = 1.5e-7
    return SphericalHarmonicModel(
        4.90280011526323e12,
        1.738e6,
        c,
        s,
        name="synthetic uncertain field",
        sigma_c=sigma_c,
        sigma_s=sigma_s,
        mu_sigma_m3_s2=25.0,
    )


def _clone_fixture(*, incomplete: bool = False) -> StringIO:
    rows = [
        "    1,    0, 1.0000000000000000E-08, 0.0000000000000000E+00",
        "    1,    1, 2.0000000000000000E-08,-1.0000000000000000E-08",
        "    2,    0,-9.0000000000000000E-05, 0.0000000000000000E+00",
        "    2,    1, 3.0000000000000000E-07, 4.0000000000000000E-07",
        "    2,    2, 2.2000000000000000E-05,-1.7000000000000000E-05",
    ]
    if incomplete:
        rows.pop()
    return StringIO("clone realization\n" + "\n".join(rows) + "\n")


def test_model_without_uncertainties_remains_backward_compatible() -> None:
    c = np.zeros((2, 2))
    s = np.zeros_like(c)
    c[0, 0] = 1.0
    model = SphericalHarmonicModel(4.9028e12, 1.738e6, c, s)
    assert not model.has_coefficient_uncertainty
    assert model.mu_sigma_m3_s2 is None
    with pytest.raises(ValueError, match="does not include coefficient uncertainties"):
        model.coefficient_uncertainty(1, 0)


def test_model_validates_and_protects_uncertainty_arrays() -> None:
    model = _uncertain_model()
    assert model.coefficient_uncertainty(2, 2) == pytest.approx((3.0e-7, 4.0e-7))
    with pytest.raises(ValueError):
        model.sigma_c[2, 2] = 0.0  # type: ignore[index]

    c = np.zeros((2, 2))
    s = np.zeros_like(c)
    c[0, 0] = 1.0
    with pytest.raises(ValueError, match="both be provided"):
        SphericalHarmonicModel(4.9028e12, 1.738e6, c, s, sigma_c=np.zeros_like(c))
    with pytest.raises(ValueError, match="non-negative"):
        SphericalHarmonicModel(
            4.9028e12,
            1.738e6,
            c,
            s,
            sigma_c=np.array([[0.0, 0.0], [-1.0, 0.0]]),
            sigma_s=np.zeros_like(c),
        )


def test_shadr_parser_preserves_coefficient_and_gm_uncertainties() -> None:
    text = (
        _pds_header_record(mu_sigma=2.5e-8)
        + _pds_coefficient_record(1, 0, 7.0e-8, 0.0, 1.1e-12, 0.0)
        + _pds_coefficient_record(2, 0, -9.0e-5, 0.0, 2.2e-12, 0.0)
        + _pds_coefficient_record(2, 2, 2.0e-5, -3.0e-5, 3.3e-12, 4.4e-12)
        + _pds_coefficient_record(3, 1, 4.0e-6, 5.0e-6, 5.5e-12, 6.6e-12)
    )
    model = read_shadr(StringIO(text), max_degree=2, name="uncertain synthetic SHADR")
    assert model.has_coefficient_uncertainty
    assert model.coefficient_uncertainty(2, 2) == pytest.approx((3.3e-12, 4.4e-12))
    assert model.sigma_c[1, 0] == pytest.approx(1.1e-12)  # type: ignore[index]
    assert model.mu_sigma_m3_s2 == pytest.approx(25.0)


def test_truncation_preserves_uncertainties() -> None:
    model = _uncertain_model()
    truncated = model.truncated(2, 1)
    assert truncated.has_coefficient_uncertainty
    assert truncated.mu_sigma_m3_s2 == pytest.approx(model.mu_sigma_m3_s2)
    assert truncated.sigma_c[2, 0] == pytest.approx(model.sigma_c[2, 0])  # type: ignore[index]
    assert truncated.sigma_c[2, 2] == 0.0  # type: ignore[index]
    assert truncated.sigma_s[2, 2] == 0.0  # type: ignore[index]


def test_independent_sigma_sampling_is_explicit_and_reproducible() -> None:
    model = _uncertain_model()
    with pytest.raises(ValueError, match="assume_independent=True"):
        sample_independent_coefficient_uncertainty(model, seed=1234, count=2)

    first = sample_independent_coefficient_uncertainty(
        model, seed=1234, count=3, assume_independent=True, include_mu=True
    )
    second = sample_independent_coefficient_uncertainty(
        model, seed=1234, count=3, assume_independent=True, include_mu=True
    )
    other = sample_independent_coefficient_uncertainty(
        model, seed=1235, count=1, assume_independent=True
    )
    for left, right in zip(first, second, strict=True):
        assert left.c == pytest.approx(right.c)
        assert left.s == pytest.approx(right.s)
        assert left.mu_m3_s2 == pytest.approx(right.mu_m3_s2)
        assert left.c[0, 0] == 1.0
        assert left.s[0, 0] == 0.0
    assert not np.array_equal(first[0].c, other[0].c)

    zero = sample_independent_coefficient_uncertainty(
        model, seed=1, sigma_scale=0.0, assume_independent=True
    )[0]
    assert zero.c == pytest.approx(model.c)
    assert zero.s == pytest.approx(model.s)


def test_clone_url_matches_archived_pds_grouping() -> None:
    assert GRGM1200A_CLONE_COUNT == 500
    assert GRGM1200A_CLONE_EXPECTED_SIZE_BYTES == 44_029_817
    assert grgm1200a_clone_url(1).endswith(
        "/gggrx_1200a_clones_0001_0100/gggrx_1200a_clone0001_sha.tab"
    )
    assert grgm1200a_clone_url(101).endswith(
        "/gggrx_1200a_clones_0101_0200/gggrx_1200a_clone0101_sha.tab"
    )
    assert grgm1200a_clone_url(500).endswith(
        "/gggrx_1200a_clones_0401_0500/gggrx_1200a_clone0500_sha.tab"
    )
    with pytest.raises(ValueError, match="within"):
        grgm1200a_clone_url(0)


def test_clone_parser_loads_complete_correlated_realization() -> None:
    model = read_grgm1200a_clone(_clone_fixture(), max_degree=2, name="clone test")
    assert model.max_degree == 2
    assert model.c[1, 1] == pytest.approx(2.0e-8)
    assert model.s[2, 1] == pytest.approx(4.0e-7)
    assert model.c[2, 2] == pytest.approx(2.2e-5)
    assert model.s[2, 2] == pytest.approx(-1.7e-5)
    assert not model.has_coefficient_uncertainty


def test_clone_parser_rejects_incomplete_realization() -> None:
    with pytest.raises(ValueError, match="incomplete through degree 2"):
        read_grgm1200a_clone(_clone_fixture(incomplete=True), max_degree=2)


def test_ensemble_summary_returns_percentiles_and_impact_fraction() -> None:
    samples = (
        OrbitUncertaintySample("a", 10.0, 100.0, 5.0, 110.0, 0.01, 0.009, 1000.0, False),
        OrbitUncertaintySample("b", 20.0, 120.0, 8.0, 130.0, 0.02, 0.019, 800.0, True),
        OrbitUncertaintySample("c", 30.0, 140.0, 9.0, 150.0, 0.03, 0.029, 1000.0, False),
    )
    result = summarize_ensemble(samples, percentile_levels=(0.0, 50.0, 100.0))
    assert result.impact_fraction == pytest.approx(1.0 / 3.0)
    assert result.percentiles["minimum_altitude_m"] == pytest.approx(
        {0.0: 10.0, 50.0: 20.0, 100.0: 30.0}
    )
    assert result.percentiles["lifetime_s"][50.0] == pytest.approx(1000.0)
    assert result.metric_values("maximum_eccentricity") == pytest.approx([0.01, 0.02, 0.03])


def test_gravity_ensemble_propagates_same_initial_state_through_realizations() -> None:
    mu = 4.90280011526323e12
    radius = MOON_MEAN_RADIUS_M + 100_000.0
    speed = np.sqrt(mu / radius)
    initial_state = np.array([radius, 0.0, 0.0, 0.0, speed, 0.0])
    period = 2.0 * np.pi * np.sqrt(radius**3 / mu)

    models = []
    for index, c20 in enumerate((-8.8e-5, -9.2e-5), start=1):
        c = np.zeros((3, 3))
        s = np.zeros_like(c)
        c[0, 0] = 1.0
        c[2, 0] = c20
        models.append(SphericalHarmonicModel(mu, 1.738e6, c, s, name=f"realization {index}"))

    result = propagate_gravity_ensemble(
        initial_state,
        0.25 * period,
        models,
        lambda _time_s: np.eye(3),
        reference_radius_m=MOON_MEAN_RADIUS_M,
        max_degree=2,
        sample_count=33,
        settings=PropagationSettings(
            rtol=1e-9,
            position_atol_m=1e-3,
            velocity_atol_m_s=1e-6,
            max_step_s=60.0,
        ),
    )
    assert len(result.samples) == 2
    assert result.impact_fraction == 0.0
    assert all(sample.lifetime_s == pytest.approx(0.25 * period) for sample in result.samples)
    assert all(np.isfinite(sample.minimum_altitude_m) for sample in result.samples)
    assert all(np.isfinite(sample.maximum_eccentricity) for sample in result.samples)
    assert set(result.percentiles["minimum_altitude_m"]) == {5.0, 50.0, 95.0}
