import numpy as np
import pytest

from lunar_astrodynamics import (
    GRGM1200A_J2,
    ClassicalElements,
    ModifiedEquinoctialElements,
    classical_from_modified_equinoctial,
    elements_from_state,
    modified_equinoctial_from_classical,
    modified_equinoctial_from_state,
    orbital_vectors_from_state,
    state_from_elements,
    state_from_modified_equinoctial,
)

MU = GRGM1200A_J2.mu_m3_s2


def _classical(eccentricity: float, inclination_deg: float) -> ClassicalElements:
    return ClassicalElements(
        semi_major_axis_m=1_900_000.0,
        eccentricity=eccentricity,
        inclination_rad=np.deg2rad(inclination_deg),
        raan_rad=np.deg2rad(123.0),
        argument_of_periapsis_rad=np.deg2rad(71.0),
        true_anomaly_rad=np.deg2rad(231.0),
    )


def test_element_state_round_trip() -> None:
    original = ClassicalElements(
        semi_major_axis_m=1_900_000.0,
        eccentricity=0.03,
        inclination_rad=np.deg2rad(52.0),
        raan_rad=np.deg2rad(21.0),
        argument_of_periapsis_rad=np.deg2rad(67.0),
        true_anomaly_rad=np.deg2rad(123.0),
    )
    state = state_from_elements(original, MU)
    recovered = elements_from_state(state, MU)
    assert recovered.semi_major_axis_m == pytest.approx(original.semi_major_axis_m, rel=1e-12)
    assert recovered.eccentricity == pytest.approx(original.eccentricity, abs=1e-12)
    assert recovered.inclination_rad == pytest.approx(original.inclination_rad, abs=1e-12)
    assert recovered.raan_rad == pytest.approx(original.raan_rad, abs=1e-12)
    assert recovered.argument_of_periapsis_rad == pytest.approx(original.argument_of_periapsis_rad, abs=1e-12)
    assert recovered.true_anomaly_rad == pytest.approx(original.true_anomaly_rad, abs=1e-12)


@pytest.mark.parametrize(
    ("eccentricity", "inclination_deg"),
    [
        (0.0, 0.0),
        (1.0e-12, 1.0e-6),
        (1.0e-8, 10.0),
        (1.0e-4, 89.999999),
        (0.01, 90.0),
        (0.2, 120.0),
        (0.7, 179.0),
    ],
)
def test_modified_equinoctial_cartesian_round_trip_across_orbit_regimes(
    eccentricity: float,
    inclination_deg: float,
) -> None:
    state = state_from_elements(_classical(eccentricity, inclination_deg), MU)
    equinoctial = modified_equinoctial_from_state(state, MU)
    recovered = state_from_modified_equinoctial(equinoctial, MU)
    np.testing.assert_allclose(recovered[:3], state[:3], rtol=2e-12, atol=5e-7)
    np.testing.assert_allclose(recovered[3:], state[3:], rtol=2e-12, atol=5e-10)


def test_modified_equinoctial_matches_standard_classical_definition() -> None:
    classical = _classical(0.13, 52.0)
    state = state_from_elements(classical, MU)
    actual = modified_equinoctial_from_state(state, MU)
    expected = modified_equinoctial_from_classical(classical)
    assert actual.semilatus_rectum_m == pytest.approx(expected.semilatus_rectum_m, rel=2e-15)
    assert actual.f == pytest.approx(expected.f, abs=2e-15)
    assert actual.g == pytest.approx(expected.g, abs=2e-15)
    assert actual.h == pytest.approx(expected.h, abs=2e-15)
    assert actual.k == pytest.approx(expected.k, abs=2e-15)
    assert actual.true_longitude_rad == pytest.approx(expected.true_longitude_rad, abs=2e-15)


def test_orbital_vectors_remain_defined_for_circular_equatorial_orbit() -> None:
    radius = 1_900_000.0
    speed = np.sqrt(MU / radius)
    state = np.array([radius, 0.0, 0.0, 0.0, speed, 0.0])
    vectors = orbital_vectors_from_state(state, MU)
    assert vectors.eccentricity == pytest.approx(0.0, abs=5e-16)
    np.testing.assert_allclose(vectors.eccentricity_vector, np.zeros(3), atol=5e-16)
    np.testing.assert_allclose(vectors.orbital_plane_normal, [0.0, 0.0, 1.0], atol=1e-15)
    assert vectors.semi_major_axis_m == pytest.approx(radius, rel=2e-15)
    assert vectors.periselene_radius_m == pytest.approx(radius, rel=2e-15)
    assert vectors.aposelene_radius_m == pytest.approx(radius, rel=2e-15)
    assert vectors.inclination_rad == pytest.approx(0.0, abs=1e-15)


def test_modified_equinoctial_circular_equatorial_state_is_well_defined() -> None:
    elements = ModifiedEquinoctialElements(
        semilatus_rectum_m=1_900_000.0,
        f=0.0,
        g=0.0,
        h=0.0,
        k=0.0,
        true_longitude_rad=np.deg2rad(73.0),
    )
    state = state_from_modified_equinoctial(elements, MU)
    recovered = modified_equinoctial_from_state(state, MU)
    assert recovered.eccentricity == pytest.approx(0.0, abs=5e-16)
    assert recovered.inclination_rad == pytest.approx(0.0, abs=5e-16)
    assert recovered.true_longitude_rad == pytest.approx(elements.true_longitude_rad, abs=5e-15)


def test_classical_angles_are_not_invented_for_singular_orbits() -> None:
    equatorial_eccentric = state_from_elements(_classical(0.03, 0.0), MU)
    with pytest.raises(ValueError, match="RAAN is undefined"):
        elements_from_state(equatorial_eccentric, MU)

    circular_inclined = state_from_elements(_classical(0.0, 45.0), MU)
    with pytest.raises(ValueError, match="argument of periapsis is undefined"):
        elements_from_state(circular_inclined, MU)

    circular_mee = modified_equinoctial_from_state(circular_inclined, MU)
    with pytest.raises(ValueError, match="argument of periapsis is undefined"):
        classical_from_modified_equinoctial(circular_mee)


def test_prograde_modified_equinoctial_reports_retrograde_equatorial_singularity() -> None:
    state = state_from_elements(_classical(0.1, 180.0), MU)
    with pytest.raises(ValueError, match="retrograde-equatorial"):
        modified_equinoctial_from_state(state, MU)


def test_modified_equinoctial_exposes_angle_definedness() -> None:
    circular_equatorial = ModifiedEquinoctialElements(1.9e6, 0.0, 0.0, 0.0, 0.0, 1.2)
    assert circular_equatorial.longitude_of_periapsis_rad is None
    assert circular_equatorial.raan_rad is None

    nonsingular = modified_equinoctial_from_classical(_classical(0.03, 45.0))
    assert nonsingular.longitude_of_periapsis_rad is not None
    assert nonsingular.raan_rad is not None


def test_nonfinite_orbital_elements_are_rejected() -> None:
    with pytest.raises(ValueError, match="finite"):
        ClassicalElements(
            semi_major_axis_m=1_900_000.0,
            eccentricity=0.03,
            inclination_rad=np.nan,
            raan_rad=0.0,
            argument_of_periapsis_rad=0.0,
            true_anomaly_rad=0.0,
        )
    with pytest.raises(ValueError, match="finite"):
        ModifiedEquinoctialElements(1.9e6, 0.0, 0.0, np.inf, 0.0, 0.0)


def test_nonfinite_mu_is_rejected() -> None:
    elements = _classical(0.03, 45.0)
    with pytest.raises(ValueError, match="finite and positive"):
        state_from_elements(elements, np.nan)
    with pytest.raises(ValueError, match="finite and positive"):
        modified_equinoctial_from_state(state_from_elements(elements, MU), np.nan)
