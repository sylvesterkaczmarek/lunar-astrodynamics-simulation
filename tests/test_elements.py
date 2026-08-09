import numpy as np
import pytest

from lunar_astrodynamics import GRGM1200A_J2, ClassicalElements, elements_from_state, state_from_elements


def test_element_state_round_trip() -> None:
    original = ClassicalElements(
        semi_major_axis_m=1_900_000.0,
        eccentricity=0.03,
        inclination_rad=np.deg2rad(52.0),
        raan_rad=np.deg2rad(21.0),
        argument_of_periapsis_rad=np.deg2rad(67.0),
        true_anomaly_rad=np.deg2rad(123.0),
    )
    state = state_from_elements(original, GRGM1200A_J2.mu_m3_s2)
    recovered = elements_from_state(state, GRGM1200A_J2.mu_m3_s2)
    assert recovered.semi_major_axis_m == pytest.approx(original.semi_major_axis_m, rel=1e-12)
    assert recovered.eccentricity == pytest.approx(original.eccentricity, abs=1e-12)
    assert recovered.inclination_rad == pytest.approx(original.inclination_rad, abs=1e-12)
    assert recovered.raan_rad == pytest.approx(original.raan_rad, abs=1e-12)
    assert recovered.argument_of_periapsis_rad == pytest.approx(original.argument_of_periapsis_rad, abs=1e-12)
    assert recovered.true_anomaly_rad == pytest.approx(original.true_anomaly_rad, abs=1e-12)


def test_singular_equatorial_orbit_is_rejected() -> None:
    state = np.array([1.9e6, 0.0, 0.0, 0.0, 1600.0, 0.0])
    with pytest.raises(ValueError, match="RAAN is undefined"):
        elements_from_state(state, GRGM1200A_J2.mu_m3_s2)


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
        ClassicalElements(
            semi_major_axis_m=1_900_000.0,
            eccentricity=0.03,
            inclination_rad=np.deg2rad(45.0),
            raan_rad=np.inf,
            argument_of_periapsis_rad=0.0,
            true_anomaly_rad=0.0,
        )


def test_nonfinite_mu_is_rejected() -> None:
    elements = ClassicalElements(
        semi_major_axis_m=1_900_000.0,
        eccentricity=0.03,
        inclination_rad=np.deg2rad(45.0),
        raan_rad=0.0,
        argument_of_periapsis_rad=0.0,
        true_anomaly_rad=0.0,
    )
    with pytest.raises(ValueError, match="finite and positive"):
        state_from_elements(elements, np.nan)
