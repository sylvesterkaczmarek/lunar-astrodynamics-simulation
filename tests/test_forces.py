import numpy as np
import pytest

from lunar_astrodynamics.constants import (
    ASTRONOMICAL_UNIT_M,
    MOON_MEAN_RADIUS_M,
    SOLAR_RADIATION_PRESSURE_1_AU_N_M2,
)
from lunar_astrodynamics.forces import (
    CallableForce,
    CompositeForceModel,
    SolarRadiationPressure,
    ThirdBodyGravity,
    apparent_disk_illumination_fraction,
    lunar_eclipse_illumination_fraction,
    third_body_acceleration,
)


def test_composite_force_sums_named_components_and_exposes_contributions() -> None:
    one = CallableForce("one", lambda _t, _r: np.array([1.0, 0.0, 0.0]))
    two = CallableForce("two", lambda _t, _r: np.array([0.0, 2.0, 0.0]))
    model = CompositeForceModel((one, two))
    position = np.array([2.0, 3.0, 4.0])

    contributions = model.component_accelerations(12.0, position)
    assert contributions["one"] == pytest.approx([1.0, 0.0, 0.0])
    assert contributions["two"] == pytest.approx([0.0, 2.0, 0.0])
    assert model(12.0, position) == pytest.approx([1.0, 2.0, 0.0])


def test_composite_force_rejects_duplicate_names() -> None:
    component = CallableForce("duplicate", lambda _t, _r: np.zeros(3))
    with pytest.raises(ValueError, match="unique"):
        CompositeForceModel((component, component))


def test_differential_third_body_acceleration_matches_collinear_analytic_geometry() -> None:
    spacecraft = np.array([1.0, 0.0, 0.0])
    body = np.array([10.0, 0.0, 0.0])
    actual = third_body_acceleration(spacecraft, body, 1.0)
    expected_x = 1.0 / 9.0**2 - 1.0 / 10.0**2
    assert actual == pytest.approx([expected_x, 0.0, 0.0], rel=1e-15, abs=1e-15)


def test_differential_third_body_acceleration_is_zero_at_central_body_origin() -> None:
    body = np.array([3.0e8, -1.0e8, 2.0e8])
    assert third_body_acceleration(np.zeros(3), body, 3.986e14) == pytest.approx(
        np.zeros(3), abs=1e-18
    )


def test_distant_third_body_tidal_acceleration_tends_to_zero_with_inverse_cube_scaling() -> None:
    spacecraft = np.array([2.0e6, 0.0, 0.0])
    mu = 3.986e14
    near = np.linalg.norm(
        third_body_acceleration(spacecraft, np.array([1.0e9, 0.0, 0.0]), mu)
    )
    far = np.linalg.norm(
        third_body_acceleration(spacecraft, np.array([2.0e9, 0.0, 0.0]), mu)
    )
    assert far < near
    assert far / near == pytest.approx(1.0 / 8.0, rel=4e-3)


def test_third_body_component_uses_position_provider() -> None:
    component = ThirdBodyGravity(
        "Earth third body",
        3.986e14,
        lambda time_s: np.array([1.0e9 + time_s, 2.0e8, 0.0]),
    )
    position = np.array([1.8e6, 0.0, 0.0])
    assert component(12.0, position) == pytest.approx(
        third_body_acceleration(position, np.array([1.0e9 + 12.0, 2.0e8, 0.0]), 3.986e14)
    )


def test_apparent_disk_model_full_sun_total_annular_and_partial_cases() -> None:
    sun = 0.01
    body = 0.02
    assert apparent_disk_illumination_fraction(sun, body, 0.04) == pytest.approx(1.0)
    assert apparent_disk_illumination_fraction(sun, body, 0.0) == pytest.approx(0.0)

    annular = apparent_disk_illumination_fraction(0.02, 0.01, 0.0)
    assert annular == pytest.approx(0.75)

    partial = apparent_disk_illumination_fraction(0.01, 0.01, 0.01)
    assert 0.0 < partial < 1.0


def test_apparent_disk_shadow_transition_is_continuous_and_monotonic() -> None:
    sun = 0.005
    body = 0.010
    separations = np.linspace(body - sun, body + sun, 101)
    fractions = np.array(
        [apparent_disk_illumination_fraction(sun, body, value) for value in separations]
    )
    assert fractions[0] == pytest.approx(0.0)
    assert fractions[-1] == pytest.approx(1.0)
    assert np.all(np.diff(fractions) >= -1e-13)
    assert np.max(np.abs(np.diff(fractions))) < 0.03


def test_lunar_eclipse_model_distinguishes_sunward_full_sun_and_antisun_umbra() -> None:
    sun = np.array([ASTRONOMICAL_UNIT_M, 0.0, 0.0])
    sunward = np.array([MOON_MEAN_RADIUS_M + 100_000.0, 0.0, 0.0])
    antisun = -sunward

    assert lunar_eclipse_illumination_fraction(sunward, sun) == pytest.approx(1.0)
    assert lunar_eclipse_illumination_fraction(antisun, sun) == pytest.approx(0.0)


def test_srp_full_sun_magnitude_and_direction_at_one_au() -> None:
    srp = SolarRadiationPressure(
        "SRP",
        lambda _t: np.array([ASTRONOMICAL_UNIT_M, 0.0, 0.0]),
        spacecraft_mass_kg=100.0,
        illuminated_area_m2=10.0,
        reflectivity_coefficient=1.5,
        include_lunar_shadow=False,
    )
    actual = srp(0.0, np.zeros(3))
    expected = SOLAR_RADIATION_PRESSURE_1_AU_N_M2 * 1.5 * 10.0 / 100.0
    assert actual == pytest.approx([-expected, 0.0, 0.0], rel=1e-14, abs=1e-14)


def test_srp_is_zero_in_lunar_umbra() -> None:
    sun = np.array([ASTRONOMICAL_UNIT_M, 0.0, 0.0])
    srp = SolarRadiationPressure(
        "SRP",
        lambda _t: sun,
        spacecraft_mass_kg=200.0,
        illuminated_area_m2=5.0,
        reflectivity_coefficient=1.3,
    )
    antisun = np.array([-(MOON_MEAN_RADIUS_M + 100_000.0), 0.0, 0.0])
    assert srp.illumination_fraction(0.0, antisun) == pytest.approx(0.0)
    assert srp(0.0, antisun) == pytest.approx(np.zeros(3), abs=1e-20)


def test_srp_scales_with_inverse_square_sun_distance() -> None:
    parameters = dict(
        spacecraft_mass_kg=100.0,
        illuminated_area_m2=2.0,
        reflectivity_coefficient=1.0,
        include_lunar_shadow=False,
    )
    near = SolarRadiationPressure(
        "near", lambda _t: np.array([ASTRONOMICAL_UNIT_M, 0.0, 0.0]), **parameters
    )
    far = SolarRadiationPressure(
        "far", lambda _t: np.array([2.0 * ASTRONOMICAL_UNIT_M, 0.0, 0.0]), **parameters
    )
    near_magnitude = np.linalg.norm(near(0.0, np.zeros(3)))
    far_magnitude = np.linalg.norm(far(0.0, np.zeros(3)))
    assert far_magnitude / near_magnitude == pytest.approx(0.25, rel=1e-14)
