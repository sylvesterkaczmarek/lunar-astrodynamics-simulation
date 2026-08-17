import numpy as np
import pytest

from lunar_astrodynamics import (
    GRGM1200A_J2,
    MOON_MEAN_RADIUS_M,
    ClassicalElements,
    RegularLatLonTerrain,
    element_history,
    orbit_history,
    scalar_evolution_statistics,
    state_from_elements,
)

MU = GRGM1200A_J2.mu_m3_s2


def _circular_equatorial_history(radius_m: float, angles_rad: np.ndarray) -> np.ndarray:
    speed = np.sqrt(MU / radius_m)
    states = []
    for angle in angles_rad:
        c = np.cos(angle)
        s = np.sin(angle)
        states.append([radius_m * c, radius_m * s, 0.0, -speed * s, speed * c, 0.0])
    return np.asarray(states, dtype=float).T


def test_orbit_history_handles_exact_circular_equatorial_trajectory() -> None:
    time_s = np.arange(5, dtype=float) * 100.0
    angles = np.linspace(0.0, 1.0, time_s.size)
    radius = MOON_MEAN_RADIUS_M + 100_000.0
    states = _circular_equatorial_history(radius, angles)

    history = orbit_history(time_s, states, MU)
    assert history.eccentricity == pytest.approx(np.zeros(time_s.size), abs=8e-16)
    assert not np.any(history.apsis_defined)
    assert np.all(np.isnan(history.apsidal_direction_change_rad))
    np.testing.assert_allclose(history.orbital_plane_normal, [[0.0] * 5, [0.0] * 5, [1.0] * 5], atol=1e-15)
    assert history.minimum_reference_altitude_m == pytest.approx(100_000.0, abs=1e-9)
    assert history.maximum_reference_altitude_m == pytest.approx(100_000.0, abs=1e-9)
    assert history.statistics.apsidal_direction.defined_fraction == 0.0
    assert history.statistics.orbital_plane_direction.maximum_change_rad == pytest.approx(0.0, abs=1e-15)

    with pytest.raises(ValueError, match="use orbit_history"):
        element_history(time_s, states, MU)


def test_near_circular_polar_history_has_stable_nonsingular_quantities() -> None:
    time_s = np.linspace(0.0, 3600.0, 13)
    states = []
    for anomaly in np.linspace(0.0, 2.0 * np.pi, time_s.size, endpoint=False):
        elements = ClassicalElements(
            semi_major_axis_m=MOON_MEAN_RADIUS_M + 120_000.0,
            eccentricity=1.0e-8,
            inclination_rad=np.deg2rad(90.0),
            raan_rad=np.deg2rad(40.0),
            argument_of_periapsis_rad=np.deg2rad(110.0),
            true_anomaly_rad=float(anomaly),
        )
        states.append(state_from_elements(elements, MU))
    history = orbit_history(time_s, np.asarray(states).T, MU)

    assert np.all(history.apsis_defined)
    assert history.statistics.eccentricity.peak_to_peak < 2e-15
    assert history.statistics.modified_equinoctial_f.peak_to_peak < 2e-15
    assert history.statistics.modified_equinoctial_g.peak_to_peak < 2e-15
    assert history.statistics.modified_equinoctial_h.peak_to_peak < 2e-15
    assert history.statistics.modified_equinoctial_k.peak_to_peak < 2e-15
    assert history.statistics.orbital_plane_direction.maximum_change_rad < 2e-8
    assert history.statistics.apsidal_direction.maximum_change_rad < 2e-8
    assert history.modified_equinoctial_true_longitude_rad_unwrapped[-1] > history.modified_equinoctial_true_longitude_rad_unwrapped[0]


def test_scalar_evolution_statistics_separate_drift_from_bounded_oscillation() -> None:
    time_s = np.linspace(0.0, 100.0, 1001)
    values = 4.0 + 0.02 * time_s + 0.3 * np.sin(2.0 * np.pi * time_s / 10.0)
    stats = scalar_evolution_statistics(time_s, values)
    assert stats.linear_rate_per_s == pytest.approx(0.02, abs=2e-5)
    assert stats.linear_drift_over_span == pytest.approx(2.0, abs=2e-3)
    assert stats.detrended_peak_to_peak == pytest.approx(0.6, abs=5e-3)
    assert 0.20 < stats.detrended_rms < 0.22


def test_orbit_history_distinguishes_reference_altitude_from_terrain_clearance() -> None:
    terrain = RegularLatLonTerrain(
        latitude_deg=np.array([-90.0, 0.0, 90.0]),
        longitude_deg_east=np.array([0.0, 180.0, 360.0]),
        elevation_grid_m=np.full((3, 3), 1_000.0),
        reference_radius_m=MOON_MEAN_RADIUS_M,
        frame="TEST_BODY",
    )
    time_s = np.array([0.0, 100.0, 200.0])
    radius = MOON_MEAN_RADIUS_M + 10_000.0
    states = _circular_equatorial_history(radius, np.array([0.0, 0.2, 0.4]))
    history = orbit_history(
        time_s,
        states,
        MU,
        terrain=terrain,
        terrain_body_fixed_from_inertial=lambda _time_s: np.eye(3),
        terrain_frame="TEST_BODY",
    )
    np.testing.assert_allclose(history.reference_radius_altitude_m, 10_000.0, atol=1e-8)
    assert history.terrain_clearance_m is not None
    np.testing.assert_allclose(history.terrain_clearance_m, 9_000.0, atol=1e-8)
    assert history.minimum_terrain_clearance_m == pytest.approx(9_000.0, abs=1e-8)
    assert history.statistics.terrain_clearance_m is not None
    assert history.statistics.terrain_clearance_m.mean == pytest.approx(9_000.0, abs=1e-8)


def test_orbit_history_reports_osculating_apsides() -> None:
    time_s = np.array([0.0, 100.0, 200.0])
    states = []
    elements = ClassicalElements(
        semi_major_axis_m=1_900_000.0,
        eccentricity=0.1,
        inclination_rad=np.deg2rad(60.0),
        raan_rad=np.deg2rad(20.0),
        argument_of_periapsis_rad=np.deg2rad(30.0),
        true_anomaly_rad=0.0,
    )
    for anomaly in (0.0, 1.0, 2.0):
        sample = ClassicalElements(
            elements.semi_major_axis_m,
            elements.eccentricity,
            elements.inclination_rad,
            elements.raan_rad,
            elements.argument_of_periapsis_rad,
            anomaly,
        )
        states.append(state_from_elements(sample, MU))
    history = orbit_history(time_s, np.asarray(states).T, MU)
    expected_periselene = elements.semi_major_axis_m * (1.0 - elements.eccentricity)
    expected_aposelene = elements.semi_major_axis_m * (1.0 + elements.eccentricity)
    np.testing.assert_allclose(history.periselene_radius_m, expected_periselene, rtol=2e-15)
    np.testing.assert_allclose(history.aposelene_radius_m, expected_aposelene, rtol=2e-15)
    np.testing.assert_allclose(
        history.periselene_altitude_m,
        expected_periselene - MOON_MEAN_RADIUS_M,
        atol=1e-8,
    )
    np.testing.assert_allclose(
        history.aposelene_altitude_m,
        expected_aposelene - MOON_MEAN_RADIUS_M,
        atol=1e-8,
    )
