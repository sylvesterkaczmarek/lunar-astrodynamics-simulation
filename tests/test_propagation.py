import numpy as np
import pytest

from lunar_astrodynamics import (
    GRGM1200A_J2,
    ClassicalElements,
    analytical_j2_secular_rates,
    element_history,
    linear_rate,
    orbital_period_s,
    PropagationSettings,
    propagate,
    state_from_elements,
)


def _validation_orbit() -> ClassicalElements:
    return ClassicalElements(
        semi_major_axis_m=GRGM1200A_J2.collision_radius_m + 150_000.0,
        eccentricity=0.02,
        inclination_rad=np.deg2rad(45.0),
        raan_rad=np.deg2rad(30.0),
        argument_of_periapsis_rad=np.deg2rad(40.0),
        true_anomaly_rad=0.0,
    )


def test_two_body_conserves_energy_and_angular_momentum() -> None:
    elements = _validation_orbit()
    state0 = state_from_elements(elements, GRGM1200A_J2.mu_m3_s2)
    period = orbital_period_s(elements.semi_major_axis_m, GRGM1200A_J2.mu_m3_s2)
    times = np.linspace(0.0, 10.0 * period, 1001)
    solution = propagate(state0, times[-1], include_j2=False, sample_times_s=times)
    assert solution.success

    r = solution.y[:3].T
    v = solution.y[3:].T
    energy = 0.5 * np.sum(v * v, axis=1) - GRGM1200A_J2.mu_m3_s2 / np.linalg.norm(r, axis=1)
    h = np.cross(r, v)

    energy_rel_span = (energy.max() - energy.min()) / abs(energy[0])
    h_rel_span = np.max(np.linalg.norm(h - h[0], axis=1)) / np.linalg.norm(h[0])
    assert energy_rel_span < 1e-10
    assert h_rel_span < 1e-10


def test_j2_preserves_axial_angular_momentum() -> None:
    elements = _validation_orbit()
    state0 = state_from_elements(elements, GRGM1200A_J2.mu_m3_s2)
    period = orbital_period_s(elements.semi_major_axis_m, GRGM1200A_J2.mu_m3_s2)
    times = np.linspace(0.0, 12.0 * period, 1201)
    solution = propagate(state0, times[-1], sample_times_s=times)
    assert solution.success

    h_z = np.cross(solution.y[:3].T, solution.y[3:].T)[:, 2]
    relative_span = (h_z.max() - h_z.min()) / abs(h_z[0])
    assert relative_span < 1e-10


def test_j2_secular_rates_match_first_order_theory() -> None:
    elements = _validation_orbit()
    state0 = state_from_elements(elements, GRGM1200A_J2.mu_m3_s2)
    period = orbital_period_s(elements.semi_major_axis_m, GRGM1200A_J2.mu_m3_s2)
    times = np.linspace(0.0, 40.0 * period, 3201)
    solution = propagate(state0, times[-1], sample_times_s=times)
    assert solution.success

    history = element_history(solution.t, solution.y, GRGM1200A_J2.mu_m3_s2)
    numerical_raan = linear_rate(history.time_s, history.raan_rad_unwrapped)
    numerical_argp = linear_rate(history.time_s, history.argument_of_periapsis_rad_unwrapped)
    analytical_raan, analytical_argp = analytical_j2_secular_rates(
        elements,
        GRGM1200A_J2.mu_m3_s2,
        GRGM1200A_J2.reference_radius_m,
        GRGM1200A_J2.j2,
    )

    assert numerical_raan == pytest.approx(analytical_raan, rel=0.01)
    assert numerical_argp == pytest.approx(analytical_argp, rel=0.02)


def test_initial_state_below_surface_is_rejected() -> None:
    state = np.array([GRGM1200A_J2.collision_radius_m - 1.0, 0.0, 0.0, 0.0, 100.0, 0.0])
    with pytest.raises(ValueError, match="surface"):
        propagate(state, 100.0)


def test_surface_impact_event_terminates_inbound_trajectory() -> None:
    radius = GRGM1200A_J2.collision_radius_m + 1_000.0
    state = np.array([radius, 0.0, 0.0, -10.0, 0.0, 0.0])
    solution = propagate(state, 300.0)
    assert solution.success
    assert solution.t_events[0].size == 1
    impact_state = solution.y_events[0][0]
    assert np.linalg.norm(impact_state[:3]) == pytest.approx(
        GRGM1200A_J2.collision_radius_m, abs=1e-3
    )


def test_default_integration_converges_to_tighter_reference() -> None:
    elements = _validation_orbit()
    state0 = state_from_elements(elements, GRGM1200A_J2.mu_m3_s2)
    period = orbital_period_s(elements.semi_major_axis_m, GRGM1200A_J2.mu_m3_s2)
    duration = 5.0 * period
    times = np.array([0.0, duration])

    default = propagate(state0, duration, sample_times_s=times)
    tighter = propagate(
        state0,
        duration,
        sample_times_s=times,
        settings=PropagationSettings(
            rtol=1e-13,
            position_atol_m=1e-9,
            velocity_atol_m_s=1e-12,
        ),
    )

    position_difference_m = np.linalg.norm(default.y[:3, -1] - tighter.y[:3, -1])
    velocity_difference_m_s = np.linalg.norm(default.y[3:, -1] - tighter.y[3:, -1])
    assert position_difference_m < 1e-3
    assert velocity_difference_m_s < 1e-6
