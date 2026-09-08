import numpy as np
import pytest

from lunar_astrodynamics.constants import GRGM1200A_J2, MOON_MEAN_RADIUS_M
from lunar_astrodynamics.elements import ClassicalElements, state_from_elements
from lunar_astrodynamics.propagation import PropagationSettings
from lunar_astrodynamics.stability import OrbitSearchPoint, SearchDynamics, j2_search_dynamics
from lunar_astrodynamics.targeting import (
    CorrectorVariable,
    DifferentialCorrectionSettings,
    FiniteDifferenceSettings,
    OrbitTargetSpecification,
    StationKeepingPolicy,
    TerminalStateTarget,
    differential_correct,
    evaluate_orbit_target,
    finite_difference_jacobian,
    finite_difference_state_transition,
    orbit_parameter_sensitivity,
    simulate_impulsive_stationkeeping,
    target_initial_state,
    target_orbit_parameters,
)

MU = GRGM1200A_J2.mu_m3_s2
R = MOON_MEAN_RADIUS_M


def _zero_dynamics() -> SearchDynamics:
    return SearchDynamics(
        name="analytic zero-acceleration dynamics",
        mu_m3_s2=MU,
        analysis_reference_radius_m=1.0,
        collision_radius_m=1.0,
        acceleration=lambda _time_s, _position_m: np.zeros(3),
        fidelity="analytic test",
        parallel_safe=True,
    )


def _central_dynamics() -> SearchDynamics:
    return j2_search_dynamics(include_j2=False)


def _propagation() -> PropagationSettings:
    return PropagationSettings(
        rtol=1e-11,
        position_atol_m=1e-5,
        velocity_atol_m_s=1e-8,
        max_step_s=30.0,
    )


def _point(a_altitude_m: float = 100_000.0, eccentricity: float = 0.01) -> OrbitSearchPoint:
    return OrbitSearchPoint(
        semi_major_axis_m=R + a_altitude_m,
        semi_major_axis_altitude_m=a_altitude_m,
        eccentricity=eccentricity,
        inclination_rad=np.deg2rad(88.0),
        raan_rad=np.deg2rad(20.0),
        periapsis_parameter_rad=np.deg2rad(270.0),
        initial_anomaly_rad=np.deg2rad(15.0),
        periapsis_parameterization="longitude_of_periapsis",
    )


def test_zero_acceleration_state_transition_matches_analytic_matrix() -> None:
    dynamics = _zero_dynamics()
    duration = 120.0
    initial = np.array([10_000_000.0, 2_000_000.0, -1_000_000.0, 100.0, -20.0, 5.0])
    result = finite_difference_state_transition(
        initial,
        duration,
        dynamics,
        position_step_m=2.0,
        velocity_step_m_s=2e-3,
        propagation=PropagationSettings(rtol=1e-12, max_step_s=10.0),
        finite_difference=FiniteDifferenceSettings(max_relative_disagreement=1e-6),
    )
    expected = np.eye(6)
    expected[:3, 3:] = duration * np.eye(3)
    np.testing.assert_allclose(result.state_transition_matrix, expected, rtol=2e-7, atol=2e-7)
    assert result.all_columns_stable
    assert all(item.selected_pair_relative_disagreement < 1e-6 for item in result.diagnostics)


def test_two_body_state_transition_step_sweep_is_stable() -> None:
    dynamics = _central_dynamics()
    point = _point(eccentricity=0.005)
    state = point.initial_state(MU)
    result = finite_difference_state_transition(
        state,
        900.0,
        dynamics,
        position_step_m=1.0,
        velocity_step_m_s=1e-3,
        propagation=_propagation(),
        finite_difference=FiniteDifferenceSettings(max_relative_disagreement=0.02),
    )
    assert result.all_columns_stable
    assert result.state_transition_matrix.shape == (6, 6)
    assert np.all(np.isfinite(result.state_transition_matrix))


def test_differential_corrector_solves_analytic_straight_line_target() -> None:
    dynamics = _zero_dynamics()
    initial = np.array([10_000_000.0, 0.0, 0.0, 10.0, 0.0, 0.0])
    result = target_initial_state(
        initial,
        100.0,
        dynamics,
        TerminalStateTarget(indices=(1,), desired_values=(250.0,), scales=(1.0,)),
        decision_indices=(4,),
        propagation=PropagationSettings(rtol=1e-12, max_step_s=10.0),
        correction=DifferentialCorrectionSettings(
            max_iterations=4,
            residual_tolerance=1e-8,
            finite_difference=FiniteDifferenceSettings(max_relative_disagreement=1e-6),
        ),
    )
    assert result.converged
    assert result.corrected_initial_state[4] == pytest.approx(2.5, abs=1e-9)
    assert result.corrected_final_state is not None
    assert result.corrected_final_state[1] == pytest.approx(250.0, abs=1e-6)
    assert result.correction.final_residual_norm < 1e-8


def test_differential_corrector_fails_cleanly_for_zero_rank_jacobian() -> None:
    definition = (CorrectorVariable("x", 1e-3, 1.0),)
    result = differential_correct(
        np.array([0.0]),
        lambda _value: np.array([1.0]),
        definition,
        settings=DifferentialCorrectionSettings(max_iterations=2),
    )
    assert not result.converged
    assert "insensitive" in result.reason
    assert result.final_residual_norm == pytest.approx(1.0)
    assert result.iterations[-1].jacobian_rank == 0
    assert not result.iterations[-1].accepted


def test_two_body_orbit_parameter_sensitivity_matches_analytic_apsis_derivatives() -> None:
    dynamics = _central_dynamics()
    point = _point(a_altitude_m=120_000.0, eccentricity=0.02)
    variables = (
        CorrectorVariable("semi_major_axis_m", 10.0, 10_000.0),
        CorrectorVariable("eccentricity", 1e-6, 0.01),
    )
    result = orbit_parameter_sensitivity(
        point,
        600.0,
        dynamics,
        variables=variables,
        outputs=("final_periselene_altitude_m", "final_aposelene_altitude_m"),
        sample_count=33,
        propagation=_propagation(),
        finite_difference=FiniteDifferenceSettings(max_relative_disagreement=1e-4),
    )
    a = point.semi_major_axis_m
    e = point.eccentricity
    expected = np.array([[1.0 - e, -a], [1.0 + e, a]])
    np.testing.assert_allclose(result.jacobian, expected, rtol=2e-5, atol=2e-3)
    assert result.all_columns_stable


def test_two_body_orbit_parameter_targeter_recovers_requested_apsides() -> None:
    dynamics = _central_dynamics()
    start = _point(a_altitude_m=108_000.0, eccentricity=0.012)
    desired_a = R + 100_000.0
    desired_e = 0.005
    desired_peri = desired_a * (1.0 - desired_e) - R
    desired_apo = desired_a * (1.0 + desired_e) - R
    variables = (
        CorrectorVariable("semi_major_axis_m", 10.0, 10_000.0, lower_bound=R + 20_000.0),
        CorrectorVariable("eccentricity", 1e-6, 0.01, lower_bound=0.0, upper_bound=0.1),
    )
    result = target_orbit_parameters(
        start,
        900.0,
        dynamics,
        OrbitTargetSpecification(
            desired_final_periselene_altitude_m=desired_peri,
            desired_final_aposelene_altitude_m=desired_apo,
        ),
        variables=variables,
        sample_count=41,
        propagation=_propagation(),
        correction=DifferentialCorrectionSettings(
            max_iterations=6,
            residual_tolerance=1e-6,
            finite_difference=FiniteDifferenceSettings(max_relative_disagreement=1e-4),
        ),
    )
    assert result.converged
    assert result.final_evaluation.final_periselene_altitude_m == pytest.approx(desired_peri, abs=1e-2)
    assert result.final_evaluation.final_aposelene_altitude_m == pytest.approx(desired_apo, abs=1e-2)
    assert result.correction.final_residual_norm < 1e-6


def test_terrain_clearance_target_fails_explicitly_without_terrain() -> None:
    dynamics = _central_dynamics()
    with pytest.raises(ValueError, match="terrain-clearance target requires a terrain model"):
        target_orbit_parameters(
            _point(),
            300.0,
            dynamics,
            OrbitTargetSpecification(minimum_terrain_clearance_m=50_000.0),
            variables=(CorrectorVariable("semi_major_axis_m", 10.0, 10_000.0),),
            sample_count=17,
            propagation=_propagation(),
        )


def test_stationkeeping_no_burns_when_thresholds_are_not_violated() -> None:
    dynamics = _central_dynamics()
    point = _point(a_altitude_m=100_000.0, eccentricity=0.002)
    state = point.initial_state(MU)
    result = simulate_impulsive_stationkeeping(
        state,
        1_800.0,
        dynamics,
        StationKeepingPolicy(
            check_interval_s=300.0,
            minimum_periselene_altitude_m=80_000.0,
            samples_per_interval=5,
        ),
        propagation=_propagation(),
        start_epoch_utc="2026-08-17T00:00:00+00:00",
    )
    assert result.maneuver_count == 0
    assert result.total_delta_v_m_s == pytest.approx(0.0)
    assert result.achieved_duration_s == pytest.approx(1_800.0)
    assert not result.terminated_early
    assert not result.impacted


@pytest.mark.parametrize("components", [("radial", "transverse"), ("transverse", "radial")])
def test_stationkeeping_applies_and_accounts_for_impulsive_correction(components) -> None:
    dynamics = _central_dynamics()
    a = R + 100_000.0
    eccentricity = 10_000.0 / a
    state = state_from_elements(
        ClassicalElements(
            semi_major_axis_m=a,
            eccentricity=eccentricity,
            inclination_rad=np.deg2rad(30.0),
            raan_rad=np.deg2rad(10.0),
            argument_of_periapsis_rad=np.deg2rad(40.0),
            true_anomaly_rad=np.deg2rad(120.0),
        ),
        MU,
    )
    result = simulate_impulsive_stationkeeping(
        state,
        900.0,
        dynamics,
        StationKeepingPolicy(
            check_interval_s=300.0,
            minimum_periselene_altitude_m=95_000.0,
            target_periselene_altitude_m=100_000.0,
            target_aposelene_altitude_m=110_000.0,
            correction_components=components,
            maximum_delta_v_per_maneuver_m_s=20.0,
            samples_per_interval=5,
        ),
        propagation=_propagation(),
        start_epoch_utc="2026-08-17T00:00:00+00:00",
    )
    assert result.maneuver_count >= 1
    first = result.maneuvers[0]
    assert first.time_s == pytest.approx(0.0)
    assert first.utc_time == "2026-08-17T00:00:00+00:00"
    assert first.delta_v_magnitude_m_s > 0.0
    radial = state[:3] / np.linalg.norm(state[:3])
    normal = np.cross(state[:3], state[3:])
    normal /= np.linalg.norm(normal)
    transverse = np.cross(normal, radial)
    assert first.delta_v_rtn_m_s.shape == (3,)
    assert first.delta_v_rtn_m_s[2] == 0.0
    np.testing.assert_allclose(
        np.column_stack((radial, transverse, normal)) @ first.delta_v_rtn_m_s,
        first.delta_v_inertial_m_s,
        atol=1e-12,
    )
    assert first.post_periselene_altitude_m > first.pre_periselene_altitude_m
    assert result.total_delta_v_m_s >= first.delta_v_magnitude_m_s
    assert result.maximum_delta_v_m_s <= result.total_delta_v_m_s
    assert not result.terminated_early


def test_stationkeeping_fails_cleanly_when_required_burn_exceeds_limit() -> None:
    dynamics = _central_dynamics()
    a = R + 100_000.0
    eccentricity = 10_000.0 / a
    state = state_from_elements(
        ClassicalElements(a, eccentricity, np.deg2rad(30.0), 0.0, 0.0, np.deg2rad(120.0)),
        MU,
    )
    result = simulate_impulsive_stationkeeping(
        state,
        900.0,
        dynamics,
        StationKeepingPolicy(
            check_interval_s=300.0,
            minimum_periselene_altitude_m=95_000.0,
            target_periselene_altitude_m=100_000.0,
            target_aposelene_altitude_m=110_000.0,
            maximum_delta_v_per_maneuver_m_s=1e-4,
            samples_per_interval=5,
        ),
        propagation=_propagation(),
    )
    assert result.terminated_early
    assert "exceeds configured limit" in result.termination_reason
    assert result.maneuver_count == 0


def test_stationkeeping_does_not_count_burns_that_leave_threshold_violated() -> None:
    state = _point(a_altitude_m=100_000.0, eccentricity=0.01).initial_state(MU)
    result = simulate_impulsive_stationkeeping(
        state, 900.0, _central_dynamics(),
        StationKeepingPolicy(
            check_interval_s=300.0,
            minimum_periselene_altitude_m=95_000.0,
            samples_per_interval=5,
        ),
        propagation=_propagation(),
    )
    # Default targets restore the initial apsides. Here those apsides already
    # breach the threshold, so their zero-residual burn cannot restore safety.
    assert result.terminated_early
    assert "leaves trigger thresholds violated" in result.termination_reason
    assert result.achieved_duration_s == 0.0
    assert result.maneuver_count == 0
    assert result.total_delta_v_m_s == 0.0
    np.testing.assert_array_equal(result.states[:, 0], state)


@pytest.mark.parametrize("explicit_targets", [True, False])
def test_stationkeeping_semimajor_trigger_uses_configured_target_or_initial_orbit(explicit_targets) -> None:
    a = R + 100_000.0
    state = state_from_elements(
        ClassicalElements(a, 10_000.0 / a, np.deg2rad(30.0), 0.0, 0.0, np.deg2rad(120.0)),
        MU,
    )
    result = simulate_impulsive_stationkeeping(
        state, 600.0, _central_dynamics(),
        StationKeepingPolicy(
            check_interval_s=300.0,
            maximum_semi_major_axis_deviation_m=1_000.0,
            target_periselene_altitude_m=100_000.0 if explicit_targets else None,
            target_aposelene_altitude_m=110_000.0 if explicit_targets else None,
            samples_per_interval=5,
        ),
        propagation=_propagation(),
    )
    assert not result.terminated_early
    assert result.achieved_duration_s == 600.0
    if explicit_targets:
        assert result.maneuver_count == 1
        burn = result.maneuvers[0]
        assert burn.time_s == 0.0
        assert burn.trigger_reasons == ("semimajor-axis deviation above threshold",)
        post_a_altitude = 0.5 * (burn.post_periselene_altitude_m + burn.post_aposelene_altitude_m)
        assert post_a_altitude == pytest.approx(105_000.0, abs=25.0)
    else:
        assert result.maneuver_count == 0


@pytest.mark.parametrize("start", [0.0, 1.0])
def test_bounded_corrector_uses_valid_one_sided_derivatives(start) -> None:
    seen = []

    def residual(values):
        value = values[0]
        assert 0.0 <= value <= 1.0
        seen.append(value)
        return np.array([value * value + value - 0.75])

    result = differential_correct(
        [start], residual,
        (CorrectorVariable("x", 0.01, 1.0, lower_bound=0.0, upper_bound=1.0),),
    )
    assert result.converged
    assert result.final_variables[0] == pytest.approx(0.5, abs=1e-5)
    assert min(seen) >= 0.0
    assert max(seen) <= 1.0


def test_finite_difference_shrinks_stencil_to_fit_narrow_bounds() -> None:
    def evaluate(values):
        assert 0.0 <= values[0] <= 1e-3
        return np.array([2.0 * values[0] + values[0] ** 2])

    result = finite_difference_jacobian(
        evaluate, [0.0], [1.0], lower_bounds=[0.0], upper_bounds=[1e-3],
    )
    assert result.jacobian[0, 0] == pytest.approx(2.0)
    assert result.all_columns_stable
    assert result.diagnostics[0].selected_step < 1e-3


def test_finite_difference_rejects_steps_lost_to_rounding() -> None:
    with pytest.raises(ValueError, match="not representable"):
        finite_difference_jacobian(lambda value: value, [1e20], [1.0])


def test_finite_difference_uses_actual_rounded_offsets() -> None:
    result = finite_difference_jacobian(lambda value: value, [1e16], [3.0])
    assert result.jacobian[0, 0] == 1.0
    assert result.all_columns_stable


@pytest.mark.parametrize("lower, upper, step", [(-1e16, -1.0, 1e16), (-1.0, 0.1, 1.0)])
def test_finite_difference_never_evaluates_past_bounds_after_rounding(lower, upper, step) -> None:
    def evaluate(values):
        assert lower <= values[0] <= upper
        return values.copy()

    # The largest forward coordinate rounds past the upper bound. Clamp it
    # before calling evaluate, then retain a correct derivative of identity.
    result = finite_difference_jacobian(
        evaluate, [lower], [step], lower_bounds=[lower], upper_bounds=[upper],
    )
    assert result.jacobian[0, 0] == pytest.approx(1.0)
    assert result.all_columns_stable


def test_circular_orbit_sensitivity_respects_eccentricity_lower_bound() -> None:
    point = _point(eccentricity=0.0)
    result = orbit_parameter_sensitivity(
        point, 60.0, _central_dynamics(),
        variables=(CorrectorVariable("eccentricity", 1e-5, 0.01, lower_bound=0.0),),
        outputs=("final_periselene_altitude_m", "final_aposelene_altitude_m"),
        sample_count=5, propagation=_propagation(),
    )
    np.testing.assert_allclose(
        result.jacobian[:, 0], [-point.semi_major_axis_m, point.semi_major_axis_m],
        rtol=1e-5,
    )
    assert result.all_columns_stable


def test_orbit_target_rejects_failed_partial_propagation(monkeypatch) -> None:
    from lunar_astrodynamics import targeting
    from lunar_astrodynamics.propagation import propagate_with_acceleration

    dynamics = _central_dynamics()
    partial = propagate_with_acceleration(
        _point().initial_state(MU), 30.0, dynamics.acceleration,
        collision_radius_m=dynamics.collision_radius_m,
        sample_times_s=[0.0, 15.0, 30.0],
    )
    partial.success = False
    partial.message = "Required step size is less than spacing between numbers."
    monkeypatch.setattr(targeting, "propagate_with_acceleration", lambda *args, **kwargs: partial)
    with pytest.raises(ValueError, match="targeting propagation failed"):
        evaluate_orbit_target(_point(), 300.0, dynamics, sample_count=5)
