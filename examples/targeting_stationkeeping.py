"""Sensitivity, local targeting and impulsive station-keeping example.

The example uses the repository's low-degree lunar J2 model so it remains
self-contained. It is a preliminary-design workflow demonstration, not a
high-fidelity station-keeping budget for a flight mission.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from lunar_astrodynamics import (
    CorrectorVariable,
    DifferentialCorrectionSettings,
    FiniteDifferenceSettings,
    MOON_MEAN_RADIUS_M,
    OrbitSearchPoint,
    OrbitTargetSpecification,
    PropagationSettings,
    StationKeepingPolicy,
    j2_search_dynamics,
    orbit_history,
    orbit_parameter_sensitivity,
    orbital_vectors_from_state,
    propagate_with_acceleration,
    simulate_impulsive_stationkeeping,
    target_orbit_parameters,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="short deterministic CI run")
    parser.add_argument("--output", type=Path, default=Path("results/targeting_stationkeeping.json"))
    return parser


def _point() -> OrbitSearchPoint:
    altitude = 105_000.0
    return OrbitSearchPoint(
        semi_major_axis_m=MOON_MEAN_RADIUS_M + altitude,
        semi_major_axis_altitude_m=altitude,
        eccentricity=0.01,
        inclination_rad=np.deg2rad(80.0),
        raan_rad=0.0,
        periapsis_parameter_rad=np.deg2rad(270.0),
        initial_anomaly_rad=0.0,
        periapsis_parameterization="longitude_of_periapsis",
    )


def _history(state: np.ndarray, duration_s: float, dynamics, settings, samples: int):
    times = np.linspace(0.0, duration_s, samples)
    solution = propagate_with_acceleration(
        state,
        duration_s,
        dynamics.acceleration,
        collision_radius_m=dynamics.collision_radius_m,
        sample_times_s=times,
        settings=settings,
    )
    if len(solution.t_events[0]):
        event_time = float(solution.t_events[0][0])
        event_state = np.asarray(solution.y_events[0][0], dtype=float)
        if abs(float(solution.t[-1]) - event_time) > 1e-9:
            times_used = np.concatenate((solution.t, np.array([event_time])))
            states_used = np.column_stack((solution.y, event_state))
        else:
            times_used = solution.t
            states_used = solution.y
    else:
        times_used = solution.t
        states_used = solution.y
    return orbit_history(
        np.asarray(times_used, dtype=float),
        np.asarray(states_used, dtype=float),
        dynamics.mu_m3_s2,
        reference_radius_m=dynamics.analysis_reference_radius_m,
    )


def _transverse_unit(state: np.ndarray) -> np.ndarray:
    radial = state[:3] / np.linalg.norm(state[:3])
    normal = np.cross(state[:3], state[3:])
    normal = normal / np.linalg.norm(normal)
    return np.cross(normal, radial)


def main() -> None:
    args = _parser().parse_args()
    dynamics = j2_search_dynamics(include_j2=True)
    propagation = PropagationSettings(
        rtol=1e-11,
        position_atol_m=1e-4,
        velocity_atol_m_s=1e-7,
        max_step_s=90.0 if args.quick else 180.0,
    )
    candidate = _point()
    target_duration_s = (0.20 if args.quick else 0.50) * 86400.0
    samples = 65 if args.quick else 193
    variables = (
        CorrectorVariable("semi_major_axis_m", 10.0, 10_000.0, lower_bound=MOON_MEAN_RADIUS_M + 20_000.0),
        CorrectorVariable("eccentricity", 1e-6, 0.01, lower_bound=0.0, upper_bound=0.1),
    )

    initial_state = candidate.initial_state(dynamics.mu_m3_s2)
    initial_vectors = orbital_vectors_from_state(initial_state, dynamics.mu_m3_s2)
    desired_periselene = initial_vectors.periselene_radius_m - dynamics.analysis_reference_radius_m
    desired_aposelene = initial_vectors.aposelene_radius_m - dynamics.analysis_reference_radius_m

    sensitivity = orbit_parameter_sensitivity(
        candidate,
        target_duration_s,
        dynamics,
        variables=variables,
        outputs=("final_periselene_altitude_m", "final_aposelene_altitude_m"),
        sample_count=samples,
        propagation=propagation,
        finite_difference=FiniteDifferenceSettings(max_relative_disagreement=0.02),
    )

    targeting = target_orbit_parameters(
        candidate,
        target_duration_s,
        dynamics,
        OrbitTargetSpecification(
            desired_final_periselene_altitude_m=desired_periselene,
            desired_final_aposelene_altitude_m=desired_aposelene,
        ),
        variables=variables,
        sample_count=samples,
        propagation=propagation,
        correction=DifferentialCorrectionSettings(
            max_iterations=8,
            residual_tolerance=2e-4,
            finite_difference=FiniteDifferenceSettings(max_relative_disagreement=0.02),
        ),
    )
    if not targeting.converged:
        raise SystemExit(f"targeting example did not converge: {targeting.correction.reason}")

    maintained_point = replace(targeting.final_parameters, initial_anomaly_rad=np.deg2rad(120.0))
    nominal_control_state = maintained_point.initial_state(dynamics.mu_m3_s2)
    nominal_vectors = orbital_vectors_from_state(nominal_control_state, dynamics.mu_m3_s2)
    nominal_periselene = nominal_vectors.periselene_radius_m - dynamics.analysis_reference_radius_m
    nominal_aposelene = nominal_vectors.aposelene_radius_m - dynamics.analysis_reference_radius_m

    # Inject a small transverse velocity error so the uncontrolled and maintained
    # cases start from the same explicit off-nominal state.
    disturbed_state = nominal_control_state.copy()
    disturbed_state[3:] -= (0.75 if args.quick else 1.0) * _transverse_unit(disturbed_state)

    maintenance_duration_s = (0.50 if args.quick else 3.0) * 86400.0
    uncontrolled = _history(
        disturbed_state,
        maintenance_duration_s,
        dynamics,
        propagation,
        129 if args.quick else 513,
    )

    stationkeeping = simulate_impulsive_stationkeeping(
        disturbed_state,
        maintenance_duration_s,
        dynamics,
        StationKeepingPolicy(
            check_interval_s=3.0 * 3600.0 if args.quick else 6.0 * 3600.0,
            minimum_periselene_altitude_m=nominal_periselene - 250.0,
            maximum_semi_major_axis_deviation_m=1_500.0,
            target_periselene_altitude_m=nominal_periselene,
            target_aposelene_altitude_m=nominal_aposelene,
            correction_components=("radial", "transverse"),
            correction_apsis_scale_m=500.0,
            maximum_delta_v_per_maneuver_m_s=20.0,
            maximum_maneuvers=30,
            samples_per_interval=9,
        ),
        propagation=propagation,
        start_epoch_utc="2026-08-17T00:00:00+00:00",
    )
    if stationkeeping.terminated_early:
        raise SystemExit(f"station-keeping example terminated early: {stationkeeping.termination_reason}")

    controlled = orbit_history(
        stationkeeping.time_s,
        stationkeeping.states,
        dynamics.mu_m3_s2,
        reference_radius_m=dynamics.analysis_reference_radius_m,
    )

    result = {
        "model": dynamics.provenance(),
        "note": "Low-degree J2 preliminary-design demonstration. Replace with compatible high-degree gravity/ephemerides for science studies.",
        "sensitivity": {
            "parameter_names": sensitivity.parameter_names,
            "output_names": sensitivity.output_names,
            "jacobian": sensitivity.jacobian.tolist(),
            "all_columns_stable": sensitivity.all_columns_stable,
            "step_diagnostics": [item.as_dict() for item in sensitivity.diagnostics],
        },
        "targeting": {
            "converged": targeting.converged,
            "iterations": len(targeting.correction.iterations),
            "initial_residual_norm": targeting.correction.initial_residual_norm,
            "final_residual_norm": targeting.correction.final_residual_norm,
            "initial_semimajor_axis_altitude_km": candidate.semi_major_axis_altitude_m / 1000.0,
            "corrected_semimajor_axis_altitude_km": targeting.final_parameters.semi_major_axis_altitude_m / 1000.0,
            "initial_eccentricity": candidate.eccentricity,
            "corrected_eccentricity": targeting.final_parameters.eccentricity,
            "target_final_periselene_km": desired_periselene / 1000.0,
            "achieved_final_periselene_km": targeting.final_evaluation.final_periselene_altitude_m / 1000.0,
            "target_final_aposelene_km": desired_aposelene / 1000.0,
            "achieved_final_aposelene_km": targeting.final_evaluation.final_aposelene_altitude_m / 1000.0,
        },
        "stationkeeping": {
            "maintenance_duration_days": maintenance_duration_s / 86400.0,
            "injected_transverse_velocity_error_m_s": 0.75 if args.quick else 1.0,
            "uncontrolled_minimum_periselene_km": float(np.min(uncontrolled.periselene_altitude_m)) / 1000.0,
            "uncontrolled_periselene_peak_to_peak_km": float(np.ptp(uncontrolled.periselene_altitude_m)) / 1000.0,
            "controlled_minimum_periselene_km": float(np.min(controlled.periselene_altitude_m)) / 1000.0,
            "controlled_periselene_peak_to_peak_km": float(np.ptp(controlled.periselene_altitude_m)) / 1000.0,
            "maneuver_count": stationkeeping.maneuver_count,
            "total_delta_v_m_s": stationkeeping.total_delta_v_m_s,
            "maximum_delta_v_m_s": stationkeeping.maximum_delta_v_m_s,
            "maneuvers": [
                {
                    "time_s": maneuver.time_s,
                    "utc_time": maneuver.utc_time,
                    "delta_v_m_s": maneuver.delta_v_magnitude_m_s,
                    "trigger_reasons": maneuver.trigger_reasons,
                    "pre_periselene_km": maneuver.pre_periselene_altitude_m / 1000.0,
                    "post_periselene_km": maneuver.post_periselene_altitude_m / 1000.0,
                }
                for maneuver in stationkeeping.maneuvers
            ],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
