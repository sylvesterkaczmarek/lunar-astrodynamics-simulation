"""Demonstrate why vector/MEE quantities are safer for nearly circular lunar orbits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from lunar_astrodynamics import (
    GRGM1200A_J2,
    MOON_MEAN_RADIUS_M,
    ClassicalElements,
    PropagationSettings,
    element_history,
    elements_from_state,
    modified_equinoctial_from_state,
    orbit_history,
    orbital_period_s,
    propagate,
    state_from_elements,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--orbits", type=float, default=10.0)
    parser.add_argument("--samples-per-orbit", type=int, default=80)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.orbits <= 0.0:
        raise SystemExit("--orbits must be positive")
    if args.samples_per_orbit < 8:
        raise SystemExit("--samples-per-orbit must be at least 8")

    mu = GRGM1200A_J2.mu_m3_s2
    exact_radius = MOON_MEAN_RADIUS_M + 100_000.0
    exact_speed = np.sqrt(mu / exact_radius)
    exact_circular_equatorial_state = np.array(
        [exact_radius, 0.0, 0.0, 0.0, exact_speed, 0.0], dtype=float
    )
    try:
        elements_from_state(exact_circular_equatorial_state, mu)
    except ValueError as exc:
        classical_singularity = str(exc)
    else:  # pragma: no cover - would indicate a scientific regression
        classical_singularity = "classical conversion unexpectedly returned angles"
    exact_mee = modified_equinoctial_from_state(exact_circular_equatorial_state, mu)

    initial = ClassicalElements(
        semi_major_axis_m=MOON_MEAN_RADIUS_M + 120_000.0,
        eccentricity=1.0e-7,
        inclination_rad=np.deg2rad(89.5),
        raan_rad=np.deg2rad(25.0),
        argument_of_periapsis_rad=np.deg2rad(40.0),
        true_anomaly_rad=0.0,
    )
    initial_state = state_from_elements(initial, mu)
    period = orbital_period_s(initial.semi_major_axis_m, mu)
    duration_s = args.orbits * period
    sample_count = int(np.ceil(args.orbits * args.samples_per_orbit)) + 1
    sample_times = np.linspace(0.0, duration_s, sample_count)
    solution = propagate(
        initial_state,
        duration_s,
        model=GRGM1200A_J2,
        include_j2=True,
        sample_times_s=sample_times,
        settings=PropagationSettings(max_step_s=period / 40.0),
    )
    history = orbit_history(solution.t, solution.y, mu)

    classical_result: dict[str, object]
    try:
        classical = element_history(solution.t, solution.y, mu)
        classical_result = {
            "available_for_every_sample": True,
            "argument_of_periapsis_peak_to_peak_deg": float(
                np.rad2deg(np.ptp(classical.argument_of_periapsis_rad_unwrapped))
            ),
            "raan_peak_to_peak_deg": float(np.rad2deg(np.ptp(classical.raan_rad_unwrapped))),
            "note": (
                "These angles are numerically defined for this e=1e-7 run, but argument of periapsis loses physical meaning as eccentricity tends to zero. "
                "Use the eccentricity vector or MEE f/g for stability decisions."
            ),
        }
    except ValueError as exc:
        classical_result = {
            "available_for_every_sample": False,
            "error": str(exc),
        }

    stats = history.statistics
    result = {
        "model": GRGM1200A_J2.name,
        "duration_orbits": args.orbits,
        "near_circular_initial_eccentricity": initial.eccentricity,
        "exact_circular_equatorial_demo": {
            "classical_conversion": classical_singularity,
            "modified_equinoctial": {
                "p_m": exact_mee.semilatus_rectum_m,
                "f": exact_mee.f,
                "g": exact_mee.g,
                "h": exact_mee.h,
                "k": exact_mee.k,
                "true_longitude_rad": exact_mee.true_longitude_rad,
            },
        },
        "near_circular_history": {
            "minimum_reference_altitude_m": history.minimum_reference_altitude_m,
            "maximum_reference_altitude_m": history.maximum_reference_altitude_m,
            "minimum_periselene_altitude_m": float(np.min(history.periselene_altitude_m)),
            "maximum_aposelene_altitude_m": float(np.max(history.aposelene_altitude_m)),
            "eccentricity_minimum": stats.eccentricity.minimum,
            "eccentricity_maximum": stats.eccentricity.maximum,
            "eccentricity_linear_drift_over_span": stats.eccentricity.linear_drift_over_span,
            "eccentricity_detrended_peak_to_peak": stats.eccentricity.detrended_peak_to_peak,
            "eccentricity_vector_linear_drift_over_span": [
                float(value) for value in stats.eccentricity_vector.linear_drift_over_span
            ],
            "eccentricity_vector_detrended_peak_to_peak": [
                float(value) for value in stats.eccentricity_vector.detrended_peak_to_peak
            ],
            "mee_f_linear_drift_over_span": stats.modified_equinoctial_f.linear_drift_over_span,
            "mee_g_linear_drift_over_span": stats.modified_equinoctial_g.linear_drift_over_span,
            "maximum_orbital_plane_change_deg": float(
                np.rad2deg(stats.orbital_plane_direction.maximum_change_rad or 0.0)
            ),
            "apsidal_direction_defined_fraction": stats.apsidal_direction.defined_fraction,
        },
        "classical_history": classical_result,
        "interpretation": (
            "Classical elements remain useful away from singular cases. For nearly circular lunar-orbit stability analysis, eccentricity-vector components and MEE f/g preserve small apsidal information without dividing the geometry into an increasingly ill-defined argument of periapsis."
        ),
    }
    text = json.dumps(result, indent=2)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
