"""Compare lunar force fidelity using real SPICE Earth/Sun ephemerides."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from lunar_astrodynamics import (
    EARTH_GM_DE440_M3_S2,
    GRGM1200A_J2,
    MOON_MEAN_RADIUS_M,
    SUN_GM_DE440_M3_S2,
    CallableForce,
    CompositeForceModel,
    PropagationSettings,
    SolarRadiationPressure,
    ThirdBodyGravity,
    propagate_with_acceleration,
    spice_ephemeris_from_utc,
    state_from_elements,
    total_acceleration,
)


def _load_kernels(kernel_dir: Path) -> None:
    try:
        import spiceypy as spice
    except ImportError as exc:
        raise SystemExit(
            "Install SPICE support first: python -m pip install -e .[spice]"
        ) from exc

    required = (kernel_dir / "naif0012.tls", kernel_dir / "de440s.bsp")
    missing = [path for path in required if not path.exists()]
    if missing:
        names = ", ".join(str(path) for path in missing)
        raise SystemExit(
            f"Missing SPICE kernels: {names}. Run scripts/download_force_model_kernels.py first."
        )
    for path in required:
        spice.furnsh(str(path.resolve()))


def _eccentricity(position_m: np.ndarray, velocity_m_s: np.ndarray, mu: float) -> float:
    radius = np.linalg.norm(position_m)
    speed_squared = float(np.dot(velocity_m_s, velocity_m_s))
    radial_velocity = float(np.dot(position_m, velocity_m_s))
    eccentricity_vector = (
        (speed_squared - mu / radius) * position_m - radial_velocity * velocity_m_s
    ) / mu
    return float(np.linalg.norm(eccentricity_vector))


def _solution_metrics(solution) -> dict[str, object]:
    positions = np.asarray(solution.y[:3].T, dtype=float)
    velocities = np.asarray(solution.y[3:].T, dtype=float)
    altitudes = np.linalg.norm(positions, axis=1) - MOON_MEAN_RADIUS_M
    eccentricities = np.array(
        [
            _eccentricity(position, velocity, GRGM1200A_J2.mu_m3_s2)
            for position, velocity in zip(positions, velocities)
        ]
    )
    impacted = bool(solution.t_events and len(solution.t_events[0]))
    return {
        "samples": int(solution.t.size),
        "integrated_duration_s": float(solution.t[-1]),
        "impacted_mean_radius_sphere": impacted,
        "impact_time_s": float(solution.t_events[0][0]) if impacted else None,
        "minimum_reference_altitude_m": float(np.min(altitudes)),
        "maximum_reference_altitude_m": float(np.max(altitudes)),
        "final_reference_altitude_m": float(altitudes[-1]),
        "minimum_osculating_eccentricity": float(np.min(eccentricities)),
        "maximum_osculating_eccentricity": float(np.max(eccentricities)),
        "final_osculating_eccentricity": float(eccentricities[-1]),
    }


def _difference_metrics(reference, candidate) -> dict[str, object]:
    count = min(reference.t.size, candidate.t.size)
    if count == 0:
        raise ValueError("propagations contain no comparable output samples")
    reference_state = np.asarray(reference.y[:, :count], dtype=float)
    candidate_state = np.asarray(candidate.y[:, :count], dtype=float)
    position_difference = np.linalg.norm(
        candidate_state[:3] - reference_state[:3], axis=0
    )
    velocity_difference = np.linalg.norm(
        candidate_state[3:] - reference_state[3:], axis=0
    )
    return {
        "common_samples": int(count),
        "final_position_difference_m": float(position_difference[-1]),
        "maximum_position_difference_m": float(np.max(position_difference)),
        "rms_position_difference_m": float(np.sqrt(np.mean(position_difference**2))),
        "final_velocity_difference_m_s": float(velocity_difference[-1]),
        "maximum_velocity_difference_m_s": float(np.max(velocity_difference)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel-dir", type=Path, default=Path("data/spice"))
    parser.add_argument("--epoch-utc", default="2026-08-17T00:00:00")
    parser.add_argument("--duration-days", type=float, default=7.0)
    parser.add_argument("--samples", type=int, default=1001)
    parser.add_argument("--spacecraft-mass-kg", type=float, default=250.0)
    parser.add_argument("--srp-area-m2", type=float, default=4.0)
    parser.add_argument("--cr", type=float, default=1.4)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.duration_days <= 0.0:
        raise SystemExit("--duration-days must be positive")
    if args.samples < 2:
        raise SystemExit("--samples must be at least two")

    _load_kernels(args.kernel_dir)
    ephemeris = spice_ephemeris_from_utc(
        args.epoch_utc,
        inertial_frame="J2000",
        observer="MOON",
    )
    earth_position = ephemeris.position_provider("EARTH")
    sun_position = ephemeris.position_provider("SUN")

    lunar_gravity = CallableForce(
        "lunar central+J2 gravity",
        lambda _time_s, position_m: total_acceleration(
            position_m,
            GRGM1200A_J2.mu_m3_s2,
            GRGM1200A_J2.reference_radius_m,
            GRGM1200A_J2.j2,
        ),
        metadata={
            "type": "central plus J2",
            "model": GRGM1200A_J2.name,
            "note": "force-isolation example; replace this component with GRAIL gravity for higher-fidelity science runs",
        },
    )
    earth = ThirdBodyGravity(
        "Earth third-body gravity",
        EARTH_GM_DE440_M3_S2,
        earth_position,
    )
    sun = ThirdBodyGravity(
        "Sun third-body gravity",
        SUN_GM_DE440_M3_S2,
        sun_position,
    )
    srp = SolarRadiationPressure(
        "solar radiation pressure",
        sun_position,
        spacecraft_mass_kg=args.spacecraft_mass_kg,
        illuminated_area_m2=args.srp_area_m2,
        reflectivity_coefficient=args.cr,
        include_lunar_shadow=True,
    )

    models = {
        "lunar_gravity_only": CompositeForceModel((lunar_gravity,)),
        "lunar_gravity_plus_earth": CompositeForceModel((lunar_gravity, earth)),
        "lunar_gravity_plus_earth_plus_sun": CompositeForceModel(
            (lunar_gravity, earth, sun)
        ),
        "lunar_gravity_plus_earth_plus_sun_plus_srp": CompositeForceModel(
            (lunar_gravity, earth, sun, srp)
        ),
    }

    initial_state = state_from_elements(
        semi_major_axis_m=MOON_MEAN_RADIUS_M + 100_000.0,
        eccentricity=0.01,
        inclination_rad=np.deg2rad(90.0),
        raan_rad=np.deg2rad(20.0),
        argument_of_periapsis_rad=np.deg2rad(30.0),
        true_anomaly_rad=0.0,
        mu_m3_s2=GRGM1200A_J2.mu_m3_s2,
    )
    duration_s = args.duration_days * 86400.0
    sample_times = np.linspace(0.0, duration_s, args.samples)
    settings = PropagationSettings(
        rtol=1e-11,
        position_atol_m=1e-5,
        velocity_atol_m_s=1e-8,
        max_step_s=300.0,
    )

    solutions = {
        name: propagate_with_acceleration(
            initial_state,
            duration_s,
            model,
            collision_radius_m=MOON_MEAN_RADIUS_M,
            sample_times_s=sample_times,
            settings=settings,
        )
        for name, model in models.items()
    }

    baseline = solutions["lunar_gravity_only"]
    full_solution = solutions["lunar_gravity_plus_earth_plus_sun_plus_srp"]
    full_illumination = np.array(
        [
            srp.illumination_fraction(time_s, full_solution.y[:3, index])
            for index, time_s in enumerate(full_solution.t)
        ]
    )

    result = {
        "epoch": ephemeris.provenance(),
        "duration_days": args.duration_days,
        "initial_state_m_m_s": [float(value) for value in initial_state],
        "spacecraft": {
            "mass_kg": args.spacecraft_mass_kg,
            "srp_area_m2": args.srp_area_m2,
            "reflectivity_coefficient": args.cr,
        },
        "models": {
            name: {
                "force_model": model.provenance(),
                "trajectory": _solution_metrics(solutions[name]),
                "difference_from_lunar_gravity_only": (
                    None if name == "lunar_gravity_only" else _difference_metrics(baseline, solutions[name])
                ),
            }
            for name, model in models.items()
        },
        "srp_illumination": {
            "minimum_fraction": float(np.min(full_illumination)),
            "maximum_fraction": float(np.max(full_illumination)),
            "full_shadow_sample_fraction": float(np.mean(full_illumination <= 1e-12)),
            "partial_shadow_sample_fraction": float(
                np.mean((full_illumination > 1e-12) & (full_illumination < 1.0 - 1e-12))
            ),
        },
        "interpretation": (
            "Differences are numerical consequences of the selected preliminary force model and spacecraft parameters. "
            "This example uses lunar central+J2 gravity to isolate perturbations; it is not a mission-grade truth model."
        ),
    }

    text = json.dumps(result, indent=2)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
