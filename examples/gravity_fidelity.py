"""Choose a gravity truncation from measured error and runtime instead of guessing.

The default mode uses a deterministic synthetic lunar-like degree-20 field so
CI is self-contained.  For a real GRAIL study pass an external SHADR file plus
an explicit caller-loaded SPICE body-fixed frame compatible with that gravity
solution.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from lunar_astrodynamics import (
    ClassicalElements,
    FidelityTolerance,
    GravityFidelityStudy,
    HarmonicTruncation,
    MOON_MEAN_RADIUS_M,
    PropagationSettings,
    SphericalHarmonicModel,
    compare_harmonic_accelerations,
    compare_harmonic_trajectories,
    default_harmonic_truncations,
    orbital_period_s,
    read_shadr,
    select_lowest_harmonic_truncation,
    spice_ephemeris_from_utc,
    spice_rotation_provider,
    state_from_elements,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="small deterministic CI workflow")
    parser.add_argument("--gravity-model", type=Path, help="external PDS SHADR gravity model")
    parser.add_argument("--kernel", action="append", type=Path, default=[], help="SPICE kernel; repeat as needed")
    parser.add_argument("--epoch-utc", default="2026-08-17T00:00:00")
    parser.add_argument("--inertial-frame", default="J2000")
    parser.add_argument("--body-fixed-frame", help="SPICE frame scientifically compatible with the supplied gravity model")
    parser.add_argument("--reference-degree", type=int)
    parser.add_argument("--reference-order", type=int)
    parser.add_argument("--duration-days", type=float)
    parser.add_argument("--max-final-position-error-m", type=float, default=250.0)
    parser.add_argument("--max-final-velocity-error-m-s", type=float, default=0.25)
    parser.add_argument("--max-periselene-variation-error-m", type=float, default=100.0)
    parser.add_argument("--max-relative-acceleration-error", type=float, default=2.0e-5)
    parser.add_argument("--output", type=Path, default=Path("results/gravity_fidelity.json"))
    parser.add_argument("--runtime-csv", type=Path, default=Path("results/gravity_fidelity_runtime.csv"))
    return parser


def _synthetic_model() -> SphericalHarmonicModel:
    degree = 20
    c = np.zeros((degree + 1, degree + 1), dtype=float)
    s = np.zeros_like(c)
    c[0, 0] = 1.0
    c[2, 0] = -2.03224e-4 / np.sqrt(5.0)
    for n in range(2, degree + 1):
        for m in range(0, n + 1):
            if n == 2 and m == 0:
                continue
            amplitude = 1.5e-5 / (n * n)
            c[n, m] = amplitude * np.cos(0.71 * n + 0.37 * m)
            if m > 0:
                s[n, m] = amplitude * np.sin(0.53 * n - 0.41 * m)
    return SphericalHarmonicModel(
        mu_m3_s2=4.90280011526323e12,
        reference_radius_m=1_738_000.0,
        c=c,
        s=s,
        name="deterministic synthetic lunar-like degree-20 field",
        frame="DEMO_FIXED",
    )


def _load_context(args: argparse.Namespace):
    if args.gravity_model is None:
        return _synthetic_model(), (lambda _time_s: np.eye(3)), {
            "mode": "self-contained synthetic convergence demonstration",
            "frame_note": "identity DEMO_FIXED frame; use external mode for science studies",
        }
    if not args.kernel or not args.body_fixed_frame:
        raise SystemExit(
            "--gravity-model requires at least one --kernel and an explicit --body-fixed-frame compatible with the gravity product"
        )
    try:
        import spiceypy as spice
    except ImportError as exc:
        raise SystemExit("external SPICE mode requires pip install -e .[spice]") from exc
    for kernel in args.kernel:
        spice.furnsh(str(kernel))
    ephemeris = spice_ephemeris_from_utc(
        args.epoch_utc,
        inertial_frame=args.inertial_frame,
        observer="MOON",
    )
    model = read_shadr(
        args.gravity_model,
        name=args.gravity_model.name,
        frame=args.body_fixed_frame,
    )
    rotation = spice_rotation_provider(
        args.inertial_frame,
        args.body_fixed_frame,
        et_offset_s=ephemeris.epoch_et_s,
    )
    return model, rotation, {
        "mode": "external SHADR/SPICE gravity fidelity study",
        "gravity_model": str(args.gravity_model),
        "epoch": ephemeris.provenance(),
        "body_fixed_frame": args.body_fixed_frame,
    }


def _reference(args: argparse.Namespace, model: SphericalHarmonicModel) -> HarmonicTruncation:
    degree = model.max_degree if args.reference_degree is None else args.reference_degree
    order = degree if args.reference_order is None else args.reference_order
    reference = HarmonicTruncation(degree, order)
    reference.validate_for(model)
    return reference


def _candidate_truncations(model: SphericalHarmonicModel, reference: HarmonicTruncation, quick: bool):
    degrees = (2, 4, 8, 12, 20) if quick and model.max_degree <= 20 else (2, 10, 20, 40, 60, 120, 300)
    candidates = [item for item in default_harmonic_truncations(model, degrees=degrees, include_model_maximum=False) if item.degree <= reference.degree]
    if reference not in candidates:
        candidates.append(reference)
    return tuple(candidates)


def _orbit_samples(model: SphericalHarmonicModel, count: int):
    a = MOON_MEAN_RADIUS_M + 105_000.0
    e = 0.01
    inc = np.deg2rad(88.0)
    raan = np.deg2rad(35.0)
    argp = np.deg2rad(270.0)
    anomalies = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
    states = np.column_stack(
        [
            state_from_elements(
                ClassicalElements(a, e, inc, raan, argp, float(anomaly)),
                model.mu_m3_s2,
            )
            for anomaly in anomalies
        ]
    )
    period = orbital_period_s(a, model.mu_m3_s2)
    times = np.linspace(0.0, period, count, endpoint=False)
    return states, times


def main() -> None:
    args = _parser().parse_args()
    model, rotation, context = _load_context(args)
    reference = _reference(args, model)
    truncations = _candidate_truncations(model, reference, args.quick)
    sample_states, sample_times = _orbit_samples(model, 5 if args.quick else 12)
    duration_days = args.duration_days if args.duration_days is not None else (0.04 if args.quick else 0.5)
    if duration_days <= 0.0:
        raise SystemExit("--duration-days must be positive")

    acceleration = compare_harmonic_accelerations(
        model,
        rotation,
        sample_states[:3].T,
        times_s=sample_times,
        velocities_m_s=sample_states[3:].T,
        truncations=truncations,
        reference=reference,
        benchmark_repetitions=1 if args.quick else 3,
    )
    initial_state = sample_states[:, 0]
    trajectory = compare_harmonic_trajectories(
        model,
        rotation,
        initial_state,
        duration_days * 86400.0,
        truncations=truncations,
        reference=reference,
        sample_count=33 if args.quick else 129,
        propagation=PropagationSettings(
            rtol=1e-10,
            position_atol_m=1e-4,
            velocity_atol_m_s=1e-7,
            max_step_s=120.0,
        ),
    )
    study = GravityFidelityStudy(acceleration, trajectory)
    acceleration_selection = select_lowest_harmonic_truncation(
        acceleration,
        FidelityTolerance(maximum_relative_acceleration_error=args.max_relative_acceleration_error),
    )
    trajectory_selection = select_lowest_harmonic_truncation(
        trajectory,
        FidelityTolerance(
            maximum_final_position_difference_m=args.max_final_position_error_m,
            maximum_final_velocity_difference_m_s=args.max_final_velocity_error_m_s,
            maximum_periselene_variation_difference_m=args.max_periselene_variation_error_m,
        ),
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = study.to_dict()
    payload["context"] = context
    payload["requested_tolerances"] = {
        "maximum_relative_acceleration_error": args.max_relative_acceleration_error,
        "maximum_final_position_difference_m": args.max_final_position_error_m,
        "maximum_final_velocity_difference_m_s": args.max_final_velocity_error_m_s,
        "maximum_periselene_variation_difference_m": args.max_periselene_variation_error_m,
    }
    payload["acceleration_selection"] = acceleration_selection.as_dict()
    payload["trajectory_selection"] = trajectory_selection.as_dict()
    payload["design_rule"] = (
        "Use the trajectory-based selection for this propagation horizon, then verify the chosen truncation over the mission-specific location/epoch envelope. "
        "Do not transfer this degree directly to a different altitude, ground track, epoch or accuracy requirement."
    )
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    study.write_runtime_csv(args.runtime_csv)

    summary = {
        "context": context,
        "reference": reference.as_dict(),
        "duration_days": duration_days,
        "acceleration_selected": None if acceleration_selection.selected_truncation is None else acceleration_selection.selected_truncation.as_dict(),
        "trajectory_selected": None if trajectory_selection.selected_truncation is None else trajectory_selection.selected_truncation.as_dict(),
        "rows": [
            {
                "truncation": entry.truncation.label,
                "max_relative_acceleration_error": entry.maximum_relative_error,
                "acceleration_seconds_per_eval": entry.runtime_seconds_per_evaluation,
                "trajectory_runtime_s": next(item.runtime_seconds for item in trajectory.entries if item.truncation == entry.truncation),
                "final_position_difference_m": next(item.final_position_difference_m for item in trajectory.entries if item.truncation == entry.truncation),
                "final_velocity_difference_m_s": next(item.final_velocity_difference_m_s for item in trajectory.entries if item.truncation == entry.truncation),
            }
            for entry in acceleration.entries
        ],
    }
    print(json.dumps(summary, indent=2))

    if args.gravity_model is not None:
        import spiceypy as spice

        spice.kclear()


if __name__ == "__main__":
    main()
