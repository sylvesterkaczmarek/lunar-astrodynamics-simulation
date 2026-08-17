"""Search a low-lunar-orbit region for stable/frozen-like candidates.

The self-contained mode uses the repository's low-degree lunar J2 model and is
intended as a workflow demonstration.  Supplying an external SHADR model plus a
compatible caller-loaded SPICE lunar body-fixed frame enables high-degree
screening with the same search API.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from lunar_astrodynamics import (
    GRGM1200A,
    PropagationSettings,
    RefinementSettings,
    StabilityConstraints,
    StabilitySearchSettings,
    StabilitySearchSpace,
    harmonic_search_dynamics,
    j2_search_dynamics,
    make_stability_map,
    read_shadr,
    run_coarse_to_fine_search,
    spice_ephemeris_from_utc,
    spice_rotation_provider,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration-days", type=float, default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--quick", action="store_true", help="small deterministic CI/demo grid")
    parser.add_argument("--gravity-model", type=Path, help="external PDS SHADR gravity model")
    parser.add_argument("--degree", type=int, default=60, help="harmonic degree when --gravity-model is supplied")
    parser.add_argument("--kernel", action="append", type=Path, default=[], help="SPICE kernel to furnsh; repeat as needed")
    parser.add_argument("--epoch-utc", default="2026-08-17T00:00:00")
    parser.add_argument("--inertial-frame", default="J2000")
    parser.add_argument(
        "--body-fixed-frame",
        help="SPICE lunar body-fixed frame compatible with the supplied gravity model",
    )
    parser.add_argument("--output", type=Path, default=Path("results/frozen_orbit_search.json"))
    parser.add_argument("--csv", type=Path, default=Path("results/frozen_orbit_search.csv"))
    parser.add_argument("--map-csv", type=Path, default=Path("results/frozen_orbit_stability_map.csv"))
    return parser


def _search_space(quick: bool) -> StabilitySearchSpace:
    if quick:
        altitudes = (90_000.0, 120_000.0)
        eccentricities = (0.01,)
        inclinations = (85.0, 95.0)
        raan = (0.0,)
        periapsis = (90.0, 270.0)
    else:
        altitudes = (80_000.0, 110_000.0, 140_000.0)
        eccentricities = (0.005, 0.015)
        inclinations = (85.0, 90.0, 95.0)
        raan = (0.0, 120.0, 240.0)
        periapsis = (90.0, 270.0)
    return StabilitySearchSpace(
        semi_major_axis_altitudes_m=altitudes,
        semi_major_axes_m=None,
        eccentricities=eccentricities,
        inclinations_rad=tuple(np.deg2rad(inclinations)),
        raan_rad=tuple(np.deg2rad(raan)),
        periapsis_rad=tuple(np.deg2rad(periapsis)),
        initial_anomaly_rad=(0.0,),
        periapsis_parameterization="longitude_of_periapsis",
    )


def _dynamics(args: argparse.Namespace):
    if args.gravity_model is None:
        return j2_search_dynamics(include_j2=True), {
            "mode": "self-contained low-degree screening",
            "note": "The J2 screen demonstrates the search workflow; it is not a high-degree frozen-orbit truth model.",
        }
    if not args.kernel or not args.body_fixed_frame:
        raise SystemExit(
            "--gravity-model requires at least one --kernel and an explicit --body-fixed-frame compatible with that gravity solution"
        )
    try:
        import spiceypy as spice
    except ImportError as exc:
        raise SystemExit("high-degree SPICE mode requires pip install -e .[spice]") from exc
    for kernel in args.kernel:
        spice.furnsh(str(kernel))
    ephemeris = spice_ephemeris_from_utc(
        args.epoch_utc,
        inertial_frame=args.inertial_frame,
        observer="MOON",
    )
    model = read_shadr(
        args.gravity_model,
        max_degree=args.degree,
        name=f"{args.gravity_model.name} n<={args.degree}",
        frame=args.body_fixed_frame,
    )
    rotation = spice_rotation_provider(
        args.inertial_frame,
        args.body_fixed_frame,
        et_offset_s=ephemeris.epoch_et_s,
    )
    dynamics = harmonic_search_dynamics(
        model,
        rotation,
        max_degree=args.degree,
        max_order=args.degree,
        parallel_safe=False,
    )
    return dynamics, {
        "mode": "external high-degree spherical harmonics",
        "gravity_model": str(args.gravity_model),
        "degree": args.degree,
        "epoch": ephemeris.provenance(),
        "product_note": f"GRGM1200A archive metadata use {GRGM1200A.body_fixed_frame}; the supplied SPICE frame must be scientifically compatible rather than merely similarly named.",
    }


def main() -> None:
    args = _parser().parse_args()
    duration_days = args.duration_days if args.duration_days is not None else (0.25 if args.quick else 2.0)
    if duration_days <= 0.0:
        raise SystemExit("--duration-days must be positive")
    dynamics, mode = _dynamics(args)
    if args.workers > 1 and not dynamics.parallel_safe:
        raise SystemExit(
            "the selected SPICE/high-degree dynamics are marked parallel_safe=False; use --workers 1"
        )

    settings = StabilitySearchSettings(
        duration_s=duration_days * 86400.0,
        sample_count=65 if args.quick else 257,
        propagation=PropagationSettings(
            rtol=1e-11,
            position_atol_m=1e-4,
            velocity_atol_m_s=1e-7,
            max_step_s=120.0 if args.quick else 300.0,
        ),
        workers=args.workers,
        apsis_eccentricity_threshold=1e-6,
        constraints=StabilityConstraints(
            require_full_duration=True,
            minimum_reference_altitude_m=10_000.0,
        ),
    )
    space = _search_space(args.quick)
    refinement = RefinementSettings(
        top_candidates=1 if args.quick else 2,
        points_per_axis=3,
        spacing_fraction=0.5,
        refine_axes=("semi_major_axis_m", "eccentricity", "inclination_rad"),
    )
    search = run_coarse_to_fine_search(
        space,
        dynamics,
        settings=settings,
        refinement=refinement,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    search.write_json(args.output)
    search.refined.write_csv(args.csv)
    stability_map = make_stability_map(
        search.coarse,
        "semi_major_axis_altitude_m",
        "inclination_deg",
        metric="periselene_altitude_peak_to_peak_m",
    )
    stability_map.write_csv(args.map_csv)

    best = search.refined.best_candidates(5, feasible_only=True)
    if not best:
        best = search.refined.best_candidates(5, feasible_only=False)
    summary = {
        "mode": mode,
        "duration_days": duration_days,
        "coarse_raw_grid_size": search.coarse.raw_grid_size,
        "coarse_unique_candidates": search.coarse.unique_candidate_count,
        "refined_unique_candidates": search.refined.unique_candidate_count,
        "seed_candidate_ids": search.seed_candidate_ids,
        "top_candidates": [
            {
                "candidate_id": candidate.candidate_id,
                "rank": candidate.rank,
                "passed_constraints": candidate.passed_constraints,
                "semi_major_axis_altitude_km": candidate.parameters.semi_major_axis_altitude_m / 1000.0,
                "eccentricity": candidate.parameters.eccentricity,
                "inclination_deg": float(np.rad2deg(candidate.parameters.inclination_rad)),
                "raan_deg": float(np.rad2deg(candidate.parameters.raan_rad)),
                "longitude_of_periapsis_deg": float(
                    np.rad2deg(candidate.parameters.longitude_of_periapsis_rad)
                ),
                "periselene_spread_m": candidate.metrics.periselene_altitude_peak_to_peak_m,
                "aposelene_spread_m": candidate.metrics.aposelene_altitude_peak_to_peak_m,
                "eccentricity_vector_drift": candidate.metrics.eccentricity_vector_linear_drift_norm,
                "eccentricity_variation": candidate.metrics.eccentricity_peak_to_peak,
                "plane_change_deg": float(np.rad2deg(candidate.metrics.orbital_plane_max_change_rad)),
                "minimum_reference_altitude_km": candidate.metrics.minimum_reference_altitude_m / 1000.0,
                "lifetime_fraction": candidate.metrics.survived_duration_fraction,
                "ranking_penalty": candidate.ranking.penalty,
                "ranking_terms": dict(candidate.ranking.normalised_terms),
            }
            for candidate in best
        ],
        "interpretation": (
            "Candidates are numerical stability/frozen-like candidates under the selected force model and search horizon. "
            "Survival alone is insufficient; inspect periselene/aposelene spread, eccentricity-vector drift, plane/apsidal evolution, clearance and uncertainty before mission use."
        ),
    }
    print(json.dumps(summary, indent=2))

    if args.gravity_model is not None:
        import spiceypy as spice

        spice.kclear()


if __name__ == "__main__":
    main()
