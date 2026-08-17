"""Quantify lunar-orbit sensitivity to gravity-field uncertainty.

By default this example uses a small synthetic uncertainty-bearing gravity field
so it runs without external data. For correlated GRGM1200A science studies,
pass downloaded PDS clone files with --clones.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from lunar_astrodynamics import (
    ClassicalElements,
    MOON_MEAN_RADIUS_M,
    PropagationSettings,
    SphericalHarmonicModel,
    constant_rate_z_rotation,
    load_grgm1200a_clone_ensemble,
    propagate_gravity_ensemble,
    sample_independent_coefficient_uncertainty,
    state_from_elements,
)

MU_MOON_M3_S2 = 4.90280011526323e12
LUNAR_SIDEREAL_PERIOD_S = 27.321661 * 86400.0


def _synthetic_uncertain_gravity() -> SphericalHarmonicModel:
    c = np.zeros((5, 5))
    s = np.zeros_like(c)
    sigma_c = np.zeros_like(c)
    sigma_s = np.zeros_like(c)
    c[0, 0] = 1.0
    c[2, 0] = -9.09e-5
    c[2, 2] = 2.2e-5
    s[2, 2] = -1.4e-5
    c[3, 1] = 7.0e-6
    s[3, 1] = 4.0e-6
    c[4, 2] = -2.0e-6
    s[4, 2] = 1.0e-6

    # Deliberately visible low-degree uncertainties for a self-contained
    # demonstration. These are synthetic values, not GRGM1200A claims.
    sigma_c[2, 0] = 3.0e-7
    sigma_c[2, 2] = 4.0e-7
    sigma_s[2, 2] = 4.0e-7
    sigma_c[3, 1] = 2.0e-7
    sigma_s[3, 1] = 2.0e-7
    sigma_c[4, 2] = 1.0e-7
    sigma_s[4, 2] = 1.0e-7
    return SphericalHarmonicModel(
        MU_MOON_M3_S2,
        1_738_000.0,
        c,
        s,
        name="synthetic uncertainty demonstration field",
        sigma_c=sigma_c,
        sigma_s=sigma_s,
    )


def _json_report(result, *, method: str, seed: int | None, duration_days: float) -> dict:
    return {
        "method": method,
        "seed": seed,
        "duration_days": duration_days,
        "realizations": len(result.samples),
        "impact_fraction": result.impact_fraction,
        "percentile_levels": list(result.percentile_levels),
        "percentiles": result.percentiles,
        "samples": [
            {
                "model_name": sample.model_name,
                "minimum_altitude_m": sample.minimum_altitude_m,
                "maximum_altitude_m": sample.maximum_altitude_m,
                "minimum_osculating_periselene_altitude_m": (
                    sample.minimum_osculating_periselene_altitude_m
                ),
                "maximum_osculating_aposelene_altitude_m": (
                    sample.maximum_osculating_aposelene_altitude_m
                ),
                "maximum_eccentricity": sample.maximum_eccentricity,
                "final_eccentricity": sample.final_eccentricity,
                "lifetime_s": sample.lifetime_s,
                "impacted": sample.impacted,
            }
            for sample in result.samples
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--duration-days", type=float, default=1.0)
    parser.add_argument("--degree", type=int, default=4)
    parser.add_argument(
        "--clones",
        type=Path,
        nargs="*",
        help="Optional downloaded PDS GRGM1200A clone files for correlated analysis",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.duration_days <= 0.0:
        parser.error("--duration-days must be positive")
    if args.degree < 0:
        parser.error("--degree must be non-negative")

    if args.clones:
        models = load_grgm1200a_clone_ensemble(args.clones, max_degree=args.degree)
        method = "PDS covariance-derived GRGM1200A clone ensemble"
        seed = None
    else:
        nominal = _synthetic_uncertain_gravity()
        models = sample_independent_coefficient_uncertainty(
            nominal,
            seed=args.seed,
            count=args.samples,
            assume_independent=True,
        )
        method = (
            "synthetic diagonal independent-sigma demonstration; "
            "not a full covariance model"
        )
        seed = args.seed

    initial_elements = ClassicalElements(
        semi_major_axis_m=MOON_MEAN_RADIUS_M + 100_000.0,
        eccentricity=0.01,
        inclination_rad=np.deg2rad(88.0),
        raan_rad=np.deg2rad(30.0),
        argument_of_periapsis_rad=np.deg2rad(45.0),
        true_anomaly_rad=0.0,
    )
    initial_state = state_from_elements(initial_elements, MU_MOON_M3_S2)
    rotation = constant_rate_z_rotation(2.0 * np.pi / LUNAR_SIDEREAL_PERIOD_S)
    duration_s = args.duration_days * 86400.0

    result = propagate_gravity_ensemble(
        initial_state,
        duration_s,
        models,
        rotation,
        max_degree=args.degree,
        sample_count=129,
        settings=PropagationSettings(
            rtol=1e-9,
            position_atol_m=1e-3,
            velocity_atol_m_s=1e-6,
            max_step_s=120.0,
        ),
    )
    report = _json_report(
        result,
        method=method,
        seed=seed,
        duration_days=args.duration_days,
    )
    text = json.dumps(report, indent=2)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
