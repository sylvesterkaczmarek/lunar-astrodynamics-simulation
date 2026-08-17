"""Compact external validation campaign for repeatable CI/archive revalidation.

This imports the independent building blocks from ``run_independent_validation``
but keeps the expensive degree-600 reconstructed-LRO comparison to roughly one
low-lunar orbital period. The separate 24-hour evolution comparison remains at
120x120. It is intended for repeatable external-data validation; the full
campaign script remains available for longer high-degree arcs.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import run_independent_validation as base


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/validation"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/independent_validation.json"),
    )
    args = parser.parse_args()

    required = (
        "naif0012.tls",
        "de421.bsp",
        "moon_pa_de421_1900-2050.bpc",
        "moon_080317.tf",
        "gggrx_0900c_sha.tab",
        "gggrx_1200b_sha.tab",
        "LRO_ES_36_GRGM900C_L600.BSP",
        "LDEM_4.IMG",
        "LDEM_4.LBL",
        "manifest.json",
    )
    missing = [name for name in required if not (args.data_dir / name).exists()]
    if missing:
        raise SystemExit(
            "Missing validation data. Run scripts/download_independent_validation_data.py first. Missing: "
            + ", ".join(missing)
        )

    manifest = json.loads((args.data_dir / "manifest.json").read_text(encoding="utf-8"))
    base.load_spice_context(args.data_dir)
    lro_spk = args.data_dir / "LRO_ES_36_GRGM900C_L600.BSP"
    base.spice.furnsh(str(lro_spk))
    object_id, coverage = base.lro_object_and_coverage(lro_spk)
    epoch = base.choose_validation_epoch(coverage[0], coverage[1], 24.0 * 3600.0)
    initial_state = base.lro_state_m(object_id, epoch)

    gravity900 = base.validate_gravity_model(
        args.data_dir / "gggrx_0900c_sha.tab",
        (10, 60, 120, 300, 600, 900),
        base.FRAME,
    )
    gravity1200 = base.validate_gravity_model(
        args.data_dir / "gggrx_1200b_sha.tab",
        (60, 120, 300, 600, 1200),
        "GRGM1200B published principal-axes coefficient frame",
    )
    frames = base.validate_frames(args.data_dir, object_id, epoch, initial_state)
    third_bodies = base.validate_third_bodies(epoch, initial_state[:3])
    terrain = base.validate_terrain(args.data_dir)
    independent_propagation = base.validate_independent_propagation(
        args.data_dir / "gggrx_0900c_sha.tab",
        initial_state,
        epoch,
        degree=120,
        horizon_s=2.0 * 3600.0,
    )

    lro_high = base.compare_to_lro_spk(
        args.data_dir / "gggrx_0900c_sha.tab",
        lro_spk,
        object_id,
        epoch,
        degree=600,
        horizon_s=2.0 * 3600.0,
        sample_count=49,
        convergence_check=True,
    )
    lro_long = base.compare_to_lro_spk(
        args.data_dir / "gggrx_0900c_sha.tab",
        lro_spk,
        object_id,
        epoch,
        degree=120,
        horizon_s=24.0 * 3600.0,
        sample_count=145,
        convergence_check=False,
    )
    lro = {
        "truth_source": "PDS LRO Radio Science GEODYN monthly reconstructed SPK",
        "truth_file": "LRO_ES_36_GRGM900C_L600.BSP",
        "lro_object_id": object_id,
        "spk_coverage_et_s": list(coverage),
        "validation_epoch_et_s": epoch,
        "validation_epoch_utc": base.spice.et2utc(epoch, "ISOC", 3),
        "archive_gravity_name_note": (
            "In the archived product name, GRGM900C_L600 denotes the degree-900 GRGM900C solution with a power-law constraint applied above degree 600; it does not mean the reconstruction used a 600x600 truncation."
        ),
        "candidate_force_model": {
            "lunar_gravity": "GRGM900C spherical harmonics",
            "high_degree_arc_degree_order": [600, 600],
            "long_arc_degree_order": [120, 120],
            "third_bodies": ["Earth point mass", "Sun point mass"],
            "planetary_ephemeris": "DE421",
            "lunar_orientation": base.FRAME,
            "excluded_relative_to_precision_OD": [
                "remaining GRGM900C degrees 601-900 on the high-degree arc",
                "solar radiation pressure and detailed attitude/area model",
                "lunar solid tides/time-variable gravity",
                "Jupiter and other planetary third bodies",
                "relativity where used by the POD solution",
                "spacecraft maneuvers and momentum management",
                "tracking-data parameter estimation",
            ],
        },
        "two_hour_degree600": asdict(lro_high),
        "twenty_four_hour_degree120": asdict(lro_long),
        "pass_fail_policy": (
            "No mission-truth pass threshold is imposed because this open-loop model deliberately differs from the GEODYN POD force/estimation setup. The reconstructed SPK quantifies physical-model residuals; numerical convergence is reported separately."
        ),
    }

    hard_checks = {
        "grgm900c_shtools": bool(gravity900["accepted"]),
        "grgm1200b_shtools": bool(gravity1200["accepted"]),
        "frame_transformations": bool(frames["accepted"]),
        "third_body_arithmetic": bool(third_bodies["accepted"]),
        "terrain_file_coordinates": bool(terrain["accepted"]),
        "independent_same_model_propagation": bool(independent_propagation["accepted"]),
    }
    result = {
        "campaign": "Independent scientific validation of lunar-astrodynamics-simulation",
        "campaign_profile": "compact external-data campaign: one-orbit degree-600 LRO arc plus 24-hour degree-120 evolution",
        "data_manifest": manifest,
        "software_versions": base.software_versions(),
        "hard_acceptance_checks": hard_checks,
        "hard_acceptance_passed": all(hard_checks.values()),
        "gravity_grgm900c_vs_shtools": gravity900,
        "gravity_grgm1200b_vs_shtools": gravity1200,
        "frame_validation": frames,
        "third_body_validation": third_bodies,
        "terrain_validation": terrain,
        "same_model_independent_propagation": independent_propagation,
        "lro_reconstructed_trajectory_comparison": lro,
        "validation_scope_note": (
            "Hard acceptance applies only to independently evaluated identical mathematical models. The LRO reconstructed-SPK comparison is an external physical-model residual and is not assigned a fabricated pass threshold."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    summary = {
        "hard_acceptance_passed": result["hard_acceptance_passed"],
        "grgm900c_max_relative_by_degree": {
            str(row["degree"]): row["max_relative_acceleration_difference"]
            for row in gravity900["degrees"]
        },
        "grgm1200b_max_relative_by_degree": {
            str(row["degree"]): row["max_relative_acceleration_difference"]
            for row in gravity1200["degrees"]
        },
        "frame_max_matrix_difference": frames["max_abs_matrix_difference"],
        "frame_max_altitude_difference_m": frames["max_altitude_difference_m"],
        "third_body_rows": third_bodies["rows"],
        "terrain_max_elevation_difference_m": terrain["max_elevation_difference_m"],
        "same_model_propagation": independent_propagation["comparison"],
        "lro_two_hour_degree600": asdict(lro_high),
        "lro_twenty_four_hour_degree120": asdict(lro_long),
        "software_versions": result["software_versions"],
    }
    print("VALIDATION_SUMMARY=" + json.dumps(summary, sort_keys=True))
    if not result["hard_acceptance_passed"]:
        raise SystemExit("one or more independent hard-validation checks failed")


if __name__ == "__main__":
    main()
