"""Compare spherical and terrain-aware clearance for a low lunar orbit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from lunar_astrodynamics import (
    LOLA_REFERENCE_RADIUS_M,
    MOON_GM_DE440_M3_S2,
    PropagationSettings,
    RegularLatLonTerrain,
    central_acceleration,
    constant_rate_z_rotation,
    load_terrain_npz,
    propagate_with_acceleration,
    propagate_with_terrain,
)

LUNAR_SIDEREAL_PERIOD_S = 27.321661 * 86400.0


def synthetic_mountain_terrain() -> RegularLatLonTerrain:
    """Small deterministic mountain field used for the self-contained example."""
    lat = np.array([-90.0, -30.0, 0.0, 30.0, 90.0])
    lon = np.array([0.0, 60.0, 90.0, 120.0, 180.0, 240.0, 300.0, 360.0])
    elevation = np.zeros((lat.size, lon.size))
    elevation[2, 1] = 2500.0
    elevation[2, 2] = 6000.0
    elevation[2, 3] = 2500.0
    elevation[:, -1] = elevation[:, 0]
    return RegularLatLonTerrain(
        lat,
        lon,
        elevation,
        reference_radius_m=LOLA_REFERENCE_RADIUS_M,
        name="synthetic 6 km equatorial mountain",
        frame="DEMO_BODY_FIXED",
        registration="gridline",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--terrain-npz",
        type=Path,
        help=(
            "Prepared external terrain grid. If omitted, a self-contained synthetic "
            "mountain is used so the example can run in CI."
        ),
    )
    parser.add_argument("--altitude-km", type=float, default=4.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    terrain = load_terrain_npz(args.terrain_npz) if args.terrain_npz else synthetic_mountain_terrain()
    radius = LOLA_REFERENCE_RADIUS_M + args.altitude_km * 1000.0
    speed = np.sqrt(MOON_GM_DE440_M3_S2 / radius)
    initial_state = np.array([radius, 0.0, 0.0, 0.0, speed, 0.0])
    period = 2.0 * np.pi * np.sqrt(radius**3 / MOON_GM_DE440_M3_S2)
    duration_s = 0.35 * period
    settings = PropagationSettings(
        rtol=1e-10,
        position_atol_m=1e-3,
        velocity_atol_m_s=1e-6,
        max_step_s=10.0,
    )
    acceleration = lambda _t, position: central_acceleration(position, MOON_GM_DE440_M3_S2)

    spherical = propagate_with_acceleration(
        initial_state,
        duration_s,
        acceleration,
        collision_radius_m=LOLA_REFERENCE_RADIUS_M,
        settings=settings,
    )
    spherical_radii = np.linalg.norm(spherical.y[:3], axis=0)
    spherical_minimum_clearance_m = float(np.min(spherical_radii - LOLA_REFERENCE_RADIUS_M))
    spherical_impacted = bool(len(spherical.t_events[0]))

    if args.terrain_npz:
        # This simple constant-rate orientation keeps the example dependency-free.
        # For science use, construct a SPICE provider for terrain.frame instead.
        rotation = constant_rate_z_rotation(2.0 * np.pi / LUNAR_SIDEREAL_PERIOD_S)
        orientation_note = "constant-rate demonstration; use SPICE for science runs"
    else:
        rotation = lambda _time: np.eye(3)
        orientation_note = "identity synthetic demonstration frame"

    terrain_result = propagate_with_terrain(
        initial_state,
        duration_s,
        acceleration,
        terrain,
        rotation,
        terrain_frame=terrain.frame,
        settings=settings,
        clearance_search_samples=1025,
    )
    report = terrain_result.clearance
    output = {
        "terrain": terrain.name,
        "terrain_frame": terrain.frame,
        "orientation_note": orientation_note,
        "reference_radius_m": terrain.reference_radius_m,
        "spherical_minimum_clearance_m": spherical_minimum_clearance_m,
        "spherical_impacted": spherical_impacted,
        "terrain_minimum_clearance_m": report.minimum_clearance_m,
        "terrain_minimum_time_s": report.minimum_time_s,
        "terrain_minimum_latitude_deg": report.minimum_location.latitude_deg,
        "terrain_minimum_longitude_deg_east": report.minimum_location.longitude_deg_east,
        "terrain_elevation_at_minimum_m": report.minimum_location.terrain_elevation_m,
        "terrain_impacted": report.impacted,
        "terrain_impact_time_s": report.impact_time_s,
        "terrain_impact_latitude_deg": None if report.impact_location is None else report.impact_location.latitude_deg,
        "terrain_impact_longitude_deg_east": None if report.impact_location is None else report.impact_location.longitude_deg_east,
        "terrain_elevation_at_impact_m": None if report.impact_location is None else report.impact_location.terrain_elevation_m,
    }
    text = json.dumps(output, indent=2)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
