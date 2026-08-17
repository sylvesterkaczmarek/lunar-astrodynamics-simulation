"""Polar lunar-orbiter ground-track, site-access, coverage, and Earth-visibility example.

The default science workflow uses NAIF's DE421 lunar orientation kernels and
NASA-published Apollo 11/17 coordinates. ``--quick`` is a deterministic,
self-contained geometry smoke test and deliberately uses synthetic sites.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from lunar_astrodynamics.access import (
    CoverageGrid,
    LunarSurfaceSite,
    analyze_earth_visibility,
    analyze_multiple_site_access,
    coverage_analysis,
    ground_track_history,
)
from lunar_astrodynamics.constants import GRGM1200A_J2, MOON_MEAN_RADIUS_M
from lunar_astrodynamics.dynamics import central_acceleration
from lunar_astrodynamics.ephemeris import spice_ephemeris_from_utc
from lunar_astrodynamics.frames import spice_rotation_provider
from lunar_astrodynamics.propagation import PropagationSettings, propagate_with_acceleration

APOLLO_11_SOURCE = "NASA Apollo 11 Lunar Surface Journal mission overview"
APOLLO_17_SOURCE = "NASA Science Taurus-Littrow Valley resource"
BODY_FIXED_FRAME = "MOON_ME_DE421"
SCIENCE_PLANE_LONGITUDE_DEG_EAST = 27.0


def _polar_state(
    rotation_body_fixed_from_inertial,
    mu_m3_s2: float,
    altitude_m: float,
    *,
    initial_longitude_deg_east: float = 0.0,
) -> np.ndarray:
    radius = MOON_MEAN_RADIUS_M + altitude_m
    speed = np.sqrt(mu_m3_s2 / radius)
    rotation = rotation_body_fixed_from_inertial(0.0)
    longitude = np.deg2rad(float(initial_longitude_deg_east))
    # At the epoch, choose an inertial circular state whose body-fixed position
    # lies on the lunar equator at the requested east longitude and whose
    # velocity lies along body-fixed +z. The instantaneous osculating plane is
    # therefore polar and crosses the chosen longitude and its antipode.
    position_b = radius * np.array([np.cos(longitude), np.sin(longitude), 0.0])
    velocity_b = np.array([0.0, 0.0, speed])
    position_i = rotation.T @ position_b
    velocity_i = rotation.T @ velocity_b
    return np.concatenate((position_i, velocity_i))


def _propagate(initial_state: np.ndarray, duration_s: float, sample_count: int):
    mu = GRGM1200A_J2.mu_m3_s2
    times = np.linspace(0.0, duration_s, sample_count)
    solution = propagate_with_acceleration(
        initial_state,
        duration_s,
        lambda _time_s, position_m: central_acceleration(position_m, mu),
        collision_radius_m=MOON_MEAN_RADIUS_M,
        sample_times_s=times,
        settings=PropagationSettings(
            rtol=1e-11,
            position_atol_m=1e-5,
            velocity_atol_m_s=1e-8,
            max_step_s=60.0,
        ),
    )
    if not solution.success:
        raise RuntimeError(solution.message)
    return solution


def _summary(
    *,
    mode: dict[str, object],
    duration_s: float,
    altitude_m: float,
    minimum_elevation_deg: float,
    track,
    access,
    coverage,
    earth,
    epoch: dict[str, object] | None,
) -> dict[str, object]:
    sites = []
    for result in access.results:
        sites.append(
            {
                "site": result.site.as_dict(),
                "access_window_count": len(result.windows),
                "total_access_time_s": result.total_access_time_s,
                "access_fraction": result.access_fraction,
                "maximum_elevation_deg": result.maximum_elevation_deg,
                "minimum_slant_range_km": (
                    None
                    if result.minimum_slant_range_m is None
                    else result.minimum_slant_range_m / 1000.0
                ),
                "mean_revisit_interval_s": (
                    None
                    if not result.revisit_intervals_s
                    else float(np.mean(result.revisit_intervals_s))
                ),
                "maximum_revisit_interval_s": (
                    None
                    if not result.revisit_intervals_s
                    else float(np.max(result.revisit_intervals_s))
                ),
            }
        )
    return {
        "mode": mode,
        "epoch": epoch,
        "body_fixed_frame": track.body_fixed_frame,
        "duration_days": duration_s / 86400.0,
        "nominal_altitude_km": altitude_m / 1000.0,
        "minimum_site_elevation_deg": minimum_elevation_deg,
        "ground_track": {
            "latitude_min_deg": float(np.min(track.latitude_deg)),
            "latitude_max_deg": float(np.max(track.latitude_deg)),
            "reference_altitude_min_km": float(np.min(track.reference_altitude_m) / 1000.0),
            "reference_altitude_max_km": float(np.max(track.reference_altitude_m) / 1000.0),
        },
        "sites": sites,
        "coverage": {
            "grid_shape": [
                int(coverage.grid.latitude_deg.size),
                int(coverage.grid.longitude_deg_east.size),
            ],
            "covered_fraction": coverage.covered_fraction,
            "mean_dwell_time_s": coverage.mean_dwell_time_s,
            "median_dwell_time_s": coverage.median_dwell_time_s,
            "maximum_dwell_time_s": coverage.maximum_dwell_time_s,
            "mean_revisit_interval_over_cells_s": coverage.mean_revisit_interval_over_cells_s,
            "maximum_revisit_interval_over_cells_s": coverage.maximum_revisit_interval_over_cells_s,
        },
        "earth_visibility": {
            "visible_time_fraction": earth.visible_time_fraction,
            "minimum_visible_fraction": float(np.min(earth.visible_fraction)),
            "fully_occulted_sample_fraction": float(np.mean(earth.visible_fraction <= 1e-12)),
            "partial_occultation_sample_fraction": float(
                np.mean((earth.visible_fraction > 1e-12) & (earth.visible_fraction < 1.0 - 1e-12))
            ),
            "visible_window_count": len(earth.visible_windows),
            "earth_range_min_km": float(np.min(earth.earth_range_m) / 1000.0),
            "earth_range_max_km": float(np.max(earth.earth_range_m) / 1000.0),
        },
        "interpretation": (
            "Access and coverage are numerical results for the stated frame, trajectory, sampling, "
            "site coordinates and elevation mask. Ground-site access uses spherical-limb/local-horizon "
            "geometry unless terrain-aware LOS is explicitly requested with a terrain model."
        ),
    }


def _quick_context(altitude_m: float):
    frame = "DEMO_BODY_FIXED"
    rotation = lambda _time_s: np.eye(3)
    state = _polar_state(rotation, GRGM1200A_J2.mu_m3_s2, altitude_m)
    sites = (
        LunarSurfaceSite("synthetic equatorial site", 0.0, 0.0, frame=frame),
        LunarSurfaceSite("synthetic northern site", 45.0, 0.0, frame=frame),
    )

    class EarthProvider:
        def __call__(self, _time_s: float) -> np.ndarray:
            return np.array([384_400_000.0, 0.0, 0.0])

        def provenance(self) -> dict[str, object]:
            return {"source": "synthetic static Earth geometry for CI"}

    return (
        frame,
        rotation,
        state,
        sites,
        EarthProvider(),
        {"mode": "self-contained polar-orbit geometry smoke"},
        None,
    )


def _science_context(kernel_dir: Path, epoch_utc: str, altitude_m: float):
    try:
        import spiceypy as spice
    except ImportError as exc:
        raise SystemExit(
            "Science mode requires SPICE support: pip install -e .[spice]"
        ) from exc
    kernel_names = (
        "naif0012.tls",
        "de421.bsp",
        "moon_pa_de421_1900-2050.bpc",
        "moon_080317.tf",
    )
    paths = tuple(kernel_dir / name for name in kernel_names)
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise SystemExit(
            "Missing DE421 lunar kernels. Run scripts/download_groundtrack_kernels.py. Missing: "
            + ", ".join(missing)
        )
    spice.kclear()
    for path in paths:
        spice.furnsh(str(path))
    ephemeris = spice_ephemeris_from_utc(
        epoch_utc,
        inertial_frame="J2000",
        observer="MOON",
        snapshot_kernels=True,
    )
    rotation = spice_rotation_provider(
        "J2000",
        BODY_FIXED_FRAME,
        et_offset_s=ephemeris.epoch_et_s,
    )
    state = _polar_state(
        rotation,
        GRGM1200A_J2.mu_m3_s2,
        altitude_m,
        initial_longitude_deg_east=SCIENCE_PLANE_LONGITUDE_DEG_EAST,
    )
    # NASA/LRO precise landing coordinates, east-positive. Elevation is kept at
    # the reference sphere in this worked example; use LunarSurfaceSite.from_terrain
    # with a compatible terrain model when local topographic height matters.
    sites = (
        LunarSurfaceSite(
            "Apollo 11 Tranquility Base",
            0.67409,
            23.47298,
            frame=BODY_FIXED_FRAME,
            coordinate_source=APOLLO_11_SOURCE,
        ),
        LunarSurfaceSite(
            "Apollo 17 Taurus-Littrow",
            20.1911,
            30.7769,
            frame=BODY_FIXED_FRAME,
            coordinate_source=APOLLO_17_SOURCE,
        ),
    )
    mode = {
        "mode": "NAIF DE421 lunar-frame worked example",
        "initial_polar_plane_longitude_deg_east": SCIENCE_PLANE_LONGITUDE_DEG_EAST,
        "site_coordinate_note": (
            "Apollo 11: NASA Lunar Surface Journal 0.67409 N, 23.47298 E; "
            "Apollo 17: NASA Science 20.1911 N, 30.7769 E."
        ),
    }
    return (
        BODY_FIXED_FRAME,
        rotation,
        state,
        sites,
        ephemeris.position_provider("EARTH"),
        mode,
        ephemeris.provenance(),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--kernel-dir", type=Path, default=Path("data/spice/de421"))
    parser.add_argument("--epoch-utc", default="2026-08-17T00:00:00")
    parser.add_argument("--duration-days", type=float)
    parser.add_argument("--samples", type=int)
    parser.add_argument("--altitude-km", type=float, default=100.0)
    parser.add_argument("--minimum-elevation-deg", type=float, default=10.0)
    parser.add_argument("--output", type=Path, default=Path("results/groundtrack_access_example.json"))
    parser.add_argument("--track-csv", type=Path, default=Path("results/groundtrack.csv"))
    parser.add_argument("--access-csv", type=Path, default=Path("results/site_access_windows.csv"))
    parser.add_argument("--coverage-csv", type=Path, default=Path("results/coverage_grid.csv"))
    parser.add_argument("--earth-csv", type=Path, default=Path("results/earth_visibility.csv"))
    args = parser.parse_args()
    altitude_m = float(args.altitude_km) * 1000.0
    if altitude_m <= 0.0:
        raise SystemExit("--altitude-km must be positive")
    if args.quick:
        duration_days = 0.12 if args.duration_days is None else float(args.duration_days)
        samples = 241 if args.samples is None else int(args.samples)
        context = _quick_context(altitude_m)
    else:
        duration_days = 1.0 if args.duration_days is None else float(args.duration_days)
        samples = 721 if args.samples is None else int(args.samples)
        context = _science_context(args.kernel_dir, args.epoch_utc, altitude_m)
    if duration_days <= 0.0 or samples < 3:
        raise SystemExit("duration must be positive and samples must be at least three")
    frame, rotation, initial_state, sites, earth_provider, mode, epoch = context
    duration_s = duration_days * 86400.0
    solution = _propagate(initial_state, duration_s, samples)
    positions = solution.y[:3].T
    track = ground_track_history(
        solution.t,
        positions,
        rotation,
        body_fixed_frame=frame,
        reference_radius_m=MOON_MEAN_RADIUS_M,
    )
    access = analyze_multiple_site_access(
        solution.t,
        positions,
        sites,
        rotation,
        body_fixed_frame=frame,
        minimum_elevation_deg=float(args.minimum_elevation_deg),
    )
    grid = CoverageGrid.regular(
        latitude_min_deg=-90.0,
        latitude_max_deg=90.0,
        latitude_step_deg=30.0 if args.quick else 15.0,
        longitude_min_deg_east=0.0,
        longitude_max_deg_east=360.0,
        longitude_step_deg=45.0 if args.quick else 30.0,
        frame=frame,
        reference_radius_m=MOON_MEAN_RADIUS_M,
    )
    coverage = coverage_analysis(
        solution.t,
        positions,
        grid,
        rotation,
        body_fixed_frame=frame,
        minimum_elevation_deg=float(args.minimum_elevation_deg),
    )
    earth = analyze_earth_visibility(solution.t, positions, earth_provider)
    summary = _summary(
        mode=mode,
        duration_s=duration_s,
        altitude_m=altitude_m,
        minimum_elevation_deg=float(args.minimum_elevation_deg),
        track=track,
        access=access,
        coverage=coverage,
        earth=earth,
        epoch=epoch,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    track.write_csv(args.track_csv)
    access.write_windows_csv(args.access_csv)
    coverage.write_csv(args.coverage_csv)
    earth.write_csv(args.earth_csv)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
