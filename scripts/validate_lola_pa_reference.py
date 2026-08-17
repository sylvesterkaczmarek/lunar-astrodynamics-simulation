"""Validate the production LOLA MOON_PA loader against the real NASA Goddard grid."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from lunar_astrodynamics import (
    LOLA_MOON_PA_DE421_FRAME,
    LOLA_REFERENCE_RADIUS_M,
    load_lola_moon_pa_grd,
    terrain_clearance_m,
)

EXPECTED_NATIVE_SHAPE = (11_521, 23_041)
SAMPLE_POINTS_DEG = (
    (0.0, 0.0),
    (0.0, 90.0),
    (0.0, 180.0),
    (0.0, 270.0),
    (45.0, 45.0),
    (-45.0, 315.0),
    (89.0, 0.0),
    (-89.0, 180.0),
)


def _cartesian_position(radius_m: float, latitude_deg: float, longitude_deg: float) -> np.ndarray:
    latitude = np.deg2rad(latitude_deg)
    longitude = np.deg2rad(longitude_deg)
    cos_latitude = np.cos(latitude)
    return radius_m * np.array(
        [
            cos_latitude * np.cos(longitude),
            cos_latitude * np.sin(longitude),
            np.sin(latitude),
        ],
        dtype=float,
    )


def _native_metadata(path: Path) -> dict[str, object]:
    try:
        from netCDF4 import Dataset
    except ImportError as exc:  # pragma: no cover - explicit live-validation dependency
        raise SystemExit("Install the terrain extra before running this script: pip install -e .[terrain]") from exc

    with Dataset(path, "r") as dataset:
        variables = dataset.variables
        lon_name = next((name for name in ("lon", "longitude", "x") if name in variables), None)
        lat_name = next((name for name in ("lat", "latitude", "y") if name in variables), None)
        z_name = next((name for name in ("z", "elevation", "height") if name in variables), None)
        if lon_name is None or lat_name is None or z_name is None:
            raise ValueError("real GMT grid does not expose recognizable longitude/latitude/elevation variables")

        longitude = np.asarray(variables[lon_name][:], dtype=float)
        latitude = np.asarray(variables[lat_name][:], dtype=float)
        z_variable = variables[z_name]
        native_shape = tuple(int(value) for value in z_variable.shape)
        if native_shape == (EXPECTED_NATIVE_SHAPE[1], EXPECTED_NATIVE_SHAPE[0]):
            native_shape = native_shape[::-1]
        if native_shape != EXPECTED_NATIVE_SHAPE:
            raise ValueError(
                f"unexpected native LOLA MOON_PA shape {z_variable.shape}; expected latitude/longitude shape {EXPECTED_NATIVE_SHAPE}"
            )
        if longitude.size != EXPECTED_NATIVE_SHAPE[1] or latitude.size != EXPECTED_NATIVE_SHAPE[0]:
            raise ValueError("native coordinate lengths do not match the documented 64 ppd gridline product")
        if not np.isclose(longitude[0], 0.0) or not np.isclose(longitude[-1], 360.0):
            raise ValueError("native longitude boundaries are not 0/360 degrees")
        if not np.isclose(latitude[0], -90.0) or not np.isclose(latitude[-1], 90.0):
            raise ValueError("native latitude boundaries are not -90/+90 degrees")
        expected_spacing = 1.0 / 64.0
        if not np.isclose(np.median(np.diff(longitude)), expected_spacing, atol=1e-12):
            raise ValueError("native longitude spacing is not 1/64 degree")
        if not np.isclose(np.median(np.diff(latitude)), expected_spacing, atol=1e-12):
            raise ValueError("native latitude spacing is not 1/64 degree")

        return {
            "longitude_variable": lon_name,
            "latitude_variable": lat_name,
            "elevation_variable": z_name,
            "elevation_dimensions": list(z_variable.dimensions),
            "native_shape": list(EXPECTED_NATIVE_SHAPE),
            "native_spacing_deg": expected_spacing,
            "longitude_bounds_deg": [float(longitude[0]), float(longitude[-1])],
            "latitude_bounds_deg": [float(latitude[0]), float(latitude[-1])],
            "elevation_dtype": str(z_variable.dtype),
            "elevation_units": getattr(z_variable, "units", None),
            "global_title": getattr(dataset, "title", None),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("grid", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--stride",
        type=int,
        default=64,
        help="Direct netCDF subsampling stride. 64 converts native 64 ppd to 1 ppd without loading the full array.",
    )
    parser.add_argument(
        "--spacecraft-reference-altitude-km",
        type=float,
        default=10.0,
        help="Instantaneous spacecraft radius above the 1737.4 km reference sphere for clearance probes.",
    )
    args = parser.parse_args()

    metadata = _native_metadata(args.grid)
    terrain = load_lola_moon_pa_grd(args.grid, registration="gridline", stride=args.stride)
    if terrain.frame != LOLA_MOON_PA_DE421_FRAME:
        raise ValueError(f"unexpected terrain frame: {terrain.frame}")
    if not np.isclose(terrain.reference_radius_m, LOLA_REFERENCE_RADIUS_M):
        raise ValueError(f"unexpected terrain reference radius: {terrain.reference_radius_m}")

    reference_altitude_m = args.spacecraft_reference_altitude_km * 1000.0
    spacecraft_radius_m = terrain.reference_radius_m + reference_altitude_m
    samples: list[dict[str, float]] = []
    for latitude_deg, longitude_deg in SAMPLE_POINTS_DEG:
        elevation_m = terrain.elevation_m(np.deg2rad(latitude_deg), np.deg2rad(longitude_deg))
        position = _cartesian_position(spacecraft_radius_m, latitude_deg, longitude_deg)
        clearance_m = terrain_clearance_m(
            0.0,
            position,
            terrain,
            lambda _time: np.eye(3),
            terrain_frame=terrain.frame,
        )
        samples.append(
            {
                "latitude_deg": latitude_deg,
                "longitude_deg_east": longitude_deg,
                "elevation_m": elevation_m,
                "spacecraft_reference_altitude_m": reference_altitude_m,
                "terrain_clearance_m": clearance_m,
            }
        )

    result = {
        "source": str(args.grid),
        "frame": terrain.frame,
        "reference_radius_m": terrain.reference_radius_m,
        "registration": terrain.registration,
        "subsample_stride": args.stride,
        "loaded_shape": [int(terrain.latitude_deg.size), int(terrain.longitude_deg_east.size)],
        "loaded_latitude_spacing_deg": terrain.latitude_spacing_deg,
        "loaded_longitude_spacing_deg": terrain.longitude_spacing_deg,
        "native": metadata,
        "samples": samples,
    }
    text = json.dumps(result, indent=2)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
