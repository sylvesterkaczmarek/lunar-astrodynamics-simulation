"""Lunar ground-track, surface-site access, coverage, and Earth-visibility analysis.

All longitude values are east-positive. Surface geometry is planetocentric in an
explicit caller-selected lunar body-fixed frame. The module deliberately keeps
reference-radius altitude separate from terrain clearance.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .constants import (
    EARTH_NOMINAL_EQUATORIAL_RADIUS_M,
    MOON_MEAN_RADIUS_M,
)
from .forces import apparent_disk_illumination_fraction
from .frames import RotationProvider, validate_rotation_matrix
from .terrain import TerrainShapeModel

FloatArray = NDArray[np.float64]
PositionProvider = Callable[[float], FloatArray]


def _times(values: ArrayLike) -> FloatArray:
    times = np.asarray(values, dtype=float)
    if times.ndim != 1 or times.size == 0 or not np.all(np.isfinite(times)):
        raise ValueError("time_s must be a non-empty finite one-dimensional array")
    if times.size > 1 and np.any(np.diff(times) <= 0.0):
        raise ValueError("time_s must be strictly increasing")
    return times


def _positions(values: ArrayLike, count: int) -> FloatArray:
    positions = np.asarray(values, dtype=float)
    # Prefer the conventional sample-major (N, 3) layout when N == 3, where
    # (N, 3) and (3, N) are otherwise shape-ambiguous.
    if positions.shape == (count, 3):
        pass
    elif positions.shape == (3, count):
        positions = positions.T
    if positions.shape != (count, 3) or not np.all(np.isfinite(positions)):
        raise ValueError("positions_inertial_m must have shape (N, 3) or (3, N) and be finite")
    return positions


def _body_fixed_positions(
    time_s: FloatArray,
    positions_inertial_m: FloatArray,
    body_fixed_from_inertial: RotationProvider,
) -> FloatArray:
    result = np.empty_like(positions_inertial_m)
    for index, (time_value, position) in enumerate(zip(time_s, positions_inertial_m, strict=True)):
        rotation = validate_rotation_matrix(body_fixed_from_inertial(float(time_value)))
        result[index] = rotation @ position
    return result


def _body_fixed_lat_lon_radius(position_body_fixed_m: ArrayLike) -> tuple[float, float, float | None]:
    position = np.asarray(position_body_fixed_m, dtype=float)
    if position.shape != (3,) or not np.all(np.isfinite(position)):
        raise ValueError("body-fixed position must be a finite three-vector")
    radius = float(np.linalg.norm(position))
    if radius == 0.0:
        raise ValueError("position cannot be at the lunar center")
    transverse = float(np.hypot(position[0], position[1]))
    latitude = float(np.arctan2(position[2], transverse))
    if transverse <= 1e-12 * radius:
        longitude = None
    else:
        longitude = float(np.arctan2(position[1], position[0]) % (2.0 * np.pi))
    return radius, latitude, longitude


def _unwrap_longitude_deg(longitude_deg_east: FloatArray) -> FloatArray:
    output = np.full(longitude_deg_east.shape, np.nan, dtype=float)
    finite = np.flatnonzero(np.isfinite(longitude_deg_east))
    if finite.size:
        output[finite] = np.rad2deg(np.unwrap(np.deg2rad(longitude_deg_east[finite])))
    return output


def _json_float(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def _json_array(values: FloatArray) -> list[float | None]:
    return [_json_float(float(value)) for value in values]


@dataclass(frozen=True)
class GroundTrackHistory:
    """Sub-spacecraft planetocentric history in an explicit lunar body-fixed frame."""

    time_s: FloatArray
    latitude_deg: FloatArray
    longitude_deg_east: FloatArray
    unwrapped_longitude_deg_east: FloatArray
    reference_altitude_m: FloatArray
    body_fixed_frame: str
    reference_radius_m: float
    terrain_clearance_m: FloatArray | None = None
    terrain_name: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "body_fixed_frame": self.body_fixed_frame,
            "reference_radius_m": self.reference_radius_m,
            "terrain_name": self.terrain_name,
            "time_s": self.time_s.tolist(),
            "latitude_deg": self.latitude_deg.tolist(),
            "longitude_deg_east": _json_array(self.longitude_deg_east),
            "unwrapped_longitude_deg_east": _json_array(self.unwrapped_longitude_deg_east),
            "reference_altitude_m": self.reference_altitude_m.tolist(),
            "terrain_clearance_m": (
                None if self.terrain_clearance_m is None else self.terrain_clearance_m.tolist()
            ),
        }

    def write_csv(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        fields = [
            "time_s",
            "latitude_deg",
            "longitude_deg_east",
            "unwrapped_longitude_deg_east",
            "reference_altitude_m",
            "terrain_clearance_m",
        ]
        with destination.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for index in range(self.time_s.size):
                writer.writerow(
                    {
                        "time_s": float(self.time_s[index]),
                        "latitude_deg": float(self.latitude_deg[index]),
                        "longitude_deg_east": _json_float(float(self.longitude_deg_east[index])),
                        "unwrapped_longitude_deg_east": _json_float(
                            float(self.unwrapped_longitude_deg_east[index])
                        ),
                        "reference_altitude_m": float(self.reference_altitude_m[index]),
                        "terrain_clearance_m": (
                            None
                            if self.terrain_clearance_m is None
                            else float(self.terrain_clearance_m[index])
                        ),
                    }
                )


def ground_track_history(
    time_s: ArrayLike,
    positions_inertial_m: ArrayLike,
    body_fixed_from_inertial: RotationProvider,
    *,
    body_fixed_frame: str,
    reference_radius_m: float = MOON_MEAN_RADIUS_M,
    terrain: TerrainShapeModel | None = None,
) -> GroundTrackHistory:
    """Convert Moon-centred inertial positions to a lunar sub-spacecraft history.

    Longitude is east-positive in ``[0, 360)``. Longitude is mathematically
    undefined at an exact pole; those samples are stored as NaN internally and
    serialized as ``null`` rather than assigning an arbitrary meridian.
    """
    if not body_fixed_frame:
        raise ValueError("body_fixed_frame must be non-empty")
    if not np.isfinite(reference_radius_m) or reference_radius_m <= 0.0:
        raise ValueError("reference_radius_m must be finite and positive")
    if terrain is not None and terrain.frame != body_fixed_frame:
        raise ValueError(
            f"terrain frame mismatch: terrain is '{terrain.frame}' but ground track is '{body_fixed_frame}'"
        )
    times = _times(time_s)
    positions = _positions(positions_inertial_m, times.size)
    body_fixed = _body_fixed_positions(times, positions, body_fixed_from_inertial)
    latitude = np.empty(times.size, dtype=float)
    longitude = np.full(times.size, np.nan, dtype=float)
    altitude = np.empty(times.size, dtype=float)
    clearance = None if terrain is None else np.empty(times.size, dtype=float)
    for index, position in enumerate(body_fixed):
        radius, latitude_rad, longitude_rad = _body_fixed_lat_lon_radius(position)
        latitude[index] = np.rad2deg(latitude_rad)
        if longitude_rad is not None:
            longitude[index] = np.rad2deg(longitude_rad) % 360.0
        altitude[index] = radius - reference_radius_m
        if terrain is not None and clearance is not None:
            lookup_longitude = 0.0 if longitude_rad is None else longitude_rad
            clearance[index] = radius - terrain.surface_radius_m(latitude_rad, lookup_longitude)
    return GroundTrackHistory(
        time_s=times,
        latitude_deg=latitude,
        longitude_deg_east=longitude,
        unwrapped_longitude_deg_east=_unwrap_longitude_deg(longitude),
        reference_altitude_m=altitude,
        body_fixed_frame=body_fixed_frame,
        reference_radius_m=float(reference_radius_m),
        terrain_clearance_m=clearance,
        terrain_name=None if terrain is None else terrain.name,
    )


@dataclass(frozen=True)
class LunarSurfaceSite:
    """A planetocentric lunar surface location in an explicit body-fixed frame."""

    name: str
    latitude_deg: float
    longitude_deg_east: float
    elevation_m: float = 0.0
    frame: str = "body-fixed"
    reference_radius_m: float = MOON_MEAN_RADIUS_M
    coordinate_source: str | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("site name must be non-empty")
        if not self.frame:
            raise ValueError("site frame must be non-empty")
        if not np.isfinite(self.latitude_deg) or not -90.0 <= self.latitude_deg <= 90.0:
            raise ValueError("site latitude must lie within [-90, 90] degrees")
        if not np.isfinite(self.longitude_deg_east):
            raise ValueError("site longitude must be finite")
        if not np.isfinite(self.elevation_m):
            raise ValueError("site elevation must be finite")
        if not np.isfinite(self.reference_radius_m) or self.reference_radius_m <= 0.0:
            raise ValueError("site reference radius must be finite and positive")
        if self.reference_radius_m + self.elevation_m <= 0.0:
            raise ValueError("site radius must be positive")
        object.__setattr__(self, "longitude_deg_east", float(self.longitude_deg_east % 360.0))

    @classmethod
    def from_terrain(
        cls,
        name: str,
        latitude_deg: float,
        longitude_deg_east: float,
        terrain: TerrainShapeModel,
        *,
        coordinate_source: str | None = None,
    ) -> "LunarSurfaceSite":
        latitude_rad = np.deg2rad(float(latitude_deg))
        longitude_rad = np.deg2rad(float(longitude_deg_east) % 360.0)
        elevation = terrain.elevation_m(latitude_rad, longitude_rad)
        return cls(
            name=name,
            latitude_deg=float(latitude_deg),
            longitude_deg_east=float(longitude_deg_east),
            elevation_m=float(elevation),
            frame=terrain.frame,
            reference_radius_m=terrain.reference_radius_m,
            coordinate_source=coordinate_source,
        )

    @property
    def position_body_fixed_m(self) -> FloatArray:
        latitude = np.deg2rad(self.latitude_deg)
        longitude = np.deg2rad(self.longitude_deg_east)
        radius = self.reference_radius_m + self.elevation_m
        cos_latitude = np.cos(latitude)
        return np.array(
            [
                radius * cos_latitude * np.cos(longitude),
                radius * cos_latitude * np.sin(longitude),
                radius * np.sin(latitude),
            ],
            dtype=float,
        )

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class SiteObservation:
    time_s: float
    visible: bool
    elevation_deg: float
    slant_range_m: float
    spherical_los_clear: bool
    terrain_los_clear: bool | None


@dataclass(frozen=True)
class AccessWindow:
    start_time_s: float
    end_time_s: float
    duration_s: float
    maximum_elevation_deg: float
    maximum_elevation_time_s: float
    minimum_slant_range_m: float
    closest_approach_time_s: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class SiteAccessResult:
    site: LunarSurfaceSite
    minimum_elevation_deg: float
    terrain_aware: bool
    windows: tuple[AccessWindow, ...]
    total_access_time_s: float
    access_fraction: float
    revisit_intervals_s: tuple[float, ...]
    outage_intervals_s: tuple[float, ...]
    maximum_elevation_deg: float | None
    minimum_slant_range_m: float | None
    observation_count: int

    def to_dict(self) -> dict[str, object]:
        return {
            "site": self.site.as_dict(),
            "minimum_elevation_deg": self.minimum_elevation_deg,
            "terrain_aware": self.terrain_aware,
            "windows": [window.as_dict() for window in self.windows],
            "total_access_time_s": self.total_access_time_s,
            "access_fraction": self.access_fraction,
            "revisit_intervals_s": list(self.revisit_intervals_s),
            "outage_intervals_s": list(self.outage_intervals_s),
            "maximum_elevation_deg": self.maximum_elevation_deg,
            "minimum_slant_range_m": self.minimum_slant_range_m,
            "observation_count": self.observation_count,
        }


def _spherical_los_clear(
    site_position_m: FloatArray,
    spacecraft_position_m: FloatArray,
    reference_radius_m: float,
) -> bool:
    direction = spacecraft_position_m - site_position_m
    norm2 = float(np.dot(direction, direction))
    if norm2 == 0.0:
        return False
    fraction = float(np.clip(-np.dot(site_position_m, direction) / norm2, 0.0, 1.0))
    closest = site_position_m + fraction * direction
    return bool(np.linalg.norm(closest) >= reference_radius_m - 1e-7)


def _terrain_los_clear(
    site_position_m: FloatArray,
    spacecraft_position_m: FloatArray,
    terrain: TerrainShapeModel,
    *,
    samples: int,
    clearance_margin_m: float,
) -> bool:
    if samples < 2:
        raise ValueError("terrain_los_samples must be at least two")
    fractions = np.linspace(0.0, 1.0, int(samples) + 2)[1:-1]
    direction = spacecraft_position_m - site_position_m
    for fraction in fractions:
        point = site_position_m + float(fraction) * direction
        radius, latitude, longitude = _body_fixed_lat_lon_radius(point)
        lookup_longitude = 0.0 if longitude is None else longitude
        if radius <= terrain.surface_radius_m(latitude, lookup_longitude) + clearance_margin_m:
            return False
    return True


def _observation_from_body_fixed(
    time_s: float,
    spacecraft_position_body_fixed_m: FloatArray,
    site: LunarSurfaceSite,
    *,
    minimum_elevation_deg: float,
    terrain: TerrainShapeModel | None,
    terrain_aware: bool,
    terrain_los_samples: int,
    terrain_clearance_margin_m: float,
) -> SiteObservation:
    site_position = site.position_body_fixed_m
    line_of_sight = spacecraft_position_body_fixed_m - site_position
    slant_range = float(np.linalg.norm(line_of_sight))
    if slant_range == 0.0:
        raise ValueError("spacecraft position cannot coincide with the surface site")
    zenith = site_position / np.linalg.norm(site_position)
    sine_elevation = float(np.dot(line_of_sight / slant_range, zenith))
    elevation = float(np.rad2deg(np.arcsin(np.clip(sine_elevation, -1.0, 1.0))))
    spherical_clear = _spherical_los_clear(
        site_position, spacecraft_position_body_fixed_m, site.reference_radius_m
    )
    terrain_clear: bool | None = None
    if terrain_aware:
        if terrain is None:
            raise ValueError("terrain-aware line of sight requires a terrain model")
        if terrain.frame != site.frame:
            raise ValueError(
                f"terrain frame mismatch: terrain is '{terrain.frame}' but site is '{site.frame}'"
            )
        site_lat = np.deg2rad(site.latitude_deg)
        site_lon = np.deg2rad(site.longitude_deg_east)
        local_surface_radius = terrain.surface_radius_m(site_lat, site_lon)
        if np.linalg.norm(site_position) < local_surface_radius - 1e-6:
            raise ValueError("site is below the supplied terrain surface")
        terrain_clear = _terrain_los_clear(
            site_position,
            spacecraft_position_body_fixed_m,
            terrain,
            samples=terrain_los_samples,
            clearance_margin_m=terrain_clearance_margin_m,
        )
    visible = (
        elevation >= minimum_elevation_deg
        and spherical_clear
        and (terrain_clear is not False)
    )
    return SiteObservation(
        time_s=float(time_s),
        visible=bool(visible),
        elevation_deg=elevation,
        slant_range_m=slant_range,
        spherical_los_clear=spherical_clear,
        terrain_los_clear=terrain_clear,
    )


def site_observation(
    time_s: float,
    spacecraft_position_inertial_m: ArrayLike,
    site: LunarSurfaceSite,
    body_fixed_from_inertial: RotationProvider,
    *,
    body_fixed_frame: str,
    minimum_elevation_deg: float = 0.0,
    terrain: TerrainShapeModel | None = None,
    terrain_aware: bool = False,
    terrain_los_samples: int = 128,
    terrain_clearance_margin_m: float = 0.0,
) -> SiteObservation:
    """Evaluate instantaneous site-to-spacecraft access geometry.

    The default is a radial local-horizon plus spherical-limb test. When
    ``terrain_aware`` is enabled, the straight line of sight is additionally
    sampled against the supplied radial terrain model. This is not a DSK mesh
    ray trace and cannot recover relief below the terrain grid resolution.
    """
    if site.frame != body_fixed_frame:
        raise ValueError(
            f"site frame mismatch: site is '{site.frame}' but rotation was declared for '{body_fixed_frame}'"
        )
    if not np.isfinite(minimum_elevation_deg) or not -90.0 <= minimum_elevation_deg <= 90.0:
        raise ValueError("minimum_elevation_deg must lie within [-90, 90]")
    if not np.isfinite(terrain_clearance_margin_m) or terrain_clearance_margin_m < 0.0:
        raise ValueError("terrain_clearance_margin_m must be finite and non-negative")
    position = np.asarray(spacecraft_position_inertial_m, dtype=float)
    if position.shape != (3,) or not np.all(np.isfinite(position)):
        raise ValueError("spacecraft position must be a finite three-vector")
    rotation = validate_rotation_matrix(body_fixed_from_inertial(float(time_s)))
    return _observation_from_body_fixed(
        float(time_s),
        rotation @ position,
        site,
        minimum_elevation_deg=float(minimum_elevation_deg),
        terrain=terrain,
        terrain_aware=terrain_aware,
        terrain_los_samples=int(terrain_los_samples),
        terrain_clearance_margin_m=float(terrain_clearance_margin_m),
    )


def _transition_time(
    time0: float,
    time1: float,
    value0: float,
    value1: float,
) -> float:
    if np.isfinite(value0) and np.isfinite(value1) and value0 != value1:
        fraction = -value0 / (value1 - value0)
        if 0.0 <= fraction <= 1.0:
            return float(time0 + fraction * (time1 - time0))
    return float(0.5 * (time0 + time1))


def _access_windows_from_samples(
    time_s: FloatArray,
    visible: NDArray[np.bool_],
    elevation_deg: FloatArray,
    slant_range_m: FloatArray,
    minimum_elevation_deg: float,
) -> tuple[AccessWindow, ...]:
    windows: list[AccessWindow] = []
    count = time_s.size
    index = 0
    while index < count:
        if not visible[index]:
            index += 1
            continue
        first = index
        while index + 1 < count and visible[index + 1]:
            index += 1
        last = index
        if first == 0:
            start = float(time_s[first])
        else:
            start = _transition_time(
                float(time_s[first - 1]),
                float(time_s[first]),
                float(elevation_deg[first - 1] - minimum_elevation_deg),
                float(elevation_deg[first] - minimum_elevation_deg),
            )
        if last == count - 1:
            end = float(time_s[last])
        else:
            end = _transition_time(
                float(time_s[last]),
                float(time_s[last + 1]),
                float(elevation_deg[last] - minimum_elevation_deg),
                float(elevation_deg[last + 1] - minimum_elevation_deg),
            )
        sample_slice = slice(first, last + 1)
        local_elevation = elevation_deg[sample_slice]
        local_range = slant_range_m[sample_slice]
        max_offset = int(np.argmax(local_elevation))
        min_offset = int(np.argmin(local_range))
        windows.append(
            AccessWindow(
                start_time_s=start,
                end_time_s=end,
                duration_s=max(0.0, end - start),
                maximum_elevation_deg=float(local_elevation[max_offset]),
                maximum_elevation_time_s=float(time_s[first + max_offset]),
                minimum_slant_range_m=float(local_range[min_offset]),
                closest_approach_time_s=float(time_s[first + min_offset]),
            )
        )
        index += 1
    return tuple(windows)


def _site_access_from_body_fixed_history(
    time_s: FloatArray,
    spacecraft_body_fixed_m: FloatArray,
    site: LunarSurfaceSite,
    *,
    minimum_elevation_deg: float,
    terrain: TerrainShapeModel | None,
    terrain_aware: bool,
    terrain_los_samples: int,
    terrain_clearance_margin_m: float,
) -> SiteAccessResult:
    visible = np.zeros(time_s.size, dtype=bool)
    elevation = np.empty(time_s.size, dtype=float)
    slant_range = np.empty(time_s.size, dtype=float)
    for index, (time_value, position) in enumerate(zip(time_s, spacecraft_body_fixed_m, strict=True)):
        observation = _observation_from_body_fixed(
            float(time_value),
            position,
            site,
            minimum_elevation_deg=minimum_elevation_deg,
            terrain=terrain,
            terrain_aware=terrain_aware,
            terrain_los_samples=terrain_los_samples,
            terrain_clearance_margin_m=terrain_clearance_margin_m,
        )
        visible[index] = observation.visible
        elevation[index] = observation.elevation_deg
        slant_range[index] = observation.slant_range_m
    windows = _access_windows_from_samples(
        time_s, visible, elevation, slant_range, minimum_elevation_deg
    )
    total = float(sum(window.duration_s for window in windows))
    span = float(time_s[-1] - time_s[0]) if time_s.size > 1 else 0.0
    revisit = tuple(
        float(windows[index + 1].start_time_s - windows[index].start_time_s)
        for index in range(len(windows) - 1)
    )
    outage = tuple(
        float(windows[index + 1].start_time_s - windows[index].end_time_s)
        for index in range(len(windows) - 1)
    )
    return SiteAccessResult(
        site=site,
        minimum_elevation_deg=float(minimum_elevation_deg),
        terrain_aware=terrain_aware,
        windows=windows,
        total_access_time_s=total,
        access_fraction=0.0 if span <= 0.0 else float(np.clip(total / span, 0.0, 1.0)),
        revisit_intervals_s=revisit,
        outage_intervals_s=outage,
        maximum_elevation_deg=(
            None if not windows else max(window.maximum_elevation_deg for window in windows)
        ),
        minimum_slant_range_m=(
            None if not windows else min(window.minimum_slant_range_m for window in windows)
        ),
        observation_count=int(time_s.size),
    )


def analyze_site_access(
    time_s: ArrayLike,
    positions_inertial_m: ArrayLike,
    site: LunarSurfaceSite,
    body_fixed_from_inertial: RotationProvider,
    *,
    body_fixed_frame: str,
    minimum_elevation_deg: float = 0.0,
    terrain: TerrainShapeModel | None = None,
    terrain_aware: bool = False,
    terrain_los_samples: int = 128,
    terrain_clearance_margin_m: float = 0.0,
) -> SiteAccessResult:
    if site.frame != body_fixed_frame:
        raise ValueError(
            f"site frame mismatch: site is '{site.frame}' but rotation was declared for '{body_fixed_frame}'"
        )
    if terrain is not None and terrain.frame != body_fixed_frame:
        raise ValueError(
            f"terrain frame mismatch: terrain is '{terrain.frame}' but access frame is '{body_fixed_frame}'"
        )
    if terrain_aware and terrain is None:
        raise ValueError("terrain-aware line of sight requires a terrain model")
    if not np.isfinite(minimum_elevation_deg) or not -90.0 <= minimum_elevation_deg <= 90.0:
        raise ValueError("minimum_elevation_deg must lie within [-90, 90]")
    times = _times(time_s)
    positions = _positions(positions_inertial_m, times.size)
    body_fixed = _body_fixed_positions(times, positions, body_fixed_from_inertial)
    return _site_access_from_body_fixed_history(
        times,
        body_fixed,
        site,
        minimum_elevation_deg=float(minimum_elevation_deg),
        terrain=terrain,
        terrain_aware=terrain_aware,
        terrain_los_samples=int(terrain_los_samples),
        terrain_clearance_margin_m=float(terrain_clearance_margin_m),
    )


@dataclass(frozen=True)
class MultiSiteAccessReport:
    body_fixed_frame: str
    minimum_elevation_deg: float
    results: tuple[SiteAccessResult, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "body_fixed_frame": self.body_fixed_frame,
            "minimum_elevation_deg": self.minimum_elevation_deg,
            "sites": [result.to_dict() for result in self.results],
        }

    def write_json(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

    def write_windows_csv(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        fields = [
            "site",
            "window_index",
            "start_time_s",
            "end_time_s",
            "duration_s",
            "maximum_elevation_deg",
            "maximum_elevation_time_s",
            "minimum_slant_range_m",
            "closest_approach_time_s",
        ]
        with destination.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for result in self.results:
                for index, window in enumerate(result.windows, start=1):
                    row = window.as_dict()
                    row["site"] = result.site.name
                    row["window_index"] = index
                    writer.writerow(row)


def analyze_multiple_site_access(
    time_s: ArrayLike,
    positions_inertial_m: ArrayLike,
    sites: Sequence[LunarSurfaceSite],
    body_fixed_from_inertial: RotationProvider,
    *,
    body_fixed_frame: str,
    minimum_elevation_deg: float = 0.0,
    terrain: TerrainShapeModel | None = None,
    terrain_aware: bool = False,
    terrain_los_samples: int = 128,
    terrain_clearance_margin_m: float = 0.0,
) -> MultiSiteAccessReport:
    if not sites:
        raise ValueError("at least one lunar surface site is required")
    if len({site.name for site in sites}) != len(sites):
        raise ValueError("site names must be unique")
    for site in sites:
        if site.frame != body_fixed_frame:
            raise ValueError(
                f"site frame mismatch: site '{site.name}' is '{site.frame}' but analysis is '{body_fixed_frame}'"
            )
    times = _times(time_s)
    positions = _positions(positions_inertial_m, times.size)
    body_fixed = _body_fixed_positions(times, positions, body_fixed_from_inertial)
    results = tuple(
        _site_access_from_body_fixed_history(
            times,
            body_fixed,
            site,
            minimum_elevation_deg=float(minimum_elevation_deg),
            terrain=terrain,
            terrain_aware=terrain_aware,
            terrain_los_samples=int(terrain_los_samples),
            terrain_clearance_margin_m=float(terrain_clearance_margin_m),
        )
        for site in sites
    )
    return MultiSiteAccessReport(
        body_fixed_frame=body_fixed_frame,
        minimum_elevation_deg=float(minimum_elevation_deg),
        results=results,
    )


@dataclass(frozen=True)
class CoverageGrid:
    """Planetocentric grid of surface points for regional or global coverage."""

    latitude_deg: FloatArray
    longitude_deg_east: FloatArray
    frame: str
    reference_radius_m: float = MOON_MEAN_RADIUS_M

    def __post_init__(self) -> None:
        latitude = np.asarray(self.latitude_deg, dtype=float)
        longitude = np.asarray(self.longitude_deg_east, dtype=float)
        if latitude.ndim != 1 or latitude.size == 0 or not np.all(np.isfinite(latitude)):
            raise ValueError("coverage latitudes must be a non-empty finite one-dimensional array")
        if longitude.ndim != 1 or longitude.size == 0 or not np.all(np.isfinite(longitude)):
            raise ValueError("coverage longitudes must be a non-empty finite one-dimensional array")
        if np.any(latitude < -90.0) or np.any(latitude > 90.0):
            raise ValueError("coverage latitudes must lie within [-90, 90]")
        longitude = np.mod(longitude, 360.0)
        if np.unique(np.round(latitude, 12)).size != latitude.size:
            raise ValueError("coverage latitudes must be unique")
        if np.unique(np.round(longitude, 12)).size != longitude.size:
            raise ValueError("coverage longitudes must be unique modulo 360 degrees")
        if not self.frame:
            raise ValueError("coverage frame must be non-empty")
        if not np.isfinite(self.reference_radius_m) or self.reference_radius_m <= 0.0:
            raise ValueError("coverage reference radius must be finite and positive")
        latitude = latitude.copy()
        longitude = longitude.copy()
        latitude.setflags(write=False)
        longitude.setflags(write=False)
        object.__setattr__(self, "latitude_deg", latitude)
        object.__setattr__(self, "longitude_deg_east", longitude)

    @classmethod
    def regular(
        cls,
        *,
        latitude_min_deg: float,
        latitude_max_deg: float,
        latitude_step_deg: float,
        longitude_min_deg_east: float,
        longitude_max_deg_east: float,
        longitude_step_deg: float,
        frame: str,
        reference_radius_m: float = MOON_MEAN_RADIUS_M,
    ) -> "CoverageGrid":
        if latitude_step_deg <= 0.0 or longitude_step_deg <= 0.0:
            raise ValueError("coverage grid steps must be positive")
        if latitude_max_deg < latitude_min_deg:
            raise ValueError("latitude_max_deg must be >= latitude_min_deg")
        if longitude_max_deg_east <= longitude_min_deg_east:
            raise ValueError("longitude_max_deg_east must be > longitude_min_deg_east")
        latitude = np.arange(
            latitude_min_deg,
            latitude_max_deg + 0.5 * latitude_step_deg,
            latitude_step_deg,
            dtype=float,
        )
        longitude = np.arange(
            longitude_min_deg_east,
            longitude_max_deg_east,
            longitude_step_deg,
            dtype=float,
        )
        return cls(latitude, longitude, frame, reference_radius_m)


@dataclass(frozen=True)
class CoverageResult:
    grid: CoverageGrid
    minimum_elevation_deg: float
    duration_s: float
    dwell_time_s: FloatArray
    access_count: NDArray[np.int64]
    maximum_elevation_deg: FloatArray
    mean_revisit_interval_s: FloatArray
    maximum_revisit_interval_s: FloatArray
    covered_fraction: float
    mean_dwell_time_s: float
    median_dwell_time_s: float
    minimum_dwell_time_s: float
    maximum_dwell_time_s: float
    mean_revisit_interval_over_cells_s: float | None
    maximum_revisit_interval_over_cells_s: float | None

    def to_dict(self) -> dict[str, object]:
        return {
            "grid": {
                "latitude_deg": self.grid.latitude_deg.tolist(),
                "longitude_deg_east": self.grid.longitude_deg_east.tolist(),
                "frame": self.grid.frame,
                "reference_radius_m": self.grid.reference_radius_m,
            },
            "minimum_elevation_deg": self.minimum_elevation_deg,
            "duration_s": self.duration_s,
            "dwell_time_s": self.dwell_time_s.tolist(),
            "access_count": self.access_count.tolist(),
            "maximum_elevation_deg": [
                [_json_float(float(value)) for value in row] for row in self.maximum_elevation_deg
            ],
            "mean_revisit_interval_s": [
                [_json_float(float(value)) for value in row]
                for row in self.mean_revisit_interval_s
            ],
            "maximum_revisit_interval_s": [
                [_json_float(float(value)) for value in row]
                for row in self.maximum_revisit_interval_s
            ],
            "covered_fraction": self.covered_fraction,
            "mean_dwell_time_s": self.mean_dwell_time_s,
            "median_dwell_time_s": self.median_dwell_time_s,
            "minimum_dwell_time_s": self.minimum_dwell_time_s,
            "maximum_dwell_time_s": self.maximum_dwell_time_s,
            "mean_revisit_interval_over_cells_s": self.mean_revisit_interval_over_cells_s,
            "maximum_revisit_interval_over_cells_s": self.maximum_revisit_interval_over_cells_s,
        }

    def write_json(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

    def write_csv(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        fields = [
            "latitude_deg",
            "longitude_deg_east",
            "dwell_time_s",
            "access_count",
            "maximum_elevation_deg",
            "mean_revisit_interval_s",
            "maximum_revisit_interval_s",
        ]
        with destination.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for i, latitude in enumerate(self.grid.latitude_deg):
                for j, longitude in enumerate(self.grid.longitude_deg_east):
                    writer.writerow(
                        {
                            "latitude_deg": float(latitude),
                            "longitude_deg_east": float(longitude),
                            "dwell_time_s": float(self.dwell_time_s[i, j]),
                            "access_count": int(self.access_count[i, j]),
                            "maximum_elevation_deg": _json_float(
                                float(self.maximum_elevation_deg[i, j])
                            ),
                            "mean_revisit_interval_s": _json_float(
                                float(self.mean_revisit_interval_s[i, j])
                            ),
                            "maximum_revisit_interval_s": _json_float(
                                float(self.maximum_revisit_interval_s[i, j])
                            ),
                        }
                    )


def coverage_analysis(
    time_s: ArrayLike,
    positions_inertial_m: ArrayLike,
    grid: CoverageGrid,
    body_fixed_from_inertial: RotationProvider,
    *,
    body_fixed_frame: str,
    minimum_elevation_deg: float = 0.0,
    terrain: TerrainShapeModel | None = None,
) -> CoverageResult:
    """Compute sampled dwell/revisit coverage over a configurable lunar grid.

    Coverage uses local-horizon plus spherical-limb geometry. If terrain is
    supplied, its elevation is used for each grid point's radial site height,
    but intervening terrain is not ray-traced for every grid cell. Use
    ``analyze_site_access(..., terrain_aware=True)`` for terrain-aware LOS to
    individual sites.
    """
    if grid.frame != body_fixed_frame:
        raise ValueError(
            f"coverage frame mismatch: grid is '{grid.frame}' but rotation is '{body_fixed_frame}'"
        )
    if terrain is not None and terrain.frame != body_fixed_frame:
        raise ValueError(
            f"terrain frame mismatch: terrain is '{terrain.frame}' but coverage is '{body_fixed_frame}'"
        )
    if not np.isfinite(minimum_elevation_deg) or not -90.0 <= minimum_elevation_deg <= 90.0:
        raise ValueError("minimum_elevation_deg must lie within [-90, 90]")
    times = _times(time_s)
    if times.size < 2:
        raise ValueError("coverage analysis requires at least two trajectory samples")
    positions = _positions(positions_inertial_m, times.size)
    body_fixed = _body_fixed_positions(times, positions, body_fixed_from_inertial)
    shape = (grid.latitude_deg.size, grid.longitude_deg_east.size)
    dwell = np.zeros(shape, dtype=float)
    access_count = np.zeros(shape, dtype=np.int64)
    max_elevation = np.full(shape, np.nan, dtype=float)
    mean_revisit = np.full(shape, np.nan, dtype=float)
    max_revisit = np.full(shape, np.nan, dtype=float)
    for i, latitude in enumerate(grid.latitude_deg):
        for j, longitude in enumerate(grid.longitude_deg_east):
            elevation_m = 0.0
            if terrain is not None:
                elevation_m = float(
                    terrain.elevation_m(np.deg2rad(latitude), np.deg2rad(longitude))
                )
            site = LunarSurfaceSite(
                name=f"coverage[{i},{j}]",
                latitude_deg=float(latitude),
                longitude_deg_east=float(longitude),
                elevation_m=elevation_m,
                frame=grid.frame,
                reference_radius_m=grid.reference_radius_m,
            )
            result = _site_access_from_body_fixed_history(
                times,
                body_fixed,
                site,
                minimum_elevation_deg=float(minimum_elevation_deg),
                terrain=None,
                terrain_aware=False,
                terrain_los_samples=2,
                terrain_clearance_margin_m=0.0,
            )
            dwell[i, j] = result.total_access_time_s
            access_count[i, j] = len(result.windows)
            if result.maximum_elevation_deg is not None:
                max_elevation[i, j] = result.maximum_elevation_deg
            if result.revisit_intervals_s:
                mean_revisit[i, j] = float(np.mean(result.revisit_intervals_s))
                max_revisit[i, j] = float(np.max(result.revisit_intervals_s))
    finite_revisit = mean_revisit[np.isfinite(mean_revisit)]
    finite_max_revisit = max_revisit[np.isfinite(max_revisit)]
    return CoverageResult(
        grid=grid,
        minimum_elevation_deg=float(minimum_elevation_deg),
        duration_s=float(times[-1] - times[0]),
        dwell_time_s=dwell,
        access_count=access_count,
        maximum_elevation_deg=max_elevation,
        mean_revisit_interval_s=mean_revisit,
        maximum_revisit_interval_s=max_revisit,
        covered_fraction=float(np.count_nonzero(dwell > 0.0) / dwell.size),
        mean_dwell_time_s=float(np.mean(dwell)),
        median_dwell_time_s=float(np.median(dwell)),
        minimum_dwell_time_s=float(np.min(dwell)),
        maximum_dwell_time_s=float(np.max(dwell)),
        mean_revisit_interval_over_cells_s=(
            None if finite_revisit.size == 0 else float(np.mean(finite_revisit))
        ),
        maximum_revisit_interval_over_cells_s=(
            None if finite_max_revisit.size == 0 else float(np.max(finite_max_revisit))
        ),
    )


def earth_visibility_fraction(
    spacecraft_position_m: ArrayLike,
    earth_position_m: ArrayLike,
    *,
    moon_radius_m: float = MOON_MEAN_RADIUS_M,
    earth_radius_m: float = EARTH_NOMINAL_EQUATORIAL_RADIUS_M,
) -> float:
    """Visible fraction of Earth's apparent circular disk after lunar occultation."""
    spacecraft = np.asarray(spacecraft_position_m, dtype=float)
    earth = np.asarray(earth_position_m, dtype=float)
    if spacecraft.shape != (3,) or earth.shape != (3,):
        raise ValueError("spacecraft and Earth positions must be three-vectors")
    if not np.all(np.isfinite(spacecraft)) or not np.all(np.isfinite(earth)):
        raise ValueError("spacecraft and Earth positions must be finite")
    if not np.isfinite(moon_radius_m) or moon_radius_m <= 0.0:
        raise ValueError("moon_radius_m must be finite and positive")
    if not np.isfinite(earth_radius_m) or earth_radius_m <= 0.0:
        raise ValueError("earth_radius_m must be finite and positive")
    spacecraft_to_moon = -spacecraft
    spacecraft_to_earth = earth - spacecraft
    moon_distance = float(np.linalg.norm(spacecraft_to_moon))
    earth_distance = float(np.linalg.norm(spacecraft_to_earth))
    if moon_distance <= moon_radius_m:
        raise ValueError("spacecraft must be outside the spherical Moon")
    if earth_distance <= earth_radius_m:
        raise ValueError("spacecraft must be outside the spherical Earth")
    if moon_distance >= earth_distance:
        return 1.0
    earth_angular_radius = float(
        np.arcsin(np.clip(earth_radius_m / earth_distance, 0.0, 1.0))
    )
    moon_angular_radius = float(
        np.arcsin(np.clip(moon_radius_m / moon_distance, 0.0, 1.0))
    )
    separation = float(
        np.arccos(
            np.clip(
                np.dot(spacecraft_to_earth, spacecraft_to_moon)
                / (earth_distance * moon_distance),
                -1.0,
                1.0,
            )
        )
    )
    return apparent_disk_illumination_fraction(
        earth_angular_radius, moon_angular_radius, separation
    )


@dataclass(frozen=True)
class EarthVisibilityWindow:
    start_time_s: float
    end_time_s: float
    duration_s: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class EarthVisibilityReport:
    time_s: FloatArray
    earth_range_m: FloatArray
    visible_fraction: FloatArray
    visible_windows: tuple[EarthVisibilityWindow, ...]
    minimum_visible_fraction: float
    visible_time_fraction: float
    position_provider_provenance: Mapping[str, object] | None

    def to_dict(self) -> dict[str, object]:
        return {
            "time_s": self.time_s.tolist(),
            "earth_range_m": self.earth_range_m.tolist(),
            "visible_fraction": self.visible_fraction.tolist(),
            "visible_windows": [window.as_dict() for window in self.visible_windows],
            "minimum_visible_fraction": self.minimum_visible_fraction,
            "visible_time_fraction": self.visible_time_fraction,
            "position_provider_provenance": self.position_provider_provenance,
        }

    def write_json(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

    def write_csv(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=["time_s", "earth_range_m", "visible_fraction"],
            )
            writer.writeheader()
            for time_value, range_value, fraction in zip(
                self.time_s, self.earth_range_m, self.visible_fraction, strict=True
            ):
                writer.writerow(
                    {
                        "time_s": float(time_value),
                        "earth_range_m": float(range_value),
                        "visible_fraction": float(fraction),
                    }
                )


def _fraction_windows(
    time_s: FloatArray,
    fraction: FloatArray,
    threshold: float,
) -> tuple[EarthVisibilityWindow, ...]:
    visible = fraction >= threshold
    windows: list[EarthVisibilityWindow] = []
    index = 0
    while index < time_s.size:
        if not visible[index]:
            index += 1
            continue
        first = index
        while index + 1 < time_s.size and visible[index + 1]:
            index += 1
        last = index
        start = (
            float(time_s[first])
            if first == 0
            else _transition_time(
                float(time_s[first - 1]),
                float(time_s[first]),
                float(fraction[first - 1] - threshold),
                float(fraction[first] - threshold),
            )
        )
        end = (
            float(time_s[last])
            if last == time_s.size - 1
            else _transition_time(
                float(time_s[last]),
                float(time_s[last + 1]),
                float(fraction[last] - threshold),
                float(fraction[last + 1] - threshold),
            )
        )
        windows.append(EarthVisibilityWindow(start, end, max(0.0, end - start)))
        index += 1
    return tuple(windows)


def analyze_earth_visibility(
    time_s: ArrayLike,
    positions_inertial_m: ArrayLike,
    earth_position_m: PositionProvider,
    *,
    minimum_visible_fraction: float = 1e-12,
    moon_radius_m: float = MOON_MEAN_RADIUS_M,
    earth_radius_m: float = EARTH_NOMINAL_EQUATORIAL_RADIUS_M,
) -> EarthVisibilityReport:
    """Evaluate Earth range and finite-disk lunar occultation over a trajectory.

    ``earth_position_m`` should normally be
    ``SpiceEphemeris.position_provider('EARTH')`` so Earth and spacecraft are in
    the same Moon-centred inertial frame and use the same epoch convention.
    """
    if (
        not np.isfinite(minimum_visible_fraction)
        or minimum_visible_fraction < 0.0
        or minimum_visible_fraction > 1.0
    ):
        raise ValueError("minimum_visible_fraction must lie within [0, 1]")
    times = _times(time_s)
    positions = _positions(positions_inertial_m, times.size)
    ranges = np.empty(times.size, dtype=float)
    fractions = np.empty(times.size, dtype=float)
    for index, (time_value, spacecraft) in enumerate(zip(times, positions, strict=True)):
        earth = np.asarray(earth_position_m(float(time_value)), dtype=float)
        if earth.shape != (3,) or not np.all(np.isfinite(earth)):
            raise ValueError("Earth position provider returned an invalid three-vector")
        ranges[index] = np.linalg.norm(earth - spacecraft)
        fractions[index] = earth_visibility_fraction(
            spacecraft,
            earth,
            moon_radius_m=moon_radius_m,
            earth_radius_m=earth_radius_m,
        )
    windows = _fraction_windows(times, fractions, float(minimum_visible_fraction))
    total_visible = float(sum(window.duration_s for window in windows))
    span = float(times[-1] - times[0]) if times.size > 1 else 0.0
    provenance_method = getattr(earth_position_m, "provenance", None)
    provenance = dict(provenance_method()) if callable(provenance_method) else None
    return EarthVisibilityReport(
        time_s=times,
        earth_range_m=ranges,
        visible_fraction=fractions,
        visible_windows=windows,
        minimum_visible_fraction=float(minimum_visible_fraction),
        visible_time_fraction=(
            0.0 if span <= 0.0 else float(np.clip(total_visible / span, 0.0, 1.0))
        ),
        position_provider_provenance=provenance,
    )
