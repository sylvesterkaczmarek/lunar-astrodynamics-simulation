"""Terrain-aware lunar shape, clearance, impact, and propagation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol
import re

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar

from .frames import RotationProvider, validate_rotation_matrix
from .propagation import AccelerationFunction, PropagationSettings

FloatArray = NDArray[np.float64]

LOLA_REFERENCE_RADIUS_M = 1_737_400.0
LOLA_MOON_PA_DE421_FRAME = "MOON_PA_DE421"
LOLA_MOON_PA_64_GRIDLINE_URL = "https://pgda.gsfc.nasa.gov/data/LOLA_PA/LDEM64_PA_gridline_202405.grd"
LOLA_MOON_PA_64_PIXEL_URL = "https://pgda.gsfc.nasa.gov/data/LOLA_PA/LDEM64_PA_pixel_202405.grd"
LOLA_PDS_LDEM4_IMG_URL = "https://imbrium.mit.edu/DATA/LOLA_GDR/CYLINDRICAL/IMG/LDEM_4.IMG"
LOLA_PDS_LDEM4_LBL_URL = "https://imbrium.mit.edu/DATA/LOLA_GDR/CYLINDRICAL/IMG/LDEM_4.LBL"
_LOLA_MOON_PA_GRID_KM_TO_M = 1000.0


class TerrainShapeModel(Protocol):
    name: str
    frame: str
    reference_radius_m: float

    def elevation_m(self, latitude_rad: float, longitude_rad: float) -> float: ...

    def surface_radius_m(self, latitude_rad: float, longitude_rad: float) -> float: ...


@dataclass(frozen=True)
class RegularLatLonTerrain:
    """Global regular latitude/longitude terrain grid with periodic longitude."""

    latitude_deg: FloatArray
    longitude_deg_east: FloatArray
    elevation_grid_m: FloatArray
    reference_radius_m: float = LOLA_REFERENCE_RADIUS_M
    name: str = "regular lunar terrain grid"
    frame: str = "body-fixed"
    registration: str = "gridline"
    source: str | None = None

    def __post_init__(self) -> None:
        lat = np.asarray(self.latitude_deg, dtype=float).copy()
        lon = np.asarray(self.longitude_deg_east, dtype=float).copy()
        elevation = np.asarray(self.elevation_grid_m, dtype=float).copy()
        registration = self.registration.lower()
        if lat.ndim != 1 or lon.ndim != 1 or lat.size < 2 or lon.size < 2:
            raise ValueError("latitude and longitude coordinates must be one-dimensional")
        if elevation.shape != (lat.size, lon.size):
            raise ValueError("elevation_grid_m must have shape (n_latitude, n_longitude)")
        if not np.all(np.isfinite(lat)) or not np.all(np.isfinite(lon)):
            raise ValueError("terrain coordinates must be finite")
        if not np.all(np.isfinite(elevation)):
            raise ValueError("terrain elevations must be finite")
        if np.any(np.diff(lat) <= 0.0) or np.any(np.diff(lon) <= 0.0):
            raise ValueError("terrain coordinates must be strictly increasing")
        if lat[0] < -90.0 - 1e-10 or lat[-1] > 90.0 + 1e-10:
            raise ValueError("terrain latitude coordinates must lie within [-90, 90]")
        if lon[0] < -1e-10 or lon[-1] > 360.0 + 1e-10:
            raise ValueError("east-positive longitude coordinates must lie within [0, 360]")
        if registration not in {"gridline", "pixel"}:
            raise ValueError("registration must be 'gridline' or 'pixel'")
        if not np.isfinite(self.reference_radius_m) or self.reference_radius_m <= 0.0:
            raise ValueError("reference_radius_m must be finite and positive")
        if registration == "gridline":
            if not np.isclose(lat[0], -90.0, atol=1e-8) or not np.isclose(lat[-1], 90.0, atol=1e-8):
                raise ValueError("global gridline terrain must include -90 and +90 degrees")
            if not np.isclose(lon[0], 0.0, atol=1e-8) or not np.isclose(lon[-1], 360.0, atol=1e-8):
                raise ValueError("global gridline terrain must include 0 and 360 degrees")
            if not np.allclose(elevation[:, 0], elevation[:, -1], rtol=0.0, atol=1e-6):
                raise ValueError("gridline 0 and 360 degree terrain boundary columns must match")
        lat.setflags(write=False)
        lon.setflags(write=False)
        elevation.setflags(write=False)
        object.__setattr__(self, "latitude_deg", lat)
        object.__setattr__(self, "longitude_deg_east", lon)
        object.__setattr__(self, "elevation_grid_m", elevation)
        object.__setattr__(self, "registration", registration)

    @property
    def latitude_spacing_deg(self) -> float:
        return float(np.median(np.diff(self.latitude_deg)))

    @property
    def longitude_spacing_deg(self) -> float:
        return float(np.median(np.diff(self.longitude_deg_east)))

    def _longitude_interpolate(self, row: FloatArray, longitude_deg: float) -> float:
        lon = float(longitude_deg % 360.0)
        x = self.longitude_deg_east
        if self.registration == "gridline":
            if lon == 0.0:
                return float(row[0])
            right = int(np.searchsorted(x, lon, side="right"))
            right = min(max(right, 1), x.size - 1)
            left = right - 1
            weight = (lon - x[left]) / (x[right] - x[left])
            return float((1.0 - weight) * row[left] + weight * row[right])
        if lon < x[0]:
            left, right = x.size - 1, 0
            x0, x1 = x[left] - 360.0, x[right]
        elif lon > x[-1]:
            left, right = x.size - 1, 0
            x0, x1 = x[left], x[right] + 360.0
        else:
            right = int(np.searchsorted(x, lon, side="right"))
            if right == 0:
                return float(row[0])
            if right >= x.size:
                return float(row[-1])
            left = right - 1
            x0, x1 = x[left], x[right]
        weight = (lon - x0) / (x1 - x0)
        return float((1.0 - weight) * row[left] + weight * row[right])

    def elevation_m(self, latitude_rad: float, longitude_rad: float) -> float:
        latitude_deg = float(np.rad2deg(latitude_rad))
        longitude_deg = float(np.rad2deg(longitude_rad))
        if not np.isfinite(latitude_deg) or not np.isfinite(longitude_deg):
            raise ValueError("latitude and longitude must be finite")
        if latitude_deg < -90.0 - 1e-10 or latitude_deg > 90.0 + 1e-10:
            raise ValueError("latitude must lie within [-90, 90] degrees")
        latitude_deg = float(np.clip(latitude_deg, -90.0, 90.0))
        if np.isclose(abs(latitude_deg), 90.0, atol=1e-12):
            row = self.elevation_grid_m[-1] if latitude_deg > 0.0 else self.elevation_grid_m[0]
            if self.registration == "gridline":
                row = row[:-1]
            return float(np.mean(row))
        lat = self.latitude_deg
        if self.registration == "pixel" and latitude_deg < lat[0]:
            ring = self._longitude_interpolate(self.elevation_grid_m[0], longitude_deg)
            pole = float(np.mean(self.elevation_grid_m[0]))
            weight = (latitude_deg + 90.0) / (lat[0] + 90.0)
            return float((1.0 - weight) * pole + weight * ring)
        if self.registration == "pixel" and latitude_deg > lat[-1]:
            ring = self._longitude_interpolate(self.elevation_grid_m[-1], longitude_deg)
            pole = float(np.mean(self.elevation_grid_m[-1]))
            weight = (90.0 - latitude_deg) / (90.0 - lat[-1])
            return float((1.0 - weight) * pole + weight * ring)
        right = int(np.searchsorted(lat, latitude_deg, side="right"))
        right = min(max(right, 1), lat.size - 1)
        left = right - 1
        if np.isclose(latitude_deg, lat[left], atol=1e-14):
            return self._longitude_interpolate(self.elevation_grid_m[left], longitude_deg)
        if np.isclose(latitude_deg, lat[right], atol=1e-14):
            return self._longitude_interpolate(self.elevation_grid_m[right], longitude_deg)
        weight = (latitude_deg - lat[left]) / (lat[right] - lat[left])
        lower = self._longitude_interpolate(self.elevation_grid_m[left], longitude_deg)
        upper = self._longitude_interpolate(self.elevation_grid_m[right], longitude_deg)
        return float((1.0 - weight) * lower + weight * upper)

    def surface_radius_m(self, latitude_rad: float, longitude_rad: float) -> float:
        return float(self.reference_radius_m + self.elevation_m(latitude_rad, longitude_rad))


@dataclass(frozen=True)
class TerrainLocation:
    latitude_deg: float
    longitude_deg_east: float
    terrain_elevation_m: float
    terrain_clearance_m: float


@dataclass(frozen=True)
class TerrainClearanceReport:
    minimum_clearance_m: float
    minimum_time_s: float
    minimum_location: TerrainLocation
    impacted: bool
    impact_time_s: float | None
    impact_location: TerrainLocation | None


@dataclass(frozen=True)
class TerrainPropagationResult:
    solution: Any
    clearance: TerrainClearanceReport


def _validate_terrain_frame(terrain: TerrainShapeModel, terrain_frame: str) -> None:
    if not terrain_frame or terrain_frame != terrain.frame:
        raise ValueError(
            f"terrain frame mismatch: terrain is '{terrain.frame}' but rotation was declared for '{terrain_frame}'"
        )


def _body_fixed_location(position_body_fixed_m: ArrayLike) -> tuple[float, float, float]:
    position = np.asarray(position_body_fixed_m, dtype=float)
    if position.shape != (3,) or not np.all(np.isfinite(position)):
        raise ValueError("position must be a finite three-vector")
    radius = float(np.linalg.norm(position))
    if radius == 0.0:
        raise ValueError("position cannot be the lunar center")
    latitude = float(np.arctan2(position[2], np.hypot(position[0], position[1])))
    longitude = float(np.arctan2(position[1], position[0]) % (2.0 * np.pi))
    return radius, latitude, longitude


def terrain_clearance_m(time_s: float, position_inertial_m: ArrayLike, terrain: TerrainShapeModel, terrain_body_fixed_from_inertial: RotationProvider, *, terrain_frame: str) -> float:
    """Return radial clearance above terrain after explicit frame transformation."""
    _validate_terrain_frame(terrain, terrain_frame)
    position = np.asarray(position_inertial_m, dtype=float)
    if position.shape != (3,) or not np.all(np.isfinite(position)):
        raise ValueError("position must be a finite three-vector")
    rotation = validate_rotation_matrix(terrain_body_fixed_from_inertial(float(time_s)))
    body_fixed = rotation @ position
    radius, latitude, longitude = _body_fixed_location(body_fixed)
    return float(radius - terrain.surface_radius_m(latitude, longitude))


def terrain_location(time_s: float, position_inertial_m: ArrayLike, terrain: TerrainShapeModel, terrain_body_fixed_from_inertial: RotationProvider, *, terrain_frame: str) -> TerrainLocation:
    _validate_terrain_frame(terrain, terrain_frame)
    position = np.asarray(position_inertial_m, dtype=float)
    if position.shape != (3,) or not np.all(np.isfinite(position)):
        raise ValueError("position must be a finite three-vector")
    rotation = validate_rotation_matrix(terrain_body_fixed_from_inertial(float(time_s)))
    body_fixed = rotation @ position
    radius, latitude, longitude = _body_fixed_location(body_fixed)
    elevation = terrain.elevation_m(latitude, longitude)
    return TerrainLocation(float(np.rad2deg(latitude)), float(np.rad2deg(longitude) % 360.0), float(elevation), float(radius - terrain.reference_radius_m - elevation))


def make_terrain_impact_event(terrain: TerrainShapeModel, terrain_body_fixed_from_inertial: RotationProvider, *, terrain_frame: str):
    """Return a terminal descending zero-clearance event for solve_ivp."""
    _validate_terrain_frame(terrain, terrain_frame)
    def terrain_event(time_s: float, state: ArrayLike, *_args: object) -> float:
        return terrain_clearance_m(time_s, np.asarray(state, dtype=float)[:3], terrain, terrain_body_fixed_from_inertial, terrain_frame=terrain_frame)
    terrain_event.terminal = True  # type: ignore[attr-defined]
    terrain_event.direction = -1.0  # type: ignore[attr-defined]
    return terrain_event


def _validate_sample_times(duration_s: float, sample_times_s: ArrayLike | None) -> FloatArray | None:
    if sample_times_s is None:
        return None
    times = np.asarray(sample_times_s, dtype=float)
    if times.ndim != 1 or times.size == 0 or not np.all(np.isfinite(times)):
        raise ValueError("sample_times_s must be a non-empty finite one-dimensional array")
    if times[0] < 0.0 or times[-1] > duration_s or np.any(np.diff(times) <= 0.0):
        raise ValueError("sample_times_s must be strictly increasing within [0, duration_s]")
    return times


def _clearance_at_solution_time(solution: Any, time_s: float, terrain: TerrainShapeModel, rotation: RotationProvider, terrain_frame: str) -> float:
    if solution.sol is None:
        raise ValueError("terrain clearance refinement requires dense output")
    state = np.asarray(solution.sol(float(time_s)), dtype=float)
    return terrain_clearance_m(float(time_s), state[:3], terrain, rotation, terrain_frame=terrain_frame)


def analyze_terrain_clearance(solution: Any, terrain: TerrainShapeModel, terrain_body_fixed_from_inertial: RotationProvider, *, terrain_frame: str, search_samples: int = 2049) -> TerrainClearanceReport:
    """Find minimum terrain clearance and impact geometry from a dense solution."""
    _validate_terrain_frame(terrain, terrain_frame)
    if search_samples < 3:
        raise ValueError("search_samples must be at least three")
    if solution.sol is None:
        raise ValueError("solution must contain dense output")
    dense_times = getattr(solution.sol, "ts", None)
    if dense_times is None or len(dense_times) < 2:
        start = float(solution.t[0])
        finish = float(solution.t[-1])
    else:
        start = float(dense_times[0])
        finish = float(dense_times[-1])
    if getattr(solution, "t_events", None) and len(solution.t_events[0]):
        finish = float(solution.t_events[0][0])
    scan_times = np.linspace(start, finish, int(search_samples))
    clearances = np.array([_clearance_at_solution_time(solution, t, terrain, terrain_body_fixed_from_inertial, terrain_frame) for t in scan_times], dtype=float)
    index = int(np.argmin(clearances))
    best_time = float(scan_times[index])
    best_clearance = float(clearances[index])
    if 0 < index < scan_times.size - 1 and finish > start:
        lower, upper = float(scan_times[index - 1]), float(scan_times[index + 1])
        refined = minimize_scalar(lambda t: _clearance_at_solution_time(solution, t, terrain, terrain_body_fixed_from_inertial, terrain_frame), bounds=(lower, upper), method="bounded", options={"xatol": max(1e-6, (upper - lower) * 1e-8)})
        if refined.success and float(refined.fun) < best_clearance:
            best_time, best_clearance = float(refined.x), float(refined.fun)
    impacted = bool(getattr(solution, "t_events", None) and len(solution.t_events[0]))
    if impacted:
        impact_time = float(solution.t_events[0][0])
        impact_state = np.asarray(solution.y_events[0][0], dtype=float)
        impact_location = terrain_location(impact_time, impact_state[:3], terrain, terrain_body_fixed_from_inertial, terrain_frame=terrain_frame)
        if impact_location.terrain_clearance_m <= best_clearance + 1e-5:
            best_time, best_clearance = impact_time, impact_location.terrain_clearance_m
    else:
        impact_time, impact_location = None, None
    minimum_state = np.asarray(solution.sol(best_time), dtype=float)
    minimum_location = terrain_location(best_time, minimum_state[:3], terrain, terrain_body_fixed_from_inertial, terrain_frame=terrain_frame)
    return TerrainClearanceReport(float(best_clearance), best_time, minimum_location, impacted, impact_time, impact_location)


def propagate_with_terrain(initial_state: ArrayLike, duration_s: float, acceleration: AccelerationFunction, terrain: TerrainShapeModel, terrain_body_fixed_from_inertial: RotationProvider, *, terrain_frame: str, sample_times_s: ArrayLike | None = None, settings: PropagationSettings = PropagationSettings(), clearance_search_samples: int = 2049) -> TerrainPropagationResult:
    """Propagate until terrain impact and return refined clearance diagnostics."""
    _validate_terrain_frame(terrain, terrain_frame)
    state0 = np.asarray(initial_state, dtype=float)
    if state0.shape != (6,) or not np.all(np.isfinite(state0)):
        raise ValueError("initial_state must be a finite six-vector")
    if not np.isfinite(duration_s) or duration_s <= 0.0:
        raise ValueError("duration_s must be finite and positive")
    if terrain_clearance_m(0.0, state0[:3], terrain, terrain_body_fixed_from_inertial, terrain_frame=terrain_frame) <= 0.0:
        raise ValueError("initial state is at or below the terrain surface")
    times = _validate_sample_times(float(duration_s), sample_times_s)
    def rhs(time_s: float, state: FloatArray) -> FloatArray:
        derivative = np.empty(6, dtype=float)
        derivative[:3] = state[3:]
        derivative[3:] = acceleration(time_s, state[:3])
        return derivative
    event = make_terrain_impact_event(terrain, terrain_body_fixed_from_inertial, terrain_frame=terrain_frame)
    solution = solve_ivp(rhs, (0.0, float(duration_s)), state0, method=settings.method, t_eval=times, rtol=settings.rtol, atol=settings.atol, max_step=settings.max_step_s, events=event, dense_output=True)
    report = analyze_terrain_clearance(solution, terrain, terrain_body_fixed_from_inertial, terrain_frame=terrain_frame, search_samples=clearance_search_samples)
    return TerrainPropagationResult(solution=solution, clearance=report)


def save_terrain_npz(terrain: RegularLatLonTerrain, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, latitude_deg=terrain.latitude_deg, longitude_deg_east=terrain.longitude_deg_east, elevation_grid_m=terrain.elevation_grid_m.astype(np.float32), reference_radius_m=np.array(terrain.reference_radius_m), name=np.array(terrain.name), frame=np.array(terrain.frame), registration=np.array(terrain.registration), source=np.array(terrain.source or ""))


def load_terrain_npz(path: str | Path) -> RegularLatLonTerrain:
    with np.load(path, allow_pickle=False) as data:
        return RegularLatLonTerrain(np.asarray(data["latitude_deg"], dtype=float), np.asarray(data["longitude_deg_east"], dtype=float), np.asarray(data["elevation_grid_m"], dtype=float), float(data["reference_radius_m"]), str(data["name"]), str(data["frame"]), str(data["registration"]), str(data["source"]) or None)


def _read_gmt_netcdf(path: str | Path, stride: int) -> tuple[FloatArray, FloatArray, FloatArray]:
    try:
        from netCDF4 import Dataset  # type: ignore[import-not-found]
    except ImportError:
        Dataset = None  # type: ignore[assignment,misc]
    if Dataset is not None:
        with Dataset(path, "r") as dataset:
            variables = dataset.variables
            lon_name = next((n for n in ("lon", "longitude", "x") if n in variables), None)
            lat_name = next((n for n in ("lat", "latitude", "y") if n in variables), None)
            z_name = next((n for n in ("z", "elevation", "height") if n in variables), None)
            if lon_name is None or lat_name is None or z_name is None:
                raise ValueError("GMT grid does not contain recognizable x/y/z variables")
            lon = np.asarray(variables[lon_name][::stride], dtype=float)
            lat = np.asarray(variables[lat_name][::stride], dtype=float)
            z_var = variables[z_name]
            dims = tuple(z_var.dimensions)
            z = np.asarray(z_var[::stride, ::stride], dtype=float)
            if dims[-2:] == (lon_name, lat_name):
                z = z.T
            elif dims[-2:] != (lat_name, lon_name):
                raise ValueError("GMT elevation dimensions do not match latitude/longitude")
            return lon, lat, z
    from scipy.io import netcdf_file
    with netcdf_file(path, "r", mmap=False) as dataset:
        variables = dataset.variables
        lon_name = next((n for n in ("lon", "longitude", "x") if n in variables), None)
        lat_name = next((n for n in ("lat", "latitude", "y") if n in variables), None)
        z_name = next((n for n in ("z", "elevation", "height") if n in variables), None)
        if lon_name is None or lat_name is None or z_name is None:
            raise ValueError("GMT grid does not contain recognizable x/y/z variables")
        lon = np.asarray(variables[lon_name].data[::stride], dtype=float)
        lat = np.asarray(variables[lat_name].data[::stride], dtype=float)
        z_var = variables[z_name]
        dims = tuple(z_var.dimensions)
        z = np.asarray(z_var.data[::stride, ::stride], dtype=float)
        if dims[-2:] == (lon_name, lat_name):
            z = z.T
        elif dims[-2:] != (lat_name, lon_name):
            raise ValueError("GMT elevation dimensions do not match latitude/longitude")
        return lon, lat, z


def _normalize_global_coordinates(longitude_deg: FloatArray, latitude_deg: FloatArray, elevation_m: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray]:
    lon, lat, z = np.asarray(longitude_deg, dtype=float), np.asarray(latitude_deg, dtype=float), np.asarray(elevation_m, dtype=float)
    if lat[0] > lat[-1]:
        lat, z = lat[::-1].copy(), z[::-1, :].copy()
    if lon[0] < -1e-8:
        normalized = np.mod(lon, 360.0)
        order = np.argsort(normalized)
        lon, z = normalized[order], z[:, order]
    return lon.astype(float), lat.astype(float), z.astype(float)


def load_lola_moon_pa_grd(path: str | Path, *, registration: str = "gridline", stride: int = 1) -> RegularLatLonTerrain:
    """Load NASA PGDA's 2024 LOLA MOON_PA global GMT/netCDF shape grid.

    The archived GMT/netCDF ``z`` array stores elevation in kilometres above
    the 1737.4 km reference sphere. The public terrain API always uses metres,
    so the product-specific loader converts those samples to metres here.
    """
    if stride < 1:
        raise ValueError("stride must be at least one")
    lon, lat, elevation_km = _read_gmt_netcdf(path, int(stride))
    lon, lat, elevation_km = _normalize_global_coordinates(lon, lat, elevation_km)
    expected_spacing_deg = float(stride) / 64.0
    if not np.isclose(np.median(np.diff(lon)), expected_spacing_deg, rtol=0.0, atol=1e-8) or not np.isclose(np.median(np.diff(lat)), expected_spacing_deg, rtol=0.0, atol=1e-8):
        raise ValueError("LOLA MOON_PA loader expects the 64 pixels/degree global product")
    if registration == "gridline" and (not np.isclose(lon[0], 0.0) or not np.isclose(lon[-1], 360.0) or not np.isclose(lat[0], -90.0) or not np.isclose(lat[-1], 90.0)):
        raise ValueError("gridline LOLA product must retain 0/360 and -90/+90 boundaries")
    elevation_m = elevation_km * _LOLA_MOON_PA_GRID_KM_TO_M
    source = LOLA_MOON_PA_64_GRIDLINE_URL if registration == "gridline" else LOLA_MOON_PA_64_PIXEL_URL
    return RegularLatLonTerrain(lat, lon, elevation_m, LOLA_REFERENCE_RADIUS_M, f"LOLA MOON_PA 64 ppd {registration} shape grid", LOLA_MOON_PA_DE421_FRAME, registration, source)


def _pds_value(label: str, key: str) -> str:
    match = re.search(rf"^\s*{re.escape(key)}\s*=\s*(.+?)\s*$", label, re.MULTILINE)
    if match is None:
        raise ValueError(f"PDS label is missing {key}")
    value = re.sub(r"\s*<[^>]+>\s*$", "", match.group(1).strip()).strip()
    return value.strip("'\"")


def load_lola_pds_global_gdr(image_path: str | Path, label_path: str | Path) -> RegularLatLonTerrain:
    """Load a signed 16-bit global pixel-registered LOLA PDS3 cylindrical GDR."""
    label = Path(label_path).read_text(encoding="ascii")
    lines, samples = int(_pds_value(label, "LINES")), int(_pds_value(label, "LINE_SAMPLES"))
    bits, sample_type = int(_pds_value(label, "SAMPLE_BITS")), _pds_value(label, "SAMPLE_TYPE").upper()
    scale, reference_radius = float(_pds_value(label, "SCALING_FACTOR")), float(_pds_value(label, "OFFSET"))
    resolution = float(_pds_value(label, "MAP_RESOLUTION"))
    longitude_direction = _pds_value(label, "POSITIVE_LONGITUDE_DIRECTION").upper()
    frame = _pds_value(label, "COORDINATE_SYSTEM_NAME")
    if bits != 16 or "INTEGER" not in sample_type or "UNSIGNED" in sample_type:
        raise ValueError("LOLA GDR reader currently supports signed 16-bit elevation products")
    if longitude_direction != "EAST":
        raise ValueError("LOLA terrain reader requires east-positive longitude")
    endian = ">" if sample_type.startswith("MSB") else "<" if sample_type.startswith("LSB") else None
    if endian is None:
        raise ValueError(f"unsupported PDS SAMPLE_TYPE: {sample_type}")
    expected_size = lines * samples * 2
    if Path(image_path).stat().st_size != expected_size:
        raise ValueError("PDS image byte size does not match label dimensions")
    raw = np.memmap(image_path, mode="r", dtype=np.dtype(f"{endian}i2"), shape=(lines, samples))
    elevation = np.asarray(raw, dtype=np.float32) * scale
    if lines != int(round(180.0 * resolution)) or samples != int(round(360.0 * resolution)):
        raise ValueError("PDS global cylindrical dimensions do not match MAP_RESOLUTION")
    half = 0.5 / resolution
    latitude = np.linspace(90.0 - half, -90.0 + half, lines, dtype=float)[::-1]
    longitude = np.linspace(half, 360.0 - half, samples, dtype=float)
    return RegularLatLonTerrain(latitude, longitude, elevation[::-1, :], reference_radius, Path(image_path).stem, frame, "pixel", str(image_path))
