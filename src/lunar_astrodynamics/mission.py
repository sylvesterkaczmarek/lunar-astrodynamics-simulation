"""Configuration-driven lunar mission-analysis workflows.

This module is deliberately a thin orchestration layer. It builds explicit
frames, forces, terrain, propagation and downstream analyses from a TOML
configuration, then calls the same scientific functions exposed by the package.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from importlib import metadata
from pathlib import Path
from typing import Any, Mapping, Sequence
import csv
import json
import sys

import numpy as np
from numpy.typing import NDArray

from .access import (
    CoverageGrid,
    CoverageResult,
    GroundTrackHistory,
    LunarSurfaceSite,
    MultiSiteAccessReport,
    analyze_earth_visibility,
    analyze_multiple_site_access,
    coverage_analysis,
    ground_track_history,
)
from .analysis import OrbitHistory, orbit_history
from .constants import (
    EARTH_GM_DE440_M3_S2,
    MOON_MEAN_RADIUS_M,
    SUN_GM_DE440_M3_S2,
)
from .elements import ClassicalElements, state_from_elements
from .ephemeris import SpiceEphemeris, spice_ephemeris_from_utc
from .fidelity import (
    AccelerationFidelityReport,
    FidelitySelectionResult,
    FidelityTolerance,
    HarmonicTruncation,
    TrajectoryFidelityReport,
    compare_harmonic_accelerations,
    compare_harmonic_trajectories,
    default_harmonic_truncations,
    select_lowest_harmonic_truncation,
)
from .forces import ForceComponent, SolarRadiationPressure, ThirdBodyGravity
from .frames import RotationProvider, spice_rotation_provider
from .harmonics import SphericalHarmonicModel, read_shadr
from .propagation import PropagationSettings, propagate_with_acceleration
from .stability import (
    CoarseToFineSearchResult,
    RefinementSettings,
    SearchDynamics,
    StabilityConstraints,
    StabilitySearchResult,
    StabilitySearchSettings,
    StabilitySearchSpace,
    harmonic_ensemble_dynamics,
    harmonic_search_dynamics,
    j2_search_dynamics,
    run_coarse_to_fine_search,
    run_stability_search,
)
from .terrain import (
    RegularLatLonTerrain,
    TerrainShapeModel,
    load_lola_moon_pa_grd,
    load_lola_pds_global_gdr,
    load_terrain_npz,
    propagate_with_terrain,
)
from .uncertainty import (
    EnsembleUncertaintyResult,
    load_grgm1200a_clone_ensemble,
    propagate_gravity_ensemble,
    sample_independent_coefficient_uncertainty,
)

FloatArray = NDArray[np.float64]


def _toml_load(path: Path) -> dict[str, Any]:
    try:
        if sys.version_info >= (3, 11):
            import tomllib
        else:  # pragma: no cover - exercised on Python 3.10 CI
            import tomli as tomllib  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Python 3.10 requires the package dependency 'tomli'") from exc
    with path.open("rb") as stream:
        data = tomllib.load(stream)
    if not isinstance(data, dict):
        raise ValueError("mission TOML root must be a table")
    return data


def _table(mapping: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = mapping.get(key, {})
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"[{key}] must be a TOML table")
    return value


def _required(mapping: Mapping[str, Any], key: str) -> Any:
    if key not in mapping:
        raise ValueError(f"missing required configuration key '{key}'")
    return mapping[key]


def _number(mapping: Mapping[str, Any], key: str, default: float | None = None) -> float:
    value = mapping.get(key, default)
    if value is None:
        raise ValueError(f"missing required numeric key '{key}'")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"configuration key '{key}' must be finite")
    return result


def _integer(mapping: Mapping[str, Any], key: str, default: int | None = None) -> int:
    value = mapping.get(key, default)
    if value is None:
        raise ValueError(f"missing required integer key '{key}'")
    result = int(value)
    if result != float(value):
        raise ValueError(f"configuration key '{key}' must be an integer")
    return result


def _boolean(mapping: Mapping[str, Any], key: str, default: bool = False) -> bool:
    value = mapping.get(key, default)
    if not isinstance(value, bool):
        raise ValueError(f"configuration key '{key}' must be true or false")
    return value


def _string(mapping: Mapping[str, Any], key: str, default: str | None = None) -> str:
    value = mapping.get(key, default)
    if value is None or not isinstance(value, str) or not value.strip():
        raise ValueError(f"configuration key '{key}' must be a non-empty string")
    return value.strip()


def _float_tuple(values: object, *, name: str) -> tuple[float, ...]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be an array")
    result = tuple(float(value) for value in values)
    if not result or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain finite values")
    return result


def _path(base: Path, value: str) -> Path:
    candidate = Path(value).expanduser()
    return candidate if candidate.is_absolute() else (base / candidate).resolve()


def _jsonable(value: object) -> object:
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "to_dict"):
        return _jsonable(value.to_dict())  # type: ignore[attr-defined]
    if hasattr(value, "as_dict"):
        return _jsonable(value.as_dict())  # type: ignore[attr-defined]
    if hasattr(value, "__dataclass_fields__"):
        return _jsonable(asdict(value))
    return str(value)


@dataclass(frozen=True)
class GravityConfig:
    model: str = "j2"
    path: Path | None = None
    degree: int | None = None
    order: int | None = None
    frame: str | None = None
    name: str | None = None

    def __post_init__(self) -> None:
        if self.model not in {"central", "j2", "shadr"}:
            raise ValueError("gravity model must be 'central', 'j2', or 'shadr'")
        if self.model == "shadr" and self.path is None:
            raise ValueError("SHADR gravity requires gravity.path")
        if self.degree is not None and self.degree < 0:
            raise ValueError("gravity degree must be non-negative")
        if self.order is not None and self.order < 0:
            raise ValueError("gravity order must be non-negative")
        if self.degree is not None and self.order is not None and self.order > self.degree:
            raise ValueError("gravity order cannot exceed degree")


@dataclass(frozen=True)
class SpiceConfig:
    enabled: bool = False
    kernels: tuple[Path, ...] = ()
    inertial_frame: str = "J2000"
    gravity_frame: str | None = None
    surface_frame: str | None = None
    observer: str = "MOON"

    def __post_init__(self) -> None:
        if self.enabled and not self.kernels:
            raise ValueError("SPICE is enabled but no kernels were configured")


@dataclass(frozen=True)
class PerturbationConfig:
    earth: bool = False
    sun: bool = False


@dataclass(frozen=True)
class SrpConfig:
    enabled: bool = False
    spacecraft_mass_kg: float | None = None
    illuminated_area_m2: float | None = None
    reflectivity_coefficient: float | None = None
    lunar_shadow: bool = True

    def __post_init__(self) -> None:
        if self.enabled:
            values = (
                self.spacecraft_mass_kg,
                self.illuminated_area_m2,
                self.reflectivity_coefficient,
            )
            if any(value is None for value in values):
                raise ValueError("enabled SRP requires mass, illuminated area and reflectivity coefficient")


@dataclass(frozen=True)
class TerrainConfig:
    kind: str = "none"
    path: Path | None = None
    image_path: Path | None = None
    label_path: Path | None = None
    registration: str = "gridline"
    stride: int = 1
    frame: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in {"none", "npz", "lola_moon_pa", "lola_pds"}:
            raise ValueError("terrain kind must be none, npz, lola_moon_pa, or lola_pds")
        if self.kind in {"npz", "lola_moon_pa"} and self.path is None:
            raise ValueError(f"terrain kind '{self.kind}' requires terrain.path")
        if self.kind == "lola_pds" and (self.image_path is None or self.label_path is None):
            raise ValueError("lola_pds terrain requires image_path and label_path")
        if self.stride < 1:
            raise ValueError("terrain stride must be at least one")


@dataclass(frozen=True)
class SiteConfig:
    name: str
    latitude_deg: float
    longitude_deg_east: float
    elevation_m: float | None = 0.0
    use_terrain_elevation: bool = False
    coordinate_source: str | None = None


@dataclass(frozen=True)
class CoverageConfig:
    enabled: bool = False
    latitude_min_deg: float = -90.0
    latitude_max_deg: float = 90.0
    latitude_step_deg: float = 10.0
    longitude_min_deg_east: float = 0.0
    longitude_max_deg_east: float = 360.0
    longitude_step_deg: float = 10.0
    minimum_elevation_deg: float = 0.0


@dataclass(frozen=True)
class MissionConfig:
    name: str
    epoch_utc: str
    duration_s: float
    output_cadence_s: float
    state_kind: str
    cartesian_state: tuple[float, ...] | None
    elements: Mapping[str, float] | None
    gravity: GravityConfig
    spice: SpiceConfig
    perturbations: PerturbationConfig
    srp: SrpConfig
    terrain: TerrainConfig
    sites: tuple[SiteConfig, ...] = ()
    coverage: CoverageConfig = CoverageConfig()
    minimum_site_elevation_deg: float = 0.0
    terrain_aware_site_los: bool = False
    propagation: PropagationSettings = PropagationSettings()
    raw: Mapping[str, object] = field(default_factory=dict, repr=False, compare=False)
    source_path: Path | None = None

    def __post_init__(self) -> None:
        if self.duration_s <= 0.0 or not np.isfinite(self.duration_s):
            raise ValueError("mission duration_s must be finite and positive")
        if self.output_cadence_s <= 0.0 or not np.isfinite(self.output_cadence_s):
            raise ValueError("output_cadence_s must be finite and positive")
        if self.state_kind not in {"cartesian", "elements"}:
            raise ValueError("initial state kind must be 'cartesian' or 'elements'")
        if self.state_kind == "cartesian":
            if self.cartesian_state is None or len(self.cartesian_state) != 6:
                raise ValueError("Cartesian initial state requires six values")
        if self.state_kind == "elements" and self.elements is None:
            raise ValueError("element initial state requires [state.elements]")


@dataclass
class MissionContext:
    config: MissionConfig
    initial_state: FloatArray
    dynamics: SearchDynamics
    rotation: RotationProvider
    surface_frame: str
    ephemeris: SpiceEphemeris | None
    terrain: TerrainShapeModel | None
    sites: tuple[LunarSurfaceSite, ...]
    gravity_model: SphericalHarmonicModel | None
    additional_forces: tuple[ForceComponent, ...]
    loaded_kernel_paths: tuple[str, ...] = ()

    def close(self) -> None:
        if self.loaded_kernel_paths:
            try:
                import spiceypy as spice  # type: ignore[import-not-found]
            except ImportError:
                return
            spice.kclear()

    def provenance(self) -> dict[str, object]:
        packages: dict[str, str] = {}
        for package in ("numpy", "scipy", "spiceypy"):
            try:
                packages[package] = metadata.version(package)
            except metadata.PackageNotFoundError:
                continue
        return {
            "mission_name": self.config.name,
            "config_source": None if self.config.source_path is None else str(self.config.source_path),
            "epoch_utc": self.config.epoch_utc,
            "initial_state_m_m_s": self.initial_state.tolist(),
            "dynamics": self.dynamics.provenance(),
            "spice": None if self.ephemeris is None else self.ephemeris.provenance(),
            "surface_frame": self.surface_frame,
            "terrain": (
                None
                if self.terrain is None
                else {
                    "name": self.terrain.name,
                    "frame": self.terrain.frame,
                    "reference_radius_m": self.terrain.reference_radius_m,
                    "source": getattr(self.terrain, "source", None),
                }
            ),
            "surface_sites": [site.as_dict() for site in self.sites],
            "propagation": asdict(self.config.propagation),
            "output_cadence_s": self.config.output_cadence_s,
            "software": {"python": sys.version.split()[0], "packages": packages},
        }


@dataclass(frozen=True)
class MissionRun:
    context_provenance: Mapping[str, object]
    time_s: FloatArray
    states: FloatArray
    history: OrbitHistory
    impacted: bool
    impact_time_s: float | None
    minimum_terrain_clearance_m: float | None
    requested_duration_s: float

    @property
    def achieved_duration_s(self) -> float:
        return float(self.impact_time_s if self.impacted and self.impact_time_s is not None else self.time_s[-1])

    def summary_dict(self) -> dict[str, object]:
        stats = self.history.statistics
        return {
            "requested_duration_s": self.requested_duration_s,
            "achieved_duration_s": self.achieved_duration_s,
            "minimum_altitude_m": self.history.minimum_reference_altitude_m,
            "maximum_altitude_m": self.history.maximum_reference_altitude_m,
            "minimum_terrain_clearance_m": self.minimum_terrain_clearance_m,
            "minimum_periselene_altitude_m": float(np.min(self.history.periselene_altitude_m)),
            "maximum_periselene_altitude_m": float(np.max(self.history.periselene_altitude_m)),
            "periselene_peak_to_peak_m": float(np.ptp(self.history.periselene_altitude_m)),
            "minimum_aposelene_altitude_m": float(np.min(self.history.aposelene_altitude_m)),
            "maximum_aposelene_altitude_m": float(np.max(self.history.aposelene_altitude_m)),
            "aposelene_peak_to_peak_m": float(np.ptp(self.history.aposelene_altitude_m)),
            "eccentricity_minimum": float(np.min(self.history.eccentricity)),
            "eccentricity_maximum": float(np.max(self.history.eccentricity)),
            "eccentricity_linear_drift_over_span": stats.eccentricity.linear_drift_over_span,
            "eccentricity_vector_linear_drift_norm": float(
                np.linalg.norm(stats.eccentricity_vector.linear_drift_over_span)
            ),
            "maximum_orbital_plane_change_deg": (
                None
                if stats.orbital_plane_direction.maximum_change_rad is None
                else float(np.rad2deg(stats.orbital_plane_direction.maximum_change_rad))
            ),
            "impact": self.impacted,
            "impact_time_s": self.impact_time_s,
            "force_model_fidelity": self.context_provenance["dynamics"],
        }

    def human_summary(self) -> str:
        summary = self.summary_dict()
        dynamics = self.context_provenance["dynamics"]
        assert isinstance(dynamics, Mapping)
        extras = dynamics.get("additional_forces", [])
        perturbations = ", ".join(
            str(item.get("name", "force")) for item in extras if isinstance(item, Mapping)
        ) or "none"
        terrain = self.context_provenance.get("terrain")
        clearance = summary["minimum_terrain_clearance_m"]
        clearance_text = "not evaluated" if clearance is None else f"{float(clearance)/1000.0:.3f} km"
        return "\n".join(
            (
                f"Mission: {self.context_provenance['mission_name']}",
                f"Duration: {self.achieved_duration_s/3600.0:.3f} h of {self.requested_duration_s/3600.0:.3f} h requested",
                f"Reference altitude: {float(summary['minimum_altitude_m'])/1000.0:.3f} to {float(summary['maximum_altitude_m'])/1000.0:.3f} km",
                f"Terrain clearance: {clearance_text}",
                f"Periselene: {float(summary['minimum_periselene_altitude_m'])/1000.0:.3f} to {float(summary['maximum_periselene_altitude_m'])/1000.0:.3f} km",
                f"Aposelene: {float(summary['minimum_aposelene_altitude_m'])/1000.0:.3f} to {float(summary['maximum_aposelene_altitude_m'])/1000.0:.3f} km",
                f"Eccentricity drift over span: {float(summary['eccentricity_linear_drift_over_span']):.6g}",
                f"Eccentricity-vector drift norm: {float(summary['eccentricity_vector_linear_drift_norm']):.6g}",
                f"Impact: {'yes' if self.impacted else 'no'}",
                f"Perturbations: {perturbations}",
                f"Gravity/fidelity: {dynamics.get('fidelity')} n={dynamics.get('harmonic_degree')} m={dynamics.get('harmonic_order')}",
                f"Terrain model: {'none' if terrain is None else terrain.get('name') if isinstance(terrain, Mapping) else terrain}",
            )
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "provenance": _jsonable(self.context_provenance),
            "summary": _jsonable(self.summary_dict()),
            "trajectory": {
                "time_s": self.time_s.tolist(),
                "state_m_m_s": self.states.T.tolist(),
                "reference_altitude_m": self.history.reference_radius_altitude_m.tolist(),
                "periselene_altitude_m": self.history.periselene_altitude_m.tolist(),
                "aposelene_altitude_m": self.history.aposelene_altitude_m.tolist(),
                "eccentricity": self.history.eccentricity.tolist(),
                "terrain_clearance_m": (
                    None if self.history.terrain_clearance_m is None else self.history.terrain_clearance_m.tolist()
                ),
            },
        }

    def write_json(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

    def write_csv(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        fields = [
            "time_s", "x_m", "y_m", "z_m", "vx_m_s", "vy_m_s", "vz_m_s",
            "reference_altitude_m", "periselene_altitude_m", "aposelene_altitude_m",
            "eccentricity", "terrain_clearance_m",
        ]
        with destination.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for index, time_value in enumerate(self.time_s):
                state = self.states[:, index]
                writer.writerow({
                    "time_s": float(time_value),
                    "x_m": float(state[0]), "y_m": float(state[1]), "z_m": float(state[2]),
                    "vx_m_s": float(state[3]), "vy_m_s": float(state[4]), "vz_m_s": float(state[5]),
                    "reference_altitude_m": float(self.history.reference_radius_altitude_m[index]),
                    "periselene_altitude_m": float(self.history.periselene_altitude_m[index]),
                    "aposelene_altitude_m": float(self.history.aposelene_altitude_m[index]),
                    "eccentricity": float(self.history.eccentricity[index]),
                    "terrain_clearance_m": None if self.history.terrain_clearance_m is None else float(self.history.terrain_clearance_m[index]),
                })

    def write_standard_plots(self, directory: str | Path) -> tuple[Path, ...]:
        output = Path(directory)
        output.mkdir(parents=True, exist_ok=True)
        paths = (output / "altitude.svg", output / "apsides.svg", output / "eccentricity.svg")
        _write_line_svg(
            paths[0], self.time_s / 3600.0,
            [self.history.reference_radius_altitude_m / 1000.0] + ([] if self.history.terrain_clearance_m is None else [self.history.terrain_clearance_m / 1000.0]),
            ["reference altitude"] + ([] if self.history.terrain_clearance_m is None else ["terrain clearance"]),
            "Elapsed time (h)", "Altitude / clearance (km)",
        )
        _write_line_svg(
            paths[1], self.time_s / 3600.0,
            [self.history.periselene_altitude_m / 1000.0, self.history.aposelene_altitude_m / 1000.0],
            ["periselene", "aposelene"], "Elapsed time (h)", "Osculating altitude (km)",
        )
        _write_line_svg(paths[2], self.time_s / 3600.0, [self.history.eccentricity], ["eccentricity"], "Elapsed time (h)", "Eccentricity")
        return paths


@dataclass(frozen=True)
class AccessWorkflowResult:
    provenance: Mapping[str, object]
    ground_track: GroundTrackHistory
    sites: MultiSiteAccessReport | None
    coverage: CoverageResult | None
    earth_visibility: object | None

    def to_dict(self) -> dict[str, object]:
        return {
            "provenance": _jsonable(self.provenance),
            "ground_track": self.ground_track.to_dict(),
            "site_access": None if self.sites is None else self.sites.to_dict(),
            "coverage": None if self.coverage is None else self.coverage.to_dict(),
            "earth_visibility": None if self.earth_visibility is None else self.earth_visibility.to_dict(),  # type: ignore[attr-defined]
        }

    def write_json(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")


@dataclass(frozen=True)
class FidelityWorkflowResult:
    provenance: Mapping[str, object]
    acceleration: AccelerationFidelityReport
    trajectory: TrajectoryFidelityReport
    acceleration_selection: FidelitySelectionResult | None
    trajectory_selection: FidelitySelectionResult | None

    def to_dict(self) -> dict[str, object]:
        return {
            "provenance": _jsonable(self.provenance),
            "acceleration": self.acceleration.to_dict(),
            "trajectory": self.trajectory.to_dict(),
            "acceleration_selection": None if self.acceleration_selection is None else self.acceleration_selection.as_dict(),
            "trajectory_selection": None if self.trajectory_selection is None else self.trajectory_selection.as_dict(),
        }


def load_mission_config(path: str | Path) -> MissionConfig:
    source = Path(path).expanduser().resolve()
    return mission_config_from_mapping(_toml_load(source), base_dir=source.parent, source_path=source)


def mission_config_from_mapping(data: Mapping[str, Any], *, base_dir: str | Path = ".", source_path: Path | None = None) -> MissionConfig:
    base = Path(base_dir).expanduser().resolve()
    mission = _table(data, "mission")
    state = _table(data, "state")
    gravity = _table(data, "gravity")
    spice = _table(data, "spice")
    perturbations = _table(data, "perturbations")
    srp = _table(data, "srp")
    terrain = _table(data, "terrain")
    integration = _table(data, "integration")
    access = _table(data, "access")
    coverage = _table(data, "coverage")

    state_kind = _string(state, "kind", "elements")
    cartesian: tuple[float, ...] | None = None
    elements: Mapping[str, float] | None = None
    if state_kind == "cartesian":
        cartesian = _float_tuple(_required(state, "cartesian"), name="state.cartesian")
        if len(cartesian) != 6:
            raise ValueError("state.cartesian must contain six values [x,y,z,vx,vy,vz]")
    elif state_kind == "elements":
        element_table = _table(state, "elements")
        elements = {key: float(value) for key, value in element_table.items()}
    else:
        raise ValueError("state.kind must be cartesian or elements")

    gravity_path = None if "path" not in gravity else _path(base, _string(gravity, "path"))
    gravity_config = GravityConfig(
        model=_string(gravity, "model", "j2").lower(),
        path=gravity_path,
        degree=None if "degree" not in gravity else _integer(gravity, "degree"),
        order=None if "order" not in gravity else _integer(gravity, "order"),
        frame=None if "frame" not in gravity else _string(gravity, "frame"),
        name=None if "name" not in gravity else _string(gravity, "name"),
    )
    kernels = tuple(_path(base, str(item)) for item in spice.get("kernels", []))
    spice_config = SpiceConfig(
        enabled=_boolean(spice, "enabled", False), kernels=kernels,
        inertial_frame=_string(spice, "inertial_frame", "J2000"),
        gravity_frame=None if "gravity_frame" not in spice else _string(spice, "gravity_frame"),
        surface_frame=None if "surface_frame" not in spice else _string(spice, "surface_frame"),
        observer=_string(spice, "observer", "MOON"),
    )
    terrain_config = TerrainConfig(
        kind=_string(terrain, "kind", "none").lower(),
        path=None if "path" not in terrain else _path(base, _string(terrain, "path")),
        image_path=None if "image_path" not in terrain else _path(base, _string(terrain, "image_path")),
        label_path=None if "label_path" not in terrain else _path(base, _string(terrain, "label_path")),
        registration=_string(terrain, "registration", "gridline"),
        stride=_integer(terrain, "stride", 1),
        frame=None if "frame" not in terrain else _string(terrain, "frame"),
    )

    raw_sites = data.get("sites", []) or []
    if not isinstance(raw_sites, Sequence) or isinstance(raw_sites, (str, bytes)):
        raise ValueError("[[sites]] must be an array of tables")
    site_configs: list[SiteConfig] = []
    for index, item in enumerate(raw_sites):
        if not isinstance(item, Mapping):
            raise ValueError(f"site {index} must be a table")
        site_configs.append(SiteConfig(
            name=_string(item, "name"), latitude_deg=_number(item, "latitude_deg"),
            longitude_deg_east=_number(item, "longitude_deg_east"),
            elevation_m=None if "elevation_m" not in item else _number(item, "elevation_m"),
            use_terrain_elevation=_boolean(item, "use_terrain_elevation", False),
            coordinate_source=None if "coordinate_source" not in item else _string(item, "coordinate_source"),
        ))

    settings = PropagationSettings(
        method=_string(integration, "method", "DOP853"),
        rtol=_number(integration, "rtol", 1e-11),
        position_atol_m=_number(integration, "position_atol_m", 1e-4),
        velocity_atol_m_s=_number(integration, "velocity_atol_m_s", 1e-7),
        max_step_s=float(integration.get("max_step_s", np.inf)),
    )
    coverage_config = CoverageConfig(
        enabled=_boolean(coverage, "enabled", False),
        latitude_min_deg=_number(coverage, "latitude_min_deg", -90.0),
        latitude_max_deg=_number(coverage, "latitude_max_deg", 90.0),
        latitude_step_deg=_number(coverage, "latitude_step_deg", 10.0),
        longitude_min_deg_east=_number(coverage, "longitude_min_deg_east", 0.0),
        longitude_max_deg_east=_number(coverage, "longitude_max_deg_east", 360.0),
        longitude_step_deg=_number(coverage, "longitude_step_deg", 10.0),
        minimum_elevation_deg=_number(coverage, "minimum_elevation_deg", 0.0),
    )
    return MissionConfig(
        name=_string(mission, "name", "lunar mission analysis"),
        epoch_utc=_string(mission, "epoch_utc", "2000-01-01T12:00:00"),
        duration_s=_number(mission, "duration_s", 86400.0),
        output_cadence_s=_number(mission, "output_cadence_s", 300.0),
        state_kind=state_kind, cartesian_state=cartesian, elements=elements,
        gravity=gravity_config, spice=spice_config,
        perturbations=PerturbationConfig(earth=_boolean(perturbations, "earth", False), sun=_boolean(perturbations, "sun", False)),
        srp=SrpConfig(
            enabled=_boolean(srp, "enabled", False),
            spacecraft_mass_kg=None if "spacecraft_mass_kg" not in srp else _number(srp, "spacecraft_mass_kg"),
            illuminated_area_m2=None if "illuminated_area_m2" not in srp else _number(srp, "illuminated_area_m2"),
            reflectivity_coefficient=None if "reflectivity_coefficient" not in srp else _number(srp, "reflectivity_coefficient"),
            lunar_shadow=_boolean(srp, "lunar_shadow", True),
        ),
        terrain=terrain_config, sites=tuple(site_configs), coverage=coverage_config,
        minimum_site_elevation_deg=_number(access, "minimum_elevation_deg", 0.0),
        terrain_aware_site_los=_boolean(access, "terrain_aware", False),
        propagation=settings, raw=dict(data), source_path=source_path,
    )


def _load_spice(config: MissionConfig) -> tuple[SpiceEphemeris | None, RotationProvider, str, tuple[str, ...]]:
    if not config.spice.enabled:
        if config.gravity.model == "shadr":
            raise ValueError("spherical-harmonic mission propagation requires explicit SPICE frame configuration")
        return None, (lambda _time_s: np.eye(3)), "INERTIAL_ALIGNED_FIXED", ()
    try:
        import spiceypy as spice  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError("SPICE mission configuration requires the optional spice dependency") from exc
    loaded: list[str] = []
    try:
        for kernel in config.spice.kernels:
            if not kernel.exists():
                raise FileNotFoundError(f"SPICE kernel not found: {kernel}")
            spice.furnsh(str(kernel)); loaded.append(str(kernel))
        ephemeris = spice_ephemeris_from_utc(config.epoch_utc, inertial_frame=config.spice.inertial_frame, observer=config.spice.observer)
        surface_frame = config.spice.surface_frame or config.spice.gravity_frame
        if not surface_frame:
            raise ValueError("SPICE configuration requires surface_frame or gravity_frame")
        rotation = spice_rotation_provider(config.spice.inertial_frame, surface_frame, et_offset_s=ephemeris.epoch_et_s)
        return ephemeris, rotation, surface_frame, tuple(loaded)
    except Exception:
        spice.kclear(); raise


def _load_terrain(config: MissionConfig, surface_frame: str) -> TerrainShapeModel | None:
    terrain = config.terrain
    if terrain.kind == "none": return None
    if terrain.kind == "npz":
        assert terrain.path is not None; model = load_terrain_npz(terrain.path)
    elif terrain.kind == "lola_moon_pa":
        assert terrain.path is not None
        model = load_lola_moon_pa_grd(terrain.path, registration=terrain.registration, stride=terrain.stride)
    else:
        assert terrain.image_path is not None and terrain.label_path is not None
        model = load_lola_pds_global_gdr(terrain.image_path, terrain.label_path)
    requested_frame = terrain.frame or surface_frame
    if model.frame == requested_frame: return model
    if model.frame == "MEAN EARTH/POLAR AXIS OF DE421" and requested_frame == "MOON_ME_DE421":
        return RegularLatLonTerrain(model.latitude_deg, model.longitude_deg_east, model.elevation_grid_m, reference_radius_m=model.reference_radius_m, name=model.name, frame=requested_frame, registration=model.registration, source=model.source)
    raise ValueError(f"terrain frame '{model.frame}' does not match configured surface frame '{requested_frame}'")


def _build_additional_forces(config: MissionConfig, ephemeris: SpiceEphemeris | None) -> tuple[ForceComponent, ...]:
    needs_ephemeris = config.perturbations.earth or config.perturbations.sun or config.srp.enabled
    if needs_ephemeris and ephemeris is None:
        raise ValueError("Earth/Sun/SRP perturbations require SPICE ephemerides")
    if ephemeris is None: return ()
    forces: list[ForceComponent] = []
    if config.perturbations.earth:
        forces.append(ThirdBodyGravity("Earth third-body gravity", EARTH_GM_DE440_M3_S2, ephemeris.position_provider("EARTH")))
    if config.perturbations.sun:
        forces.append(ThirdBodyGravity("Sun third-body gravity", SUN_GM_DE440_M3_S2, ephemeris.position_provider("SUN")))
    if config.srp.enabled:
        assert config.srp.spacecraft_mass_kg is not None and config.srp.illuminated_area_m2 is not None and config.srp.reflectivity_coefficient is not None
        forces.append(SolarRadiationPressure("solar radiation pressure", ephemeris.position_provider("SUN"), config.srp.spacecraft_mass_kg, config.srp.illuminated_area_m2, config.srp.reflectivity_coefficient, include_lunar_shadow=config.srp.lunar_shadow))
    return tuple(forces)


def _build_dynamics(config: MissionConfig, ephemeris: SpiceEphemeris | None, additional_forces: Sequence[ForceComponent]) -> tuple[SearchDynamics, SphericalHarmonicModel | None]:
    if config.gravity.model in {"central", "j2"}:
        j2_rotation = None
        if config.gravity.model == "j2" and ephemeris is not None:
            j2_frame = config.gravity.frame or config.spice.gravity_frame or config.spice.surface_frame
            if not j2_frame: raise ValueError("SPICE J2 configuration requires a lunar gravity/surface frame")
            j2_rotation = spice_rotation_provider(config.spice.inertial_frame, j2_frame, et_offset_s=ephemeris.epoch_et_s)
        base = j2_search_dynamics(include_j2=config.gravity.model == "j2", body_fixed_from_inertial=j2_rotation, analysis_reference_radius_m=MOON_MEAN_RADIUS_M)
        if not additional_forces: return base, None
        def acceleration(time_s: float, position_m: FloatArray) -> FloatArray:
            total = np.asarray(base.acceleration(time_s, position_m), dtype=float)
            for force in additional_forces: total = total + np.asarray(force(time_s, position_m), dtype=float)
            return total
        force_metadata = []
        for force in additional_forces:
            method = getattr(force, "provenance", None); force_metadata.append(dict(method()) if callable(method) else {"name": force.name})
        return SearchDynamics(name=f"{base.name} with perturbations", mu_m3_s2=base.mu_m3_s2, analysis_reference_radius_m=base.analysis_reference_radius_m, collision_radius_m=base.collision_radius_m, acceleration=acceleration, fidelity=f"{base.fidelity} + perturbation forces", harmonic_degree=base.harmonic_degree, harmonic_order=base.harmonic_order, parallel_safe=False, provenance_data={**dict(base.provenance_data), "additional_forces": force_metadata}), None

    if ephemeris is None: raise ValueError("SHADR gravity requires a SPICE epoch/frame context")
    assert config.gravity.path is not None
    gravity_frame = config.gravity.frame or config.spice.gravity_frame
    if not gravity_frame: raise ValueError("SHADR gravity requires gravity.frame or spice.gravity_frame")
    degree = config.gravity.degree
    model = read_shadr(config.gravity.path, max_degree=degree, name=config.gravity.name or config.gravity.path.name, frame=gravity_frame)
    degree = model.max_degree if degree is None else min(degree, model.max_degree)
    order = degree if config.gravity.order is None else min(config.gravity.order, degree)
    gravity_rotation = spice_rotation_provider(config.spice.inertial_frame, gravity_frame, et_offset_s=ephemeris.epoch_et_s)
    return harmonic_search_dynamics(model, gravity_rotation, max_degree=degree, max_order=order, additional_forces=additional_forces, analysis_reference_radius_m=MOON_MEAN_RADIUS_M, collision_radius_m=MOON_MEAN_RADIUS_M, parallel_safe=False), model


def _initial_state(config: MissionConfig, mu_m3_s2: float) -> FloatArray:
    if config.state_kind == "cartesian":
        assert config.cartesian_state is not None
        state = np.asarray(config.cartesian_state, dtype=float)
        if state.shape != (6,) or not np.all(np.isfinite(state)): raise ValueError("initial Cartesian state must be a finite six-vector")
        return state
    assert config.elements is not None; elements = config.elements
    if "semi_major_axis_m" in elements: axis = float(elements["semi_major_axis_m"])
    elif "semi_major_axis_altitude_m" in elements: axis = MOON_MEAN_RADIUS_M + float(elements["semi_major_axis_altitude_m"])
    else: raise ValueError("state.elements requires semi_major_axis_m or semi_major_axis_altitude_m")
    required = ("eccentricity", "inclination_deg", "raan_deg", "argument_of_periapsis_deg", "true_anomaly_deg")
    missing = [key for key in required if key not in elements]
    if missing: raise ValueError(f"state.elements is missing: {', '.join(missing)}")
    return state_from_elements(ClassicalElements(axis, float(elements["eccentricity"]), float(np.deg2rad(elements["inclination_deg"])), float(np.deg2rad(elements["raan_deg"])), float(np.deg2rad(elements["argument_of_periapsis_deg"])), float(np.deg2rad(elements["true_anomaly_deg"]))), mu_m3_s2)


def _sample_times(config: MissionConfig) -> FloatArray:
    count = max(2, int(np.floor(config.duration_s / config.output_cadence_s)) + 1)
    times = np.arange(count, dtype=float) * config.output_cadence_s
    if times[-1] < config.duration_s - 1e-9: times = np.concatenate((times, np.array([config.duration_s])))
    else: times[-1] = config.duration_s
    return times


def _build_sites(config: MissionConfig, terrain: TerrainShapeModel | None, surface_frame: str) -> tuple[LunarSurfaceSite, ...]:
    sites: list[LunarSurfaceSite] = []
    for item in config.sites:
        if item.use_terrain_elevation:
            if terrain is None: raise ValueError(f"site '{item.name}' requests terrain elevation without terrain")
            elevation = terrain.elevation_m(np.deg2rad(item.latitude_deg), np.deg2rad(item.longitude_deg_east))
        else: elevation = 0.0 if item.elevation_m is None else item.elevation_m
        sites.append(LunarSurfaceSite(item.name, item.latitude_deg, item.longitude_deg_east, elevation_m=float(elevation), frame=surface_frame, reference_radius_m=MOON_MEAN_RADIUS_M if terrain is None else terrain.reference_radius_m, coordinate_source=item.coordinate_source))
    if len({site.name for site in sites}) != len(sites): raise ValueError("surface site names must be unique")
    return tuple(sites)


def build_mission_context(config: MissionConfig) -> MissionContext:
    ephemeris, rotation, surface_frame, loaded = _load_spice(config)
    try:
        terrain = _load_terrain(config, surface_frame)
        additional = _build_additional_forces(config, ephemeris)
        dynamics, gravity_model = _build_dynamics(config, ephemeris, additional)
        state = _initial_state(config, dynamics.mu_m3_s2)
        sites = _build_sites(config, terrain, surface_frame)
        return MissionContext(config, state, dynamics, rotation, surface_frame, ephemeris, terrain, sites, gravity_model, tuple(additional), loaded)
    except Exception:
        if loaded:
            try:
                import spiceypy as spice  # type: ignore[import-not-found]
                spice.kclear()
            except ImportError: pass
        raise


def run_mission(context: MissionContext) -> MissionRun:
    config = context.config; sample_times = _sample_times(config); minimum_clearance: float | None = None
    if context.terrain is None:
        solution = propagate_with_acceleration(context.initial_state, config.duration_s, context.dynamics.acceleration, collision_radius_m=context.dynamics.collision_radius_m, sample_times_s=sample_times, settings=config.propagation)
    else:
        terrain_result = propagate_with_terrain(context.initial_state, config.duration_s, context.dynamics.acceleration, context.terrain, context.rotation, terrain_frame=context.surface_frame, sample_times_s=sample_times, settings=config.propagation, clearance_search_samples=max(257, int(2 * sample_times.size + 1)))
        solution = terrain_result.solution; minimum_clearance = float(terrain_result.clearance.minimum_clearance_m)
    if not solution.success: raise RuntimeError(f"mission propagation failed: {solution.message}")
    time = np.asarray(solution.t, dtype=float); states = np.asarray(solution.y, dtype=float); event_times = solution.t_events[0]
    impacted = bool(len(event_times)); impact_time = float(event_times[0]) if impacted else None
    if impacted:
        event_state = np.asarray(solution.y_events[0][0], dtype=float)
        if time.size == 0 or abs(float(time[-1]) - impact_time) > 1e-9:
            time = np.concatenate((time, np.array([impact_time]))); states = np.column_stack((states, event_state))
    if time.size < 2: raise RuntimeError("mission propagation returned fewer than two trajectory samples")
    history = orbit_history(time, states, context.dynamics.mu_m3_s2, reference_radius_m=context.dynamics.analysis_reference_radius_m, terrain=context.terrain, terrain_body_fixed_from_inertial=None if context.terrain is None else context.rotation, terrain_frame=None if context.terrain is None else context.surface_frame)
    if minimum_clearance is None and history.terrain_clearance_m is not None: minimum_clearance = float(np.min(history.terrain_clearance_m))
    return MissionRun(context.provenance(), time, states, history, impacted, impact_time, minimum_clearance, config.duration_s)


def run_access_workflow(context: MissionContext, mission: MissionRun | None = None) -> AccessWorkflowResult:
    run = mission or run_mission(context); positions = run.states[:3].T
    track = ground_track_history(run.time_s, positions, context.rotation, body_fixed_frame=context.surface_frame, reference_radius_m=context.dynamics.analysis_reference_radius_m, terrain=context.terrain)
    site_report = None
    if context.sites:
        site_report = analyze_multiple_site_access(run.time_s, positions, context.sites, context.rotation, body_fixed_frame=context.surface_frame, minimum_elevation_deg=context.config.minimum_site_elevation_deg, terrain=context.terrain, terrain_aware=context.config.terrain_aware_site_los)
    coverage_report = None
    if context.config.coverage.enabled:
        c = context.config.coverage
        grid = CoverageGrid.regular(latitude_min_deg=c.latitude_min_deg, latitude_max_deg=c.latitude_max_deg, latitude_step_deg=c.latitude_step_deg, longitude_min_deg_east=c.longitude_min_deg_east, longitude_max_deg_east=c.longitude_max_deg_east, longitude_step_deg=c.longitude_step_deg, frame=context.surface_frame, reference_radius_m=context.dynamics.analysis_reference_radius_m)
        coverage_report = coverage_analysis(run.time_s, positions, grid, context.rotation, body_fixed_frame=context.surface_frame, minimum_elevation_deg=c.minimum_elevation_deg, terrain=context.terrain)
    earth_visibility = None if context.ephemeris is None else analyze_earth_visibility(run.time_s, positions, context.ephemeris.position_provider("EARTH"))
    return AccessWorkflowResult(context.provenance(), track, site_report, coverage_report, earth_visibility)


def _search_space_from_config(data: Mapping[str, Any]) -> StabilitySearchSpace:
    return StabilitySearchSpace(
        semi_major_axis_altitudes_m=None if "semi_major_axis_altitudes_km" not in data else tuple(1000.0 * value for value in _float_tuple(data["semi_major_axis_altitudes_km"], name="search.semi_major_axis_altitudes_km")),
        semi_major_axes_m=None if "semi_major_axes_km" not in data else tuple(1000.0 * value for value in _float_tuple(data["semi_major_axes_km"], name="search.semi_major_axes_km")),
        eccentricities=_float_tuple(data.get("eccentricities", [0.005, 0.02]), name="search.eccentricities"),
        inclinations_rad=tuple(np.deg2rad(_float_tuple(data.get("inclinations_deg", [85.0, 90.0, 95.0]), name="search.inclinations_deg"))),
        raan_rad=tuple(np.deg2rad(_float_tuple(data.get("raan_deg", [0.0]), name="search.raan_deg"))),
        periapsis_rad=tuple(np.deg2rad(_float_tuple(data.get("periapsis_deg", [90.0, 270.0]), name="search.periapsis_deg"))),
        initial_anomaly_rad=tuple(np.deg2rad(_float_tuple(data.get("initial_anomaly_deg", [0.0]), name="search.initial_anomaly_deg"))),
        periapsis_parameterization=_string(data, "periapsis_parameterization", "longitude_of_periapsis"),
    )


def _gravity_uncertainty_models(context: MissionContext) -> tuple[SphericalHarmonicModel, ...]:
    if context.gravity_model is None: raise ValueError("gravity uncertainty requires a SHADR gravity model")
    uncertainty = _table(context.config.raw, "uncertainty")
    if not uncertainty: raise ValueError("uncertainty workflow requires an [uncertainty] table")
    mode = _string(uncertainty, "mode", "diagonal")
    if mode == "diagonal":
        return sample_independent_coefficient_uncertainty(context.gravity_model, seed=_integer(uncertainty, "seed", 20260818), count=_integer(uncertainty, "samples", 8), sigma_scale=_number(uncertainty, "sigma_scale", 1.0), assume_independent=_boolean(uncertainty, "assume_independent", False), include_mu=_boolean(uncertainty, "include_mu", False))
    if mode == "clones":
        raw_paths = uncertainty.get("clone_paths", [])
        if not isinstance(raw_paths, Sequence) or isinstance(raw_paths, (str, bytes)) or not raw_paths: raise ValueError("clone uncertainty mode requires uncertainty.clone_paths")
        base = context.config.source_path.parent if context.config.source_path is not None else Path.cwd()
        return load_grgm1200a_clone_ensemble(context.gravity_model, tuple(_path(base, str(item)) for item in raw_paths), max_degree=context.dynamics.harmonic_degree)
    raise ValueError("uncertainty.mode must be 'diagonal' or 'clones'")


def run_frozen_orbit_workflow(context: MissionContext) -> StabilitySearchResult | CoarseToFineSearchResult:
    search = _table(context.config.raw, "search")
    if not search: raise ValueError("frozen-orbit workflow requires a [search] table")
    space = _search_space_from_config(search); duration = _number(search, "duration_s", context.config.duration_s)
    settings = StabilitySearchSettings(
        duration_s=duration,
        sample_count=_integer(search, "sample_count", max(33, int(duration / context.config.output_cadence_s) + 1)),
        propagation=context.config.propagation, workers=_integer(search, "workers", 1),
        constraints=StabilityConstraints(
            require_full_duration=_boolean(search, "require_full_duration", True),
            minimum_reference_altitude_m=None if "minimum_reference_altitude_km" not in search else 1000.0 * _number(search, "minimum_reference_altitude_km"),
            minimum_terrain_clearance_m=None if "minimum_terrain_clearance_km" not in search else 1000.0 * _number(search, "minimum_terrain_clearance_km"),
            maximum_periselene_spread_m=None if "maximum_periselene_spread_km" not in search else 1000.0 * _number(search, "maximum_periselene_spread_km"),
            maximum_eccentricity_variation=None if "maximum_eccentricity_variation" not in search else _number(search, "maximum_eccentricity_variation"),
        ),
    )
    uncertainty_dynamics = ()
    if _boolean(search, "use_uncertainty", False):
        if context.gravity_model is None or context.ephemeris is None: raise ValueError("uncertainty-aware frozen search requires SHADR gravity with SPICE")
        gravity_rotation = spice_rotation_provider(context.config.spice.inertial_frame, context.gravity_model.frame, et_offset_s=context.ephemeris.epoch_et_s)
        uncertainty_dynamics = harmonic_ensemble_dynamics(_gravity_uncertainty_models(context), gravity_rotation, max_degree=context.dynamics.harmonic_degree, max_order=context.dynamics.harmonic_order, additional_forces=context.additional_forces, analysis_reference_radius_m=context.dynamics.analysis_reference_radius_m, collision_radius_m=context.dynamics.collision_radius_m, parallel_safe=False)
    common = dict(terrain=context.terrain, terrain_body_fixed_from_inertial=None if context.terrain is None else context.rotation, terrain_frame=None if context.terrain is None else context.surface_frame, uncertainty_dynamics=uncertainty_dynamics)
    if _boolean(search, "refine", True):
        axes = search.get("refine_axes", ["semi_major_axis_m", "eccentricity", "inclination_rad"])
        if not isinstance(axes, Sequence) or isinstance(axes, (str, bytes)): raise ValueError("search.refine_axes must be an array")
        refinement = RefinementSettings(top_candidates=_integer(search, "top_candidates", 1), points_per_axis=_integer(search, "points_per_axis", 3), spacing_fraction=_number(search, "spacing_fraction", 0.5), refine_axes=tuple(str(item) for item in axes))
        return run_coarse_to_fine_search(space, context.dynamics, settings=settings, refinement=refinement, **common)
    return run_stability_search(space, context.dynamics, settings=settings, **common)


def run_uncertainty_workflow(context: MissionContext) -> EnsembleUncertaintyResult:
    if context.gravity_model is None: raise ValueError("gravity uncertainty workflow requires a SHADR gravity model")
    uncertainty = _table(context.config.raw, "uncertainty")
    if not uncertainty: raise ValueError("uncertainty workflow requires an [uncertainty] table")
    models = _gravity_uncertainty_models(context)
    levels = _float_tuple(uncertainty.get("percentiles", [5.0, 50.0, 95.0]), name="uncertainty.percentiles")
    if context.ephemeris is None: raise ValueError("SHADR uncertainty workflow requires SPICE")
    rotation = spice_rotation_provider(context.config.spice.inertial_frame, context.gravity_model.frame, et_offset_s=context.ephemeris.epoch_et_s)
    return propagate_gravity_ensemble(context.initial_state, _number(uncertainty, "duration_s", context.config.duration_s), models, rotation, collision_radius_m=context.dynamics.collision_radius_m, reference_radius_m=context.dynamics.analysis_reference_radius_m, max_degree=context.dynamics.harmonic_degree, max_order=context.dynamics.harmonic_order, sample_count=_integer(uncertainty, "sample_count", 129), settings=context.config.propagation, percentile_levels=levels)


def _truncations_from_config(data: Mapping[str, Any], model: SphericalHarmonicModel) -> tuple[HarmonicTruncation, ...]:
    if "truncations" in data:
        raw = data["truncations"]
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)): raise ValueError("fidelity.truncations must be an array")
        values = []
        for item in raw:
            if isinstance(item, Sequence) and not isinstance(item, (str, bytes)) and len(item) == 2: values.append(HarmonicTruncation(int(item[0]), int(item[1])))
            else: values.append(HarmonicTruncation(int(item), int(item)))
        return tuple(values)
    if "degrees" in data: return tuple(HarmonicTruncation(int(degree), int(degree)) for degree in _float_tuple(data["degrees"], name="fidelity.degrees"))
    return default_harmonic_truncations(model)


def _selection_tolerance(data: Mapping[str, Any], *, acceleration: bool) -> FidelityTolerance | None:
    kwargs: dict[str, float | bool] = {}
    if acceleration:
        if "maximum_relative_acceleration_error" in data: kwargs["maximum_relative_acceleration_error"] = _number(data, "maximum_relative_acceleration_error")
        if "maximum_absolute_acceleration_error_m_s2" in data: kwargs["maximum_absolute_acceleration_error_m_s2"] = _number(data, "maximum_absolute_acceleration_error_m_s2")
    else:
        for name in ("maximum_final_position_difference_m", "maximum_final_velocity_difference_m_s", "maximum_periselene_variation_difference_m", "maximum_eccentricity_variation_difference", "maximum_minimum_terrain_clearance_difference_m", "maximum_lifetime_difference_s"):
            if name in data: kwargs[name] = _number(data, name)
        if kwargs: kwargs["require_impact_match"] = _boolean(data, "require_impact_match", True)
    return None if not kwargs else FidelityTolerance(**kwargs)  # type: ignore[arg-type]


def run_fidelity_workflow(context: MissionContext, mission: MissionRun | None = None) -> FidelityWorkflowResult:
    if context.gravity_model is None or context.ephemeris is None: raise ValueError("fidelity workflow requires SHADR gravity with an explicit SPICE frame")
    fidelity = _table(context.config.raw, "fidelity")
    if not fidelity: raise ValueError("fidelity workflow requires a [fidelity] table")
    run = mission or run_mission(context); model = context.gravity_model
    gravity_rotation = spice_rotation_provider(context.config.spice.inertial_frame, model.frame, et_offset_s=context.ephemeris.epoch_et_s)
    truncations = _truncations_from_config(fidelity, model); reference_degree = _integer(fidelity, "reference_degree", model.max_degree); reference = HarmonicTruncation(reference_degree, _integer(fidelity, "reference_order", reference_degree))
    stride = max(1, _integer(fidelity, "acceleration_sample_stride", max(1, run.time_s.size // 24))); indices = np.arange(0, run.time_s.size, stride, dtype=int)
    if indices[-1] != run.time_s.size - 1: indices = np.concatenate((indices, np.array([run.time_s.size - 1])))
    acceleration = compare_harmonic_accelerations(model, gravity_rotation, run.states[:3, indices].T, times_s=run.time_s[indices], velocities_m_s=run.states[3:, indices].T, truncations=truncations, reference=reference, benchmark_repetitions=_integer(fidelity, "benchmark_repetitions", 2))
    trajectory = compare_harmonic_trajectories(model, gravity_rotation, context.initial_state, _number(fidelity, "trajectory_duration_s", min(context.config.duration_s, 7200.0)), truncations=truncations, reference=reference, additional_forces=context.additional_forces, analysis_reference_radius_m=context.dynamics.analysis_reference_radius_m, collision_radius_m=context.dynamics.collision_radius_m, sample_count=_integer(fidelity, "trajectory_sample_count", 81), propagation=context.config.propagation, terrain=context.terrain, terrain_body_fixed_from_inertial=None if context.terrain is None else context.rotation, terrain_frame=None if context.terrain is None else context.surface_frame)
    acc_tolerance = _selection_tolerance(fidelity, acceleration=True); traj_tolerance = _selection_tolerance(fidelity, acceleration=False)
    return FidelityWorkflowResult(context.provenance(), acceleration, trajectory, None if acc_tolerance is None else select_lowest_harmonic_truncation(acceleration, acc_tolerance), None if traj_tolerance is None else select_lowest_harmonic_truncation(trajectory, traj_tolerance))


def ensemble_to_dict(result: EnsembleUncertaintyResult, provenance: Mapping[str, object]) -> dict[str, object]:
    return {
        "provenance": _jsonable(provenance),
        "note": "This ensemble isolates gravity-field uncertainty; deterministic third-body and SRP forces are not added by propagate_gravity_ensemble.",
        "impact_fraction": result.impact_fraction,
        "percentile_levels": list(result.percentile_levels),
        "percentiles": {key: {str(level): value for level, value in values.items()} for key, values in result.percentiles.items()},
        "samples": [asdict(sample) for sample in result.samples],
    }


def _write_line_svg(path: Path, x: FloatArray, series: Sequence[FloatArray], labels: Sequence[str], x_label: str, y_label: str) -> None:
    width, height = 800, 420; left, right, top, bottom = 70, 25, 35, 60
    x_values = np.asarray(x, dtype=float); y_values = [np.asarray(item, dtype=float) for item in series]
    finite_parts = [item[np.isfinite(item)] for item in y_values if np.any(np.isfinite(item))]
    if x_values.size == 0 or not finite_parts: raise ValueError("plot data must contain finite values")
    finite_y = np.concatenate(finite_parts); xmin, xmax = float(np.min(x_values)), float(np.max(x_values)); ymin, ymax = float(np.min(finite_y)), float(np.max(finite_y))
    if xmax == xmin: xmax = xmin + 1.0
    if ymax == ymin:
        padding = max(1.0, abs(ymin) * 0.01); ymin -= padding; ymax += padding
    def px(value: float) -> float: return left + (value - xmin) / (xmax - xmin) * (width - left - right)
    def py(value: float) -> float: return top + (ymax - value) / (ymax - ymin) * (height - top - bottom)
    styles = ("#1f77b4", "#d62728", "#2ca02c", "#9467bd")
    lines = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">', '<rect width="100%" height="100%" fill="white"/>', f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="black"/>', f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="black"/>', f'<text x="{width/2}" y="{height-15}" text-anchor="middle" font-family="sans-serif" font-size="14">{x_label}</text>', f'<text x="18" y="{height/2}" transform="rotate(-90 18 {height/2})" text-anchor="middle" font-family="sans-serif" font-size="14">{y_label}</text>']
    for index, (values, label) in enumerate(zip(y_values, labels, strict=True)):
        points = " ".join(f"{px(float(xv)):.2f},{py(float(yv)):.2f}" for xv, yv in zip(x_values, values, strict=True) if np.isfinite(xv) and np.isfinite(yv)); colour = styles[index % len(styles)]
        lines.append(f'<polyline fill="none" stroke="{colour}" stroke-width="1.5" points="{points}"/>'); lines.append(f'<text x="{width-right-150}" y="{top+16*index}" font-family="sans-serif" font-size="11" fill="{colour}">{label}</text>')
    lines.append("</svg>"); path.write_text("\n".join(lines) + "\n", encoding="utf-8")
