"""Composable lunar force components for perturbation and mission analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping, Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .constants import (
    ASTRONOMICAL_UNIT_M,
    MOON_MEAN_RADIUS_M,
    SOLAR_RADIATION_PRESSURE_1_AU_N_M2,
    SUN_NOMINAL_RADIUS_M,
)

FloatArray = NDArray[np.float64]
PositionProvider = Callable[[float], FloatArray]
AccelerationFunction = Callable[[float, FloatArray], FloatArray]


class ForceComponent(Protocol):
    """Minimal named acceleration component interface."""

    name: str

    def __call__(self, time_s: float, position_m: FloatArray) -> FloatArray: ...


def _position_vector(value: ArrayLike, *, name: str) -> FloatArray:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be a finite three-vector")
    return vector


def _validated_acceleration(value: ArrayLike, *, name: str) -> FloatArray:
    acceleration = np.asarray(value, dtype=float)
    if acceleration.shape != (3,) or not np.all(np.isfinite(acceleration)):
        raise ValueError(f"force component '{name}' returned an invalid acceleration")
    return acceleration


@dataclass(frozen=True)
class CallableForce:
    """Name and document an existing acceleration callable for composition."""

    name: str
    acceleration: AccelerationFunction
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("force name must be non-empty")

    def __call__(self, time_s: float, position_m: FloatArray) -> FloatArray:
        return _validated_acceleration(
            self.acceleration(float(time_s), _position_vector(position_m, name="position_m")),
            name=self.name,
        )

    def provenance(self) -> dict[str, object]:
        return {"name": self.name, **dict(self.metadata)}


@dataclass(frozen=True)
class CompositeForceModel:
    """Sum independent named force components without coupling their implementations."""

    components: tuple[ForceComponent, ...]
    name: str = "composite lunar force model"

    def __post_init__(self) -> None:
        if not self.components:
            raise ValueError("a composite force model requires at least one component")
        names = [component.name for component in self.components]
        if any(not item for item in names):
            raise ValueError("all force components must have non-empty names")
        if len(names) != len(set(names)):
            raise ValueError("force component names must be unique")

    def component_accelerations(
        self, time_s: float, position_m: ArrayLike
    ) -> dict[str, FloatArray]:
        position = _position_vector(position_m, name="position_m")
        return {
            component.name: _validated_acceleration(
                component(float(time_s), position), name=component.name
            )
            for component in self.components
        }

    def __call__(self, time_s: float, position_m: FloatArray) -> FloatArray:
        contributions = self.component_accelerations(time_s, position_m)
        return np.sum(np.stack(tuple(contributions.values()), axis=0), axis=0)

    def provenance(self) -> dict[str, object]:
        components: list[dict[str, object]] = []
        for component in self.components:
            provenance_method = getattr(component, "provenance", None)
            if callable(provenance_method):
                value = dict(provenance_method())
            else:
                value = {"name": component.name}
            components.append(value)
        return {"name": self.name, "components": components}


def third_body_acceleration(
    spacecraft_position_m: ArrayLike,
    third_body_position_m: ArrayLike,
    gravitational_parameter_m3_s2: float,
) -> FloatArray:
    """Differential third-body acceleration in a central-body-centred frame.

    ``third_body_position_m`` is the disturbing body's position relative to the
    central body.  The returned acceleration subtracts the acceleration of the
    central body itself, so it is suitable for Moon-centred relative dynamics.
    """
    spacecraft = _position_vector(spacecraft_position_m, name="spacecraft_position_m")
    third_body = _position_vector(third_body_position_m, name="third_body_position_m")
    mu = float(gravitational_parameter_m3_s2)
    if not np.isfinite(mu) or mu <= 0.0:
        raise ValueError("gravitational_parameter_m3_s2 must be finite and positive")

    spacecraft_to_body = third_body - spacecraft
    body_distance = float(np.linalg.norm(third_body))
    relative_distance = float(np.linalg.norm(spacecraft_to_body))
    if body_distance == 0.0 or relative_distance == 0.0:
        raise ValueError("third-body geometry is singular")

    return mu * (
        spacecraft_to_body / relative_distance**3
        - third_body / body_distance**3
    )


@dataclass(frozen=True)
class ThirdBodyGravity:
    """Point-mass third-body perturbation driven by a body-position provider."""

    name: str
    gravitational_parameter_m3_s2: float
    third_body_position_m: PositionProvider

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("force name must be non-empty")
        if (
            not np.isfinite(self.gravitational_parameter_m3_s2)
            or self.gravitational_parameter_m3_s2 <= 0.0
        ):
            raise ValueError("gravitational_parameter_m3_s2 must be finite and positive")

    def __call__(self, time_s: float, position_m: FloatArray) -> FloatArray:
        body_position = self.third_body_position_m(float(time_s))
        return third_body_acceleration(
            position_m,
            body_position,
            self.gravitational_parameter_m3_s2,
        )

    def provenance(self) -> dict[str, object]:
        result: dict[str, object] = {
            "name": self.name,
            "type": "differential point-mass third-body gravity",
            "gravitational_parameter_m3_s2": float(self.gravitational_parameter_m3_s2),
        }
        provider_provenance = getattr(self.third_body_position_m, "provenance", None)
        if callable(provider_provenance):
            result["position_provider"] = provider_provenance()
        return result


def apparent_disk_illumination_fraction(
    solar_angular_radius_rad: float,
    occulting_angular_radius_rad: float,
    angular_separation_rad: float,
) -> float:
    """Visible solar-disk fraction for two overlapping apparent circular disks."""
    sun_radius = float(solar_angular_radius_rad)
    body_radius = float(occulting_angular_radius_rad)
    separation = float(angular_separation_rad)
    if not all(np.isfinite(value) for value in (sun_radius, body_radius, separation)):
        raise ValueError("apparent disk angles must be finite")
    if sun_radius <= 0.0 or body_radius < 0.0:
        raise ValueError("solar angular radius must be positive and occulting radius non-negative")
    if separation < 0.0 or separation > np.pi:
        raise ValueError("angular separation must lie within [0, pi]")

    if body_radius == 0.0 or separation >= sun_radius + body_radius:
        return 1.0
    if body_radius >= sun_radius + separation:
        return 0.0
    if sun_radius >= body_radius + separation:
        return float(np.clip(1.0 - (body_radius / sun_radius) ** 2, 0.0, 1.0))

    d = separation
    r_s = sun_radius
    r_b = body_radius
    sun_argument = np.clip((d * d + r_s * r_s - r_b * r_b) / (2.0 * d * r_s), -1.0, 1.0)
    body_argument = np.clip((d * d + r_b * r_b - r_s * r_s) / (2.0 * d * r_b), -1.0, 1.0)
    radicand = max(
        0.0,
        (-d + r_s + r_b)
        * (d + r_s - r_b)
        * (d - r_s + r_b)
        * (d + r_s + r_b),
    )
    overlap = (
        r_s * r_s * np.arccos(sun_argument)
        + r_b * r_b * np.arccos(body_argument)
        - 0.5 * np.sqrt(radicand)
    )
    visible = 1.0 - overlap / (np.pi * r_s * r_s)
    return float(np.clip(visible, 0.0, 1.0))


def lunar_eclipse_illumination_fraction(
    spacecraft_position_m: ArrayLike,
    sun_position_m: ArrayLike,
    *,
    moon_radius_m: float = MOON_MEAN_RADIUS_M,
    sun_radius_m: float = SUN_NOMINAL_RADIUS_M,
) -> float:
    """Solar illumination fraction for a spherical Moon occulting a spherical Sun."""
    spacecraft = _position_vector(spacecraft_position_m, name="spacecraft_position_m")
    sun = _position_vector(sun_position_m, name="sun_position_m")
    moon_radius = float(moon_radius_m)
    sun_radius = float(sun_radius_m)
    if not np.isfinite(moon_radius) or moon_radius <= 0.0:
        raise ValueError("moon_radius_m must be finite and positive")
    if not np.isfinite(sun_radius) or sun_radius <= 0.0:
        raise ValueError("sun_radius_m must be finite and positive")

    spacecraft_to_moon = -spacecraft
    spacecraft_to_sun = sun - spacecraft
    moon_distance = float(np.linalg.norm(spacecraft_to_moon))
    sun_distance = float(np.linalg.norm(spacecraft_to_sun))
    if moon_distance <= moon_radius:
        raise ValueError("spacecraft must be outside the spherical Moon")
    if sun_distance <= sun_radius:
        raise ValueError("spacecraft must be outside the spherical Sun")
    if moon_distance >= sun_distance:
        return 1.0

    solar_angular_radius = float(np.arcsin(np.clip(sun_radius / sun_distance, 0.0, 1.0)))
    moon_angular_radius = float(np.arcsin(np.clip(moon_radius / moon_distance, 0.0, 1.0)))
    cosine_separation = float(
        np.dot(spacecraft_to_sun, spacecraft_to_moon) / (sun_distance * moon_distance)
    )
    angular_separation = float(np.arccos(np.clip(cosine_separation, -1.0, 1.0)))
    return apparent_disk_illumination_fraction(
        solar_angular_radius,
        moon_angular_radius,
        angular_separation,
    )


@dataclass(frozen=True)
class SolarRadiationPressure:
    """Cannonball SRP with optional finite-disk lunar eclipse attenuation."""

    name: str
    sun_position_m: PositionProvider
    spacecraft_mass_kg: float
    illuminated_area_m2: float
    reflectivity_coefficient: float
    pressure_1_au_n_m2: float = SOLAR_RADIATION_PRESSURE_1_AU_N_M2
    astronomical_unit_m: float = ASTRONOMICAL_UNIT_M
    sun_radius_m: float = SUN_NOMINAL_RADIUS_M
    moon_radius_m: float = MOON_MEAN_RADIUS_M
    include_lunar_shadow: bool = True

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("force name must be non-empty")
        if not np.isfinite(self.spacecraft_mass_kg) or self.spacecraft_mass_kg <= 0.0:
            raise ValueError("spacecraft_mass_kg must be finite and positive")
        if not np.isfinite(self.illuminated_area_m2) or self.illuminated_area_m2 < 0.0:
            raise ValueError("illuminated_area_m2 must be finite and non-negative")
        if (
            not np.isfinite(self.reflectivity_coefficient)
            or self.reflectivity_coefficient < 0.0
        ):
            raise ValueError("reflectivity_coefficient must be finite and non-negative")
        if not np.isfinite(self.pressure_1_au_n_m2) or self.pressure_1_au_n_m2 <= 0.0:
            raise ValueError("pressure_1_au_n_m2 must be finite and positive")
        if not np.isfinite(self.astronomical_unit_m) or self.astronomical_unit_m <= 0.0:
            raise ValueError("astronomical_unit_m must be finite and positive")

    def illumination_fraction(self, time_s: float, position_m: ArrayLike) -> float:
        if not self.include_lunar_shadow:
            return 1.0
        return lunar_eclipse_illumination_fraction(
            position_m,
            self.sun_position_m(float(time_s)),
            moon_radius_m=self.moon_radius_m,
            sun_radius_m=self.sun_radius_m,
        )

    def __call__(self, time_s: float, position_m: FloatArray) -> FloatArray:
        spacecraft = _position_vector(position_m, name="position_m")
        sun = _position_vector(self.sun_position_m(float(time_s)), name="sun_position_m")
        sun_to_spacecraft = spacecraft - sun
        distance = float(np.linalg.norm(sun_to_spacecraft))
        if distance == 0.0:
            raise ValueError("Sun-spacecraft geometry is singular")
        illumination = self.illumination_fraction(time_s, spacecraft)
        pressure = self.pressure_1_au_n_m2 * (self.astronomical_unit_m / distance) ** 2
        magnitude = (
            illumination
            * pressure
            * self.reflectivity_coefficient
            * self.illuminated_area_m2
            / self.spacecraft_mass_kg
        )
        return float(magnitude) * sun_to_spacecraft / distance

    def provenance(self) -> dict[str, object]:
        result: dict[str, object] = {
            "name": self.name,
            "type": "cannonball solar radiation pressure",
            "spacecraft_mass_kg": float(self.spacecraft_mass_kg),
            "illuminated_area_m2": float(self.illuminated_area_m2),
            "reflectivity_coefficient": float(self.reflectivity_coefficient),
            "pressure_1_au_n_m2": float(self.pressure_1_au_n_m2),
            "astronomical_unit_m": float(self.astronomical_unit_m),
            "sun_radius_m": float(self.sun_radius_m),
            "moon_radius_m": float(self.moon_radius_m),
            "include_lunar_shadow": bool(self.include_lunar_shadow),
            "shadow_model": "finite apparent Sun/Moon disk overlap" if self.include_lunar_shadow else "none",
        }
        provider_provenance = getattr(self.sun_position_m, "provenance", None)
        if callable(provider_provenance):
            result["position_provider"] = provider_provenance()
        return result
