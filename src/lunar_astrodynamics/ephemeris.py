"""Explicit SPICE ephemeris access with reproducibility metadata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _spice_module() -> Any:
    try:
        import spiceypy as spice  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "SPICE support requires 'pip install lunar-astrodynamics-simulation[spice]'"
        ) from exc
    return spice


@dataclass(frozen=True)
class SpiceKernelRecord:
    """One kernel visible in the SPICE kernel pool when a context was created."""

    path: str
    kind: str
    source: str

    def as_dict(self) -> dict[str, str]:
        return {"path": self.path, "kind": self.kind, "source": self.source}


def loaded_spice_kernels() -> tuple[SpiceKernelRecord, ...]:
    """Snapshot all kernels currently loaded through SPICE ``furnsh`` calls."""
    spice = _spice_module()
    records: list[SpiceKernelRecord] = []
    for index in range(int(spice.ktotal("ALL"))):
        data = spice.kdata(index, "ALL")
        if len(data) < 3:
            raise RuntimeError("unexpected SpiceyPy kdata return value")
        records.append(SpiceKernelRecord(str(data[0]), str(data[1]), str(data[2])))
    return tuple(records)


@dataclass(frozen=True)
class SpiceEphemeris:
    """Moon-centred SPICE position context for deterministic force evaluation.

    ``epoch_et_s`` is SPICE ephemeris time in seconds past J2000 TDB.  Dynamics
    time ``time_s`` is added directly to that epoch.  The default aberration
    correction is ``NONE`` because gravitational and SRP force models require
    geometric, same-epoch positions rather than apparent light-time-corrected
    directions.
    """

    epoch_et_s: float
    inertial_frame: str = "J2000"
    observer: str = "MOON"
    aberration_correction: str = "NONE"
    epoch_utc: str | None = None
    kernels: tuple[SpiceKernelRecord, ...] = ()

    def __post_init__(self) -> None:
        if not np.isfinite(self.epoch_et_s):
            raise ValueError("epoch_et_s must be finite")
        if not self.inertial_frame:
            raise ValueError("inertial_frame must be non-empty")
        if not self.observer:
            raise ValueError("observer must be non-empty")
        if self.aberration_correction.upper() != "NONE":
            raise ValueError(
                "force-model ephemerides require geometric SPICE positions with aberration_correction='NONE'"
            )
        object.__setattr__(self, "aberration_correction", "NONE")

    def ephemeris_time_s(self, time_s: float) -> float:
        if not np.isfinite(time_s):
            raise ValueError("time_s must be finite")
        return float(self.epoch_et_s + time_s)

    def position_m(self, target: str, time_s: float) -> FloatArray:
        """Return target position relative to ``observer`` in the inertial frame."""
        if not target:
            raise ValueError("target must be non-empty")
        spice = _spice_module()
        position_km, _light_time_s = spice.spkpos(
            target,
            self.ephemeris_time_s(time_s),
            self.inertial_frame,
            self.aberration_correction,
            self.observer,
        )
        position = np.asarray(position_km, dtype=float) * 1000.0
        if position.shape != (3,) or not np.all(np.isfinite(position)):
            raise ValueError("SPICE returned an invalid position vector")
        return position

    def position_provider(self, target: str) -> "SpicePositionProvider":
        return SpicePositionProvider(target=target, ephemeris=self)

    def provenance(self) -> dict[str, object]:
        return {
            "source": "SPICE",
            "epoch_et_s": float(self.epoch_et_s),
            "epoch_utc": self.epoch_utc,
            "inertial_frame": self.inertial_frame,
            "observer": self.observer,
            "aberration_correction": self.aberration_correction,
            "kernels": [record.as_dict() for record in self.kernels],
        }


@dataclass(frozen=True)
class SpicePositionProvider:
    """Callable SPICE target-position provider carrying target provenance."""

    target: str
    ephemeris: SpiceEphemeris

    def __post_init__(self) -> None:
        if not self.target:
            raise ValueError("target must be non-empty")

    def __call__(self, time_s: float) -> FloatArray:
        return self.ephemeris.position_m(self.target, time_s)

    def provenance(self) -> dict[str, object]:
        result = self.ephemeris.provenance()
        result["target"] = self.target
        return result


def spice_ephemeris_from_et(
    epoch_et_s: float,
    *,
    inertial_frame: str = "J2000",
    observer: str = "MOON",
    snapshot_kernels: bool = True,
) -> SpiceEphemeris:
    """Create a geometric SPICE context from an explicit ET epoch."""
    kernels = loaded_spice_kernels() if snapshot_kernels else ()
    return SpiceEphemeris(
        float(epoch_et_s),
        inertial_frame=inertial_frame,
        observer=observer,
        aberration_correction="NONE",
        epoch_utc=None,
        kernels=kernels,
    )


def spice_ephemeris_from_utc(
    epoch_utc: str,
    *,
    inertial_frame: str = "J2000",
    observer: str = "MOON",
    snapshot_kernels: bool = True,
) -> SpiceEphemeris:
    """Create a geometric SPICE context from a UTC epoch using a loaded LSK."""
    if not epoch_utc:
        raise ValueError("epoch_utc must be non-empty")
    spice = _spice_module()
    epoch_et_s = float(spice.str2et(epoch_utc))
    kernels = loaded_spice_kernels() if snapshot_kernels else ()
    return SpiceEphemeris(
        epoch_et_s,
        inertial_frame=inertial_frame,
        observer=observer,
        aberration_correction="NONE",
        epoch_utc=epoch_utc,
        kernels=kernels,
    )
