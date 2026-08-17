import sys
from types import SimpleNamespace

import numpy as np
import pytest

from lunar_astrodynamics.ephemeris import (
    SpiceEphemeris,
    loaded_spice_kernels,
    spice_ephemeris_from_et,
    spice_ephemeris_from_utc,
)


def _fake_spice() -> SimpleNamespace:
    calls: list[tuple[object, ...]] = []

    def ktotal(kind: str) -> int:
        assert kind == "ALL"
        return 2

    def kdata(index: int, kind: str):
        assert kind == "ALL"
        values = [
            ("/kernels/naif0012.tls", "TEXT", "", 0),
            ("/kernels/de440s.bsp", "SPK", "", 1),
        ]
        return values[index]

    def str2et(epoch: str) -> float:
        calls.append(("str2et", epoch))
        return 123_456.0

    def spkpos(target: str, et: float, frame: str, abcorr: str, observer: str):
        calls.append(("spkpos", target, et, frame, abcorr, observer))
        return np.array([1.0, 2.0, 3.0]), 1.23

    return SimpleNamespace(
        calls=calls,
        ktotal=ktotal,
        kdata=kdata,
        str2et=str2et,
        spkpos=spkpos,
    )


def test_loaded_spice_kernels_snapshots_paths_types_and_sources(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _fake_spice()
    monkeypatch.setitem(sys.modules, "spiceypy", fake)
    records = loaded_spice_kernels()
    assert [record.path for record in records] == [
        "/kernels/naif0012.tls",
        "/kernels/de440s.bsp",
    ]
    assert [record.kind for record in records] == ["TEXT", "SPK"]


def test_spice_utc_context_uses_explicit_epoch_geometric_positions_and_si_units(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _fake_spice()
    monkeypatch.setitem(sys.modules, "spiceypy", fake)
    context = spice_ephemeris_from_utc(
        "2026-08-17T00:00:00",
        inertial_frame="J2000",
        observer="MOON",
    )

    position = context.position_m("EARTH", 12.5)
    assert position == pytest.approx([1000.0, 2000.0, 3000.0])
    assert ("str2et", "2026-08-17T00:00:00") in fake.calls
    assert (
        "spkpos",
        "EARTH",
        123_468.5,
        "J2000",
        "NONE",
        "MOON",
    ) in fake.calls

    provenance = context.provenance()
    assert provenance["epoch_utc"] == "2026-08-17T00:00:00"
    assert provenance["epoch_et_s"] == pytest.approx(123_456.0)
    assert provenance["inertial_frame"] == "J2000"
    assert provenance["observer"] == "MOON"
    assert provenance["aberration_correction"] == "NONE"
    assert len(provenance["kernels"]) == 2


def test_spice_position_provider_records_target(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _fake_spice()
    monkeypatch.setitem(sys.modules, "spiceypy", fake)
    context = spice_ephemeris_from_utc("2026-08-17T00:00:00")
    provider = context.position_provider("SUN")
    assert provider(4.0) == pytest.approx([1000.0, 2000.0, 3000.0])
    assert provider.provenance()["target"] == "SUN"


def test_et_context_can_be_created_without_kernel_snapshot() -> None:
    context = spice_ephemeris_from_et(
        42.0,
        inertial_frame="ECLIPJ2000",
        snapshot_kernels=False,
    )
    assert context.epoch_et_s == pytest.approx(42.0)
    assert context.epoch_utc is None
    assert context.inertial_frame == "ECLIPJ2000"
    assert context.kernels == ()


def test_force_model_ephemeris_rejects_apparent_aberration_corrections() -> None:
    with pytest.raises(ValueError, match="geometric SPICE positions"):
        SpiceEphemeris(
            0.0,
            inertial_frame="J2000",
            observer="MOON",
            aberration_correction="LT+S",
        )
