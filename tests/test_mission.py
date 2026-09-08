from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import lunar_astrodynamics.mission as mission_module
from lunar_astrodynamics import SphericalHarmonicModel
from lunar_astrodynamics.cli import main as cli_main
from lunar_astrodynamics.mission import (
    build_mission_context,
    load_mission_config,
    mission_config_from_mapping,
    run_access_workflow,
    run_frozen_orbit_workflow,
    run_mission,
)


def _mapping() -> dict[str, object]:
    return {
        "mission": {"name": "test polar mission", "epoch_utc": "2026-08-18T00:00:00", "duration_s": 3600.0, "output_cadence_s": 120.0},
        "state": {"kind": "elements", "elements": {"semi_major_axis_altitude_m": 100000.0, "eccentricity": 0.005, "inclination_deg": 90.0, "raan_deg": 0.0, "argument_of_periapsis_deg": 270.0, "true_anomaly_deg": 0.0}},
        "gravity": {"model": "j2"},
        "access": {"minimum_elevation_deg": 10.0},
        "sites": [{"name": "equatorial site", "latitude_deg": 0.0, "longitude_deg_east": 0.0}],
        "coverage": {"enabled": True, "latitude_min_deg": -30.0, "latitude_max_deg": 30.0, "latitude_step_deg": 30.0, "longitude_min_deg_east": 0.0, "longitude_max_deg_east": 180.0, "longitude_step_deg": 60.0, "minimum_elevation_deg": 5.0},
        "integration": {"rtol": 1e-10, "position_atol_m": 1e-3, "velocity_atol_m_s": 1e-6, "max_step_s": 120.0},
    }


def test_config_build_and_mission_run() -> None:
    context = build_mission_context(mission_config_from_mapping(_mapping()))
    try:
        run = run_mission(context)
        assert run.time_s[0] == 0.0
        assert run.time_s[-1] == pytest.approx(3600.0)
        assert not run.impacted
        assert run.history.minimum_reference_altitude_m > 80000.0
        assert run.history.maximum_reference_altitude_m < 120000.0
        assert run.summary_dict()["force_model_fidelity"]["harmonic_degree"] == 2
        assert "test polar mission" in run.human_summary()
    finally:
        context.close()


def test_cartesian_state_alternative() -> None:
    mapping = _mapping()
    mapping["state"] = {"kind": "cartesian", "cartesian": [1837400.0, 0.0, 0.0, 0.0, 0.0, 1633.0]}
    context = build_mission_context(mission_config_from_mapping(mapping))
    try:
        assert context.initial_state.shape == (6,)
        assert np.all(np.isfinite(context.initial_state))
    finally:
        context.close()


def test_access_and_coverage_use_same_mission_trajectory() -> None:
    context = build_mission_context(mission_config_from_mapping(_mapping()))
    try:
        mission = run_mission(context)
        access = run_access_workflow(context, mission)
        assert access.ground_track.time_s.size == mission.time_s.size
        assert access.sites is not None
        assert len(access.sites.results) == 1
        assert access.coverage is not None
        assert access.coverage.dwell_time_s.shape == (3, 3)
        assert access.earth_visibility is None
    finally:
        context.close()


def test_tiny_configured_frozen_search() -> None:
    mapping = _mapping()
    mapping["search"] = {"semi_major_axis_altitudes_km": [95.0, 105.0], "eccentricities": [0.005], "inclinations_deg": [85.0, 95.0], "raan_deg": [0.0], "periapsis_deg": [90.0, 270.0], "initial_anomaly_deg": [0.0], "duration_s": 1800.0, "sample_count": 33, "workers": 1, "refine": False, "minimum_reference_altitude_km": 20.0}
    context = build_mission_context(mission_config_from_mapping(mapping))
    try:
        result = run_frozen_orbit_workflow(context)
        assert result.raw_grid_size == 8
        assert result.unique_candidate_count == 8
        assert len(result.candidates) == 8
        assert all(np.isfinite(candidate.ranking.penalty) for candidate in result.candidates)
    finally:
        context.close()


def test_shadr_configuration_requires_explicit_spice(tmp_path: Path) -> None:
    mapping = _mapping()
    mapping["gravity"] = {"model": "shadr", "path": str(tmp_path / "model.tab"), "degree": 20, "frame": "MOON_PA_DE421"}
    with pytest.raises(ValueError, match="SPICE"):
        build_mission_context(mission_config_from_mapping(mapping))


def test_cli_analyse_writes_standard_outputs(tmp_path: Path) -> None:
    config = tmp_path / "mission.toml"
    config.write_text(
        """
[mission]
name = "CLI polar mission"
epoch_utc = "2026-08-18T00:00:00"
duration_s = 1800.0
output_cadence_s = 120.0

[state]
kind = "elements"

[state.elements]
semi_major_axis_altitude_m = 100000.0
eccentricity = 0.005
inclination_deg = 90.0
raan_deg = 0.0
argument_of_periapsis_deg = 270.0
true_anomaly_deg = 0.0

[gravity]
model = "j2"

[integration]
rtol = 1e-10
position_atol_m = 1e-3
velocity_atol_m_s = 1e-6
max_step_s = 120.0
""".strip() + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "out"
    assert cli_main(["analyse", str(config), "--output-dir", str(output)]) == 0
    assert (output / "mission.json").exists()
    assert (output / "trajectory.csv").exists()
    assert (output / "summary.txt").exists()
    assert (output / "plots" / "altitude.svg").exists()
    payload = json.loads((output / "mission.json").read_text(encoding="utf-8"))
    assert payload["provenance"]["mission_name"] == "CLI polar mission"
    assert payload["summary"]["impact"] is False


def test_cli_rejects_missing_config(tmp_path: Path) -> None:
    assert cli_main(["analyse", str(tmp_path / "missing.toml"), "--output-dir", str(tmp_path / "out")]) == 2


def test_load_mission_config_resolves_relative_paths(tmp_path: Path) -> None:
    config = tmp_path / "mission.toml"
    config.write_text(
        """
[mission]
name = "path test"
epoch_utc = "2026-08-18T00:00:00"
duration_s = 600.0
output_cadence_s = 60.0

[state]
kind = "cartesian"
cartesian = [1837400.0, 0.0, 0.0, 0.0, 0.0, 1633.0]

[gravity]
model = "shadr"
path = "gravity.tab"
degree = 20
frame = "MOON_PA_DE421"
""".strip() + "\n",
        encoding="utf-8",
    )
    loaded = load_mission_config(config)
    assert loaded.gravity.path == (tmp_path / "gravity.tab").resolve()


def test_shadr_mission_retains_full_model_for_fidelity(monkeypatch, tmp_path: Path) -> None:
    c = np.zeros((5, 5))
    s = np.zeros_like(c)
    c[0, 0] = 1.0
    c[2, 0] = -9.0e-5
    c[4, 2] = 1.0e-7
    full_model = SphericalHarmonicModel(
        4.9028e12, 1.738e6, c, s, name="full synthetic field", frame="MOON_PA_DE421"
    )
    monkeypatch.setattr(mission_module, "read_shadr", lambda *args, **kwargs: full_model)
    monkeypatch.setattr(
        mission_module,
        "spice_rotation_provider",
        lambda *args, **kwargs: (lambda _time_s: np.eye(3)),
    )
    mapping = _mapping()
    mapping["gravity"] = {
        "model": "shadr",
        "path": str(tmp_path / "gravity.tab"),
        "degree": 2,
        "order": 2,
        "frame": "MOON_PA_DE421",
    }
    config = mission_config_from_mapping(mapping)
    dynamics, retained = mission_module._build_dynamics(
        config, SimpleNamespace(epoch_et_s=0.0), ()
    )
    assert retained is full_model
    assert retained.max_degree == 4
    assert dynamics.harmonic_degree == 2
    assert dynamics.harmonic_order == 2


def _spice_config(tmp_path: Path, *, kernels=None, **spice_options):
    mapping = _mapping()
    kernel = tmp_path / "mission.tf"
    kernel.write_text("KPL/FK\n\\begindata\nMISSION_TEST_VALUE = 42\n\\begintext\n")
    mapping["spice"] = {
        "enabled": True,
        "kernels": [str(kernel)] if kernels is None else kernels,
        "surface_frame": "IAU_MOON",
        **spice_options,
    }
    return mission_config_from_mapping(mapping)


def _fake_spice_epoch(monkeypatch):
    monkeypatch.setattr(
        mission_module, "spice_ephemeris_from_utc",
        lambda *args, **kwargs: SimpleNamespace(epoch_et_s=0.0),
    )


def test_mission_close_preserves_callers_kernels_and_is_idempotent(tmp_path, monkeypatch):
    spice = pytest.importorskip("spiceypy")
    config = _spice_config(tmp_path)
    _fake_spice_epoch(monkeypatch)
    path = str(config.spice.kernels[0])
    # A caller and two sequential contexts can use the same file.
    spice.furnsh(path)
    before = spice.ktotal("ALL")
    first = second = None
    try:
        first = build_mission_context(config)
        second = build_mission_context(config)
        assert spice.ktotal("ALL") == before + 2
        second.close()
        second.close()
        assert spice.ktotal("ALL") == before + 1
        first.close()
        assert spice.ktotal("ALL") == before
        assert spice.gdpool("MISSION_TEST_VALUE", 0, 1)[0] == 42
    finally:
        if second is not None:
            second.close()
        if first is not None:
            first.close()
        spice.unload(path)


@pytest.mark.parametrize("failure", ["missing_kernel", "initial_state", "partial_meta_kernel"])
def test_failed_mission_setup_preserves_callers_kernels(tmp_path, monkeypatch, failure):
    spice = pytest.importorskip("spiceypy")
    config = _spice_config(tmp_path)
    _fake_spice_epoch(monkeypatch)
    path = str(config.spice.kernels[0])
    spice.furnsh(path)
    before = spice.ktotal("ALL")
    try:
        if failure == "missing_kernel":
            config = _spice_config(tmp_path, kernels=[path, str(tmp_path / "missing.tf")])
        elif failure == "partial_meta_kernel":
            meta = tmp_path / "partial.tm"
            meta.write_text(
                "KPL/MK\n\\begindata\nKERNELS_TO_LOAD = (\n'"
                + path + "',\n'" + str(tmp_path / "missing.tf") + "'\n)\n\\begintext\n"
            )
            config = _spice_config(tmp_path, kernels=[str(meta)])
        else:
            monkeypatch.setattr(mission_module, "_initial_state", lambda *args: (_ for _ in ()).throw(ValueError("bad state")))
        with pytest.raises(Exception):
            build_mission_context(config)
        assert spice.ktotal("ALL") == before
        assert spice.gdpool("MISSION_TEST_VALUE", 0, 1)[0] == 42
    finally:
        spice.unload(path)


@pytest.mark.parametrize("failure", ["malformed_text", "partial_meta_kernel", "initial_state"])
def test_failed_mission_setup_restores_callers_kernel_pool(tmp_path, monkeypatch, failure):
    spice = pytest.importorskip("spiceypy")
    _fake_spice_epoch(monkeypatch)
    caller = tmp_path / "caller.tf"
    caller.write_text(
        "KPL/FK\n\\begindata\n"
        "MISSION_CALLER_NUMERIC = 1\n"
        "MISSION_CALLER_STRING = 'from file'\n\\begintext\n"
    )
    spice.furnsh(str(caller))
    # Caller overrides must survive reloading text kernels during UNLOAD.
    spice.pdpool("MISSION_CALLER_NUMERIC", [1.25, -3.0])
    spice.pcpool("MISSION_CALLER_STRING", ["caller", "override"])
    before = spice.ktotal("ALL")
    mission_kernel = tmp_path / "overrides.tf"
    assignments = (
        "KPL/FK\n\\begindata\n"
        "MISSION_CALLER_NUMERIC = (99, 100)\n"
        "MISSION_CALLER_STRING = ('mission', 'replacement')\n"
        "MISSION_NEW_NUMBER = 77\n"
        "MISSION_NEW_STRING = 'temporary'\n"
    )
    if failure == "malformed_text":
        # FURNSH retains the valid assignments, but never registers this file.
        mission_kernel.write_text(assignments + "BROKEN_VALUE = (1, 'unterminated\n")
        kernels = [str(mission_kernel)]
        expected_error = spice.SpiceTYPEMISMATCH
    elif failure == "partial_meta_kernel":
        mission_kernel.write_text(assignments + "\\begintext\n")
        meta = tmp_path / "partial_pool.tm"
        meta.write_text(
            "KPL/MK\n\\begindata\nKERNELS_TO_LOAD = (\n'"
            + str(mission_kernel) + "',\n'" + str(tmp_path / "missing.tf") + "'\n)\n\\begintext\n"
        )
        kernels = [str(meta)]
        expected_error = spice.SpiceNOSUCHFILE
    else:
        mission_kernel.write_text(assignments + "\\begintext\n")
        kernels = [str(mission_kernel)]
        expected_error = ValueError
        monkeypatch.setattr(
            mission_module, "_initial_state",
            lambda *args: (_ for _ in ()).throw(ValueError("bad state")),
        )
    try:
        with pytest.raises(expected_error):
            build_mission_context(_spice_config(tmp_path, kernels=kernels))
        assert spice.ktotal("ALL") == before
        assert spice.kdata(before - 1, "ALL")[0] == str(caller)
        np.testing.assert_array_equal(spice.gdpool("MISSION_CALLER_NUMERIC", 0, 2), [1.25, -3.0])
        assert spice.gcpool("MISSION_CALLER_STRING", 0, 2) == ["caller", "override"]
        for name in ("MISSION_NEW_NUMBER", "MISSION_NEW_STRING"):
            with pytest.raises(spice.NotFoundError):
                spice.dtpool(name)
    finally:
        spice.unload(str(caller))
        for name in ("MISSION_CALLER_NUMERIC", "MISSION_CALLER_STRING", "MISSION_NEW_NUMBER", "MISSION_NEW_STRING"):
            spice.dvpool(name)


@pytest.mark.parametrize("options, message", [
    ({"observer": "EARTH"}, "observer"),
    ({"inertial_frame": "IAU_MOON"}, "inertial"),
    ({"inertial_frame": "UNKNOWN_TEST_FRAME"}, "inertial"),
])
def test_mission_rejects_incompatible_spice_origin_and_axes(tmp_path, monkeypatch, options, message):
    spice = pytest.importorskip("spiceypy")
    config = _spice_config(tmp_path, **options)
    _fake_spice_epoch(monkeypatch)
    before = spice.ktotal("ALL")
    with pytest.raises(ValueError, match=message):
        build_mission_context(config)
    assert spice.ktotal("ALL") == before


def test_mission_accepts_moon_id_and_alternative_inertial_frame(tmp_path, monkeypatch):
    pytest.importorskip("spiceypy")
    _fake_spice_epoch(monkeypatch)
    context = build_mission_context(_spice_config(tmp_path, observer="301", inertial_frame="ECLIPJ2000"))
    context.close()


def test_terrain_frame_override_cannot_bypass_surface_rotation_check(tmp_path, monkeypatch):
    mapping = _mapping()
    mapping["terrain"] = {"kind": "npz", "path": str(tmp_path / "terrain.npz"), "frame": "MOON_PA_DE421"}
    config = mission_config_from_mapping(mapping)
    monkeypatch.setattr(mission_module, "load_terrain_npz", lambda _: SimpleNamespace(frame="MOON_PA_DE421"))
    with pytest.raises(ValueError, match="must match configured surface frame"):
        mission_module._load_terrain(config, "MOON_ME_DE421")
