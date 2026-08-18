from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

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
