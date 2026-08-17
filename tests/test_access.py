import json

import numpy as np
import pytest

from lunar_astrodynamics.access import (
    CoverageGrid,
    LunarSurfaceSite,
    analyze_earth_visibility,
    analyze_multiple_site_access,
    analyze_site_access,
    coverage_analysis,
    earth_visibility_fraction,
    ground_track_history,
    site_observation,
)
from lunar_astrodynamics.constants import MOON_MEAN_RADIUS_M
from lunar_astrodynamics.terrain import RegularLatLonTerrain

R = MOON_MEAN_RADIUS_M
FRAME = "TEST_BODY_FIXED"
IDENTITY = lambda _time_s: np.eye(3)


def _position(latitude_deg: float, longitude_deg: float, altitude_m: float) -> np.ndarray:
    latitude = np.deg2rad(latitude_deg)
    longitude = np.deg2rad(longitude_deg)
    radius = R + altitude_m
    return radius * np.array(
        [
            np.cos(latitude) * np.cos(longitude),
            np.cos(latitude) * np.sin(longitude),
            np.sin(latitude),
        ]
    )


def test_ground_track_cardinal_longitudes_and_altitude() -> None:
    times = np.arange(4.0)
    positions = np.array(
        [
            _position(0.0, 0.0, 100_000.0),
            _position(0.0, 90.0, 100_000.0),
            _position(0.0, 180.0, 100_000.0),
            _position(0.0, 270.0, 100_000.0),
        ]
    )
    track = ground_track_history(
        times,
        positions,
        IDENTITY,
        body_fixed_frame=FRAME,
    )
    np.testing.assert_allclose(track.latitude_deg, 0.0, atol=1e-12)
    np.testing.assert_allclose(track.longitude_deg_east, [0.0, 90.0, 180.0, 270.0], atol=1e-12)
    np.testing.assert_allclose(track.reference_altitude_m, 100_000.0, atol=1e-9)


def test_ground_track_wraps_longitude_and_unwraps_continuously() -> None:
    track = ground_track_history(
        [0.0, 1.0, 2.0],
        np.array(
            [
                _position(0.0, 358.0, 50_000.0),
                _position(0.0, 359.0, 50_000.0),
                _position(0.0, 1.0, 50_000.0),
            ]
        ),
        IDENTITY,
        body_fixed_frame=FRAME,
    )
    np.testing.assert_allclose(track.longitude_deg_east, [358.0, 359.0, 1.0], atol=1e-12)
    np.testing.assert_allclose(track.unwrapped_longitude_deg_east, [358.0, 359.0, 361.0], atol=1e-12)


def test_ground_track_does_not_invent_longitude_at_exact_pole() -> None:
    track = ground_track_history(
        [0.0],
        [_position(90.0, 123.0, 80_000.0)],
        IDENTITY,
        body_fixed_frame=FRAME,
    )
    assert track.latitude_deg[0] == pytest.approx(90.0)
    assert np.isnan(track.longitude_deg_east[0])
    assert track.to_dict()["longitude_deg_east"] == [None]


def test_site_observation_overhead_and_far_side_geometry() -> None:
    site = LunarSurfaceSite("equator", 0.0, 0.0, frame=FRAME)
    overhead = site_observation(
        0.0,
        _position(0.0, 0.0, 100_000.0),
        site,
        IDENTITY,
        body_fixed_frame=FRAME,
    )
    assert overhead.visible
    assert overhead.elevation_deg == pytest.approx(90.0, abs=1e-12)
    assert overhead.slant_range_m == pytest.approx(100_000.0, abs=1e-9)
    assert overhead.spherical_los_clear

    far_side = site_observation(
        0.0,
        _position(0.0, 180.0, 100_000.0),
        site,
        IDENTITY,
        body_fixed_frame=FRAME,
    )
    assert not far_side.visible
    assert far_side.elevation_deg < 0.0
    assert not far_side.spherical_los_clear


def test_terrain_aware_line_of_sight_can_block_local_horizon_access() -> None:
    elevation = np.zeros((3, 4), dtype=float)
    elevation[1, 1] = 120_000.0
    terrain = RegularLatLonTerrain(
        np.array([-90.0, 0.0, 90.0]),
        np.array([0.0, 5.0, 10.0, 360.0]),
        elevation,
        reference_radius_m=R,
        name="synthetic ridge",
        frame=FRAME,
    )
    site = LunarSurfaceSite("site", 0.0, 0.0, frame=FRAME)
    spacecraft = _position(0.0, 10.0, 100_000.0)
    local = site_observation(
        0.0,
        spacecraft,
        site,
        IDENTITY,
        body_fixed_frame=FRAME,
    )
    terrain_aware = site_observation(
        0.0,
        spacecraft,
        site,
        IDENTITY,
        body_fixed_frame=FRAME,
        terrain=terrain,
        terrain_aware=True,
        terrain_los_samples=100,
    )
    assert local.visible
    assert local.elevation_deg > 0.0
    assert not terrain_aware.visible
    assert terrain_aware.terrain_los_clear is False


def test_access_windows_report_closest_approach_and_duration() -> None:
    times = np.array([0.0, 10.0, 20.0, 30.0, 40.0])
    positions = np.array(
        [
            _position(0.0, 60.0, 500_000.0),
            _position(0.0, 30.0, 500_000.0),
            _position(0.0, 0.0, 500_000.0),
            _position(0.0, 330.0, 500_000.0),
            _position(0.0, 300.0, 500_000.0),
        ]
    )
    result = analyze_site_access(
        times,
        positions,
        LunarSurfaceSite("equator", 0.0, 0.0, frame=FRAME),
        IDENTITY,
        body_fixed_frame=FRAME,
    )
    assert len(result.windows) == 1
    window = result.windows[0]
    assert 0.0 < window.start_time_s < 20.0
    assert 20.0 < window.end_time_s < 40.0
    assert window.duration_s > 0.0
    assert window.maximum_elevation_deg == pytest.approx(90.0, abs=1e-12)
    assert window.maximum_elevation_time_s == pytest.approx(20.0)
    assert window.minimum_slant_range_m == pytest.approx(500_000.0, abs=1e-9)
    assert window.closest_approach_time_s == pytest.approx(20.0)
    assert result.revisit_intervals_s == ()


def test_multiple_sites_are_analyzed_independently() -> None:
    times = [0.0, 100.0]
    positions = np.array([_position(0.0, 0.0, 100_000.0)] * 2)
    report = analyze_multiple_site_access(
        times,
        positions,
        (
            LunarSurfaceSite("near", 0.0, 0.0, frame=FRAME),
            LunarSurfaceSite("far", 0.0, 180.0, frame=FRAME),
        ),
        IDENTITY,
        body_fixed_frame=FRAME,
    )
    by_name = {result.site.name: result for result in report.results}
    assert by_name["near"].access_fraction == pytest.approx(1.0)
    assert by_name["far"].access_fraction == pytest.approx(0.0)


def test_coverage_reports_dwell_fraction_and_cell_metrics() -> None:
    times = [0.0, 100.0]
    positions = np.array([_position(0.0, 0.0, 100_000.0)] * 2)
    grid = CoverageGrid(
        np.array([0.0]),
        np.array([0.0, 180.0]),
        FRAME,
        R,
    )
    result = coverage_analysis(
        times,
        positions,
        grid,
        IDENTITY,
        body_fixed_frame=FRAME,
    )
    assert result.covered_fraction == pytest.approx(0.5)
    assert result.dwell_time_s[0, 0] == pytest.approx(100.0)
    assert result.dwell_time_s[0, 1] == pytest.approx(0.0)
    assert result.access_count[0, 0] == 1
    assert result.access_count[0, 1] == 0
    assert result.maximum_elevation_deg[0, 0] == pytest.approx(90.0)
    assert np.isnan(result.maximum_elevation_deg[0, 1])


def test_earth_visibility_fraction_known_near_and_far_side_geometry() -> None:
    earth = np.array([384_400_000.0, 0.0, 0.0])
    near_side_spacecraft = np.array([R + 100_000.0, 0.0, 0.0])
    far_side_spacecraft = np.array([-(R + 100_000.0), 0.0, 0.0])
    assert earth_visibility_fraction(near_side_spacecraft, earth) == pytest.approx(1.0)
    assert earth_visibility_fraction(far_side_spacecraft, earth) == pytest.approx(0.0)


def test_earth_visibility_report_uses_position_provider_and_occultation_windows() -> None:
    class EarthProvider:
        def __call__(self, _time_s: float) -> np.ndarray:
            return np.array([384_400_000.0, 0.0, 0.0])

        def provenance(self) -> dict[str, object]:
            return {"source": "analytic Earth fixture"}

    times = [0.0, 10.0, 20.0]
    positions = np.array(
        [
            [-(R + 100_000.0), 0.0, 0.0],
            [R + 100_000.0, 0.0, 0.0],
            [R + 100_000.0, 0.0, 0.0],
        ]
    )
    report = analyze_earth_visibility(times, positions, EarthProvider())
    np.testing.assert_allclose(report.visible_fraction, [0.0, 1.0, 1.0])
    assert report.position_provider_provenance == {"source": "analytic Earth fixture"}
    assert len(report.visible_windows) == 1
    assert 0.0 < report.visible_windows[0].start_time_s < 10.0
    assert report.visible_windows[0].end_time_s == pytest.approx(20.0)


def test_structured_exports_are_valid_json_and_csv(tmp_path) -> None:
    times = [0.0, 100.0]
    positions = np.array([_position(0.0, 0.0, 100_000.0)] * 2)
    access = analyze_multiple_site_access(
        times,
        positions,
        (LunarSurfaceSite("site", 0.0, 0.0, frame=FRAME),),
        IDENTITY,
        body_fixed_frame=FRAME,
    )
    json_path = tmp_path / "access.json"
    csv_path = tmp_path / "access.csv"
    access.write_json(json_path)
    access.write_windows_csv(csv_path)
    payload = json.loads(json_path.read_text())
    assert payload["sites"][0]["site"]["name"] == "site"
    assert "start_time_s" in csv_path.read_text()

    coverage = coverage_analysis(
        times,
        positions,
        CoverageGrid(np.array([0.0]), np.array([0.0]), FRAME, R),
        IDENTITY,
        body_fixed_frame=FRAME,
    )
    coverage_json = tmp_path / "coverage.json"
    coverage_csv = tmp_path / "coverage.csv"
    coverage.write_json(coverage_json)
    coverage.write_csv(coverage_csv)
    json.loads(coverage_json.read_text())
    assert "dwell_time_s" in coverage_csv.read_text()
